"""
OpenAI-compatible STT server using FunASR.

Supports SenseVoice (with rich_transcription_postprocess) and
Paraformer (with ct-punc punctuation model) architectures.

Endpoint: POST /v1/audio/transcriptions
Compatible with VoiceMode MCP and OpenAI Python SDK.
"""

import os
import json
import subprocess
import tomllib
import tempfile
import logging
from datetime import date

import httpx
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import PlainTextResponse, JSONResponse
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("funasr-server")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.INFO)
    _handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG_PATH = os.environ.get("CONFIG_PATH", "/app/config.toml")

_DEFAULTS = {
    "funasr": {
        "model": "iic/SenseVoiceSmall",
        "punc_model": "",
        "vad_model": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "device": "cpu",
        "host": "0.0.0.0",
        "port": 2023,
    },
    "azure": {
        "enabled": False,
        "region": "japaneast",
        "key_env": "AZURE_SPEECH_KEY",
    },
    "llm": {
        "enabled": False,
        "model": "qwen3.5:4b",
        "base_url": "http://localhost:11434",
        "timeout": 15.0,
        "prompt": {
            "system": (
                "你是一个专业的语音转文字后处理助手。请对以下 ASR 转录文本进行润色：\n"
                "1. 修正中英文错别字（保持原意，不要改写）\n"
                "2. 添加合适的标点符号（逗号、句号、问号等）\n"
                "3. 合理断句\n"
                "只输出润色后的文本，不要添加任何解释。"
            )
        },
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config() -> dict:
    cfg = _DEFAULTS
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "rb") as f:
            user_cfg = tomllib.load(f)
        cfg = _deep_merge(_DEFAULTS, user_cfg)
        logger.info("Loaded config from %s", CONFIG_PATH)
    else:
        logger.info("No config file at %s, using defaults", CONFIG_PATH)
    return cfg


cfg = load_config()
funasr_cfg = cfg["funasr"]
llm_cfg = cfg["llm"]
azure_cfg = cfg["azure"]
azure_key = os.environ.get(azure_cfg["key_env"], "") if azure_cfg["enabled"] else ""

# Tracks the date (YYYY-MM-DD) when Azure was degraded due to quota errors.
# Reset on container restart or when a new day begins.
_azure_degraded_date: str | None = None

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="FunASR STT Server")
model = None
punc_model = None


def _is_sensevoice() -> bool:
    return "sensevoice" in funasr_cfg["model"].lower()


def _model_short_name() -> str:
    model_id = funasr_cfg["model"]
    if "sensevoice" in model_id.lower():
        return model_id.rsplit("/", 1)[-1]
    if "paraformer" in model_id.lower():
        return "paraformer-large"
    return model_id.rsplit("/", 1)[-1]


def get_model():
    global model, punc_model
    if model is None:
        logger.info(
            "Loading model: %s (VAD: %s, device: %s)",
            funasr_cfg["model"], funasr_cfg["vad_model"], funasr_cfg["device"],
        )
        model = AutoModel(
            model=funasr_cfg["model"],
            vad_model=funasr_cfg["vad_model"],
            vad_kwargs={"max_single_segment_time": 30000},
            device=funasr_cfg["device"],
            trust_remote_code=True,
        )
        logger.info("ASR model loaded successfully")

        punc_model_id = funasr_cfg.get("punc_model", "")
        if punc_model_id:
            logger.info("Loading punctuation model: %s", punc_model_id)
            punc_model = AutoModel(
                model=punc_model_id,
                device=funasr_cfg["device"],
            )
            logger.info("Punctuation model loaded successfully")
    return model


def _to_wav_bytes(audio_data: bytes, suffix: str) -> bytes:
    """Convert audio to 16kHz mono PCM WAV via ffmpeg. Returns wav bytes."""
    if suffix.lower() == ".wav":
        return audio_data
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as src:
        src.write(audio_data)
        src_path = src.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as dst:
        dst_path = dst.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src_path, "-ar", "16000", "-ac", "1", "-f", "wav", dst_path],
            capture_output=True, check=True,
        )
        with open(dst_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(src_path)
        os.unlink(dst_path)


def _today_str() -> str:
    """Return today's date string in UTC+8 (Asia/Shanghai)."""
    return date.today().isoformat()


async def transcribe_azure(audio_data: bytes, suffix: str) -> str | None:
    """Try Azure Fast Transcription. Returns text on success, None on failure/skip."""
    global _azure_degraded_date

    if not azure_cfg["enabled"] or not azure_key:
        return None

    today = _today_str()
    if _azure_degraded_date == today:
        return None

    try:
        wav_data = _to_wav_bytes(audio_data, suffix)
    except Exception as e:
        logger.warning("ffmpeg WAV conversion failed, falling back to FunASR: %s", e)
        return None

    url = (
        f"https://{azure_cfg['region']}.api.cognitive.microsoft.com"
        "/speechtotext/transcriptions:transcribe?api-version=2025-10-15"
    )
    definition = json.dumps({"locales": ["zh-CN"]})

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                url,
                headers={
                    "Ocp-Apim-Subscription-Key": azure_key,
                    "Accept": "application/json",
                },
                files={"audio": ("audio.wav", wav_data, "audio/wav")},
                data={"definition": definition},
            )

        if resp.status_code in (429, 403):
            _azure_degraded_date = today
            logger.warning(
                "Azure quota exceeded (HTTP %d), degrading to FunASR for today (%s)",
                resp.status_code, today,
            )
            return None

        resp.raise_for_status()

        combined = resp.json().get("combinedPhrases", [])
        text = combined[0].get("text", "") if combined else ""
        logger.info("Azure transcription succeeded (%d chars), skipping FunASR", len(text))
        return text

    except httpx.HTTPStatusError as e:
        if e.response.status_code in (429, 403):
            _azure_degraded_date = today
            logger.warning(
                "Azure quota exceeded (HTTP %d), degrading to FunASR for today (%s)",
                e.response.status_code, today,
            )
        else:
            logger.warning("Azure transcription failed (HTTP %d), falling back to FunASR", e.response.status_code)
        return None
    except Exception as e:
        logger.warning("Azure transcription failed, falling back to FunASR: %s", e)
        return None


async def polish_text(text: str) -> str:
    """Call local Ollama to polish ASR output. Falls back to raw text on failure."""
    if not llm_cfg["enabled"] or not text:
        return text
    try:
        async with httpx.AsyncClient(timeout=llm_cfg["timeout"]) as http:
            resp = await http.post(
                f"{llm_cfg['base_url']}/api/chat",
                json={
                    "model": llm_cfg["model"],
                    "think": False,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": llm_cfg["prompt"]["system"]},
                        {"role": "user", "content": text},
                    ],
                },
            )
        result = resp.json().get("message", {}).get("content", "")
        return result.strip() if result.strip() else text
    except Exception as e:
        logger.warning("LLM polish failed, using raw ASR text: %s", e)
        return text


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    get_model()


@app.get("/health")
async def health():
    info = {"status": "ok", "model": funasr_cfg["model"]}
    if funasr_cfg.get("punc_model"):
        info["punc_model"] = funasr_cfg["punc_model"]
    info["azure_enabled"] = azure_cfg["enabled"] and bool(azure_key)
    info["azure_degraded"] = _azure_degraded_date == _today_str() if _azure_degraded_date else False
    return info


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": _model_short_name(), "object": "model", "owned_by": "funasr"}],
    }


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("auto"),
    response_format: str = Form("json"),
    language: str = Form("auto"),
    prompt: str = Form(None),
):
    audio_data = await file.read()
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_data)
        tmp_path = tmp.name

    try:
        # Try Azure first (if enabled and not degraded)
        azure_text = await transcribe_azure(audio_data, suffix)
        if azure_text is not None:
            text = await polish_text(azure_text)
            if response_format == "text":
                return PlainTextResponse(text)
            return JSONResponse({"text": text})

        # Fall back to local FunASR
        asr_model = get_model()
        res = asr_model.generate(
            input=tmp_path,
            cache={},
            language=language,
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )

        if res and len(res) > 0:
            raw_text = res[0].get("text", "")
        else:
            raw_text = ""

        if _is_sensevoice():
            text = rich_transcription_postprocess(raw_text)
        elif punc_model is not None and raw_text:
            punc_res = punc_model.generate(input=raw_text)
            text = punc_res[0].get("text", raw_text) if punc_res else raw_text
        else:
            text = raw_text

        text = await polish_text(text)

        if response_format == "text":
            return PlainTextResponse(text)
        return JSONResponse({"text": text})

    except Exception as e:
        logger.error("Transcription failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    uvicorn.run(app, host=funasr_cfg["host"], port=funasr_cfg["port"])
