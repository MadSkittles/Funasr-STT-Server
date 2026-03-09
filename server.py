"""
OpenAI-compatible STT server using FunASR.

Supports SenseVoice (with rich_transcription_postprocess) and
Paraformer (with ct-punc punctuation model) architectures.

Endpoint: POST /v1/audio/transcriptions
Compatible with VoiceMode MCP and OpenAI Python SDK.
"""

import os
import tomllib
import tempfile
import logging

import httpx
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import PlainTextResponse, JSONResponse
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("funasr-server")

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
    "llm": {
        "enabled": True,
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
