# FunASR STT Server

OpenAI-compatible Speech-to-Text server powered by [FunASR](https://github.com/modelscope/FunASR). Supports both **Paraformer-Large** (default) and **SenseVoice** model architectures, with automatic punctuation restoration and optional LLM polish layer.

## Features

- **OpenAI-compatible API** — drop-in replacement for `POST /v1/audio/transcriptions`
- **Paraformer-Large** (default) — high-accuracy Chinese ASR with ct-punc punctuation model
- **SenseVoice** support — configurable alternative with built-in rich transcription postprocessing
- **Multi-language** — Chinese, English, and mixed-language audio
- **LLM polish layer** — optional Ollama integration (qwen3.5:4b) for typo correction and sentence segmentation
- **Graceful fallback** — if LLM is unavailable, returns ASR text without error
- **Config-driven** — all settings in `config.toml`, model switching without image rebuild

## Quick Start

### Prerequisites

- Docker (OrbStack or Docker Desktop)
- ~3GB RAM for the container (Paraformer-Large + ct-punc + VAD)
- Ollama with `qwen3.5:4b` pulled (optional, for polish layer)

The image is published to Docker Hub as [`likanwen/funasr-stt`](https://hub.docker.com/r/likanwen/funasr-stt) with multi-platform support (linux/arm64, linux/amd64). No local build required.

### 1. Start the server

```bash
cd MacBook/funasr
docker compose up -d
```

On first run, FunASR automatically downloads models from [ModelScope](https://modelscope.cn) and caches them at `~/.cache/modelscope` on the host.

**Download size:** ~2.5GB total (Paraformer-Large 840MB + ct-punc 278MB + FSMN-VAD). Takes a few minutes on first start.

Check progress:

```bash
docker compose logs -f
```

Once you see `Punctuation model loaded successfully`, the server is ready.

### 2. Verify it's running

```bash
curl http://127.0.0.1:2023/health
# {"status":"ok","model":"iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch","punc_model":"iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"}
```

### 3. Transcribe audio

```bash
curl http://127.0.0.1:2023/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=auto"
# {"text":"今天天气真好，我们一起去公园里跑步。"}
```

## API Reference

### `POST /v1/audio/transcriptions`

Compatible with the [OpenAI Audio API](https://platform.openai.com/docs/api-reference/audio).

**Form fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | file | required | Audio file (wav, mp3, m4a, flac, etc.) |
| `model` | string | `auto` | Model name (for compatibility; actual model set in config.toml) |
| `response_format` | string | `json` | `json` or `text` |
| `language` | string | `auto` | Language hint (`zh`, `en`, `ja`, `ko`, `yue`, `auto`) |

**Response:**
```json
{"text": "transcribed text here"}
```

### `GET /health`

```json
{
  "status": "ok",
  "model": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
  "punc_model": "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
}
```

### `GET /v1/models`

```json
{"object": "list", "data": [{"id": "paraformer-large", "object": "model", "owned_by": "funasr"}]}
```

## Configuration

All settings are in `config.toml` (mounted into the container as read-only):

```toml
[funasr]
model = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
punc_model = "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
vad_model = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
device = "cpu"
host = "0.0.0.0"
port = 2023

[llm]
enabled = false
model = "qwen3.5:4b"
base_url = "http://localhost:11434"
timeout = 15.0
```

### Switching Models

To use SenseVoiceSmall instead, edit `config.toml`:

```toml
[funasr]
model = "iic/SenseVoiceSmall"
punc_model = ""
vad_model = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
```

Then restart: `docker compose restart`

The server auto-detects the model type:
- **Paraformer** — uses `punc_model` (ct-punc) for punctuation restoration
- **SenseVoice** — uses `rich_transcription_postprocess` to strip special tokens

### Enable LLM Polish Layer

1. Pull the model in Ollama:
   ```bash
   ollama pull qwen3.5:4b
   ```

2. Set `enabled = true` in `config.toml` under `[llm]`

3. Restart:
   ```bash
   docker compose restart
   ```

The polish layer corrects typos, adds punctuation, and fixes sentence segmentation. If Ollama is unreachable or times out, the ASR text is returned as-is.

## Building the Image

```bash
docker build \
  --build-arg http_proxy=http://host.internal:7897 \
  --build-arg https_proxy=http://host.internal:7897 \
  -t funasr-stt .
```

> The `http_proxy` build args are only needed if pip requires a proxy to reach PyPI. Remove them if your network can access PyPI directly.

### Multi-platform push to Docker Hub

```bash
HTTPS_PROXY=http://127.0.0.1:7897 HTTP_PROXY=http://127.0.0.1:7897 \
docker buildx build \
  --builder multiarch \
  --platform linux/arm64,linux/amd64 \
  --build-arg http_proxy=http://127.0.0.1:7897 \
  --build-arg https_proxy=http://127.0.0.1:7897 \
  -t likanwen/funasr-stt:latest \
  --push .
```

## Benchmark

### ASR Model Comparison

Tested with 3 real-world Chinese audio samples (short/middle/long):

| Metric | SenseVoiceSmall | Paraformer-Large | Change |
|--------|----------------|------------------|--------|
| Avg CER | 9.4% | **3.9%** | -59% |
| Avg Latency | 1.57s | 2.02s | +29% |

Paraformer-Large significantly improves recognition of English words (feature, memory, CPU) and proper nouns in Chinese context.

Run the benchmark yourself:
```bash
python benchmark/benchmark_asr.py
```

Requires test audio in `benchmark/test_audio/` with matching `.txt` ground truth files.

## Integration

### VoiceMode (Claude CLI)

Add to `~/.voicemode/voicemode.env`:
```
VOICEMODE_STT_BASE_URLS=http://127.0.0.1:2023/v1
VOICEMODE_PREFER_LOCAL=true
VOICEMODE_ALWAYS_TRY_LOCAL=true
```

### [OpenWhispr](https://openwhispr.com/)

OpenWhispr is a macOS app that provides system-wide voice input via a global hotkey. It captures audio from the microphone, sends it to a transcription service, and pastes the result into the currently focused app.

This server can serve as OpenWhispr's transcription backend, with Ollama providing the "Intelligence" layer for cleanup and polish.

**Architecture:**

```
Microphone → OpenWhispr → FunASR STT Server (transcription) → paste into app
                              ↕
                        Ollama qwen3.5:4b (intelligence / cleanup)
```

**Setup:**

1. **Transcription Service** — In OpenWhispr settings, set the custom transcription endpoint:
   ```
   http://127.0.0.1:2023/v1
   ```

2. **Intelligence (LLM polish)** — OpenWhispr has its own Intelligence feature for text cleanup. You can either:

   - **Use OpenWhispr's built-in Intelligence** (recommended): Configure Ollama (`http://localhost:11434`) as the Intelligence provider directly in OpenWhispr, with model `qwen3.5:4b`. This keeps transcription and polish decoupled.

   - **Use this server's built-in polish layer**: Set `enabled = true` under `[llm]` in `config.toml` and `docker compose restart`. The server will return already-polished text, so you can leave OpenWhispr's Intelligence disabled.

3. **Ollama setup** (for either approach):
   ```bash
   # Install Ollama (https://ollama.com)
   ollama pull qwen3.5:4b
   ```

4. **Permissions** — Grant Accessibility permission:

   System Settings → Privacy & Security → Accessibility → enable OpenWhispr

   (Required for auto-paste of transcription result into the focused app)

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:2023/v1",
    api_key="dummy",
)

with open("audio.wav", "rb") as f:
    result = client.audio.transcriptions.create(
        model="auto",
        file=f,
    )
print(result.text)
```

## Management

```bash
# Start
docker compose up -d

# Stop
docker compose down

# View logs
docker compose logs -f

# Restart (after config changes)
docker compose restart
```

## Notes

- **Model cache** — `~/.cache/modelscope` stores downloaded model weights. Persistent across container restarts. Not auto-cleaned by macOS.
- **Volume mounts** — `server.py` and `config.toml` are mounted into the container, so code/config changes take effect on restart without rebuilding the image.
- Container restarts automatically unless manually stopped (`restart: unless-stopped`)
- Audio files are written to a temp file, processed, then deleted
