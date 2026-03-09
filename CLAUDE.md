# FunASR STT Server — Project Notes

## Project Overview

OpenAI-compatible Speech-to-Text server using FunASR.
Currently runs **Paraformer-Large** (ASR) + **ct-punc** (punctuation restoration).
Also supports SenseVoice models via config switch.
Optionally applies an LLM polish layer (Ollama qwen3.5:4b) for typo correction and sentence segmentation.

## Key Files

| File | Purpose |
|------|---------|
| `server.py` | FastAPI server, main logic (supports both Paraformer and SenseVoice) |
| `config.toml` | Runtime config (model selection, LLM settings) |
| `Dockerfile` | Container build |
| `docker-compose.yml` | Deployment config |
| `benchmark/` | Benchmark scripts and results |
| `benchmark/benchmark_asr.py` | ASR model CER + latency benchmark script |
| `benchmark/benchmark_polish.py` | LLM polish layer benchmark script |

## Architecture

```
audio → FunASR Paraformer-Large → ct-punc (punctuation) → polish_text (Ollama) → response
```

- Model type is auto-detected from `config.toml`:
  - **Paraformer**: raw text → `ct-punc` model adds punctuation
  - **SenseVoice**: raw text → `rich_transcription_postprocess` strips special tokens
- `polish_text()` is skipped when `llm.enabled = false` or text is empty
- On LLM failure, falls back to ASR text (no error returned to client)

## Model Selection

Current: `iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` + `iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch`

### Benchmark Results (3 real-world Chinese test cases)

| Metric | SenseVoiceSmall | Paraformer-Large | Change |
|--------|----------------|------------------|--------|
| Avg CER | 9.4% | **3.9%** | -59% |
| Avg Latency | 1.57s | 2.02s | +29% |

Key improvements: English words (feature, memory, CPU), proper nouns (树莓派).
Latency increase due to additional ct-punc model inference.

### Switching Models

Edit `config.toml` `[funasr]` section. Commented-out alternative config is included.
After switching, restart the container: `docker compose restart`.
First run with a new model triggers download from ModelScope (~840MB for Paraformer-Large).

## Configuration

All settings are in `config.toml` (mounted into container at `/app/config.toml`).
The only environment variable is `CONFIG_PATH` (default `/app/config.toml`) to override the config file location.

Key config fields:
- `funasr.model` — ASR model ID (ModelScope)
- `funasr.punc_model` — punctuation model ID (empty string to disable; required for Paraformer)
- `funasr.vad_model` — VAD model ID
- `llm.enabled` — enable/disable LLM polish layer

## Model Cache

Models are cached at `~/.cache/modelscope` on the host (mounted into container).
Total cache size: ~2.5GB (Paraformer-Large 840MB + ct-punc 278MB + VAD + SenseVoice if cached).
Not auto-cleaned by macOS.

## Docker Build

### Multi-platform (arm64 + amd64) — push to Docker Hub

Requires `multiarch` buildx builder (docker-container driver, host network).
Both `HTTPS_PROXY`/`HTTP_PROXY` env vars AND `--build-arg` are needed:
- Env vars: cover BuildKit daemon itself (pulling base image from Docker Hub)
- `--build-arg`: cover pip inside the RUN layer

```bash
# One-time: create builder (if not exists)
docker buildx create --name multiarch --driver docker-container --driver-opt network=host --use --bootstrap

# Build & push
cd "~/Documents/Machine Config Hepler/MacBook/funasr"
HTTPS_PROXY=http://127.0.0.1:7897 HTTP_PROXY=http://127.0.0.1:7897 \
docker buildx build \
  --builder multiarch \
  --platform linux/arm64,linux/amd64 \
  --build-arg http_proxy=http://127.0.0.1:7897 \
  --build-arg https_proxy=http://127.0.0.1:7897 \
  -t likanwen/funasr-stt:latest \
  --push .
```

> Port `7897` is Clash Verge Rev mixed proxy port. Adjust if different.
> apt uses USTC mirror (`mirrors.ustc.edu.cn`) — no proxy needed for apt.

### Cross-compile on remote x86_64 server (for arm64 target)
```bash
# On remote server — install QEMU first if needed:
docker run --privileged --rm tonistiigi/binfmt --install arm64
docker buildx create --name multiarch --use

# Build arm64 image:
docker buildx build --builder multiarch --platform linux/arm64 -t funasr-stt:arm64 --load .

# Export and transfer:
docker save funasr-stt:arm64 | gzip > funasr-stt-arm64.tar.gz
scp funasr-stt-arm64.tar.gz local:~

# On Mac:
docker load < funasr-stt-arm64.tar.gz
docker tag funasr-stt:arm64 funasr-stt
```

## Benchmark Notes

### ASR Model Benchmark
- `benchmark/benchmark_asr.py`: compares CER and latency using audio files in `benchmark/test_audio/`
- Paraformer-Large selected over SenseVoiceSmall: 59% CER reduction, acceptable latency increase
- Results: `benchmark/benchmark_asr_results_sensevoice.json`, `benchmark/benchmark_asr_results_paraformer.json`

### LLM Polish Benchmark
- `qwen3.5:4b` selected: 100% accuracy on all 11 test cases, avg latency ~1.65s
- Ollama native API used (`/api/chat` with `think: false`) — avoids thinking-mode token waste
- GitHub Models tested (gpt-4.1-nano, gpt-5 series) but hit rate limits quickly
- Results saved in `benchmark/benchmark_results.json`

## Current Deployment State

- ASR model: **Paraformer-Large** with **ct-punc** punctuation restoration
- LLM polish layer: **implemented but disabled** (`llm.enabled = false` in config.toml)
- Reason: OpenWhispr handles polish externally; re-enable when needed
- To enable: set `enabled = true` in `config.toml` `[llm]` section and ensure Ollama is running with `qwen3.5:4b`
- `server.py` is volume-mounted into the container (no rebuild needed for code changes)
