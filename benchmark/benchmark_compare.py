"""
FunASR vs Azure Speech STT benchmark comparison.

Usage:
    1. Place test audio files (*.wav or *.m4a) in test_audio/
    2. For each audio, create a matching *.txt with the ground truth text
    3. Create benchmark/.env with Azure credentials:
         AZURE_SPEECH_KEY=your-key
         AZURE_SPEECH_REGION=japaneast
       Or set them as environment variables.
    4. Ensure FunASR is running: docker compose up -d
    5. Run: python benchmark_compare.py
"""

import os
import json
import subprocess
import sys
import tempfile
import time

import httpx

from benchmark_common import compute_cer, discover_tests


# ── Load .env ────────────────────────────────────────────────────────────────

def _load_dotenv(path: str):
    """Load KEY=VALUE pairs from a .env file into os.environ (no overwrite)."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key and key not in os.environ:
                os.environ[key] = value

_load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# ── Config ────────────────────────────────────────────────────────────────────

TEST_DIR = os.path.join(os.path.dirname(__file__), "test_audio")
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "benchmark_compare_results.json")

FUNASR_URL = "http://127.0.0.1:2023/v1/audio/transcriptions"

AZURE_SPEECH_KEY = os.environ.get("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION", "japaneast")
AZURE_STT_URL = (
    f"https://{AZURE_SPEECH_REGION}.api.cognitive.microsoft.com"
    "/speechtotext/transcriptions:transcribe?api-version=2025-10-15"
)
HTTP_PROXY = os.environ.get("HTTP_PROXY", "") or os.environ.get("http_proxy", "")


# ── Transcription backends ──────────────────────────────────────────────────

def transcribe_funasr(audio_path: str) -> tuple[str, float]:
    """Send audio to local FunASR server."""
    with open(audio_path, "rb") as f:
        audio_data = f.read()

    fname = os.path.basename(audio_path)
    t0 = time.perf_counter()
    resp = httpx.post(
        FUNASR_URL,
        files={"file": (fname, audio_data)},
        data={"model": "auto", "response_format": "json"},
        timeout=120.0,
    )
    latency = time.perf_counter() - t0
    resp.raise_for_status()
    text = resp.json().get("text", "")
    return text, latency


def _to_wav_bytes(audio_path: str) -> bytes:
    """Convert audio to 16kHz mono PCM WAV via ffmpeg. Returns wav bytes."""
    ext = os.path.splitext(audio_path)[1].lower()
    if ext == ".wav":
        with open(audio_path, "rb") as f:
            return f.read()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", "-f", "wav", tmp_path],
            capture_output=True, check=True,
        )
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)


def transcribe_azure(audio_data: bytes) -> tuple[str, float]:
    """Send pre-converted WAV bytes to Azure Fast Transcription API."""
    proxy = HTTP_PROXY or None
    definition = json.dumps({"locales": ["zh-CN"]})

    t0 = time.perf_counter()
    with httpx.Client(proxy=proxy, timeout=120.0) as client:
        resp = client.post(
            AZURE_STT_URL,
            headers={
                "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
                "Accept": "application/json",
            },
            files={
                "audio": ("audio.wav", audio_data, "audio/wav"),
            },
            data={
                "definition": definition,
            },
        )
    latency = time.perf_counter() - t0
    resp.raise_for_status()

    data = resp.json()
    # Fast Transcription: combinedPhrases[0].text has the full result
    combined = data.get("combinedPhrases", [])
    if combined:
        text = combined[0].get("text", "")
    else:
        text = ""

    return text, latency


# ── Run comparison ──────────────────────────────────────────────────────────

def run_single(
    backend_name: str,
    transcribe_fn,
    tests: list[dict],
    preconverted: dict[str, bytes] | None = None,
) -> list[dict]:
    """Run all test cases against a single backend.

    If preconverted is provided, pass the bytes to transcribe_fn instead of the path.
    """
    results = []
    for tc in tests:
        try:
            if preconverted is not None:
                hypothesis, latency = transcribe_fn(preconverted[tc["name"]])
            else:
                hypothesis, latency = transcribe_fn(tc["audio_path"])
            cer = compute_cer(tc["ground_truth"], hypothesis)
            results.append({
                "name": tc["name"],
                "ground_truth": tc["ground_truth"],
                "hypothesis": hypothesis,
                "cer": round(cer, 4),
                "latency": round(latency, 3),
            })
        except Exception as e:
            print(f"  [{backend_name}] {tc['name']}: ERROR — {e}")
            results.append({
                "name": tc["name"],
                "ground_truth": tc["ground_truth"],
                "hypothesis": f"[ERROR] {e}",
                "cer": 1.0,
                "latency": 0.0,
            })
    return results


def summarize(results: list[dict]) -> dict:
    """Compute avg CER and latency from results."""
    valid = [r for r in results if not r["hypothesis"].startswith("[ERROR]")]
    if not valid:
        return {"avg_cer": None, "avg_latency": None, "ok": 0, "errors": len(results)}
    avg_cer = sum(r["cer"] for r in valid) / len(valid)
    avg_lat = sum(r["latency"] for r in valid) / len(valid)
    return {
        "avg_cer": round(avg_cer, 4),
        "avg_latency": round(avg_lat, 3),
        "ok": len(valid),
        "errors": len(results) - len(valid),
    }


# ── Display ─────────────────────────────────────────────────────────────────

def print_results(tests: list[dict], funasr_results: list[dict], azure_results: list[dict]):
    """Print per-case comparison and summary table."""
    sep = "=" * 64
    print(f"\n{sep}")
    print("  FunASR vs Azure Speech STT Comparison")
    print(sep)

    for i, tc in enumerate(tests):
        fr = funasr_results[i]
        ar = azure_results[i]
        print(f"\nTest Case: {tc['name']}")
        print(f"  Ground Truth: {tc['ground_truth'][:70]}{'...' if len(tc['ground_truth']) > 70 else ''}")

        for label, r in [("FunASR", fr), ("Azure ", ar)]:
            if r["hypothesis"].startswith("[ERROR]"):
                print(f"  {label}:  ERROR")
            else:
                cer_pct = r["cer"] * 100
                print(f"  {label}:  CER: {cer_pct:5.1f}%  Latency: {r['latency']:.2f}s")
                print(f"    hyp: {r['hypothesis'][:80]}{'...' if len(r['hypothesis']) > 80 else ''}")

    # Summary
    fs = summarize(funasr_results)
    az = summarize(azure_results)

    print(f"\n{sep}")
    print("  SUMMARY")
    print(sep)
    print(f"{'':14s} {'Avg CER':>10s}  {'Avg Latency':>12s}")

    for label, s in [("FunASR", fs), ("Azure", az)]:
        cer_str = f"{s['avg_cer'] * 100:.1f}%" if s["avg_cer"] is not None else "N/A"
        lat_str = f"{s['avg_latency']:.2f}s" if s["avg_latency"] is not None else "N/A"
        print(f"  {label:12s} {cer_str:>10s}  {lat_str:>12s}")

    print(sep)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Validate Azure config
    if not AZURE_SPEECH_KEY:
        print("Error: AZURE_SPEECH_KEY environment variable is not set.")
        print("  export AZURE_SPEECH_KEY='your-subscription-key'")
        sys.exit(1)

    print(f"Azure region: {AZURE_SPEECH_REGION}")
    print(f"Azure proxy:  {HTTP_PROXY or '(direct)'}")
    print(f"FunASR URL:   {FUNASR_URL}")

    # Discover test cases
    print("\nDiscovering test cases...")
    tests = discover_tests(TEST_DIR)
    print(f"Found {len(tests)} test case(s)")

    # Pre-convert audio to WAV for Azure (exclude conversion time from latency)
    print("\nPre-converting audio to WAV for Azure...")
    wav_cache: dict[str, bytes] = {}
    for tc in tests:
        wav_cache[tc["name"]] = _to_wav_bytes(tc["audio_path"])
    print(f"Converted {len(wav_cache)} file(s)")

    # Run FunASR
    print("\n--- Running FunASR ---")
    funasr_results = run_single("FunASR", transcribe_funasr, tests)

    # Run Azure
    print("\n--- Running Azure Speech ---")
    azure_results = run_single("Azure", transcribe_azure, tests, preconverted=wav_cache)

    # Display
    print_results(tests, funasr_results, azure_results)

    # Save JSON
    output = {
        "funasr": {
            "results": funasr_results,
            "summary": summarize(funasr_results),
        },
        "azure": {
            "region": AZURE_SPEECH_REGION,
            "results": azure_results,
            "summary": summarize(azure_results),
        },
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
