"""
ASR model benchmark: compare CER and latency across models.

Usage:
    1. Place test audio files (*.wav) in test_audio/
    2. For each audio, create a matching *.txt with the ground truth text
    3. Run: python benchmark_asr.py

The script calls the STT server at http://127.0.0.1:2023/v1/audio/transcriptions
and compares the result against ground truth using Character Error Rate (CER).
"""

import os
import json
import time

import httpx

from benchmark_common import compute_cer, discover_tests

# ── Config ────────────────────────────────────────────────────────────────────

STT_URL = "http://127.0.0.1:2023/v1/audio/transcriptions"
TEST_DIR = os.path.join(os.path.dirname(__file__), "test_audio")
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "benchmark_asr_results.json")


# ── Run benchmark ────────────────────────────────────────────────────────────

def transcribe(audio_path: str) -> tuple[str, float]:
    """Send audio to STT server, return (text, latency_seconds)."""
    with open(audio_path, "rb") as f:
        audio_data = f.read()

    fname = os.path.basename(audio_path)
    t0 = time.perf_counter()
    resp = httpx.post(
        STT_URL,
        files={"file": (fname, audio_data)},
        data={"model": "auto", "response_format": "json"},
        timeout=120.0,
    )
    latency = time.perf_counter() - t0
    resp.raise_for_status()
    text = resp.json().get("text", "")
    return text, latency


def run_benchmark(tests: list[dict]) -> list[dict]:
    """Run all test cases and collect results."""
    results = []
    for tc in tests:
        try:
            hypothesis, latency = transcribe(tc["audio_path"])
            cer = compute_cer(tc["ground_truth"], hypothesis)
            results.append({
                "name": tc["name"],
                "ground_truth": tc["ground_truth"],
                "hypothesis": hypothesis,
                "cer": round(cer, 4),
                "latency": round(latency, 3),
            })
            cer_pct = cer * 100
            print(f"  {tc['name']:30s}  CER: {cer_pct:5.1f}%  Latency: {latency:.2f}s")
            if hypothesis != tc["ground_truth"]:
                print(f"    ref: {tc['ground_truth'][:80]}")
                print(f"    hyp: {hypothesis[:80]}")
        except Exception as e:
            print(f"  {tc['name']:30s}  ERROR: {e}")
            results.append({
                "name": tc["name"],
                "ground_truth": tc["ground_truth"],
                "hypothesis": f"[ERROR] {e}",
                "cer": 1.0,
                "latency": 0.0,
            })
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Discovering test cases...")
    tests = discover_tests(TEST_DIR)
    print(f"Found {len(tests)} test case(s)\n")

    # Get server info
    try:
        health = httpx.get("http://127.0.0.1:2023/health", timeout=5.0).json()
        model_name = health.get("model", "unknown")
        punc_name = health.get("punc_model", "")
    except Exception as e:
        print(f"Error: cannot reach STT server at 127.0.0.1:2023 — {e}")
        raise SystemExit(1)

    print(f"Model: {model_name}")
    if punc_name:
        print(f"Punc:  {punc_name}")
    print(f"{'=' * 60}\n")

    results = run_benchmark(tests)

    # Summary
    valid = [r for r in results if not r["hypothesis"].startswith("[ERROR]")]
    if valid:
        avg_cer = sum(r["cer"] for r in valid) / len(valid)
        avg_lat = sum(r["latency"] for r in valid) / len(valid)
        print(f"\n{'=' * 60}")
        print(f"  Avg CER:     {avg_cer * 100:.1f}%")
        print(f"  Avg Latency: {avg_lat:.2f}s")
        print(f"  Cases:       {len(valid)} OK, {len(results) - len(valid)} failed")
        print(f"{'=' * 60}")

    # Save results
    output = {
        "model": model_name,
        "punc_model": punc_name,
        "results": results,
        "summary": {
            "avg_cer": round(avg_cer, 4) if valid else None,
            "avg_latency": round(avg_lat, 3) if valid else None,
            "total": len(results),
            "ok": len(valid),
            "errors": len(results) - len(valid),
        },
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
