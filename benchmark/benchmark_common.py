"""
Shared utilities for ASR benchmarking: CER calculation, text normalization, test discovery.
"""

import os
import sys
import unicodedata


# ── Text normalization ──────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Normalize text for CER: remove punctuation and whitespace, lowercase."""
    out = []
    for ch in unicodedata.normalize("NFKC", text):
        cat = unicodedata.category(ch)
        if cat.startswith("P") or cat.startswith("Z") or cat.startswith("C"):
            continue
        out.append(ch.lower())
    return "".join(out)


# ── CER calculation ────────────────────────────────────────────────────────

def _edit_distance(ref: str, hyp: str) -> int:
    """Levenshtein distance at character level."""
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def compute_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate: edit_distance(ref, hyp) / len(ref)."""
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)


# ── Test discovery ──────────────────────────────────────────────────────────

def discover_tests(test_dir: str) -> list[dict]:
    """Find paired .wav + .txt files in the given directory."""
    if not os.path.isdir(test_dir):
        print(f"Error: test directory not found: {test_dir}")
        print("Create test_audio/ with .wav files and matching .txt ground truth files.")
        sys.exit(1)

    tests = []
    for fname in sorted(os.listdir(test_dir)):
        if not fname.lower().endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg")):
            continue
        base = os.path.splitext(fname)[0]
        audio_path = os.path.join(test_dir, fname)
        txt_path = os.path.join(test_dir, base + ".txt")
        if not os.path.exists(txt_path):
            print(f"Warning: no ground truth for {fname}, skipping")
            continue
        with open(txt_path, encoding="utf-8") as f:
            ground_truth = f.read().strip()
        tests.append({
            "name": base,
            "audio_path": audio_path,
            "ground_truth": ground_truth,
        })

    if not tests:
        print("Error: no test cases found. Need .wav + .txt pairs in test_audio/")
        sys.exit(1)

    return tests
