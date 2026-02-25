"""
benchmark_context.py — Context continuity validation benchmark.

Compares translation quality with and without the rolling context window
by running a 10-sentence scripted conversation through the Translator.

Metrics
-------
- Repetitions: consecutive outputs sharing >60% of their words
- Grammar breaks: outputs that are unusually short (<3 chars) or identical
  to the English input (failed translation)

Usage
-----
    python benchmarks/benchmark_context.py

Output
------
    results/context_results.csv
"""

from __future__ import annotations

import collections
import csv
import difflib
import os
import sys
import time
from typing import Deque

# Project root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config  # noqa: E402

import argostranslate.package  # noqa: E402
import argostranslate.translate  # noqa: E402

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
CONVERSATION_SCRIPT = os.path.join(
    PROJECT_ROOT, "test_clips", "conversation_script.txt"
)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "context_results.csv")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_conversation() -> list[str]:
    """Load conversation sentences from script file."""
    sentences = []
    with open(CONVERSATION_SCRIPT, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            sentences.append(line)
    return sentences


def _translate(text: str) -> str:
    """Single Argos translation en→hi."""
    return argostranslate.translate.translate(text, "en", "hi")


def _word_overlap(a: str, b: str) -> float:
    """Fraction of words in common between two strings."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / max(len(words_a), len(words_b))


def _is_repetition(prev_output: str, curr_output: str) -> bool:
    """Check if current output is too similar to previous (>60% word overlap)."""
    return _word_overlap(prev_output, curr_output) > 0.60


def _is_grammar_break(source: str, output: str) -> bool:
    """Check for grammar break indicators."""
    # Output too short
    if len(output.strip()) < 3:
        return True
    # Output identical to source (translation did nothing useful)
    if output.strip().lower() == source.strip().lower():
        return True
    # Output is just punctuation
    if not any(c.isalnum() for c in output):
        return True
    return False


# ── Context-aware translation (mirrors translator.py logic) ─────────────────

def _translate_with_context(
    new_text: str,
    context_source: Deque[str],
    context_translated: Deque[str],
) -> str:
    """Translate with rolling context prefix — same logic as Translator class."""
    prefix_source = " ".join(context_source).strip()
    prefix_translated = " ".join(context_translated).strip()

    if prefix_source:
        combined_input = f"{prefix_source} {new_text}"
    else:
        combined_input = new_text

    full_translation = _translate(combined_input)

    if prefix_translated:
        trimmed = _trim_prefix(full_translation, prefix_translated)
    else:
        trimmed = full_translation

    context_source.append(new_text)
    result = trimmed.strip() if trimmed.strip() else full_translation.strip()
    context_translated.append(result)
    return result


def _trim_prefix(full: str, prefix_tr: str) -> str:
    """Remove translated prefix — mirrors Translator._trim_prefix()."""
    if not prefix_tr:
        return full

    if full.startswith(prefix_tr):
        return full[len(prefix_tr):].strip()

    matcher = difflib.SequenceMatcher(None, full, prefix_tr, autojunk=False)
    blocks = matcher.get_matching_blocks()
    trimmed_end = 0
    for block in blocks:
        if block.a == trimmed_end and block.b == 0:
            trimmed_end = block.a + block.size
        else:
            break

    overlap_ratio = trimmed_end / max(len(prefix_tr), 1)
    if trimmed_end > 0 and overlap_ratio >= 0.30:
        return full[trimmed_end:].strip()

    return full


# ── Main benchmark ───────────────────────────────────────────────────────────

def run_benchmark() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    sentences = _load_conversation()
    print(f"Loaded {len(sentences)} conversation sentences.\n")

    # Verify Argos model
    installed = argostranslate.package.get_installed_packages()
    if not any(p.from_code == "en" and p.to_code == "hi" for p in installed):
        print("ERROR: Argos en→hi package not installed. Run setup_models.py.")
        sys.exit(1)

    # ── Run WITH context ─────────────────────────────────────────────────────
    print(f"{'=' * 60}")
    print("  Condition: WITH context window")
    print(f"{'=' * 60}")

    context_source: Deque[str] = collections.deque(maxlen=config.CONTEXT_MAXLEN)
    context_translated: Deque[str] = collections.deque(maxlen=config.CONTEXT_MAXLEN)

    outputs_with: list[str] = []
    latencies_with: list[float] = []

    for idx, sentence in enumerate(sentences, 1):
        t0 = time.perf_counter()
        output = _translate_with_context(sentence, context_source, context_translated)
        latency = time.perf_counter() - t0
        outputs_with.append(output)
        latencies_with.append(latency)
        print(f"  [{idx:>2}] EN:  {sentence}")
        print(f"       HI:  {output}  ({latency:.3f}s)")

    # ── Run WITHOUT context ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Condition: WITHOUT context window")
    print(f"{'=' * 60}")

    outputs_without: list[str] = []
    latencies_without: list[float] = []

    for idx, sentence in enumerate(sentences, 1):
        t0 = time.perf_counter()
        output = _translate(sentence)
        latency = time.perf_counter() - t0
        outputs_without.append(output)
        latencies_without.append(latency)
        print(f"  [{idx:>2}] EN:  {sentence}")
        print(f"       HI:  {output}  ({latency:.3f}s)")

    # ── Count repetitions and grammar breaks ─────────────────────────────────
    reps_with = 0
    reps_without = 0
    breaks_with = 0
    breaks_without = 0

    for i in range(len(sentences)):
        # Grammar breaks
        if _is_grammar_break(sentences[i], outputs_with[i]):
            breaks_with += 1
        if _is_grammar_break(sentences[i], outputs_without[i]):
            breaks_without += 1

        # Repetitions (check against previous output)
        if i > 0:
            if _is_repetition(outputs_with[i - 1], outputs_with[i]):
                reps_with += 1
            if _is_repetition(outputs_without[i - 1], outputs_without[i]):
                reps_without += 1

    # ── Results ──────────────────────────────────────────────────────────────
    import numpy as np

    results = {
        "repetitions_with_context": reps_with,
        "repetitions_without_context": reps_without,
        "grammar_breaks_with_context": breaks_with,
        "grammar_breaks_without_context": breaks_without,
        "avg_latency_with_context": f"{np.mean(latencies_with):.4f}",
        "avg_latency_without_context": f"{np.mean(latencies_without):.4f}",
    }

    # Write per-sentence detail CSV
    detail_csv = os.path.join(RESULTS_DIR, "context_detail.csv")
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sentence_id", "source", "output_with_context", "output_without_context",
            "latency_with_s", "latency_without_s",
            "is_repetition_with", "is_repetition_without",
            "is_grammar_break_with", "is_grammar_break_without",
        ])
        writer.writeheader()
        for i in range(len(sentences)):
            is_rep_with = _is_repetition(outputs_with[i-1], outputs_with[i]) if i > 0 else False
            is_rep_without = _is_repetition(outputs_without[i-1], outputs_without[i]) if i > 0 else False
            writer.writerow({
                "sentence_id": i + 1,
                "source": sentences[i],
                "output_with_context": outputs_with[i],
                "output_without_context": outputs_without[i],
                "latency_with_s": f"{latencies_with[i]:.4f}",
                "latency_without_s": f"{latencies_without[i]:.4f}",
                "is_repetition_with": is_rep_with,
                "is_repetition_without": is_rep_without,
                "is_grammar_break_with": _is_grammar_break(sentences[i], outputs_with[i]),
                "is_grammar_break_without": _is_grammar_break(sentences[i], outputs_without[i]),
            })

    print(f"\n✓ Detail saved to {detail_csv}")

    # Write summary CSV
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "metric", "with_context", "without_context",
        ])
        writer.writeheader()
        writer.writerow({
            "metric": "repetitions",
            "with_context": reps_with,
            "without_context": reps_without,
        })
        writer.writerow({
            "metric": "grammar_breaks",
            "with_context": breaks_with,
            "without_context": breaks_without,
        })
        writer.writerow({
            "metric": "avg_latency_s",
            "with_context": results["avg_latency_with_context"],
            "without_context": results["avg_latency_without_context"],
        })

    print(f"✓ Summary saved to {RESULTS_CSV}")

    # Print paper-ready table
    print(f"\n{'=' * 60}")
    print("  Context Continuity Results (for Paper Section 5)")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<25} {'With Context':>15} {'Without Context':>18}")
    print(f"  {'─' * 25} {'─' * 15} {'─' * 18}")
    print(f"  {'Repetitions':<25} {reps_with:>15} {reps_without:>18}")
    print(f"  {'Grammar Breaks':<25} {breaks_with:>15} {breaks_without:>18}")
    print(f"  {'Avg Latency (s)':<25} {results['avg_latency_with_context']:>15} {results['avg_latency_without_context']:>18}")


if __name__ == "__main__":
    run_benchmark()
