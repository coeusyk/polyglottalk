"""
translator.py — Machine-translation thread using Argos Translate.

Context continuity
------------------
Maintains a rolling deque of the last CONTEXT_MAXLEN source-language
segments.  Before each translation the previous segments are prepended
to the new text so the model sees sentence-boundary context.  After
translation the re-translated prefix is trimmed from the output (exact
match first, fuzzy difflib fallback).

The Argos model is loaded ONCE in __init__; run() never imports or loads
anything.
"""

from __future__ import annotations

import collections
import difflib
import logging
import queue
import threading
import time
from typing import Deque

import argostranslate.package
import argostranslate.translate

import config
from models import TextSegment, TranslatedSegment

logger = logging.getLogger(__name__)


class Translator:
    """Translates TextSegment objects into TranslatedSegment objects.

    Context continuity
    ------------------
    self._context_source  — deque[str], last N source-text segments
    self._cache_key       — the prefix string whose translation is cached
    self._cache_val       — cached translation of that prefix string
    """

    def __init__(
        self,
        text_queue: queue.Queue,
        tts_queue: queue.Queue,
        stop_event: threading.Event,
        source_lang: str = config.SOURCE_LANG,
        target_lang: str = config.TARGET_LANG,
        context_maxlen: int = config.CONTEXT_MAXLEN,
    ) -> None:
        self._text_queue = text_queue
        self._tts_queue = tts_queue
        self._stop_event = stop_event
        self._source_lang = source_lang
        self._target_lang = target_lang

        self._context_source: Deque[str] = collections.deque(
            maxlen=context_maxlen
        )
        self._cache_key: str = ""
        self._cache_val: str = ""

        logger.info(
            "Loading translation model (%s → %s)…", source_lang, target_lang
        )
        t0 = time.perf_counter()
        self._load_model()
        logger.info("Translation model loaded in %.1fs", time.perf_counter() - t0)

    # ── Thread target ───────────────────────────────────────────────────────

    def run(self) -> None:
        """Consume TextSegments, translate with context, push TranslatedSegments."""
        while not self._stop_event.is_set():
            try:
                item = self._text_queue.get(timeout=config.QUEUE_GET_TIMEOUT)
            except queue.Empty:
                continue

            if item is None:  # shutdown sentinel
                break

            assert isinstance(item, TextSegment)

            if not item.text.strip():
                logger.debug("Chunk #%d skipped — empty text.", item.chunk_id)
                continue

            t0 = time.perf_counter()
            translated = self._translate_with_context(item.text)
            elapsed = time.perf_counter() - t0

            if not translated:
                logger.warning(
                    "Chunk #%d produced empty translation, skipping.", item.chunk_id
                )
                continue

            logger.info(
                "Translation done (%.3fs) chunk #%d: %r",
                elapsed,
                item.chunk_id,
                translated,
            )

            segment = TranslatedSegment(
                chunk_id=item.chunk_id,
                text=translated,
                timestamp=time.perf_counter(),
            )
            self._put(segment)

        logger.info("Translator stopped.")

    # ── Context-aware translation ────────────────────────────────────────────

    def _translate_with_context(self, new_text: str) -> str:
        """Translate new_text with rolling context prefix for continuity.

        Steps
        -----
        1. Build ``prefix_source`` from context deque.
        2. Concatenate: ``combined_input = prefix_source + " " + new_text``.
        3. Translate ``combined_input``.
        4. Translate ``prefix_source`` alone (cached) → ``prefix_translated``.
        5. Strip ``prefix_translated`` from start of full output (exact/fuzzy).
        6. Update context deque with ``new_text``.
        7. Return trimmed translation (fallback to full if trim fails).
        """
        prefix_source = " ".join(self._context_source).strip()

        if prefix_source:
            combined_input = f"{prefix_source} {new_text}"
        else:
            combined_input = new_text

        full_translation = self._translate(combined_input)

        if prefix_source:
            prefix_translated = self._translate_prefix(prefix_source)
            trimmed = self._trim_prefix(full_translation, prefix_translated)
        else:
            trimmed = full_translation

        # Update rolling context AFTER translation (use source text)
        self._context_source.append(new_text)

        # Safety: never return empty
        result = trimmed.strip() if trimmed.strip() else full_translation.strip()
        return result

    def _trim_prefix(self, full: str, prefix_tr: str) -> str:
        """Remove translated prefix from start of full translation.

        1. Exact match: strip ``prefix_tr`` from start of ``full``.
        2. Fuzzy fallback: use difflib to find longest matching prefix.
        3. If overlap < 30% of prefix_tr length, return ``full`` unchanged.
        """
        if not prefix_tr:
            return full

        # ── Exact match ────────────────────────────────────────────────────
        if full.startswith(prefix_tr):
            return full[len(prefix_tr) :].strip()

        # ── Fuzzy match via SequenceMatcher ───────────────────────────────
        # Find the longest common prefix block at the start of both strings
        matcher = difflib.SequenceMatcher(None, full, prefix_tr, autojunk=False)
        # opcodes: list of (tag, i1, i2, j1, j2)
        # We look at the first block; if it's 'equal' and starts at 0, trim it.
        blocks = matcher.get_matching_blocks()  # sorted by a
        trimmed_end = 0
        for block in blocks:
            if block.a == trimmed_end and block.b == 0:
                trimmed_end = block.a + block.size
            else:
                break

        overlap_ratio = trimmed_end / max(len(prefix_tr), 1)
        if trimmed_end > 0 and overlap_ratio >= 0.30:
            logger.debug(
                "Fuzzy prefix trim: removed %d chars (%.0f%% overlap)",
                trimmed_end,
                overlap_ratio * 100,
            )
            return full[trimmed_end:].strip()

        logger.debug(
            "Prefix trim skipped — overlap %.0f%% < 30%%", overlap_ratio * 100
        )
        return full

    # ── Translation helpers ─────────────────────────────────────────────────

    def _translate(self, text: str) -> str:
        """Call Argos Translate for the given text."""
        return argostranslate.translate.translate(
            text, self._source_lang, self._target_lang
        )

    def _translate_prefix(self, prefix_source: str) -> str:
        """Translate the context prefix, using a simple one-entry cache."""
        if prefix_source == self._cache_key:
            return self._cache_val
        result = self._translate(prefix_source)
        self._cache_key = prefix_source
        self._cache_val = result
        return result

    # ── Model loading ───────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Verify the Argos language package is installed.

        Raises
        ------
        RuntimeError
            If the required language package is not found.
            Instructs the user to run setup_models.py first.
        """
        installed = argostranslate.package.get_installed_packages()
        found = any(
            p.from_code == self._source_lang and p.to_code == self._target_lang
            for p in installed
        )
        if not found:
            raise RuntimeError(
                f"Argos Translate package not found for "
                f"{self._source_lang!r} → {self._target_lang!r}. "
                f"Run 'python setup_models.py' first to download models."
            )
        logger.debug(
            "Argos package %s→%s verified.", self._source_lang, self._target_lang
        )

    # ── Queue helper ────────────────────────────────────────────────────────

    def _put(self, segment: TranslatedSegment) -> None:
        """Push to tts_queue; retry (with stop_event check) on Full."""
        while not self._stop_event.is_set():
            try:
                self._tts_queue.put(segment, timeout=config.QUEUE_PUT_TIMEOUT)
                return
            except queue.Full:
                logger.debug(
                    "tts_queue full — retrying put for chunk #%d", segment.chunk_id
                )
