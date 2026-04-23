"""Text-to-speech synthesis using Microsoft Edge TTS (free, Hebrew-capable).

Per-user preferences (voice/rate/pitch/mode) live on the ``users`` table and
are passed in at synth time. This module is stateless apart from a disk cache
keyed by (text, voice, rate, pitch) so repeated replies don't re-synthesize.

Design notes:
- We shell out to the ``edge-tts`` CLI rather than importing the library so
  the Python dependency is optional. If the CLI isn't installed the caller
  gets ``TTSUnavailableError`` and should fall back to text-only delivery.
- Failures are *always* swallowed by the caller — TTS is an enhancement,
  never the sole delivery channel.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)

# Optional speed boost applied *only* to English segments during multilingual
# synthesis. Matti finds Jenny a touch slow vs. Hila's pace; a small positive
# boost keeps the two voices feeling balanced in a mixed reply. Values are
# signed edge-tts rate strings ("+15%", "-5%", "0%"). Override via the
# TTS_ENGLISH_RATE_BOOST env var (read from .env by systemd).
ENGLISH_RATE_BOOST: str = os.environ.get("TTS_ENGLISH_RATE_BOOST", "+15%")


def _sum_signed_pct(a: str, b: str) -> str:
    """Sum two signed edge-tts percentages, e.g. '+15%' + '-10%' -> '+5%'.

    Accepts unsigned values as well ('15%'); a leading '+' is implied.
    Returns a normalized signed string. Used to combine a user's base rate
    pref with a per-language boost without losing sign correctness.
    """
    def _parse(s: str) -> int:
        s = (s or "0%").strip()
        if s.endswith("%"):
            s = s[:-1]
        if not s:
            return 0
        if s[0] not in "+-":
            s = "+" + s
        try:
            return int(s)
        except ValueError:
            return 0

    total = _parse(a) + _parse(b)
    sign = "+" if total >= 0 else "-"
    return f"{sign}{abs(total)}%"


# Mapping short voice keys (what we store per-user) -> Edge TTS voice IDs.
# Extend this map when new providers/locales are added.
VOICE_MAP: dict[str, str] = {
    "hila": "he-IL-HilaNeural",
    "avri": "he-IL-AvriNeural",
    "jenny": "en-US-JennyNeural",
    "guy": "en-US-GuyNeural",
}

# Pair each Hebrew voice with a same-gender English counterpart so a single
# per-user ``tts_voice`` pref implies both languages with consistent persona.
ENGLISH_PAIR: dict[str, str] = {
    "hila": "jenny",
    "avri": "guy",
}

# Unicode ranges (rough) used for per-segment language detection.
_HEBREW_RE = re.compile(r"[\u0590-\u05FF]")
_LATIN_RE = re.compile(r"[A-Za-z]")
# Split on sentence terminators (.!?…) + whitespace, or on newlines. We keep
# the splitter simple; over-split is fine because we merge consecutive
# same-language segments afterward.
_SENT_SPLIT_RE = re.compile(r"(?<=[\.!\?…:;])\s+|\n+")


def split_by_language(text: str, default_lang: str = "he") -> List[Tuple[str, str]]:
    """Split ``text`` into (segment, lang) pairs where lang is 'he' or 'en'.

    Strategy: sentence-granularity. For each sentence, count Hebrew vs Latin
    letters and tag with the dominant language. Digit/punctuation-only
    sentences inherit ``default_lang``. Consecutive same-language segments are
    merged so edge-tts runs once per language run, not once per sentence.
    """
    segments: List[Tuple[str, str]] = []
    for raw in _SENT_SPLIT_RE.split(text):
        sentence = raw.strip()
        if not sentence:
            continue
        heb = len(_HEBREW_RE.findall(sentence))
        lat = len(_LATIN_RE.findall(sentence))
        if heb == 0 and lat == 0:
            lang = default_lang
        elif heb == 0:
            lang = "en"
        elif lat == 0:
            lang = "he"
        else:
            # Mixed sentence — use dominant script. Hebrew wins ties because
            # a Hebrew voice mispronouncing a short English token (e.g. "OK")
            # is less jarring than Jenny trying to read a Hebrew sentence.
            lang = "he" if heb >= lat else "en"
        segments.append((sentence, lang))

    # Merge consecutive same-language segments so we issue one edge-tts call
    # per language run.
    merged: List[Tuple[str, str]] = []
    for sentence, lang in segments:
        if merged and merged[-1][1] == lang:
            merged[-1] = (merged[-1][0] + " " + sentence, lang)
        else:
            merged.append((sentence, lang))
    return merged


class TTSUnavailableError(RuntimeError):
    """Raised when the edge-tts binary is missing or cannot be invoked."""


class TTSSynthesisError(RuntimeError):
    """Raised when synthesis fails for a specific input."""


@dataclass(frozen=True)
class TTSRequest:
    """A single synthesis request."""

    text: str
    voice: str = "hila"       # key into VOICE_MAP
    rate: str = "0%"
    pitch: str = "0Hz"


def should_speak(text: str, mode: str, long_threshold: int = 200) -> bool:
    """Apply the 'when to speak' rule for a user's configured mode.

    - ``always``: always speak.
    - ``long_only``: speak when text is longer than ``long_threshold`` chars.
    - ``on_request``: speak only when the reply ends with a voice-request
      marker. We accept the 🎙️ emoji or the Hebrew keyword 'תקריא' at the
      end of the message (case-insensitive, whitespace-trimmed).
    """
    if not text:
        return False
    if mode == "always":
        return True
    if mode == "long_only":
        return len(text) >= long_threshold
    if mode == "on_request":
        stripped = text.rstrip()
        return stripped.endswith("🎙️") or stripped.lower().endswith("תקריא")
    return False


class TTSSynthesizer:
    """Synthesize speech via edge-tts with a simple on-disk LRU cache."""

    # Cap how long a single synth is allowed to take before we give up.
    SUBPROCESS_TIMEOUT_S: int = 30

    def __init__(
        self,
        cache_dir: Path,
        *,
        cache_max_bytes: int = 500 * 1024 * 1024,
        binary: str = "edge-tts",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_max_bytes = cache_max_bytes
        self.binary = binary

    # ------------------------------------------------------------------ utils

    def _resolve_invocation(self) -> List[str]:
        """Return the argv prefix used to invoke edge-tts.

        Tries in order:
        1. ``self.binary`` on PATH (e.g. system-wide install).
        2. ``<venv>/bin/edge-tts`` next to the running interpreter.
        3. ``python -m edge_tts`` if the package imports cleanly.

        Raises ``TTSUnavailableError`` if none work.
        """
        path_match = shutil.which(self.binary)
        if path_match:
            return [path_match]

        venv_bin = Path(sys.executable).parent / self.binary
        if venv_bin.is_file():
            return [str(venv_bin)]

        try:
            import edge_tts  # noqa: F401
        except ImportError as e:
            raise TTSUnavailableError(
                f"'{self.binary}' CLI not found and edge_tts package not "
                f"importable: {e}. Install with `pip install edge-tts`."
            ) from e
        return [sys.executable, "-m", "edge_tts"]

    def _check_binary(self) -> None:
        # Resolution is the actual liveness check.
        self._resolve_invocation()

    @staticmethod
    def _normalize_signed(value: str) -> str:
        """edge-tts requires rate/pitch to be signed ('+0%', '-15%').

        Accept either form from callers so user-facing config can be terse
        ('0%', '15%'). Bare values get a leading '+' so edge-tts validation
        passes.
        """
        if not value:
            return value
        return value if value[0] in "+-" else f"+{value}"

    @staticmethod
    def _cache_key(req: TTSRequest, voice_id: str) -> str:
        payload = f"{voice_id}|{req.rate}|{req.pitch}|{req.text}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.mp3"

    def _touch(self, path: Path) -> None:
        """Bump mtime so LRU eviction treats this entry as recently used."""
        try:
            path.touch(exist_ok=True)
        except OSError:
            logger.debug("Failed to touch cache entry", path=str(path))

    def _evict_if_needed(self) -> None:
        """Trim cache directory to ``cache_max_bytes`` using LRU by mtime."""
        try:
            entries = [
                (p.stat().st_mtime, p.stat().st_size, p)
                for p in self.cache_dir.glob("*.mp3")
                if p.is_file()
            ]
        except OSError:
            return

        total = sum(sz for _, sz, _ in entries)
        if total <= self.cache_max_bytes:
            return

        # Sort oldest-first, delete until under cap.
        entries.sort(key=lambda t: t[0])
        for _, sz, path in entries:
            try:
                path.unlink()
                total -= sz
            except OSError:
                continue
            if total <= self.cache_max_bytes:
                break

    # -------------------------------------------------------------- synthesis

    async def synthesize(self, req: TTSRequest) -> Path:
        """Synthesize ``req.text`` and return the path to the MP3 file.

        Uses cache when possible. Raises ``TTSUnavailableError`` if the CLI
        is missing and ``TTSSynthesisError`` on synth failure.
        """
        voice_id = VOICE_MAP.get(req.voice)
        if voice_id is None:
            raise TTSSynthesisError(
                f"Unknown voice '{req.voice}'. Available: {list(VOICE_MAP)}"
            )

        self._check_binary()

        key = self._cache_key(req, voice_id)
        out_path = self._cache_path(key)

        if out_path.exists() and out_path.stat().st_size > 0:
            self._touch(out_path)
            logger.debug("TTS cache hit", key=key[:12], bytes=out_path.stat().st_size)
            return out_path

        logger.info(
            "Synthesizing TTS",
            voice=voice_id,
            rate=req.rate,
            pitch=req.pitch,
            chars=len(req.text),
        )

        # edge-tts writes to --write-media. Text is passed via --text to
        # avoid stdin encoding quirks with Hebrew on some shells.
        # edge-tts uses argparse, which treats a separate arg like "-15%" as
        # a flag. Pass rate/pitch joined with "=" so the minus is kept as a
        # value. voice/text/output are safe as separate tokens.
        invocation = self._resolve_invocation()
        cmd = [
            *invocation,
            "-v", voice_id,
            f"--rate={self._normalize_signed(req.rate)}",
            f"--pitch={self._normalize_signed(req.pitch)}",
            "--text", req.text,
            "--write-media", str(out_path),
        ]

        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.SUBPROCESS_TIMEOUT_S
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise TTSSynthesisError(
                    f"edge-tts timed out after {self.SUBPROCESS_TIMEOUT_S}s"
                )
        except FileNotFoundError as e:
            raise TTSUnavailableError(str(e)) from e

        elapsed_ms = int((time.monotonic() - start) * 1000)

        if proc.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
            # Clean up partial artifact so we don't serve a corrupt cache entry.
            if out_path.exists():
                try:
                    out_path.unlink()
                except OSError:
                    pass
            err_text = (stderr or b"").decode("utf-8", errors="replace")[:500]
            raise TTSSynthesisError(
                f"edge-tts failed (rc={proc.returncode}): {err_text}"
            )

        logger.info(
            "TTS synthesis complete",
            elapsed_ms=elapsed_ms,
            bytes=out_path.stat().st_size,
        )

        # Best-effort eviction; never fatal.
        try:
            self._evict_if_needed()
        except Exception as evict_err:  # pragma: no cover - belt & braces
            logger.debug("Cache eviction skipped", error=str(evict_err))

        return out_path

    # ----------------------------------------------------------- multilingual

    async def synthesize_multilingual(
        self,
        text: str,
        *,
        hebrew_voice: str,
        rate: str,
        pitch: str,
    ) -> Path:
        """Synthesize ``text``, routing each sentence to a voice that matches
        its language (Hebrew → hebrew_voice, English → its ENGLISH_PAIR
        counterpart). Returns the path to a concatenated MP3.

        When the whole text is a single language run, we delegate to the plain
        ``synthesize`` path so the result is cache-hit-friendly.
        """
        english_voice = ENGLISH_PAIR.get(hebrew_voice, "jenny")
        segments = split_by_language(text, default_lang="he")

        if not segments:
            raise TTSSynthesisError("Empty text after language segmentation")

        # Single-language fast path — identical to the old behavior, keeps
        # cache keys stable for previously-synthesized strings. Pure-English
        # replies still get the English boost so the pace matches Hila.
        if len(segments) == 1:
            seg_text, lang = segments[0]
            if lang == "he":
                voice = hebrew_voice
                seg_rate = rate
            else:
                voice = english_voice
                seg_rate = _sum_signed_pct(rate, ENGLISH_RATE_BOOST)
            return await self.synthesize(
                TTSRequest(text=seg_text, voice=voice, rate=seg_rate, pitch=pitch)
            )

        # Multi-segment — synthesize each piece, then concat via ffmpeg.
        # English segments get ENGLISH_RATE_BOOST added to the user's base rate
        # so Jenny doesn't sound slower than Hila in a mixed reply.
        #
        # FIX 2026-04-23: per-segment resilience. Previously any single edge-tts
        # failure (e.g. a 4-char Hebrew punctuation chunk rc=1'ing on edge-tts)
        # aborted the whole reply, caller saw None, voice silently dropped.
        # Now we skip failed segments and continue; only abort if ALL fail.
        english_rate = _sum_signed_pct(rate, ENGLISH_RATE_BOOST)
        part_paths: List[Path] = []
        skipped: List[tuple] = []
        for seg_text, lang in segments:
            if lang == "he":
                voice = hebrew_voice
                seg_rate = rate
            else:
                voice = english_voice
                seg_rate = english_rate
            try:
                part_path = await self.synthesize(
                    TTSRequest(text=seg_text, voice=voice, rate=seg_rate, pitch=pitch)
                )
                part_paths.append(part_path)
            except TTSSynthesisError as exc:
                logger.warning(
                    "Per-segment TTS synth failed — skipping",
                    lang=lang,
                    chars=len(seg_text),
                    voice=voice,
                    preview=seg_text[:40],
                    error=str(exc)[:200],
                )
                skipped.append((lang, len(seg_text), seg_text[:40]))

        if not part_paths:
            # All segments failed — propagate so caller can fall back to text.
            raise TTSSynthesisError(
                f"All {len(segments)} segments failed synth; first_skipped={skipped[:1]}"
            )

        if skipped:
            logger.info(
                "Multilingual TTS partial synth",
                total=len(segments),
                ok=len(part_paths),
                skipped=len(skipped),
                skipped_sample=skipped[:3],
            )

        # Stable output filename keyed on the full multi-part input + voices
        # so consecutive identical replies hit cache on the joined file too.
        joined_key = hashlib.sha256(
            (
                f"MULTI|{hebrew_voice}|{english_voice}|{rate}|{pitch}|{text}"
            ).encode("utf-8")
        ).hexdigest()
        out_path = self.cache_dir / f"{joined_key}.mp3"
        if out_path.exists() and out_path.stat().st_size > 0:
            self._touch(out_path)
            return out_path

        # ffmpeg concat demuxer — robust for MP3s from the same encoder.
        # edge-tts always emits 24 kHz mono MP3 at ~48 kbps, so ``-c copy``
        # would work, but we re-encode to be safe against voice-specific
        # header variations.
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, dir=str(self.cache_dir)
        ) as list_file:
            list_path = Path(list_file.name)
            for p in part_paths:
                # ffmpeg concat list format: `file '<path>'` with single quotes
                # doubled to escape.
                escaped = str(p).replace("'", "'\\''")
                list_file.write(f"file '{escaped}'\n")

        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(list_path),
                "-c:a", "libmp3lame",
                "-b:a", "48k",
                "-ar", "24000",
                "-ac", "1",
                str(out_path),
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.SUBPROCESS_TIMEOUT_S
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise TTSSynthesisError(
                    f"ffmpeg concat timed out after {self.SUBPROCESS_TIMEOUT_S}s"
                )
        except FileNotFoundError as e:
            raise TTSUnavailableError(
                f"ffmpeg required for multilingual TTS concat: {e}"
            ) from e
        finally:
            try:
                list_path.unlink()
            except OSError:
                pass

        if proc.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
            err_text = (stderr or b"").decode("utf-8", errors="replace")[:500]
            if out_path.exists():
                try:
                    out_path.unlink()
                except OSError:
                    pass
            raise TTSSynthesisError(
                f"ffmpeg concat failed (rc={proc.returncode}): {err_text}"
            )

        logger.info(
            "Multilingual TTS concat complete",
            segments=len(segments),
            languages=[lang for _, lang in segments],
            bytes=out_path.stat().st_size,
        )
        # Eviction covers both per-segment and joined entries.
        try:
            self._evict_if_needed()
        except Exception:  # pragma: no cover
            pass
        return out_path


async def synthesize_for_user(
    synth: TTSSynthesizer,
    text: str,
    *,
    voice: str,
    rate: str,
    pitch: str,
) -> Optional[Path]:
    """Convenience wrapper that returns ``None`` on any failure.

    Routes through ``synthesize_multilingual`` so mixed Hebrew/English replies
    are spoken with the appropriate voice per sentence. ``voice`` is treated
    as the Hebrew voice choice; the English counterpart is picked from
    ``ENGLISH_PAIR`` (same gender persona).

    Use this from the reply-flow hook where TTS must never block delivery.
    """
    try:
        return await synth.synthesize_multilingual(
            text, hebrew_voice=voice, rate=rate, pitch=pitch
        )
    except TTSUnavailableError as e:
        logger.warning("TTS unavailable, skipping", error=str(e))
        return None
    except TTSSynthesisError as e:
        logger.warning("TTS synthesis failed, skipping", error=str(e))
        return None
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Unexpected TTS error, skipping", error=str(e))
        return None
