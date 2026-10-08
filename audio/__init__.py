"""Hearing the rhyme, not just reading it.

    heard = audio.heard_vowels("song.webm", lyrics_text)
    rhyme.detect(lyrics_text, heard=heard)

Rappers change their accent to make things rhyme — "film" sung to meet
"tell", "caveat" pushed toward "got". The lyrics sheet cannot show that;
the vocal can. `heard_vowels` returns, for every syllable of the lyrics it
could place in the audio, how strongly the voice sounded each vowel class,
keyed "line:char_start:syllable". The rhyme engine uses it as a second
pronunciation beside the dictionary's (rhyme_engine.analyze, `heard`).

Everything is cached per song in the rhymescope cache (~/.cache/rhymescope):
the separated vocal, the phoneme posteriors and the result. The first run
on a song takes a few minutes on a CPU; after that it is instant.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

VERSION = 3
_MIN_FIT = -5.0          # a bar whose alignment fits worse is not used
_BAND = 3.0              # seconds a line may drift from the line sync


def _disk(audio: str, text: str) -> pathlib.Path:
    import stt
    tag = hashlib.sha1(text.encode()).hexdigest()[:12]
    return stt.cached("heard-vowels.json", audio, VERSION, tag)


def cached_heard(audio: str | None, text: str) -> dict | None:
    """heard_vowels' result if it was already computed, else None — for
    callers that must not start minutes of work (the scorecard, the UI
    before the user asks)."""
    if not audio or not pathlib.Path(audio).exists():
        return None
    disk = _disk(audio, text)
    try:
        return json.loads(disk.read_text()) if disk.exists() else None
    except (OSError, ValueError):
        return None


def find_audio(base: pathlib.Path) -> str | None:
    """The downloaded audio beside a lyrics/source path, if any."""
    stem = base.name.split(".lyrics")[0]
    stem = stem[:-len(base.suffix)] if base.suffix and stem.endswith(base.suffix) else stem
    for ext in (".webm", ".m4a", ".mp3", ".opus", ".wav", ".flac"):
        p = base.with_name(stem + ext)
        if p.exists():
            return str(p)
    return None


def heard_vowels(audio: str, text: str, progress=None) -> dict:
    disk = _disk(audio, text)
    if disk.exists():
        try:
            return json.loads(disk.read_text())
        except ValueError:
            pass
    from lexicon import WORD_RE, phones_for
    from timing import sync_from_alignment
    from . import phonemes
    from .align import align_song, targets, vowel_scores
    from .phonemes import FRAME
    lp = phonemes.posteriors(audio, progress)
    vocab = phonemes.vocab()
    lines = text.splitlines()
    rep: dict = {}
    sync = sync_from_alignment(pathlib.Path(audio), lines, report=rep,
                               progress=progress) or []
    starts = [t for t, _l in sync] + [None] * (len(lines) - len(sync))
    tgts, wheres, bands = [], [], []
    for ln, line in enumerate(lines):
        tgt, where = [], []
        for m in WORD_RE.finditer(line):
            ph, _oov = phones_for(m.group(0))
            if not ph:
                continue
            for ids, k in targets(ph, vocab):
                tgt.append((ids, k))
                where.append(None if k is None else f"{ln}:{m.start()}:{k}")
        tgts.append(tgt)
        wheres.append(where)
        t0 = starts[ln] if starts[ln] is not None else 0.0
        nxt = next((starts[j] for j in range(ln + 1, len(lines))
                    if starts[j] is not None and lines[j].strip()), None)
        t1 = nxt if nxt is not None else len(lp) * FRAME
        bands.append((max(int((t0 - _BAND) / FRAME), 0),
                      min(int((t1 + _BAND) / FRAME), len(lp))))
    aligned = align_song(lp, tgts, bands)
    out: dict = {}
    fits = []
    line_fit: dict = {}
    for ln, res in enumerate(aligned):
        if res is None:
            continue
        spans, fit = res
        fits.append(fit)
        line_fit[str(ln)] = round(fit, 2)       # JSON keys: str either way
        if fit < _MIN_FIT:
            continue
        for (a_, b_), key in zip(spans, wheres[ln]):
            if key:
                out[key] = {c: round(v, 2)
                            for c, v in vowel_scores(lp, a_, b_, vocab).items()}
    result = {"syllables": out, "sync": rep, "line_fit": line_fit,
              "fit": round(sum(fits) / len(fits), 2) if fits else None}
    disk.write_text(json.dumps(result))
    return result
