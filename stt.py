"""
stt.py — speech-to-text fallback (faster-whisper) + audio format registry.

Only used when no real lyrics are found online, which makes it the weakest
link in the pipeline: everything downstream — the rhyme map, the sync, the
video — is only as right as the words it starts from.

Whisper was trained on speech, and a rap record is not speech. The vocal
sits under an 808 and a hi-hat pattern, doubled and ad-libbed, delivered
at three times conversational speed. Three things follow, and this module
does all three:

1. SEPARATE THE VOCAL FIRST — when transcribing. Real source separation
   (demucs) lifts the voice off the beat and is by far the largest single
   improvement for a song with no lyrics online (73% → 99% of the song
   heard). It is optional because it is a heavy dependency.

   It is NOT used to time lyrics that are already known: measured, the
   full mix anchors more words than the separated vocal does, and
   separating costs minutes (see video._listen and listen_batched).

   What does NOT work, measured rather than assumed: band-passing the
   audio to the vocal range. It sounds like it should help and it makes
   the transcript materially worse (-0.42 → -0.73 mean confidence on the
   test track) — Whisper expects a natural full-band spectrum and a
   filtered one is out of distribution.

2. DECODE FOR MUSIC. Whisper's defaults are tuned to decide "is anyone
   talking?", and over a loud beat the answer keeps coming back no: on
   the test track the default settings silently dropped 27% of the song,
   hook included. Raising the silence bar and loosening the confidence
   floor recovers most of it (73% → 89% coverage).

3. SAY SO WHEN IT IS BAD. A transcript can be confidently wrong, and a
   wrong transcript is worse than none — it produces a whole video of a
   song nobody sang. Every run is scored and reported so the caller can
   tell the user what they are looking at.
"""

from __future__ import annotations

import pathlib
import statistics
import subprocess
import sys
import tempfile

AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac",
             ".wma", ".mp4", ".webm", ".mkv"}

# `base` is the faster-whisper default and it is not good enough here: on
# the test track it hears "Get a free voucher with the VD" where `small`
# hears the line. `small` roughly doubles decode time and is worth it —
# this path only runs when there are no real lyrics to be had, and the
# whole output is built on its words.
DEFAULT_MODEL = "small"

_models: dict = {}


def get_model(size: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("faster-whisper is not installed:  pip install faster-whisper")
    if size not in _models:
        _models[size] = WhisperModel(size, device="auto", compute_type="auto")
    return _models[size]


def have_separator() -> bool:
    """Whether vocal separation is available in this environment."""
    try:
        import demucs  # noqa: F401
        return True
    except ImportError:
        return False


_vocal_cache: dict = {}

# Everything expensive is kept ON DISK, keyed by the audio's content. The
# separated vocal and Whisper's word timings used to live only in memory,
# so opening a song you had already aligned — to fix a rhyme, to render
# another format — paid the full five-to-six minutes again. Now a song is
# listened to once, ever: reopening it is instant. Delete the folder to
# reclaim the space; nothing else depends on it.
CACHE = pathlib.Path.home() / ".cache" / "rhymescope"


def audio_key(path: str) -> str:
    """A fingerprint of the audio itself (not its name or location), so a
    renamed or re-downloaded copy of the same file still hits the cache."""
    import hashlib
    h = hashlib.sha1()
    p = pathlib.Path(path)
    with p.open("rb") as f:
        h.update(f.read(4 << 20))           # the first 4 MB …
        f.seek(max(0, p.stat().st_size - (1 << 20)))
        h.update(f.read())                  # … the last 1 MB, and the size
    h.update(str(p.stat().st_size).encode())
    return h.hexdigest()[:20]


def cached(kind: str, path: str, *parts) -> pathlib.Path:
    """Where a cached result for this audio lives."""
    tag = "-".join(str(x) for x in parts if x is not None)
    d = CACHE / audio_key(path)
    d.mkdir(parents=True, exist_ok=True)
    return d / (f"{kind}-{tag}" if tag else kind)


def isolate_vocal(path: str, progress=None) -> tuple[str, bool]:
    """Strip the instrumental, returning (path_to_use, separated?).

    Separation is the expensive step in the whole pipeline (minutes, on a
    laptop CPU), and both the transcription pass and the word-alignment
    pass want the same clean vocal — so it is done once per song and kept
    in the cache for good.

    Falls back to the original audio on any failure — a transcript from
    the full mix is worse than one from a clean vocal, but it is far
    better than no transcript because a separator crashed."""
    if not have_separator():
        return path, False
    if path in _vocal_cache:
        return _vocal_cache[path], True
    try:
        keep = cached("vocals.wav", path)
    except OSError:
        keep = None
    if keep is not None and keep.exists() and keep.stat().st_size > 0:
        _vocal_cache[path] = str(keep)
        return str(keep), True
    if progress:
        progress("separating the vocal from the beat …")
    try:
        tmp = tempfile.mkdtemp(prefix="rs-demucs-")
        subprocess.run(
            [sys.executable, "-m", "demucs", "--two-stems=vocals",
             *SEPARATE_ARGS, "-o", tmp, path],
            check=True, capture_output=True, timeout=1800)
        found = sorted(pathlib.Path(tmp).rglob("vocals.*"))
        if found:
            out = str(found[0])
            if keep is not None:
                import shutil
                shutil.move(out, keep)
                shutil.rmtree(tmp, ignore_errors=True)
                out = str(keep)
            _vocal_cache[path] = out
            return out, True
    except Exception:
        pass
    return path, False


# Separation in four parallel jobs: 285 s → 181 s on a 5½-minute song on
# a laptop CPU, with no change in what comes out.
SEPARATE_ARGS: list = ["-j", "4"]

# bump when the listening pass changes in a way that makes old cached
# word timings stale (model, decode settings, windowing …)
LISTEN_VERSION = 2

LISTEN_WINDOW = 28.0        # seconds per batched window (see listen_batched)


def listen_batched(model, path: str, language: str | None = None):
    """Word-timestamped decode, batched over fixed windows.

    faster-whisper's batched pipeline decodes several windows at once, but
    by default it chooses them with a voice-activity detector — which, on a
    full mix, throws away bars it takes for music. Fixed windows keep every
    second of the song. Measured: 68 s and 614 anchored words, against
    139 s / 627 unbatched and 57 s / 501 with the voice detector; 20-second
    windows cut too many words at their edges (485)."""
    from faster_whisper import BatchedInferencePipeline
    from faster_whisper.audio import decode_audio
    dur = len(decode_audio(path)) / 16000 - 0.05
    clips, t = [], 0.0
    while t < dur - 0.5:
        clips.append({"start": t, "end": min(dur, t + LISTEN_WINDOW)})
        t += LISTEN_WINDOW
    kw = {k: v for k, v in MUSIC_DECODE.items()
          if k not in ("condition_on_previous_text", "vad_filter")}
    if not clips:                       # a clip shorter than a window
        return model.transcribe(path, language=language,
                                word_timestamps=True, **MUSIC_DECODE)
    return BatchedInferencePipeline(model=model).transcribe(
        path, language=language, batch_size=8, word_timestamps=True,
        vad_filter=False, clip_timestamps=clips, **kw)


# Decode settings for singing over a beat, shared by every Whisper pass.
# Whisper's defaults are asking "is anyone talking?", and over a loud mix
# the answer keeps coming back no: on the test track they silently dropped
# 27% of the song, hook included. That matters twice over — those bars go
# missing from the transcript, AND the alignment pass gets no word
# timestamps there, so the sync in exactly the loudest passages falls back
# to interpolation and drifts.
MUSIC_DECODE = {
    "vad_filter": False,
    "beam_size": 5,
    "condition_on_previous_text": False,   # a repeated hook must not
                                           # convince it to loop forever
    "no_speech_threshold": 0.9,
    "log_prob_threshold": -1.5,
}


def _score(segments, duration: float) -> dict:
    """How much to trust this transcript.

    Two independent ways a music transcription goes wrong, so two numbers:
    it can hear the wrong words (low confidence), and it can skip whole
    passages it decided were instrumental (low coverage). A transcript can
    look fine on one and be useless on the other."""
    if not segments or not duration:
        return {"confidence": 0.0, "coverage": 0.0, "ok": False,
                "note": "nothing was transcribed"}
    conf = statistics.mean(s.avg_logprob for s in segments)
    cov = min(sum(s.end - s.start for s in segments) / duration, 1.0)
    # a decoder that has lost the thread repeats itself; a real song has
    # repeated hooks, so this has to be loud before it means anything
    lines = [s.text.strip().lower() for s in segments if s.text.strip()]
    rep = 1 - (len(set(lines)) / len(lines)) if lines else 0.0
    note = ""
    if conf < -1.0:
        note = "the vocal was hard to make out — expect wrong words"
    elif cov < 0.45:
        note = "much of the track came back as instrumental"
    elif rep > 0.6:
        note = "the transcript repeats itself — it may have looped"
    return {"confidence": conf, "coverage": cov, "repetition": rep,
            "ok": not note, "note": note}


def transcribe(path: str, model_size: str = DEFAULT_MODEL,
               language: str | None = None, progress=None) -> str:
    """Transcribe an audio file. `progress(pct)` gets 0..1 callbacks."""
    return transcribe_with_timing(path, model_size, language, progress)[0]


def transcribe_with_timing(path: str, model_size: str = DEFAULT_MODEL,
                           language: str | None = None, progress=None,
                           separate: bool = True
                           ) -> tuple[str, list | None, dict]:
    """Transcribe audio and return (text, sync_list, report).

    `sync_list` is [(seconds, line_text), …], or None if timing fails.
    `report` carries confidence/coverage and a plain-language note when
    the result should not be trusted — see `_score`."""
    src, separated = (isolate_vocal(path, progress) if separate
                      else (path, False))
    model = get_model(model_size)
    segments, info = model.transcribe(src, language=language, **MUSIC_DECODE)
    lines, sync, kept = [], [], []
    for seg in segments:
        t = seg.text.strip()
        if t:
            lines.append(t)
            sync.append((seg.start, t))
            kept.append(seg)
        if progress and info.duration:
            progress(min(seg.end / info.duration, 1.0))

    report = _score(kept, info.duration)
    report["separated"] = separated
    report["model"] = model_size
    return "\n".join(lines), (sync or None), report
