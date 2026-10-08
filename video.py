"""
video.py — render a rhyme-map video (mp4) from an Analysis + synced lyrics.

No screen recording: frames are PIL-drawn at 1920x1080 and ffmpeg assembles
them with the audio using exact timings.

Word-by-word timing: LRC gives line timestamps; word start times inside a
line are interpolated from syllable counts, then (optionally) snapped to the
actual vocal via Whisper word-level timestamps aligned against the lyrics.

Themes map to the approved short-form variants of the video spec
(rhymescope-video-spec.md): "dark" is variant 1a (rhymes on ink),
"light" is variant 1b (paper, black header band, lime diagonal cut).

The 16:9 lyrics-only render follows the brand kit's §3a YouTube frame spec
instead — a fixed off-white paper look with three frames (01 title,
02 verse, 04 end screen), no theme variants. See the YT_ section below.
"""

from __future__ import annotations

import bisect
import colorsys
import difflib
import functools
import math
import multiprocessing
import os
import pathlib
import re as _re
import statistics
import subprocess
import tempfile

from PIL import Image, ImageColor, ImageDraw, ImageFont

from term_render import syllable_map
from rhyme_engine import Analysis

# Video formats: width, height — the layout adapts to the orientation
# (landscape: stats + wide window; portrait: a 16:9 slot up top, the rhyme
# scheme filling everything under it)
FORMATS = {
    "youtube": (1920, 1080),
    "tiktok": (1080, 1920),
    "instagram": (1080, 1920),
}

# RhymeScope brand (Rhymescope Brand Kit.dc.html · rhymescope-video-spec.md):
#   Ink #0A0A0A · Paper #FAFAF8 · Lime #CBF135 · Badge blue #2E5BD8
#   Anton titles/wordmark · Space Grotesk 300/500 lyrics · Space Mono labels
#   Lime marks at most one text element per frame (bar counter / CTA /
#   artist name); fixed accents (1b diagonal, lower-third bar) don't count.
INK = "#0A0A0A"
PAPER = "#FAFAF8"
LIME = "#CBF135"
BADGE = "#2E5BD8"
DIMTX = "#5E5E66"

# Highlighter set — one colour per rhyme sound. palette.assign() reuses
# them adjacency-aware when a song has more families than colours.
HIGHLIGHTERS = ["#FBA6B6", "#29C8DE", "#12A67C",
                "#B79EE9", "#F5893F", "#C9BA3E"]

# The 9:16 vertical spec's own palette. Light is the spec as written;
# dark is the same frame on ink, with every grey contrast-matched to its
# light counterpart against the page rather than guessed — an unsaid bar
# sits 2.4:1 off the page on ink exactly as #A8A8A2 sits 2.35:1 off paper,
# so the two themes read equally dim. The lime rule and the six chip
# colours are unchanged: they are the brand, not the theme.
RSV_THEMES = {
    "light": {                  # the spec as written — rhymes on paper
        "bg": "#FDFDFC",        # page
        "head": "#0A0A0A",      # header band
        "head_fg": "#FAFAF8",   # wordmark
        "head_dim": "#6E6E76",  # EP. 041
        "slot": "#EDEDE8",      # clip band fill
        "slot_fg": "#B4B4AC",   # clip band label
        "rule": LIME,           # the only saturated colour outside the chips
        "label": "#86867E",     # scheme label
        "dim": "#9A9A93",       # a bar not said yet — a step darker than
                                # the spec's #A8A8A2, for the smaller type
        "fg": "#0A0A0A",        # the bar being said
        "hair": "#E6E6E0",      # footer rule
        "handle": "#C2C2BA",    # @rhymescope
    },
    "dark": {                   # the same frame on ink (brand variant 1a)
        "bg": INK,
        "head": INK,            # the band dissolves into the page: a paper
                                # slab at the top of a dark frame glares on
                                # a phone, and the clip window right under
                                # it already divides the frame
        "head_fg": PAPER,
        "head_dim": "#6E6E76",
        "slot": "#1C1C1E",      # lifted off the page by the same 1.16:1
        "slot_fg": "#46464C",   # that #EDEDE8 sits under #FDFDFC on paper
        "rule": LIME,
        "label": "#6E6E76",
        "dim": "#5C5C64",
        "fg": PAPER,
        "hair": "#202024",
        "handle": "#3C3C42",
    },
}

THEMES = {
    "dark": {                       # spec variant 1a: rhymes on ink
        "variant": "1a",
        "bg": INK, "band": INK, "slot": "#141414", "edge": "#2A2A2E",
        "fg": PAPER, "past": DIMTX, "next": "#3C3C42",
        "wordmark": PAPER, "tag": LIME, "dim": "#8A8A92",
        "rsv": RSV_THEMES["dark"],
    },
    "light": {                      # spec variant 1b: paper + black band
        "variant": "1b",
        "bg": PAPER, "band": INK, "slot": INK, "edge": "#E3E3DE",
        "fg": INK, "past": DIMTX, "next": "#C9C9CE",
        "wordmark": PAPER, "tag": LIME, "dim": "#7A7A82",
        "rsv": RSV_THEMES["light"],
    },
}


from palette import (assign as _assign, ink_for as _ink_for,
                     highlighter_pool as _hpool, sound_colors as _sound_colors)

# _ink_for measures WCAG contrast against both candidates and returns the
# one that actually reads on a given chip fill. All six brand highlighters
# are light enough that ink wins, so the words stay ink-dark on every chip
# — but the measurement stays, so a future palette change can't silently
# produce white-on-mid-tone mud.


def _mix(c1: str, c2: str, t: float) -> str:
    """Blend hex color c1 toward c2 by t (0..1)."""
    a = [int(c1[k:k + 2], 16) for k in (1, 3, 5)]
    b = [int(c2[k:k + 2], 16) for k in (1, 3, 5)]
    return "#{:02x}{:02x}{:02x}".format(
        *(round(x + (y - x) * t) for x, y in zip(a, b)))


def _palette(analysis, theme: dict | None = None,
             ref: dict | None = None,
             line_map: list | None = None) -> list[str]:
    """Per-family colors, adjacency-aware: two families that share a line
    can never be handed the same colour. The brand highlighters come
    first; past them the pool keeps generating, so a song with more rhymes
    than the brand has chips gets more colours rather than repeats.
    `ref` pins colours from a whole-song map (see _color_ref)."""
    return _assign(analysis, lambda n: _hpool(n, tuple(HIGHLIGHTERS)),
                   ref=ref, line_map=line_map)


def _color_ref(analysis) -> dict:
    """The whole song's colouring, keyed by rhyme sound.

    Every excerpt renderer re-analyzes its own slice — that is what makes
    a verse's colours describe the verse — but re-analysis also renumbers
    the families, so without this the same rhyme comes out pink in verse 1
    and green in verse 3. Handing each excerpt the song-level map keeps
    one rhyme one colour across every clip and every format."""
    return _sound_colors(analysis, _palette(analysis))

_fonts: dict = {}
_FONT_DIR = pathlib.Path(__file__).resolve().parent / "fonts"


def _font(kind: str = "body", size: int = 44,
          weight: int = 600, width: int = 100) -> ImageFont.FreeTypeFont:
    """Brand type per the video spec: Anton for titles/wordmark, Space
    Grotesk 300/500 for lyrics and captions, Space Mono for labels and
    timecodes. `archivo` is the 16:9 spec's face (§3a), the only kind that
    uses `width` — Archivo's axes are [Weight, Width], in that order.
    Falls back to the previous brand files, then system fonts."""
    key = (kind, size, weight, width)
    if key in _fonts:
        return _fonts[key]
    tries = {
        # (file, variable-font axes to set — Space Grotesk is [wght]
        #  300–700, Manrope [wght], Archivo [wght, wdth], Inter
        #  [opsz, wght])
        "display": [("Anton-Regular.ttf", None), ("SpaceGrotesk.ttf", [700])],
        "body": [("SpaceGrotesk.ttf", [min(max(weight, 300), 700)]),
                 ("Manrope.ttf", [weight]),
                 ("Archivo.ttf", [weight, 100]),
                 ("Inter.ttf", [28, weight])],
        "archivo": [("Archivo.ttf", [weight, width]),
                    ("SpaceGrotesk.ttf", [min(max(weight, 300), 700)]),
                    ("Inter.ttf", [28, weight])],
        "mono": [("SpaceMono-Bold.ttf" if weight >= 700
                  else "SpaceMono-Regular.ttf", None)],
    }[kind]
    f = None
    for name, axes in tries:
        try:
            f = ImageFont.truetype(str(_FONT_DIR / name), size)
            if axes:
                f.set_variation_by_axes(axes)
            break
        except OSError:
            continue
    if f is None:
        for path in ([f"/usr/share/fonts/noto/NotoSans-{'Bold' if weight >= 700 else 'Regular'}.ttf",
                      f"/usr/share/fonts/TTF/DejaVuSans{'-Bold' if weight >= 700 else ''}.ttf"]):
            try:
                f = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
    _fonts[key] = f or ImageFont.load_default()
    return _fonts[key]


def _logo(d: ImageDraw.ImageDraw, x: int, y: int, size: int,
          badge: str = BADGE, mark: str = PAPER,
          radius: float | None = None) -> None:
    """rhymescope-icon: blue rounded square, crosshair + waveform — a PIL
    redraw of rhymescope-icon.svg on its 64-unit viewBox. `radius` overrides
    the corner radius in px (the 16:9 spec asks for ~27% of the size)."""
    k = size / 64
    lw = max(2, round(2.5 * k))
    d.rounded_rectangle([x, y, x + size, y + size],
                        radius=12 * k if radius is None else radius,
                        fill=badge)

    def R(x0, y0, w0, h0):              # rounded waveform bar
        d.rounded_rectangle([x + x0 * k, y + y0 * k,
                             x + (x0 + w0) * k, y + (y0 + h0) * k],
                            radius=1.5 * k, fill=mark)

    d.ellipse([x + 12 * k, y + 12 * k, x + 52 * k, y + 52 * k],
              outline=mark, width=lw)
    for x1, y1, x2, y2 in ((32, 6, 32, 13), (32, 51, 32, 58),
                           (6, 32, 13, 32), (51, 32, 58, 32)):
        d.line([x + x1 * k, y + y1 * k, x + x2 * k, y + y2 * k],
               fill=mark, width=lw)
    R(23, 28, 3, 9)
    R(28.5, 22, 3, 20)
    R(34, 26, 3, 13)
    R(39.5, 24, 3, 17)


def _audio_duration(path: pathlib.Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)], text=True)
    return float(out.strip())


# --------------------------------------------------------------------------
# word-level timing (shared with the terminal karaoke)
# --------------------------------------------------------------------------

def word_schedule(sync: list, words_by_line: dict,
                  tail: float = 4.0) -> list[list[float]]:
    """Per lyrics line: the start time of each word, interpolated inside the
    line's LRC window proportional to syllable counts. Rap runs ~4-5
    syllables/second, so the sung part of a window is capped instead of
    stretching words across instrumental gaps."""
    out: list[list[float]] = []
    for i, (t0, _txt) in enumerate(sync):
        wl = words_by_line.get(i, [])
        if not wl:
            out.append([])
            continue
        t1 = sync[i + 1][0] if i + 1 < len(sync) else t0 + tail
        gap = max(t1 - t0, 0.3)
        syls = [max(len(w.syllables), 1) for _, w in wl]
        total = sum(syls)
        sung = min(gap, max(1.0, 0.24 * total))
        starts, acc = [], 0
        for s in syls:
            starts.append(t0 + sung * acc / total)
            acc += s
        out.append(starts)
    return out


def _norm(word: str) -> str:
    w = _re.sub(r"[^a-z0-9]", "", word.lower())
    # Whisper writes standard spellings, lyrics use dropped-G ("runnin'"):
    # canonicalize -ing → -in so those words still anchor. Short words
    # (sing, king, thing) keep their g to avoid false matches.
    if len(w) >= 6 and w.endswith("ing"):
        w = w[:-1]
    return w


@functools.lru_cache(maxsize=None)
def _akey(tok: str) -> str:
    """Alignment key: the word's phonetic signature, so spelling variants
    anchor ('cuz'/'cause', 'runnin''/'running', homophones, numbers) —
    the ear heard sounds, not spellings."""
    from rhyme_engine import phones_for, ARPABET_VOWELS, CONS_CLASS
    if not tok:
        return tok
    try:
        ph, _ = phones_for(tok)
    except Exception:
        return tok
    if not ph:
        return tok
    from rhyme_engine import VOWEL_CLASS
    out = []
    for p in ph.split():
        b = p.rstrip("012")
        if b in ARPABET_VOWELS:
            out.append(VOWEL_CLASS.get(b, b))    # class level: cuz == cause
        else:
            out.append(CONS_CLASS.get(b, b) or "")
    return ".".join(out)


_heard_cache: dict = {}
_listen_locks: dict = {}


def _listen(audio: pathlib.Path, model_size: str, language: str | None,
            progress=None) -> list[tuple[str, float]]:
    """Whisper word-level pass over the audio → [(token, start_s), …].
    Cached per file so timing + alignment share one transcription."""
    key = (str(audio), model_size, language)
    if key in _heard_cache:
        return _heard_cache[key]
    # the app starts this pass in the background as soon as a song loads;
    # a render that asks for it meanwhile waits for that work to finish
    # rather than starting a second copy of it
    import threading
    lock = _listen_locks.setdefault(key, threading.Lock())
    with lock:
        if key in _heard_cache:
            return _heard_cache[key]
        return _listen_now(audio, model_size, language, progress, key)


def _listen_now(audio, model_size, language, progress, key):
    import json
    import stt
    # heard once per song, ever: the result is kept on disk beside the
    # separated vocal (stt.CACHE), so reopening a song skips all of this
    try:
        disk = stt.cached("heard.json", str(audio), model_size, language,
                          stt.LISTEN_VERSION)
        if disk.exists():
            heard = [tuple(x) for x in json.loads(disk.read_text())]
            _heard_cache[key] = heard
            return heard
    except (OSError, ValueError):
        disk = None
    # The same clean vocal the transcriber used (cached, so this is free
    # when lyrics came from Whisper) and the same music-tuned decode. Both
    # matter more here than they look: an anchor can only be placed where
    # a word was heard, so every bar Whisper skips under a loud beat is a
    # stretch of lyrics with nothing to pin it to, left to interpolation.
    # Measured on a 5½-minute song (see stt.py): when the LYRICS ARE KNOWN,
    # separating the vocal first does not help this pass — the full mix
    # anchored MORE words (614 of 794 against ~517) — and it cost 4-5
    # minutes. Separation stays where it earns its keep: transcribing a
    # song with no lyrics online. And batching over fixed 28-second windows
    # halves the decode without the voice detector, which drops sung bars
    # it takes for music: ~7 minutes → ~1, with better sync.
    if progress:
        progress("listening")
    model = stt.get_model(model_size)
    segments, info = stt.listen_batched(model, str(audio), language)
    heard: list[tuple[str, float]] = []
    for seg in segments:
        for w in seg.words or []:
            tok = _norm(w.word)
            if tok:
                heard.append((tok, w.start))
        if progress and info.duration:
            progress(min(seg.end / info.duration, 1.0))
    _heard_cache[key] = heard
    if disk is not None and heard:
        try:
            disk.write_text(json.dumps(heard))
        except OSError:
            pass
    return heard


def _even_timing(lines: list[str], t0: float, t1: float,
                 syl) -> list[tuple[float, str]]:
    """Timing with nothing to go on: spread the lyrics across the window
    in proportion to how long each line takes to say.

    This is the floor, not the goal. It exists because the alternative —
    what this used to do — was to refuse: a song whose lyrics could not
    be matched to its vocal got no timing at all, which meant no karaoke
    and no video, for a song whose lyrics were sitting right there. Bars
    are not evenly spaced in time, so this drifts; but it drifts around
    the right place, every line lands in the right order, and the render
    is something you can watch and nudge with --offset rather than an
    error message."""
    load = [max(sum(syl(w) for w in ln.split()), 1) for ln in lines]
    cum, run = [0], 0
    for v in load:
        run += v
        cum.append(run)
    span = max(t1 - t0, 1.0)
    return [(t0 + span * cum[i] / max(cum[-1], 1), lines[i])
            for i in range(len(lines))]


def sync_from_alignment(audio: pathlib.Path, lines: list[str],
                        model_size: str = "base", language: str | None = None,
                        progress=None, report: dict | None = None
                        ) -> list[tuple[float, str]] | None:
    """Build [(seconds, line_text), …] for plain lyrics (Genius etc. — no
    LRC timestamps) by listening to the audio: Whisper word timestamps are
    sequence-matched against the lyrics and each line starts at its first
    matched word; unmatched lines interpolate between anchored neighbors.

    This ALWAYS returns a timing when there are lyrics and audio. Where
    the two line up it is anchored to the vocal, word by word; where they
    do not it falls back to spreading the lines across the track, and says
    so in `report` — {"anchors": n, "lines": n, "matched": 0..1,
    "mode": "vocal"|"partial"|"even"} — so the caller can tell you how
    much to trust it instead of the song simply failing."""
    rep = report if report is not None else {}
    rep.update(anchors=0, lines=len(lines), matched=0.0, mode="even")
    from rhyme_engine import phones_for, ARPABET_VOWELS

    @functools.lru_cache(maxsize=None)
    def _syl(word: str) -> int:
        ph, _ = phones_for(word)
        if not ph:
            return 1
        return max(1, sum(1 for x in ph.split()
                          if x.rstrip("012") in ARPABET_VOWELS))

    tokens: list[str] = []
    tok_line: list[int] = []
    for i, ln in enumerate(lines):
        for wd in ln.split():
            t = _norm(wd)
            if t:
                tokens.append(t)
                tok_line.append(i)
    if not tokens:
        return None

    duration = _audio_duration(audio)
    heard = _listen(audio, model_size, language, progress)
    if len(heard) < 20:
        # nothing was heard to anchor against — an instrumental-heavy
        # mix, a language Whisper could not follow, a very short clip
        return _even_timing(lines, 0.0, duration or 60.0, _syl)

    sm = difflib.SequenceMatcher(None, [_akey(t) for t, _ in heard],
                                 [_akey(t) for t in tokens],
                                 autojunk=False)
    line_t: dict[int, float] = {}
    matched = 0
    for blk in sm.get_matching_blocks():
        for off in range(blk.size):
            ln = tok_line[blk.b + off]
            ts = heard[blk.a + off][1]
            if ln not in line_t or ts < line_t[ln]:
                line_t[ln] = ts
            matched += 1
    n = len(lines)
    rep["matched"] = matched / max(len(tokens), 1)
    if len(line_t) < max(4, n // 8) or matched < len(tokens) // 5:
        # The lyrics and the vocal do not line up. That can mean a
        # different edit, a heavy mix, a language Whisper mangled — or
        # lyrics for a different song altogether, which is why `matched`
        # is reported rather than swallowed. Either way, the song still
        # gets a timing.
        first = heard[0][1] if heard else 0.0
        last = heard[-1][1] if heard else (duration or 60.0)
        return _even_timing(lines, max(first - 0.5, 0.0),
                            max(last + 2.0, first + 5.0), _syl)

    # syllables per line — timing must respect ARTICULATION RATE: nobody
    # raps faster than ~7 syllables/second, so any anchor set implying
    # that is wrong, however well the words matched
    line_syl = [sum(_syl(w) for w in ln.split()) for ln in lines]
    cum = [0]
    for v in line_syl:
        cum.append(cum[-1] + v)
    MIN_SPS = 0.14                     # s per syllable, physical floor

    # keep the longest chain of anchors that is monotonic AND physically
    # possible — one mis-matched anchor must not crush the lines after it
    cand = sorted(line_t)
    best = [1] * len(cand)
    prev = [-1] * len(cand)
    for j in range(len(cand)):
        for i in range(j):
            need = MIN_SPS * (cum[cand[j]] - cum[cand[i]])
            if (line_t[cand[j]] - line_t[cand[i]] >= need
                    and best[i] + 1 > best[j]):
                best[j] = best[i] + 1
                prev[j] = i
    k = max(range(len(cand)), key=lambda x: best[x])
    chain = []
    while k >= 0:
        chain.append(cand[k])
        k = prev[k]
    anchors = sorted(chain)
    if not anchors:
        first = heard[0][1] if heard else 0.0
        last = heard[-1][1] if heard else (duration or 60.0)
        return _even_timing(lines, first, max(last + 2.0, first + 5.0), _syl)
    rep.update(anchors=len(anchors),
               mode="vocal" if len(anchors) >= max(4, n // 4) else "partial")

    # observed pace of the anchored region, for lead-in and tail
    spans = [(line_t[b] - line_t[a]) / max(cum[b] - cum[a], 1)
             for a, b in zip(anchors, anchors[1:])]
    sps = min(max(statistics.median(spans), 0.16), 0.45) if spans else 0.28

    times: list[float | None] = [None] * n
    for i in anchors:
        times[i] = line_t[i]
    first, final = anchors[0], anchors[-1]
    for i in range(first - 1, -1, -1):            # lead-in at observed pace
        times[i] = max(0.0, times[i + 1] - sps * max(line_syl[i], 2))
    # interpolate between anchors weighted by syllable load
    for a, b in zip(anchors, anchors[1:]):
        span = times[b] - times[a]
        total = max(cum[b] - cum[a], 1)
        for m in range(a + 1, b):
            times[m] = times[a] + span * (cum[m] - cum[a]) / total
    # tail at observed pace, stretched toward where the vocal really ends
    vocal_end = heard[-1][1] + 2.0
    tail_syl = cum[n] - cum[final]
    tail_time = max(sps * tail_syl,
                    min(vocal_end - times[final], 0.45 * tail_syl))
    for i in range(final + 1, n):
        times[i] = times[final] + tail_time * (cum[i] - cum[final]) \
            / max(tail_syl, 1)
    for i in range(1, n):             # strictly increasing
        if times[i] <= times[i - 1]:
            times[i] = times[i - 1] + 0.3
    if progress:
        progress(("anchors", len(anchors), n, 0.0))
    return [(times[i], lines[i]) for i in range(n)]


def align_word_times(audio: pathlib.Path, sync: list, words_by_line: dict,
                     model_size: str = "base", language: str | None = None,
                     progress=None) -> list[list[float]] | None:
    """Refine the interpolated schedule by listening to the audio: Whisper
    emits word-level timestamps, those words are sequence-aligned against
    the real lyrics, and matched words snap to the time they were actually
    sung (unmatched words re-interpolate between the snapped anchors).
    Returns a schedule like word_schedule(), or None if alignment failed."""
    base = word_schedule(sync, words_by_line)
    flat: list[tuple[int, int]] = [(ln, k)
                                   for ln, starts in enumerate(base)
                                   for k in range(len(starts))]
    if not flat:
        return None
    base_t = [base[ln][k] for ln, k in flat]
    lyric_tok = [_norm(words_by_line[ln][k][1].text) for ln, k in flat]

    heard = _listen(audio, model_size, language, progress)
    if len(heard) < 20:
        return None

    sm = difflib.SequenceMatcher(None, [_akey(t) for t, _ in heard],
                                 [_akey(t) for t in lyric_tok],
                                 autojunk=False)
    anchors: dict[int, float] = {}
    for blk in sm.get_matching_blocks():
        for off in range(blk.size):
            anchors[blk.b + off] = heard[blk.a + off][1]

    # sanity: if Whisper's timing wildly disagrees with the LRC skeleton,
    # the audio is probably a different edit — keep the interpolation
    good = {i: t for i, t in anchors.items() if abs(t - base_t[i]) <= 5.0}
    if len(good) < max(10, len(flat) // 10):
        return None

    # smoothness: drop anchors that disagree with their neighbors — one
    # bad Whisper timestamp must not warp the words around it
    idxs = sorted(good)
    offs = {i: good[i] - base_t[i] for i in idxs}
    keep: dict = {}
    for k, i in enumerate(idxs):
        lo, hi = max(0, k - 2), min(len(idxs), k + 3)
        local = sorted(offs[idxs[m]] for m in range(lo, hi))
        if abs(offs[i] - local[len(local) // 2]) <= 1.2:
            keep[i] = good[i]
    good = keep or good
    drift = statistics.median(t - base_t[i] for i, t in good.items())

    # ------------------------------------------------------------------
    # Global monotone fit. Per-line interpolation lets one bad anchor bend
    # only its own neighbourhood, which is why timing used to wobble bar
    # to bar. Instead every anchor is treated as a sample of one warp
    # curve — lyric position → audio time — and the curve is forced
    # monotone (time never runs backwards) by pooling adjacent violators.
    # A word's time is then read off the curve, so the whole verse drifts
    # together instead of snapping around individual anchors.
    # ------------------------------------------------------------------
    idxs = sorted(good)
    # pool-adjacent-violators: every anchor is KEPT (its position is real
    # information); only its time is corrected, by averaging it with the
    # neighbours it contradicts until the sequence never runs backwards.
    blocks: list[list] = []                    # [mean, weight, [indices]]
    for i in idxs:
        blocks.append([good[i], 1.0, [i]])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            m2, w2, i2 = blocks.pop()
            m1, w1, i1 = blocks.pop()
            wt = w1 + w2
            blocks.append([(m1 * w1 + m2 * w2) / wt, wt, i1 + i2])
    good = {i: b[0] for b in blocks for i in b[2]}
    idxs = sorted(good)

    # piecewise re-interpolation between anchors
    refined = list(base_t)
    for a, b in zip(idxs, idxs[1:]):
        ta, tb = good[a], good[b]
        refined[a], refined[b] = ta, tb
        span = base_t[b] - base_t[a]
        for m in range(a + 1, b):
            frac = (base_t[m] - base_t[a]) / span if span > 0 else 0
            refined[m] = ta + (tb - ta) * frac
    for m in range(idxs[0]):                       # before the first anchor
        refined[m] = max(0.0, base_t[m] + (good[idxs[0]] - base_t[idxs[0]]))
    for m in range(idxs[-1] + 1, len(flat)):       # after the last one
        refined[m] = base_t[m] + (good[idxs[-1]] - base_t[idxs[-1]])
    for m in range(1, len(refined)):               # keep time moving forward,
        # with a floor between words so reveals never pile up in one frame
        refined[m] = max(refined[m], refined[m - 1] + 0.05)

    out = [list(starts) for starts in base]
    for (ln, k), t in zip(flat, refined):
        out[ln][k] = t
    if progress:
        progress(("anchors", len(good), len(flat), drift))
    return out


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------

def _word_runs(w, colors, fam_at, w_i):
    """One word as [(syllable_text, family_color_or_None), …]."""
    if w.syllables and any((w_i, k) in fam_at for k in range(len(w.syllables))):
        return [(w.text[s:e],
                 colors[fam_at[(w_i, k)]] if (w_i, k) in fam_at else None)
                for k, (s, e) in enumerate(w.spans)]
    return [(w.text, None)]


def _fit(d, text, font, max_w):
    """Ellipsize text that would overflow max_w pixels."""
    if d.textlength(text, font=font) <= max_w:
        return text
    while text and d.textlength(text + "…", font=font) > max_w:
        text = text[:-1]
    return text + "…"


def _draw_line(d, x, y, colors, fam_at, wl, raw, active, plain,
               font, T, max_width, stroke=0):
    """Draw one lyrics line; words with index >= active are dimmed and
    unmarked. active=None means the whole line is revealed. `stroke`
    thickens every glyph outward by that many px — Space Grotesk tops out
    at weight 700, so the active bar wears a 1px stroke to read ~800.

    Chips (spec: padding 2px 10px, radius 6) are painted in a first pass
    and every word in a second, so a chip's side padding can never cover
    the tail of the word drawn before it."""
    # Long lines WRAP (see _wrap_rows) instead of shrinking: shrinking a
    # dense bar to fit produced a row half the size of its neighbours,
    # which is unreadable on a phone. Only a single unbreakable word
    # still falls back to shrinking.
    f = font
    if len(wl) == 1:                    # measure THIS row's word, not the
        seg = raw[wl[0][1].start:wl[0][1].end]   # whole lyric line — a
        total = d.textlength(seg, font=f)        # wrapped one-word row is
        if total > max_width and total > 0:      # not an overflow
            f = _font(size=max(24, int(font.size * max_width / total)))
    ascent, _ = f.getmetrics()

    # `raw` is the FULL lyric line while `wl` may be just one wrapped row,
    # so the cursor starts at this row's first word and the trailing text
    # stops at its last — otherwise a row re-draws the rest of the line.
    cursor = wl[0][1].start if wl else 0
    raw_stop = wl[-1][1].end if wl else len(raw)
    x_limit = x + max_width

    chips: list[tuple[float, float, str]] = []
    texts: list[tuple[float, str, str]] = []
    clipped = False

    for k, (w_i, w) in enumerate(wl):
        gap_txt = raw[cursor:w.start]
        if gap_txt:
            gap_w = d.textlength(gap_txt, font=f)
            if x + gap_w > x_limit:
                clipped = True
                break
            texts.append((x, gap_txt, plain))
            x += gap_w
        cursor = w.end

        revealed = active is None or k < active

        if revealed:
            for t, color in _word_runs(w, colors, fam_at, w_i):
                seg_w = d.textlength(t, font=f)
                if x + seg_w > x_limit:
                    clipped = True
                    break
                if color and t.strip():
                    chips.append((x, x + seg_w, color))
                    texts.append((x, t, _ink_for(color, dark=INK,
                                                 light=PAPER)))
                else:
                    texts.append((x, t, plain))
                x += seg_w
            if clipped:
                break
        else:
            word_w = d.textlength(w.text, font=f)
            if x + word_w > x_limit:
                clipped = True
                break
            texts.append((x, w.text, T["next"]))
            x += word_w

    if not clipped:                     # trailing text, this row only
        tail_txt = raw[cursor:raw_stop]
        if tail_txt and x + d.textlength(tail_txt, font=f) <= x_limit:
            texts.append((x, tail_txt, plain))

    hpad = max(4, round(f.size * 0.14))     # chip side padding scales with
    for x0, x1, color in chips:             # the type so it can't swallow
        d.rounded_rectangle([x0 - hpad, y + 3, x1 + hpad, y + ascent + 9],
                            radius=6, fill=color)
    for tx, t, fill in texts:
        d.text((tx, y), t, font=f, fill=fill,
               stroke_width=stroke, stroke_fill=fill)


def _wrap_rows(d, wl, raw, font, max_width):
    """Split one lyrics line into visual rows that each fit max_width.
    Returns [(word_slice, row_text), …] — breaking at word boundaries so
    type size never has to shrink."""
    if not wl:
        return [(wl, raw)]
    rows, cur = [], []
    for item in wl:
        trial = cur + [item]
        seg = raw[trial[0][1].start:trial[-1][1].end]
        if cur and d.textlength(seg, font=font) > max_width:
            rows.append((cur, raw[cur[0][1].start:cur[-1][1].end]))
            cur = [item]
        else:
            cur = trial
    if cur:
        rows.append((cur, raw[cur[0][1].start:cur[-1][1].end]))
    return rows


_TAG_WORDS = ("official", "audio", "video", "lyric", "visuali", "hd", "hq",
              "4k", "explicit", "clean", "remaster", "music video", "mv")


def clean_title(title: str) -> str:
    """A song title without the upload's packaging: "KILLSHOT [Official
    Audio]" → "KILLSHOT", "Kool G Rap - Executioner Style + Lyrics" →
    "Kool G Rap - Executioner Style". Brackets that are part of the song
    ("(feat. J. Cole)", "Pt. II") stay."""
    def tag(m):
        inner = m.group(1).lower()
        return "" if any(w in inner for w in _TAG_WORDS) else m.group(0)
    t = _re.sub(r"\s*[\(\[（【]([^\)\]）】]*)[\)\]）】]", tag, title)
    t = _re.sub(r"\s*[+|]\s*lyrics?\b.*$", "", t, flags=_re.I)
    t = _re.sub(r"\s{2,}", " ", t).strip(" -–—|")
    return t or title


def _title_split(title: str) -> tuple[str, str]:
    """'Artist - Song' → ('ARTIST', 'SONG'); plain titles keep one part."""
    for sep in (" - ", " – ", " — "):
        if sep in title:
            a, b = title.split(sep, 1)
            if a.strip() and b.strip():
                return a.strip().upper(), b.strip().upper()
    return title.strip().upper(), ""


# --------------------------------------------------------------------------
# 9:16 vertical spec (rhymescope-vertical-spec.md) — the short-form frame
# --------------------------------------------------------------------------
# One vertical flex column, five bands, nothing absolutely positioned. Only
# the band heights below are given; every y on the frame falls out of them,
# which is why they are constants and not literals — move a band and the
# rhyme stack follows it. The look is fixed (paper, black header, one lime
# rule); `theme` does not reach it, exactly as the 16:9 §3a frames.
#
# Band 2 is a 16:9 window held open whether or not there is a clip:
# make_video's ffmpeg overlay drops a performance video straight into it,
# and a render without one leaves the same rectangle free (carrying the
# title) so a video cut in later, in an editor, lands on it without
# covering a single bar — adding the video never re-flows the lyrics.
RSV_PAD = 64                # every band's horizontal padding
RSV_HEAD_H = 128            # band 1 — header
RSV_SLOT_H = 608            # band 2 — 16:9 at 1080 wide, full-bleed
RSV_RULE_H = 6              # band 3 — the lime accent rule
RSV_FOOT_H = 104            # band 5 — footer
RSV_BAND_TOP = 56           # band 4's padding-top
RSV_GAP = 22                # scheme label → the first bar
RSV_LINE_GAP = 14           # bar → bar. Tighter than the label's gap so
                            # nine bars reach the band's full depth: the
                            # 16:9 window above is fixed, so the only room
                            # for more scheme is between the bars.
RSV_LYRIC = 40              # lyric size …
RSV_LYRIC_MIN = 34          # … and the floor, for a bar that alone
                            # cannot fit the band; nine 48px bars end at
                            # y 1495, a hair inside the safe line
RSV_LINE_H = 1.30
RSV_ROW_GAP = 4             # `gap: 6px 11px` — between one line's rows …
RSV_WORD_GAP = 10           # … and between its words
RSV_TRACK = 0.0            # lyric letter-spacing, em
RSV_CHIP_PAD = 5            # a syllable box's side padding — tighter for the smaller type
RSV_PAGE = 9                # bars on screen
RSV_SAFE_BOT = 1500         # TikTok's UI: no lyric may cross this

RSV_W, RSV_H = FORMATS["tiktok"]
RSV_SLOT_TOP = RSV_HEAD_H                               # 128
RSV_SLOT_BOT = RSV_SLOT_TOP + RSV_SLOT_H                # 736
RSV_BAND_Y = RSV_SLOT_BOT + RSV_RULE_H + RSV_BAND_TOP   # 798



def _base(title: str, T: dict, dims: tuple = (1920, 1080),
          has_clip: bool = False, ep: str | None = None) -> Image.Image:
    """Static background per the video spec: portrait is the 9:16 vertical
    spec's band stack (header / 16:9 clip / lime rule / rhymes / footer);
    landscape without a clip is the brand kit's §3a frame 02 (paper +
    bottom hairline bar), and with a clip keeps spec §4's clip top-left +
    lower third. Portrait reserves its clip band either way; `has_clip`
    only says whether ffmpeg is about to overlay a performance video
    there, in which case the title is left out of it."""
    w, h = dims
    if w > h and not has_clip:
        return _yt_verse_base(title, dims)
    if w < h:
        return _rsv_base(title, ep=ep, has_clip=has_clip, rsv=T["rsv"])
    img = Image.new("RGB", (w, h), T["bg"])
    d = ImageDraw.Draw(img)
    artist, song = _title_split(title)

    # ---- landscape with a clip, spec §4. (The lyrics-only 16:9 returned at
    # the top of this function — it is the brand kit's frame 02.)
    # clip top-left: keep the lower third under it + badge
    af, sf = _font("display", 44), _font("mono", 22)
    pw = max(d.textlength(artist, font=af),
             (d.textlength(song, font=sf) if song else 0)) + 72
    x0, y0 = 96, 912
    ph = 118 if song else 84
    d.rectangle([x0, y0, x0 + pw, y0 + 10], fill=LIME)      # fixed accent
    d.rectangle([x0, y0 + 10, x0 + pw, y0 + 10 + ph], fill=INK,
                outline=T["edge"], width=1)
    d.text((x0 + 36, y0 + 24), artist, font=af, fill=LIME)
    if song:
        d.text((x0 + 36, y0 + 84), song, font=sf, fill=PAPER)
    _logo(d, w - 64 - 84, h - 64 - 84, 84)
    return img


def _stat(d, x, value, label, T):
    """Right-aligned stat block for the landscape end-card overlay."""
    vf, lf = _font("display", 44), _font("mono", 20)
    vw = d.textlength(value, font=vf)
    lw = d.textlength(label, font=lf)
    bw = max(vw, lw)
    d.text((x - bw + (bw - vw) / 2, 58), value, font=vf, fill=T["fg"])
    d.text((x - bw + (bw - lw) / 2, 116), label, font=lf, fill=T["dim"])
    return x - bw - 70


# --------------------------------------------------------------------------
# YouTube 16:9 — brand kit §3a: frame 01 title · 02 verse · 04 end screen
# --------------------------------------------------------------------------
#
# One fixed look (off-white paper), no theme variants: the 16:9 lyrics-only
# render is this spec, `theme` only reaches the portrait and with-clip
# layouts. Type is Archivo (variable, [Weight, Width]) for everything the
# viewer reads and Space Mono for every small label.

YT = {
    "bg": "#FDFDFC",            # the only background colour in the spec
    "ink": "#0A0A0A",           # active text
    "dim": "#9A9A93",           # inactive lyric line
    "label": "#86867E",         # label grey
    "label2": "#C2C2BA",        # secondary label grey
    "hair": "#E6E6E0",          # hairline
    "zone_fill": "#F1F1EC",     # end-screen guide zones
    "zone_edge": "#C2C2BA",
}

# rhyme sounds A / B / C — the palette bars on frames 01 and 04 show exactly
# these three, in this order, which is also the head of HIGHLIGHTERS so a
# song's first three families get them.
YT_SOUNDS = HIGHLIGHTERS[:3]

YT_BAR_H = 92               # bottom hairline bar, all frames
YT_PAD = 120                # frame 02 / 04 horizontal padding (140 on 01)
YT_PAD_TOP = 76             # frame 02 top padding
YT_LYRIC = 60               # spec lyric size …
YT_LYRIC_MIN = 46           # … and the auto-fit floor for dense bars
YT_LINE_H = 1.30
YT_LINE_GAP = 10            # between lyric lines
YT_WORD_GAP = 16            # the flex row's `gap: 0 20px`
YT_TRACK = 0.0             # lyric letter-spacing, em
YT_PAGE = 8                 # lines on screen; past 8 the frame cuts

YT_END_HEADLINE = ("NEXT VERSE", "NEXT WEEK")
YT_HANDLE = "@RHYMESCOPE"
YT_NAME = "Rhymescope"

# sentinel line indices for the two static cards on the frame timeline
_TITLE, _END = -2, -3


def _tw(d, text: str, font, track: float = 0.0) -> float:
    """Rendered width of `text` including letter-spacing between glyphs."""
    if not text:
        return 0.0
    return d.textlength(text, font=font) + track * (len(text) - 1)


def _ttext(d, x, baseline, text, font, fill, track: float = 0.0) -> None:
    """Draw `text` from x, sitting on `baseline`, with CSS letter-spacing.
    PIL has no tracking, so tracked text goes out glyph by glyph — that
    drops kerning, which is exactly what letter-spacing does in a browser."""
    if not track:
        d.text((x, baseline), text, font=font, fill=fill, anchor="ls")
        return
    for ch in text:
        d.text((x, baseline), ch, font=font, fill=fill, anchor="ls")
        x += d.textlength(ch, font=font) + track


def _bl(top: float, line_h: float, font) -> float:
    """Baseline of a CSS line box of height `line_h` whose top is at `top`
    (half-leading split above and below the font's content area)."""
    asc, desc = font.getmetrics()
    return top + (line_h - (asc + desc)) / 2 + asc


def _cbl(mid: float, font) -> float:
    """Baseline that centres a single line of type on `mid`."""
    asc, desc = font.getmetrics()
    return mid + (asc - desc) / 2


def _yt_swatches(d, x: float, y: float) -> None:
    """The three palette bars: 64 × 10px, 8px apart."""
    for i, c in enumerate(YT_SOUNDS):
        d.rectangle([x + i * 72, y, x + i * 72 + 64, y + 10], fill=c)


def _yt_bar(d, dims, left: str | None, right: str | None,
            pad: int = YT_PAD) -> None:
    """The bottom hairline bar shared by every frame: icon + label left,
    label right, both Space Mono 24 uppercase at 0.22em."""
    w, h = dims
    top = h - YT_BAR_H
    d.line([(0, top), (w, top)], fill=YT["hair"], width=1)
    mid = top + YT_BAR_H / 2
    _logo(d, pad, round(mid - 26), 52, radius=14)
    f = _font("mono", 24)
    tr = 0.22 * 24
    if left:
        _ttext(d, pad + 52 + 20, _cbl(mid, f), left.upper(), f,
               YT["label"], tr)
    if right:
        r = right.upper()
        _ttext(d, w - pad - _tw(d, r, f, tr), _cbl(mid, f), r, f,
               YT["label2"], tr)


def _yt_greedy(d, text: str, font, track: float, max_w: float) -> list[str]:
    """Wrap one string onto rows that each fit max_w, at word boundaries."""
    rows, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cur and _tw(d, trial, font, track) > max_w:
            rows.append(cur)
            cur = word
        else:
            cur = trial
    return rows + ([cur] if cur else [])


def _yt_title_card(title: str, dims=(1920, 1080)) -> Image.Image:
    """Frame 01 — the opening card: palette bars, artist, song title."""
    w, h = dims
    img = Image.new("RGB", (w, h), YT["bg"])
    d = ImageDraw.Draw(img)
    artist, song = _title_split(title)
    if not song:                        # no "Artist - Song" split: the whole
        artist, song = "", artist       # string is the title

    pad = 140
    max_w = w - 2 * pad
    words = song.split()
    # hard break after the first word, then shrink (spec: 190px once a title
    # needs three lines) until every row fits
    parts = [words[0], " ".join(words[1:])] if len(words) > 1 else [song]
    for size in (232, 190, 160, 132, 108):
        f = _font("archivo", size, 200, 100)
        tr = -0.05 * size
        rows = [r for p in parts for r in _yt_greedy(d, p, f, tr, max_w)]
        if len(rows) <= 3 and all(_tw(d, r, f, tr) <= max_w for r in rows):
            break

    af = _font("archivo", 96, 200, 104)
    lh = round(size * 0.88)
    total = 10 + 30 + len(rows) * lh + (30 + 96 if artist else 0)
    y = 120 + max(0, (h - YT_BAR_H - 120 - total) / 2)

    _yt_swatches(d, pad, y)
    y += 10 + 30
    if artist:
        _ttext(d, pad, _bl(y, 96, af), artist, af, YT["label"], 0.02 * 96)
        y += 96 + 30
    for r in rows:
        _ttext(d, pad, _bl(y, lh, f), r, f, YT["ink"], tr)
        y += lh

    _yt_bar(d, dims, "RHYMESCOPE", "RHYME SCHEME BREAKDOWN", pad=pad)
    return img


def _yt_verse_base(title: str, dims=(1920, 1080)) -> Image.Image:
    """Frame 02 background: paper + the bottom bar. The right-hand label
    moves with the bars on screen, so it is drawn per frame."""
    w, h = dims
    img = Image.new("RGB", (w, h), YT["bg"])
    d = ImageDraw.Draw(img)
    artist, song = _title_split(title)
    left = f"{artist} — {song}" if song else artist
    f = _font("mono", 24)
    _yt_bar(d, dims, _fit(d, left, f, w - 2 * YT_PAD - 72 - 420), None)
    return img


def _yt_place(d, ln, sync, wbl, colors, fam_at, font, track, max_w, pad,
              gap: float = YT_WORD_GAP, breaks: set | None = None):
    """Lay one lyric line out the way the spec's flex row does: every word
    its own box, `gap` apart, wrapping when the row is full.

    A chip's side padding is part of its word's box, so a highlighted word
    takes up more room and the gap stays between the chips — not between
    the letters, which would let two chips touch.

    `breaks` pins the wrap: a set of word positions that each start a new
    row, measured once at one weight and replayed at another. The 9:16
    spec steps a bar from weight 300 to 400 in place and nothing on the
    frame may move, so its rows are broken at the heavier weight and every
    line reuses those breaks.

    Returns [[(x, [(dx, text, width, colour, first, last), …], k_w), …], …]
    — one list per visual row, so a chip can be drawn per rhyming run with
    the run's position inside its word known."""
    raw = sync[ln][1]
    words = wbl.get(ln, [])
    if not words:
        txt = raw.strip() or "· · ·"
        return [[(0.0, [(0.0, txt, _tw(d, txt, font, track), None,
                         True, True)], 0)]]
    rows, cur, x = [], [], 0.0
    for k_w, (w_i, w) in enumerate(words):
        runs = _word_runs(w, colors, fam_at, w_i)
        # only a run touching the word's edge pushes the box out — and the
        # box is sized as if the chip were already painted, so a word can
        # never shift sideways at the moment its highlight lands
        lead = pad if runs[0][1] else 0.0
        segs, off = [], lead
        for k, (t, color) in enumerate(runs):
            sw = _tw(d, t, font, track)
            segs.append((off, t, sw, color, k == 0, k == len(runs) - 1))
            off += sw
        box = off + (pad if runs[-1][1] else 0.0)
        wrap = k_w in breaks if breaks is not None else x + box > max_w
        if cur and wrap:
            rows.append(cur)
            cur, x = [], 0.0
        cur.append((x, segs, k_w))
        x += box + gap
    if cur:
        rows.append(cur)
    return rows


# --------------------------------------------------------------------------
# the strip: one long column of bars, and a window gliding down it
# --------------------------------------------------------------------------
#
# The passage is laid out ONCE as a single tall column; every bar keeps
# one position relative to its neighbours for the whole render, and what
# moves is the window onto it. That part was right. What made the motion
# look wrong was how the window moved:
#
#   · it eased to a DEAD STOP at the start of every bar — each bar was its
#     own ease-in/ease-out segment — so the column pulsed, bar by bar,
#     instead of travelling;
#   · frames only changed at word onsets plus a few in-betweens, and were
#     stitched together with uneven durations, so even the moving parts
#     stepped at a ragged 10-15 frames a second;
#   · every position was rounded to a whole pixel, so a slow crawl moved
#     in visible one-pixel ticks.
#
# Now the window moves like a teleprompter. Its path is planned once for
# the whole song: a straight line through every bar's start — so the
# column travels at the pace the bars are said, never stopping between
# them — with the corners rounded off by a Gaussian so speed changes are
# gradual. Every frame at a true 30 fps is drawn at its own moment, and
# the column is placed to the sub-pixel, blending the two rows it falls
# between, so a crawl of half a pixel per frame is as smooth as a fast
# one.

FPS = 60                   # a phone scrolls at 60: 30 read as a judder
LIVE_WEIGHT = 400          # the bar being said — thin, set apart by ink
DIM_WEIGHT = 400           # every other bar: same cut, grey ink — a lighter
                           # cut is narrower and left gaps between its words
FADE_S = 0.12              # a lit word / a bar going live fades in this long


class Strip:
    """A passage as one tall column, plus the window's path down it.

    `layout` maps a lyric line to (y in the column, height, rows at the
    live weight, rows at the dim weight). Bars are drawn once per state
    and kept as sprites, so a frame is a handful of pastes rather than a
    page of text rendering — which is what makes 30 real frames a second
    affordable."""

    def __init__(self, layout, total, size, view_w, view_h, read_y, lh,
                 row_gap, line_gap, pad, track, colours):
        self.layout = layout
        self.total = total
        self.size = size
        self.view_w = view_w
        self.view_h = view_h
        self.read_y = read_y
        self.lh = lh
        self.row_gap = row_gap
        self.line_gap = line_gap
        self.pad = pad
        self.track = track
        self.bg, self.fg, self.dim_fg = colours
        self.act = _font("archivo", size, LIVE_WEIGHT, 96)
        self.dim = _font("archivo", size, DIM_WEIGHT, 96)
        self.path = None           # scroll per frame, planned by plan()
        self._sprites: dict = {}
        self._live: dict = {}

    # ---- where the window goes ------------------------------------------

    def max_scroll(self) -> float:
        return max(0.0, self.total - self.view_h)

    def target(self, ln) -> float:
        """Where the column sits when bar `ln` is at the read line —
        clamped, so the first bars do not scroll the column off its own
        top and the window stops when the column's bottom arrives."""
        if ln not in self.layout:
            return 0.0
        return min(max(self.layout[ln][0] - self.read_y, 0.0),
                   self.max_scroll())

    def plan(self, starts: list, duration: float, sigma: float = 1.1,
             lead: float = 0.3) -> None:
        """The window's path for the whole song, one value per frame.

        `starts` is (time, line) for every bar as it is said. The path is
        piecewise LINEAR through those points — the column never stops
        between bars — and then smoothed with a Gaussian of `sigma`
        seconds, so a change of pace is a curve rather than a corner.
        `lead` starts each move a touch early: the eye arrives at the
        next bar as the voice does, not after it."""
        import numpy as np
        nfr = max(1, int(round(duration * FPS)))
        pts = sorted((t - lead, self.target(ln)) for t, ln in starts
                     if ln in self.layout)
        if not pts:
            self.path = np.zeros(nfr)
            return
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.maximum.accumulate(np.array([p[1] for p in pts], dtype=float))
        raw = np.interp(np.arange(nfr) / FPS, xs, ys)
        s = sigma * FPS
        k = np.arange(-int(3 * s), int(3 * s) + 1)
        g = np.exp(-0.5 * (k / s) ** 2)
        g /= g.sum()
        edge = len(k) // 2
        padded = np.concatenate([np.full(edge, raw[0]), raw,
                                 np.full(edge, raw[-1])])
        smooth = np.convolve(padded, g, mode="valid")
        # never backwards, never past either end
        self.path = np.maximum.accumulate(np.clip(smooth, 0.0,
                                                  self.max_scroll()))

    def at(self, t: float) -> float:
        if self.path is None or not len(self.path):
            return 0.0
        i = min(max(int(round(t * FPS)), 0), len(self.path) - 1)
        return float(self.path[i])

    # ---- how a bar looks --------------------------------------------------

    def _render_bar(self, ln, state, active) -> Image.Image:
        y0, h, ref, soft = self.layout[ln]
        img = Image.new("RGB", (self.view_w, max(int(h), 1)), self.bg)
        d = ImageDraw.Draw(img)
        live = state == "live"
        f = self.act if live else self.dim
        fill = self.fg if live else self.dim_fg
        asc, desc = f.getmetrics()
        seam = max(2, round(self.pad / 4))
        y = 0
        for row in (ref if live else soft):
            base_y = _bl(y, self.lh, f)
            chips, texts = [], []
            for wx, segs, k_w in row:
                # said already: every bar above the live one, and the live
                # bar up to the word being said
                lit = state == "past" or (
                    live and (active is None or k_w < active))
                for dx, t, sw, color, first, last in segs:
                    x = wx + dx
                    if color and t.strip() and lit:
                        chips.append((x - (self.pad if first else seam),
                                      x + sw + (self.pad if last else seam),
                                      color))
                        texts.append((x, t, _ink_for(color, dark=INK,
                                                     light=PAPER)))
                    else:
                        texts.append((x, t, fill if lit else self.dim_fg))
            for x0, x1, color in chips:        # flat rectangles, no radius
                d.rectangle([x0, base_y - asc, x1, base_y + desc],
                            fill=color)
            for x, t, colour in texts:
                _ttext(d, x, base_y, t, f, colour, self.track)
            y += self.lh + self.row_gap
        return img

    def sprite(self, ln, state, active=None) -> Image.Image:
        """Bar `ln` as it looks in `state` — drawn once and kept. The live
        bar changes word by word, so only its most recent looks are kept."""
        if state != "live":
            key = (ln, state)
            img = self._sprites.get(key)
            if img is None:
                img = self._sprites[key] = self._render_bar(ln, state, None)
            return img
        key = (ln, active)
        img = self._live.get(key)
        if img is None:
            if len(self._live) > 24:
                self._live.clear()
            img = self._live[key] = self._render_bar(ln, "live", active)
        return img


def _lay_strip(lines, sync, wbl, colors, fam_at, act, dim, track, lh,
               row_gap, line_gap, pad, word_gap, max_w):
    """Measure every bar once, at both weights, and stack them. Every
    position is a whole pixel so a bar's sprite lands exactly."""
    d = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    layout, y = {}, 0
    for ln in lines:
        ref = _yt_place(d, ln, sync, wbl, colors, fam_at, act, track,
                        max_w, pad, word_gap)
        # the dim weight replays the LIVE weight's wrap, so a bar stepping
        # live cannot re-break its own rows under the reader
        soft = _yt_place(d, ln, sync, wbl, colors, fam_at, dim, track,
                         max_w, pad, word_gap, {r[0][2] for r in ref[1:]})
        h = len(ref) * lh + (len(ref) - 1) * row_gap
        layout[ln] = (y, h, ref, soft)
        y += h + line_gap
    return layout, max(0, y - line_gap)


@functools.lru_cache(maxsize=8)
def _fade_rows(depth: int):
    """0..256 weights for the `depth` rows at a window edge: the top and
    bottom dissolve into the page, because a strip has no page break to
    hide a hard cut across a letter behind."""
    import numpy as np
    m = np.clip((np.arange(depth, dtype=np.float32) + 0.5) / depth, 0, 1)
    m = m * m * (3 - 2 * m)                    # soft, not linear
    return (m * 256).astype(np.uint16)[:, None, None]


def _draw_strip(strip: Strip, scroll: float, line_idx: int, active_words,
                *, width: int, height: int, fade: int = 0,
                fade_from=None, alpha: float = 1.0) -> Image.Image:
    """The window at `scroll`; during a change of state (`fade_from` =
    the previous (line, words lit), `alpha` how far into the change) the
    old and new looks are cross-faded, so a word's chip and a bar going
    live ease in instead of popping."""
    new = _draw_strip_once(strip, scroll, line_idx, active_words,
                           width=width, height=height, fade=fade)
    if fade_from is None or alpha >= 1.0:
        return new
    old = _draw_strip_once(strip, scroll, fade_from[0], fade_from[1],
                           width=width, height=height, fade=fade)
    return Image.blend(old, new, max(0.0, alpha))


def _draw_strip_once(strip: Strip, scroll: float, line_idx: int,
                     active_words, *, width: int, height: int,
                     fade: int = 0) -> Image.Image:
    """The window onto the column at `scroll`, placed to the sub-pixel.

    The two whole-pixel positions either side of `scroll` are blended by
    how far between them it falls, so the column can move by a fraction
    of a pixel per frame without ticking. Integer arithmetic throughout,
    and the fade touches only the rows it changes — this runs thirty
    times a second of video."""
    import numpy as np
    y0 = int(math.floor(scroll))
    f8 = int(round((scroll - y0) * 256))
    band = Image.new("RGB", (width, height + 1), strip.bg)
    for ln, (y, h, _ref, _soft) in strip.layout.items():
        top = y - y0
        if top + h < 0 or top > height + 1:
            continue
        state = ("past" if ln < line_idx else
                 "live" if ln == line_idx else "future")
        band.paste(strip.sprite(ln, state, active_words
                                if state == "live" else None), (0, top))
    a = np.asarray(band)
    if 0 < f8 < 256:
        out = ((a[:-1].astype(np.uint16) * (256 - f8)
                + a[1:].astype(np.uint16) * f8 + 128) >> 8).astype(np.uint8)
    else:
        out = a[1:].copy() if f8 >= 256 else a[:-1].copy()
    if fade and height > 2 * fade:
        bg = np.array(ImageColor.getrgb(strip.bg), dtype=np.uint16)
        m = _fade_rows(fade)
        top, bot = out[:fade].astype(np.uint16), out[-fade:].astype(np.uint16)
        out[:fade] = ((top * m + bg * (256 - m)) >> 8).astype(np.uint8)
        mb = m[::-1]
        out[-fade:] = ((bot * mb + bg * (256 - mb)) >> 8).astype(np.uint8)
    return Image.fromarray(out)


def _yt_strip(sync, words_by_line, colors, fam_at, bar_no,
              dims=(1920, 1080)) -> Strip:
    """Lay the 16:9 verse out as one column. Computed once per render.

    The type holds one size for the whole render and only falls back if
    some single bar cannot fit the window on its own."""
    w, h = dims
    max_w = w - 2 * YT_PAD
    view_h = h - YT_BAR_H - YT_PAD_TOP
    lines = sorted(bar_no)
    d = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    size = YT_LYRIC
    for size in range(YT_LYRIC, YT_LYRIC_MIN - 1, -2):
        f = _font("archivo", size, LIVE_WEIGHT, 96)
        lh = round(size * YT_LINE_H)
        tallest = max(
            (len(_yt_place(d, ln, sync, words_by_line, colors, fam_at, f,
                           YT_TRACK * size, max_w, size * 8 / 76)) * lh
             for ln in lines), default=0)
        if tallest <= view_h:
            break
    act = _font("archivo", size, LIVE_WEIGHT, 96)
    dim = _font("archivo", size, DIM_WEIGHT, 96)
    lh = round(size * YT_LINE_H)
    pad = size * 8 / 76
    layout, total = _lay_strip(lines, sync, words_by_line, colors, fam_at,
                               act, dim, YT_TRACK * size, lh, 0,
                               YT_LINE_GAP, pad, YT_WORD_GAP, max_w)
    return Strip(layout, total, size, max_w, int(view_h),
                 round(view_h * 0.38), lh, 0, YT_LINE_GAP, pad,
                 YT_TRACK * size, (YT["bg"], YT["ink"], YT["dim"]))


def _yt_verse_frame(base, sync, words_by_line, colors, fam_at, bar_no,
                    line_idx, active_words=None, verse_label=None,
                    fade_from=None, alpha: float = 1.0,
                    dims=(1920, 1080), strip: Strip | None = None,
                    scroll: float | None = None):
    """Frame 02 — the working frame: the verse as one column, gliding.

    The highlights build as the verse is said: a bar above the live one
    keeps its chips, the live bar lights up to `active_words`, the bars
    below are still bare grey. `active_words=None` lights the whole line.
    `verse_label` fills the bottom-right slot; with none given the slot
    carries the range of bars on screen."""
    img = base.copy()
    d = ImageDraw.Draw(img)
    w, h = dims
    if strip is None:
        strip = _yt_strip(sync, words_by_line, colors, fam_at, bar_no,
                          dims=dims)
    if scroll is None:
        scroll = strip.target(line_idx)
    if not strip.layout:
        return img
    img.paste(_draw_strip(strip, scroll, line_idx, active_words,
                          width=strip.view_w, height=strip.view_h, fade=48,
                          fade_from=fade_from, alpha=alpha),
              (YT_PAD, YT_PAD_TOP))
    if verse_label is None:
        vis = [ln for ln, (y0, hh, _r, _s) in strip.layout.items()
               if y0 + hh > scroll and y0 < scroll + strip.view_h
               and ln in bar_no]
        if vis:
            a_, b_ = bar_no[min(vis)], bar_no[max(vis)]
            verse_label = f"BAR {a_}" if a_ == b_ else f"BAR {a_}–{b_}"
    if verse_label:
        f = _font("mono", 24)
        tr = 0.22 * 24
        r = verse_label.upper()
        _ttext(d, w - YT_PAD - _tw(d, r, f, tr),
               _cbl(h - YT_BAR_H / 2, f), r, f, YT["label2"], tr)
    return img


def _yt_end_card(dims=(1920, 1080), guides: bool = False) -> Image.Image:
    """Frame 04 — the end screen. The right third stays empty on purpose:
    YouTube draws its own end-screen elements there. `guides=True` renders
    the dashed placeholders for checking the layout; the exported frame
    must not carry them."""
    w, h = dims
    img = Image.new("RGB", (w, h), YT["bg"])
    d = ImageDraw.Draw(img)

    right_w = 496 * 2 + 36
    col_x = YT_PAD
    box_top, box_h = 100, h - YT_BAR_H - 100

    # The spec's own numbers collide here: 132px caps need ~770px but the
    # reserved zones leave the left column 562px. "Keep all Rhymescope copy
    # out of the right column" is the rule that matters (YouTube draws over
    # it), so 132 is a ceiling and the headline shrinks to clear the zones.
    left_w = w - 2 * YT_PAD - 90 - right_w
    for size in (132, 120, 108, 96, 88, 80, 72):
        hf = _font("archivo", size, 200, 100)
        htr = -0.05 * size
        if all(_tw(d, r, hf, htr) <= left_w for r in YT_END_HEADLINE):
            break
    hlh = round(size * 0.92)
    total = 10 + 40 + len(YT_END_HEADLINE) * hlh + 40 + 80
    y = box_top + (box_h - total) / 2

    _yt_swatches(d, col_x, y)
    y += 10 + 40
    for row in YT_END_HEADLINE:
        _ttext(d, col_x, _bl(y, hlh, hf), row, hf, YT["ink"], htr)
        y += hlh
    y += 40

    _logo(d, col_x, round(y), 80, radius=21)
    nf, hnd = _font("archivo", 34, 400, 100), _font("mono", 20)
    stack = 34 + 6 + 20
    ty = y + (80 - stack) / 2
    _ttext(d, col_x + 80 + 24, _bl(ty, 34, nf), YT_NAME, nf, YT["ink"],
           -0.01 * 34)
    _ttext(d, col_x + 80 + 24, _bl(ty + 34 + 6, 20, hnd), YT_HANDLE, hnd,
           YT["label"], 0.18 * 20)

    if guides:
        zx, zy = w - YT_PAD - right_w, box_top + (box_h - (279 + 36 + 196)) / 2
        zf = _font("mono", 18)
        for i, label in enumerate(("NEXT EPISODE", "BEST OF PLAYLIST")):
            x0 = zx + i * (496 + 36)
            _yt_zone(d, [x0, zy, x0 + 496, zy + 279])
            _ttext(d, x0 + 20, zy + 279 - 20, label, zf, YT["dim"],
                   0.18 * 18)
        cy = zy + 279 + 36
        _yt_zone(d, [zx, cy, zx + 196, cy + 196], ellipse=True)
        sf = _font("mono", 17)
        tr = 0.14 * 17
        _ttext(d, zx + (196 - _tw(d, "SUBSCRIBE", sf, tr)) / 2,
               _cbl(cy + 98, sf), "SUBSCRIBE", sf, YT["dim"], tr)
        nf2 = _font("mono", 19)
        for k, line in enumerate(("Leave these three zones empty.",
                                  "YouTube draws its own cards here.")):
            d.text((zx + 196 + 32, cy + 98 - 15 + k * 30), line, font=nf2,
                   fill=YT["dim"], anchor="lm")

    _yt_bar(d, dims, "RHYME SCHEME BREAKDOWN", "LAST 20 SECONDS")
    return img


def _yt_zone(d, box, ellipse: bool = False) -> None:
    """A dashed guide zone (end-screen placeholder, never exported)."""
    x0, y0, x1, y1 = box
    shape = d.ellipse if ellipse else d.rectangle
    shape(box, fill=YT["zone_fill"])
    step, on = 22, 12
    if ellipse:                         # dashes on a circle: short arcs
        for a in range(0, 360, 12):
            d.arc(box, a, a + 7, fill=YT["zone_edge"], width=2)
        return
    for x in range(int(x0), int(x1), step):
        d.line([x, y0, min(x + on, x1), y0], fill=YT["zone_edge"], width=2)
        d.line([x, y1, min(x + on, x1), y1], fill=YT["zone_edge"], width=2)
    for y in range(int(y0), int(y1), step):
        d.line([x0, y, x0, min(y + on, y1)], fill=YT["zone_edge"], width=2)
        d.line([x1, y, x1, min(y + on, y1)], fill=YT["zone_edge"], width=2)


# --------------------------------------------------------------------------
# 9:16 vertical spec — the frame builders
# --------------------------------------------------------------------------

def _rsv_fit(d, text: str, font, track: float, max_w: float) -> str:
    """_fit, but measuring with letter-spacing — every label on this frame
    is tracked, so measuring without it under-reads the width."""
    if _tw(d, text, font, track) <= max_w:
        return text
    while text and _tw(d, text + "…", font, track) > max_w:
        text = text[:-1]
    return text + "…"


def _ep_tag(ep) -> str:
    """`41` / `ep41` / `EP. 041` → `EP. 041` for the header's right slot.
    A tag with no digits in it passes through uppercased, so a one-off
    label still lands there; nothing at all leaves the slot empty."""
    if not ep:
        return ""
    digits = "".join(c for c in str(ep) if c.isdigit())
    return f"EP. {int(digits):03d}" if digits else str(ep).strip().upper()


def _scheme_label(analysis: Analysis) -> str:
    """Band 4's label — what the passage's pattern IS, derived from the
    analysis so an automated render still carries one.

    A family's members are the syllables of one rhyme sound, so a run of
    consecutive stream indices inside a family is a multi and its length
    is the chain's depth: 'MULTI, 3 SYLLABLE CHAIN'. With no chain deeper
    than a syllable the label falls back to the shape of the scheme —
    whether the sounds land on line ends or sit inside the bars. Pass
    `scheme=` to write it by hand instead."""
    fam = analysis.family_of
    deepest = run = 1
    for i in range(1, len(analysis.stream)):
        if fam.get(i) is not None and fam.get(i) == fam.get(i - 1):
            run += 1
            deepest = max(deepest, run)
        else:
            run = 1
    if deepest >= 2:
        return f"MULTI, {deepest} SYLLABLE CHAIN"
    n = len(analysis.families)
    if not n or not fam:
        return ""
    last: dict = {}
    for w_i, wd in enumerate(analysis.words):
        last[wd.line] = w_i
    ends = sum(1 for i in fam
               if last.get(analysis.words[analysis.stream[i].word_index].line)
               == analysis.stream[i].word_index)
    kind = "END RHYME" if ends * 2 >= len(fam) else "INTERNAL"
    return f"{kind}, {n} SOUND{'S' if n != 1 else ''}"


def _rsv_base(title: str, ep: str | None = None, has_clip: bool = False,
              placeholder: bool = False, rsv: dict | None = None) -> Image.Image:
    """Bands 1, 2, 3 and 5 — everything on the 9:16 frame that does not
    change bar to bar. Band 4 is drawn per frame by _rsv_frame.

    `placeholder` renders band 2 the way the template does (flat fill +
    `YOUTUBE CLIP · 16 : 9`); a real render without a clip carries the
    title there instead, so the frame opens on something rather than on
    a label meant for the template. Either way the rectangle is the same
    1080 × 608, so dropping a video in covers it exactly."""
    rsv = rsv or RSV_THEMES["light"]
    w, h = FORMATS["tiktok"]
    img = Image.new("RGB", (w, h), rsv["bg"])
    d = ImageDraw.Draw(img)
    artist, song = _title_split(title)

    # ---- band 1: header (0–128)
    d.rectangle([0, 0, w, RSV_HEAD_H - 1], fill=rsv["head"])
    _logo(d, RSV_PAD, (RSV_HEAD_H - 60) // 2, 60, radius=16)
    f = _font("mono", 28)
    _ttext(d, RSV_PAD + 60 + 22, _cbl(RSV_HEAD_H / 2, f), "RHYMESCOPE",
           f, rsv["head_fg"], 0.2 * 28)
    tag = _ep_tag(ep)
    if tag:
        f, tr = _font("mono", 24), 0.18 * 24
        _ttext(d, w - RSV_PAD - _tw(d, tag, f, tr), _cbl(RSV_HEAD_H / 2, f),
               tag, f, rsv["head_dim"], tr)

    # ---- band 2: the 16:9 clip window (128–736), full-bleed, no radius
    d.rectangle([0, RSV_SLOT_TOP, w, RSV_SLOT_BOT - 1], fill=rsv["slot"])
    mid = (RSV_SLOT_TOP + RSV_SLOT_BOT) / 2
    if placeholder:
        f, tr = _font("mono", 24), 0.18 * 24
        t = "YOUTUBE CLIP · 16 : 9"
        _ttext(d, (w - _tw(d, t, f, tr)) / 2, _cbl(mid, f), t, f,
               rsv["slot_fg"], tr)
    elif not has_clip:
        tf = _font("archivo", 84, 400, 96)
        head = _fit(d, song or artist, tf, w - 2 * RSV_PAD)
        if song:
            f, tr = _font("mono", 26), 0.18 * 26
            a = _rsv_fit(d, artist, f, tr, w - 2 * RSV_PAD)
            _ttext(d, (w - _tw(d, a, f, tr)) / 2, _cbl(mid - 64, f), a, f,
                   rsv["label"], tr)
            d.text((w / 2, mid + 26), head, font=tf, fill=rsv["fg"],
                   anchor="mm")
        else:
            d.text((w / 2, mid), head, font=tf, fill=rsv["fg"], anchor="mm")

    # ---- band 3: the accent rule (736–742)
    d.rectangle([0, RSV_SLOT_BOT, w, RSV_SLOT_BOT + RSV_RULE_H - 1],
                fill=rsv["rule"])

    # ---- band 5: footer (1816–1920). Expendable — TikTok's own UI may
    # cover it — so nothing but the credit lives here.
    top = h - RSV_FOOT_H
    d.line([(0, top), (w, top)], fill=rsv["hair"], width=1)
    mid = top + RSV_FOOT_H / 2
    hf, htr = _font("mono", 24), 0.18 * 24
    handle_w = _tw(d, YT_HANDLE, hf, htr)
    f, tr = _font("archivo", 28, 400, 96), 0.02 * 28
    foot = f"{artist} — {song}" if song else artist
    _ttext(d, RSV_PAD, _cbl(mid, f),
           _rsv_fit(d, foot, f, tr, w - 2 * RSV_PAD - handle_w - 40),
           f, rsv["fg"], tr)
    _ttext(d, w - RSV_PAD - handle_w, _cbl(mid, hf), YT_HANDLE, hf,
           rsv["handle"], htr)
    return img


def _rsv_band_top(has_label: bool) -> float:
    """Where band 4's first lyric row starts — under the scheme label and
    its flex gap, or at the band's own padding when there is no label."""
    if not has_label:
        return RSV_BAND_Y
    asc, desc = _font("mono", 22).getmetrics()
    return RSV_BAND_Y + asc + desc + RSV_GAP


def _rsv_strip(sync, words_by_line, colors, fam_at, bar_no,
               has_label: bool = True, rsv: dict | None = None) -> Strip:
    """Lay band 4 out as one column. Computed once per render.

    The type holds at RSV_LYRIC for the whole video and only falls back
    toward RSV_LYRIC_MIN if some single bar cannot fit the band on its
    own. A dense passage simply makes the column longer."""
    rsv = rsv or RSV_THEMES["light"]
    d = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    max_w = RSV_W - 2 * RSV_PAD
    view_h = int(RSV_SAFE_BOT - _rsv_band_top(has_label))
    lines = sorted(bar_no)
    size = RSV_LYRIC
    for size in range(RSV_LYRIC, RSV_LYRIC_MIN - 1, -2):
        f = _font("archivo", size, LIVE_WEIGHT, 96)
        lh = round(size * RSV_LINE_H)
        tallest = max(
            (len(_yt_place(d, ln, sync, words_by_line, colors, fam_at, f,
                           RSV_TRACK * size, max_w,
                           size * RSV_CHIP_PAD / RSV_LYRIC, RSV_WORD_GAP))
             * (lh + RSV_ROW_GAP) for ln in lines), default=0)
        if tallest <= view_h:
            break
    act = _font("archivo", size, LIVE_WEIGHT, 96)
    dim = _font("archivo", size, DIM_WEIGHT, 96)
    lh = round(size * RSV_LINE_H)
    pad = size * RSV_CHIP_PAD / RSV_LYRIC
    layout, total = _lay_strip(lines, sync, words_by_line, colors, fam_at,
                               act, dim, RSV_TRACK * size, lh, RSV_ROW_GAP,
                               RSV_LINE_GAP, pad, RSV_WORD_GAP, max_w)
    return Strip(layout, total, size, max_w, view_h, round(view_h * 0.36),
                 lh, RSV_ROW_GAP, RSV_LINE_GAP, pad, RSV_TRACK * size,
                 (rsv["bg"], rsv["fg"], rsv["dim"]))


def _rsv_frame(base, sync, words_by_line, colors, fam_at, bar_no,
               line_idx, active_words=None, scheme: str | None = None,
               strip: Strip | None = None, rsv: dict | None = None,
               scroll: float | None = None, fade_from=None,
               alpha: float = 1.0):
    """Band 4 — the rhyme scheme: the scheme label, then the column of
    bars, gliding so the bar being said sits at the read line.

    The scheme BUILDS as the verse is said: bars above the live one keep
    their chips, the live bar lights word by word, the bars below are
    still bare. Rows are broken at the live weight whether a bar is live
    or not, so a bar stepping live moves nothing under the reader.
    `strip` comes from _rsv_strip, computed once per render; None
    recomputes it for a one-off still."""
    rsv = rsv or RSV_THEMES["light"]
    img = base.copy()
    d = ImageDraw.Draw(img)
    label = (scheme or "").strip().upper()
    if label and not label.startswith("SCHEME"):
        label = f"SCHEME — {label}"
    if strip is None:
        strip = _rsv_strip(sync, words_by_line, colors, fam_at, bar_no,
                           bool(label), rsv=rsv)
    if not strip.layout:
        return img
    if scroll is None:
        scroll = strip.target(line_idx)
    if label:
        f, tr = _font("mono", 22), 0.18 * 22
        _ttext(d, RSV_PAD, RSV_BAND_Y + f.getmetrics()[0],
               _rsv_fit(d, label, f, tr, strip.view_w), f, rsv["label"], tr)
    img.paste(_draw_strip(strip, scroll, line_idx, active_words,
                          width=strip.view_w, height=strip.view_h, fade=44,
                          fade_from=fade_from, alpha=alpha),
              (RSV_PAD, int(_rsv_band_top(bool(label)))))
    return img


def _frame(base, analysis, colors, fam_at, words_by_line, sync,
           line_idx, active_words, t_now, t_total, T, dims=(1920, 1080),
           bar_no=None, has_clip=False, cards=None, scheme=None,
           strip=None, scroll=None, fade_from=None, alpha: float = 1.0):
    w, h = dims
    # where the column sits right now — read from the planned path by the
    # CLOCK, not from the live bar, so it keeps moving between bars
    if scroll is None and strip is not None and t_now is not None:
        scroll = strip.at(t_now)

    # ---- 16:9 lyrics-only: the brand kit's three frames. The title and end
    # cards are rendered once by make_video and handed over in `cards`.
    if cards is not None:
        if line_idx == _TITLE:
            return cards["title"]
        if line_idx == _END:
            return cards["end"]
        if fade_from is not None and fade_from[0] in (_TITLE, _END):
            fade_from = None             # cards cut, they do not dissolve
        return _yt_verse_frame(base, sync, words_by_line, colors, fam_at,
                               bar_no, line_idx, active_words,
                               cards["label"], dims=dims, strip=strip,
                               scroll=scroll, fade_from=fade_from,
                               alpha=alpha)

    # ---- 9:16: the vertical spec's band 4, under the bands _rsv_base drew
    if w < h:
        return _rsv_frame(base, sync, words_by_line, colors, fam_at,
                          bar_no, line_idx, active_words, scheme=scheme,
                          strip=strip, rsv=T["rsv"], scroll=scroll,
                          fade_from=fade_from, alpha=alpha)

    img = base.copy()
    d = ImageDraw.Draw(img)

    # The rhyme stack is a static plate of bars around the active one:
    # dim bars Archivo 600, the active bar Archivo 700 a step larger, in
    # the spec's text band.
    if has_clip:                        # clip top-left, verse to the right
        above, below = 3, 3
        act, dim = _font("body", 46, 400), _font("body", 38, 300)
        gap = 28
        x0, max_w = 1010, w - 1010 - 96
        top, bottom = 96, 880
    else:                               # lyrics own the frame; slim strip
        above, below = 3, 3
        act, dim = _font("body", 52, 400), _font("body", 42, 300)
        gap = 34
        x0, max_w = 96, w - 192
        top, bottom = 96, 940

    # stats appear only in the closing seconds: the final scheme numbers
    # as an end-card, not a running counter
    if t_total and t_now >= t_total - 5.0:
        st = analysis.stats
        x = w - 120
        x = _stat(d, x, f"{st['density']:.0%}", "DENSITY", T)
        x = _stat(d, x, str(st['rhymed_syllables']), "RHYMED SYLLABLES", T)
        _stat(d, x, str(st['families']), "FAMILIES", T)

    # Plan the rows: a long bar WRAPS onto extra rows rather than
    # shrinking, so every row keeps its readable size. The active bar is
    # planned first, then neighbours join alternately (previous bar first)
    # while the text band has room — the plate keeps context on both sides
    # and never runs past the band.
    def rows_for(ln, f):
        text = sync[ln][1]
        wl = words_by_line.get(ln, [])
        if not text.strip():
            return [([], "· · ·")]
        return _wrap_rows(d, wl, text, f, max_w)

    budget = bottom - top
    cur_rows = rows_for(line_idx, act)
    used = len(cur_rows) * (act.size + gap)
    plan = [(line_idx, act, cur_rows)]
    ok_up, ok_down = True, True
    for off in [s * k for k in range(1, max(above, below) + 1) for s in (-1, 1)]:
        ln = line_idx + off
        if off < 0 and (not ok_up or -off > above):
            continue
        if off > 0 and (not ok_down or off > below):
            continue
        if not 0 <= ln < len(sync):
            continue
        rows = rows_for(ln, dim)
        need = len(rows) * (dim.size + gap)
        if used + need > budget:        # a gap in the plate reads as a cut
            if off < 0:                 # lyric — stop that direction cold
                ok_up = False
            else:
                ok_down = False
            continue
        used += need
        plan.append((ln, dim, rows))
    plan.sort(key=lambda p: p[0])

    # the plate is centred in its field so the frame stays balanced at any
    # verse length
    y = top + max(0, (bottom - top - used) // 2)
    drawn: list[int] = []
    for ln, f, rows in plan:
        rh = f.size + gap
        full = sync[ln][1]
        for wsl, rtext in rows:
            if y + f.size > bottom:
                break
            drawn.append(ln)
            if not wsl:
                d.text((x0, y), rtext, font=f, fill=T["next"])
                y += rh
                continue
            if ln > line_idx:
                d.text((x0, y), _fit(d, rtext, f, max_w), font=f,
                       fill=T["next"])
            elif ln < line_idx:
                _draw_line(d, x0, y, colors, fam_at, wsl, full, None,
                           T["past"], f, T, max_w)
            else:
                # `active` counts words in the whole line; offset it into
                # this row so the reveal continues across the wrap
                base_i = words_by_line.get(ln, []).index(wsl[0])
                sub_active = (None if active_words is None
                              else max(0, active_words - base_i))
                _draw_line(d, x0, y, colors, fam_at, wsl, full, sub_active,
                           T["fg"], f, T, max_w)
            y += rh

    # lime bar counter — counted from the bars actually on screen (spec:
    # the one lime text element), in the bottom strip left of the badge
    if bar_no and drawn and not has_clip:
        vis = sorted({bar_no[ln] for ln in drawn if ln in bar_no})
        if vis:
            tag = (f"BAR {vis[0]}" if vis[0] == vis[-1]
                   else f"BAR {vis[0]}–{vis[-1]}")
            d.text((w - 64 - 56 - 28, h - 48), tag,
                   font=_font("mono", 24, 700), fill=T["tag"], anchor="rm")
    return img


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

_JOB: dict = {}     # shared read-only render state (inherited via fork)


def _state_at(t: float):
    """(line, words lit) at time `t`, from the word events."""
    J = _JOB
    i = bisect.bisect_right(J["ev_t"], t) - 1
    e = J["events"][max(i, 0)]
    return e[1], e[2]


def _fade_at(t: float):
    """(previous (line, words lit), 0..1) while the state that began most
    recently is still fading in, else (None, 1.0)."""
    J = _JOB
    i = bisect.bisect_right(J["ev_t"], t) - 1
    if i <= 0:
        return None, 1.0
    a = (t - J["ev_t"][i]) / FADE_S
    if a >= 1.0:
        return None, 1.0
    prev = J["events"][i - 1]
    return (prev[1], prev[2]), a


def _render_chunk(job) -> str:
    """Render frames [a, b) in order and pipe them straight to their own
    encoder. Each worker owns a contiguous stretch of the video, so no
    frame ever crosses a process boundary and no PNG is ever written —
    and consecutive identical frames (a held title card, a still passage)
    are sent again without being drawn again."""
    k0, k1, path = job
    J = _JOB
    w, h = J["dims"]
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
         "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
         "-pix_fmt", "yuv420p", "-g", str(FPS * 2), "-threads", "2",
         str(path)], stdin=subprocess.PIPE)
    last_key, last_bytes = None, None
    try:
        for k in range(k0, k1):
            t = k / FPS
            ln, active = _state_at(t)
            strip = J["strip"]
            scroll = float(strip.path[min(k, len(strip.path) - 1)]) \
                if strip is not None and strip.path is not None else None
            fade_from, alpha = (_fade_at(t) if strip is not None
                                else (None, 1.0))
            key = (ln, active, None if scroll is None else round(scroll * 16),
                   fade_from, round(alpha * 16))
            if key != last_key:
                img = _frame(J["base"], J["analysis"], J["colors"],
                             J["fam_at"], J["wbl"], J["sync"], ln, active, t,
                             J["duration"], J["T"], dims=J["dims"],
                             bar_no=J["bar_no"], has_clip=J["has_clip"],
                             cards=J["cards"], scheme=J["scheme"],
                             strip=strip, scroll=scroll,
                             fade_from=fade_from, alpha=alpha)
                last_bytes = img.convert("RGB").tobytes()
                last_key = key
            enc.stdin.write(last_bytes)
    finally:
        enc.stdin.close()
        enc.wait()
    if enc.returncode:
        raise RuntimeError(f"encoder failed on frames {k0}-{k1}")
    return str(path)


def make_video(analysis: Analysis, sync: list, audio: pathlib.Path,
               title: str, out: pathlib.Path, theme: str = "light",
               schedule: list | None = None, format: str = "youtube",
               clip: pathlib.Path | None = None, verse_label: str | None = None,
               title_secs: float = 3.0, end_secs: float = 20.0,
               ep: str | None = None, scheme: str | None = None,
               color_ref: dict | None = None,
               color_lines: list | None = None,
               progress=print) -> pathlib.Path:
    """`clip` is an optional performance video whose timeline matches the
    audio; ffmpeg composites it into the spec's clip slot (portrait:
    1080×608 at y 210 · landscape: 960×540 top-left). Portrait holds that
    slot open with no clip too — a bare 16:9 window to drop a video into
    later — so the lyrics sit in the same band either way.

    16:9 with no clip renders the brand kit's frame set: the title card
    holds `title_secs`, the end screen the last `end_secs`, the verse
    everything between. Both cards yield to the lyrics rather than covering
    a bar, and each is capped on short excerpts (title 15% of the run, end
    screen 25%) so a 30s verse clip doesn't open and close on cards.
    `verse_label` fills the bottom-right slot (`VERSE 2`); without one the
    slot carries the bar range on screen.

    9:16 renders the vertical spec, whose look is fixed — `theme` does not
    reach it. `ep` fills the header's `EP. 041`; `scheme` overrides band
    4's label, which is otherwise derived from the analysis."""
    if format not in FORMATS:
        raise ValueError(f"format must be one of {list(FORMATS)}, got {format}")
    title = clean_title(title)
    w, h = FORMATS[format]

    T = THEMES[theme]
    colors = _palette(analysis, T, ref=color_ref, line_map=color_lines)
    fam_at, words_by_line = syllable_map(analysis)
    duration = _audio_duration(audio)

    # bar numbers for the header counter: content lines only
    bar_no, c = {}, 0
    for j, (_t, txt) in enumerate(sync):
        if txt.strip():
            c += 1
            bar_no[j] = c
    if schedule is None:
        schedule = word_schedule(sync, words_by_line)

    yt = w > h and clip is None         # brand kit §3a frame set
    vert = h > w                        # the 9:16 vertical spec
    cards = None
    strip = None
    if vert:
        if scheme is None:
            scheme = _scheme_label(analysis)
        # The column is a property of the VERSE, not of the frame: laid
        # out once here, so a bar's height and its wrap can never depend
        # on which bar is live.
        strip = _rsv_strip(sync, words_by_line, colors, fam_at, bar_no,
                           bool(scheme))
    elif yt:
        strip = _yt_strip(sync, words_by_line, colors, fam_at, bar_no,
                          dims=(w, h))
    # flatten into events: (time, line, words_active_in_line, word_index)
    events: list[tuple[float, int, int, int | None]] = [(0.0, -1, 0, None)]
    for ln, (t0, _txt) in enumerate(sync):
        starts = schedule[ln]
        if not starts:
            events.append((t0, ln, 0, None))
        else:
            for k, ts in enumerate(starts):
                w_i = words_by_line[ln][k][0]
                events.append((ts, ln, k + 1, w_i))
    events.sort(key=lambda e: e[0])

    if yt:
        # the title card holds its full time even when the verse starts
        # straight away — the frame it cuts to is whatever the verse had
        # reached by then, so no bar is lost, it just isn't watched landing
        hold = min(title_secs, 0.15 * duration)
        resume = max((e for e in events if e[0] <= hold),
                     key=lambda e: e[0], default=(0.0, -1, 0, None))
        events = [e for e in events if e[0] > hold]
        events[:0] = [(0.0, _TITLE, None, None),
                      (hold, resume[1], resume[2], resume[3])]
        t_end = duration - min(end_secs, 0.25 * duration)
        if t_end > (events[-1][0] + 1.5):   # never cut a bar short
            events = [e for e in events if e[0] < t_end]
            events.append((t_end, _END, None, None))
        cards = {"title": _yt_title_card(title, dims=(w, h)),
                 "end": _yt_end_card(dims=(w, h)),
                 "label": verse_label}

    # The window's path, planned once for the whole song from when each
    # bar starts — see Strip.plan.
    if strip is not None:
        starts = []
        for ln in sorted(bar_no):
            st = schedule[ln][0] if schedule[ln] else sync[ln][0]
            starts.append((st, ln))
        strip.plan(starts, duration)

    base = _base(title, T, dims=(w, h), has_clip=clip is not None, ep=ep)
    nframes = max(1, round(duration * FPS))

    with tempfile.TemporaryDirectory(prefix="rhymescope-") as td:
        tdir = pathlib.Path(td)
        _JOB.update(base=base, analysis=analysis, colors=colors,
                    fam_at=fam_at, wbl=words_by_line, sync=sync,
                    duration=duration, T=T, dims=(w, h), bar_no=bar_no,
                    has_clip=clip is not None, cards=cards, scheme=scheme,
                    strip=strip, events=events,
                    ev_t=[e[0] for e in events])
        workers = max(1, min(8, os.cpu_count() or 1,
                             nframes // (FPS * 4) or 1))
        bounds = [round(nframes * i / workers) for i in range(workers + 1)]
        jobs = [(bounds[i], bounds[i + 1], tdir / f"part{i:02d}.mp4")
                for i in range(workers) if bounds[i + 1] > bounds[i]]
        if progress:
            progress(f"    rendering {nframes} frames at {FPS} fps "
                     f"on {len(jobs)} core{'s' if len(jobs) > 1 else ''} …")
        try:
            if len(jobs) > 1:
                ctx = multiprocessing.get_context("fork")
                with ctx.Pool(len(jobs)) as pool:
                    parts = pool.map(_render_chunk, jobs)
            else:
                parts = [_render_chunk(j) for j in jobs]
        finally:
            _JOB.clear()

        listing = tdir / "parts.txt"
        listing.write_text("".join(f"file '{p}'\n" for p in parts),
                           encoding="utf-8")
        if progress:
            progress("    muxing audio …")
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-f", "concat", "-safe", "0", "-i", str(listing),
               "-i", str(audio)]
        if clip is not None:
            cw, ch, cx, cy = (1080, 608, 0, RSV_SLOT_TOP) if w < h else \
                             (960, 540, 0, 0)
            cmd += ["-i", str(clip), "-filter_complex",
                    f"[2:v]scale={cw}:{ch}:force_original_aspect_ratio="
                    f"increase,crop={cw}:{ch},setpts=PTS-STARTPTS[perf];"
                    f"[0:v][perf]overlay={cx}:{cy}:eof_action=pass[v]",
                    "-map", "[v]", "-map", "1:a",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-pix_fmt", "yuv420p"]
        else:
            # the parts are already the final picture: copy, don't re-encode
            cmd += ["-map", "0:v", "-map", "1:a", "-c:v", "copy"]
        cmd += ["-c:a", "aac", "-b:a", "192k", "-t", f"{duration:.3f}",
                "-movflags", "+faststart", str(out)]
        subprocess.run(cmd, check=True)
    return out


# --------------------------------------------------------------------------
# per-verse clips (TikTok / Reels)
# --------------------------------------------------------------------------

def split_verses(sync: list, min_lines: int = 10, min_secs: float = 30.0,
                 max_secs: float = 110.0,
                 gap_s: float = 8.0) -> list[tuple[int, int]]:
    """Group sync lines into verse ranges [(first, last), …] (inclusive)
    for per-verse clips. Sections split at blank lines / instrumental gaps;
    then intros, hooks and bridges (short sections — fewer than min_lines
    lines or under min_secs) merge into a neighboring verse, so a 3-verse
    song yields ~3 clips (verse + its hook), each capped near max_secs."""
    content = [i for i, (_, txt) in enumerate(sync) if txt.strip()]
    if not content:
        return []
    groups: list[list[int]] = [[content[0]]]
    for prev, i in zip(content, content[1:]):
        blank_between = any(not sync[j][1].strip() for j in range(prev + 1, i))
        if blank_between or (sync[i][0] - sync[prev][0]) > gap_s:
            groups.append([i])
        else:
            groups[-1].append(i)

    def dur(g):
        return sync[g[-1]][0] + 4.0 - sync[g[0]][0]

    def small(g):
        return len(g) < min_lines or dur(g) < min_secs

    changed = True
    while changed and len(groups) > 1:
        changed = False
        for k, g in enumerate(groups):
            if not small(g):
                continue
            span = lambda a, b: sync[b[-1]][0] + 4.0 - sync[a[0]][0]
            if k > 0 and span(groups[k - 1], g) <= max_secs:
                groups[k - 1].extend(g)          # hook joins its verse
            elif k + 1 < len(groups) and span(g, groups[k + 1]) <= max_secs:
                groups[k + 1][:0] = g            # intro joins the next verse
            else:
                continue
            groups.pop(k)
            changed = True
            break

    # lyrics with no blank lines / gaps collapse into one giant group —
    # a clip must stay clip-sized, so oversized groups split themselves
    # at their largest internal pauses
    final: list = []
    for g in groups:
        while dur(g) > max_secs and len(g) >= 2 * min_lines:
            t0 = sync[g[0]][0]
            cands = [i for i in range(min_lines - 1, len(g) - min_lines)
                     if min_secs <= sync[g[i + 1]][0] - t0 <= max_secs]
            if not cands:
                break
            cut = max(cands, key=lambda i: (sync[g[i + 1]][0]
                                            - sync[g[i]][0],
                                            sync[g[i + 1]][0] - t0))
            final.append(g[:cut + 1])
            g = g[cut + 1:]
        final.append(g)
    return [(g[0], g[-1]) for g in final]


def parse_timecode(s: str) -> float:
    """'1:02' → 62.0 · '1:02:30' → 3750.0 · '75' → 75.0 · '1:02.5' → 62.5.

    Raises ValueError on anything else, so a typo is reported rather than
    silently rendering the wrong part of the song."""
    parts = str(s).strip().split(":")
    if not 1 <= len(parts) <= 3 or any(p.strip() == "" for p in parts):
        raise ValueError(f"bad timecode {s!r} — use m:ss, h:mm:ss or seconds")
    total = 0.0
    for p in parts:
        try:
            total = total * 60 + float(p)
        except ValueError:
            raise ValueError(
                f"bad timecode {s!r} — use m:ss, h:mm:ss or seconds") from None
    return total


def parse_range(s: str, duration: float) -> tuple[float, float]:
    """'1:02-3:07' → (62.0, 187.0). Also accepts '1:02 3:07', '1:02..3:07',
    and an open end ('2:10-' → to the end of the song). The window is
    clamped to the audio and checked for order."""
    txt = str(s).strip().replace("..", "-").replace("–", "-")
    if " " in txt and "-" not in txt:
        txt = txt.replace(" ", "-", 1)
    bits = [b.strip() for b in txt.split("-") if b.strip()]
    if not bits or len(bits) > 2:
        raise ValueError(f"bad range {s!r} — use 1:02-3:07")
    t0 = parse_timecode(bits[0])
    t1 = parse_timecode(bits[1]) if len(bits) == 2 else duration
    if t1 < t0:
        raise ValueError(f"range {s!r} runs backwards — start must come first")
    if t0 >= duration:
        raise ValueError(f"range {s!r} starts past the end of the song "
                         f"({int(duration) // 60}:{int(duration) % 60:02d})")
    t0 = max(0.0, min(t0, duration))
    t1 = max(0.0, min(t1, duration))
    if t1 - t0 < 1.0:
        raise ValueError(f"range {s!r} is under a second long")
    return t0, t1


def make_clip(analysis: Analysis, sync: list, audio: pathlib.Path,
              title: str, out: pathlib.Path, t0: float, t1: float,
              theme: str = "light", schedule: list | None = None,
              format: str = "tiktok", lines_range: tuple | None = None,
              clip: pathlib.Path | None = None, verse_label: str | None = None,
              ep: str | None = None, scheme: str | None = None,
              song_colors: list | None = None,
              progress=print) -> pathlib.Path:
    """Render just the window [t0, t1] of a song.

    The audio is cut to the window, and both the rhyme map and the word
    timing are SLICED out of the whole song's — nothing is analysed again.
    That is what makes a clip trustworthy: your corrections from the rhyme
    editor are in it, its colours are the full video's colours, and the
    Whisper alignment carries over unchanged.

    A line is included when it OVERLAPS the window, not merely when it starts
    inside it — so asking for 1:02 mid-sentence still opens with the line
    being sung there on screen, instead of silent audio over a blank card.
    Pass `lines_range` to force an explicit inclusive line span instead of
    time-overlap selection (used by the per-verse path)."""
    from rhyme_engine import slice_analysis
    from term_render import syllable_map as _smap

    if schedule is None:
        _fam, wbl = _smap(analysis)
        schedule = word_schedule(sync, wbl)

    if lines_range is not None:
        i0, i1 = lines_range
        idx = list(range(i0, i1 + 1))
    else:
        # overlap test: line j runs from its own start to the next line's
        idx = []
        for j, (start, _txt) in enumerate(sync):
            nxt = next((sync[k][0] for k in range(j + 1, len(sync))), t1 + 1.0)
            if start < t1 and nxt > t0:
                idx.append(j)
    if not idx:
        raise RuntimeError(f"no lyrics between {t0:.0f}s and {t1:.0f}s")

    if song_colors is None:
        song_colors = _palette(analysis)
    sub = slice_analysis(analysis, idx, colors=song_colors)
    sub_sync = [(max(sync[j][0] - t0, 0.0), sync[j][1]) for j in idx]
    sub_sched = [[max(t - t0, 0.0) for t in schedule[j]] for j in idx]

    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tf:
        cut = pathlib.Path(tf.name)
    clip_cut: pathlib.Path | None = None
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-ss", f"{t0:.3f}", "-i", str(audio), "-t", f"{t1 - t0:.3f}",
             "-vn", "-c:a", "aac", "-b:a", "192k", str(cut)],
            check=True)
        if clip is not None:
            # the performance video is cut to the same window, assuming its
            # timeline matches the audio (nudge with --offset if it drifts)
            with tempfile.NamedTemporaryFile(suffix=".mp4",
                                             delete=False) as cf:
                clip_cut = pathlib.Path(cf.name)
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-ss", f"{t0:.3f}", "-i", str(clip),
                 "-t", f"{t1 - t0:.3f}", "-an",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                 str(clip_cut)],
                check=True)
        make_video(sub, sub_sync, cut, title, out, theme=theme,
                   schedule=sub_sched, format=format, clip=clip_cut,
                   verse_label=verse_label, ep=ep, scheme=scheme,
                   progress=progress)
    finally:
        cut.unlink(missing_ok=True)
        if clip_cut is not None:
            clip_cut.unlink(missing_ok=True)
    return out


def make_verse_videos(analysis: Analysis, sync: list, audio: pathlib.Path,
                      title: str, out_base: pathlib.Path,
                      theme: str = "light", schedule: list | None = None,
                      format: str = "tiktok",
                      clip: pathlib.Path | None = None,
                      ep: str | None = None, scheme: str | None = None,
                      progress=print) -> list[pathlib.Path]:
    """Render one portrait clip per verse: the audio is cut to the verse's
    window, the verse is re-analyzed on its own (colors reflect the verse's
    scheme), and word timing is sliced from the full-song schedule so the
    Whisper alignment carries over. Returns the list of written files."""
    from term_render import syllable_map as _smap

    if schedule is None:
        _fam, wbl = _smap(analysis)
        schedule = word_schedule(sync, wbl)
    duration = _audio_duration(audio)
    verses = split_verses(sync)
    # one colouring for the whole song, so verse 3 paints a rhyme the
    # same colour verse 1 did — and the whole song's reading, corrections
    # included, is what every verse is cut from
    song_colors = _palette(analysis)
    outs: list[pathlib.Path] = []

    for k, (i0, i1) in enumerate(verses, start=1):
        first_t = min((schedule[i0][0] if schedule[i0] else sync[i0][0]),
                      sync[i0][0])
        last_words = schedule[i1] or [sync[i1][0]]
        # next content line bounds the tail; cap it so long instrumentals
        # don't pad the clip
        nxt = next((sync[j][0] for j in range(i1 + 1, len(sync))
                    if sync[j][1].strip()), duration)
        t0 = max(0.0, first_t - 1.2)
        t1 = min(duration, min(nxt, last_words[-1] + 4.0))
        if t1 - t0 < 3.0:
            continue

        out = out_base.with_suffix(f".verse{k}.{format}.mp4")
        if progress:
            progress(f"  verse {k}/{len(verses)} "
                     f"({int(t0)}s–{int(t1)}s, {i1 - i0 + 1} lines)")
        try:
            # the verse path knows its own line span, so it passes it rather
            # than letting make_clip re-derive one from the timestamps
            make_clip(analysis, sync, audio, title, out, t0, t1, theme=theme,
                      schedule=schedule, format=format,
                      lines_range=(i0, i1), clip=clip,
                      verse_label=f"VERSE {k}", ep=ep, scheme=scheme,
                      song_colors=song_colors, progress=progress)
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            # one bad verse (e.g. the audio file moved mid-run) shouldn't
            # abort the remaining clips
            if progress:
                progress(f"  ✗ verse {k} failed: {exc}")
            continue
        outs.append(out)
    return outs


# --------------------------------------------------------------------------
# still frames (16:9 spec, §Export)
# --------------------------------------------------------------------------

def export_frames(analysis: Analysis, sync: list, title: str,
                  out_dir: pathlib.Path, ep: str | None = None,
                  line: int = 3, verse_label: str | None = None,
                  guides: bool = False, stem: str = "rs") -> list[pathlib.Path]:
    """Write the three 16:9 frames as 1920×1080 PNGs, named the way the spec
    asks: `rs-ep041-01-title.png`, `-02-verse.png`, `-04-end.png`. `line` is
    the 1-based bar to show as the active one on the verse frame."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fam_at, words_by_line = syllable_map(analysis)
    colors = _palette(analysis, THEMES["light"])
    bar_no, c = {}, 0
    for j, (_t, txt) in enumerate(sync):
        if txt.strip():
            c += 1
            bar_no[j] = c
    active = next((j for j, n in bar_no.items() if n == line), -1)

    pre = "-".join(p for p in (stem, ep) if p)
    frames = {
        f"{pre}-01-title.png": _yt_title_card(title),
        # the still shows the verse as the video has it at that bar: lit
        # down to the active line, the bars below it still bare
        f"{pre}-02-verse.png": _yt_verse_frame(
            _yt_verse_base(title), sync, words_by_line, colors, fam_at,
            bar_no, active, verse_label=verse_label),
        f"{pre}-04-end.png": _yt_end_card(guides=guides),
    }
    written = []
    for name, img in frames.items():
        img.save(out_dir / name)
        written.append(out_dir / name)
    return written


def export_vertical(analysis: Analysis, sync: list, title: str,
                    out_dir: pathlib.Path, ep: str | None = None,
                    line: int = 5, scheme: str | None = None,
                    placeholder: bool = True, theme: str = "light",
                    stem: str = "rs") -> pathlib.Path:
    """Write the 9:16 frame as a 1080×1920 PNG, named the way the vertical
    spec asks: `rs-ep041-vertical.png`. `line` is the 1-based bar shown
    active (the spec's static frame is line 5); `placeholder` fills the
    clip band the way the template does rather than with the title.
    `theme` picks the vertical palette — "light" (paper) or "dark" (ink);
    a dark export is named `rs-ep041-vertical-dark.png` so the two can sit
    in one folder."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fam_at, words_by_line = syllable_map(analysis)
    colors = _palette(analysis, THEMES["light"])
    bar_no, c = {}, 0
    for j, (_t, txt) in enumerate(sync):
        if txt.strip():
            c += 1
            bar_no[j] = c
    active = next((j for j, n in bar_no.items() if n == line), -1)

    rsv = RSV_THEMES[theme]
    name = "-".join(p for p in (stem, ep, "vertical",
                                None if theme == "light" else theme)
                    if p) + ".png"
    # the still shows the page as the video has it at that bar: lit down
    # to the active line, the bars below it still bare
    img = _rsv_frame(_rsv_base(title, ep=ep, placeholder=placeholder,
                               rsv=rsv),
                     sync, words_by_line, colors, fam_at, bar_no, active,
                     scheme=scheme or _scheme_label(analysis), rsv=rsv)
    img.save(out_dir / name)
    return out_dir / name


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Export the 16:9 spec frames (01 title / 02 verse / "
                    "04 end) as 1920×1080 PNGs, or the 9:16 vertical "
                    "frame as a 1080×1920 PNG (--vertical).")
    ap.add_argument("lyrics", type=pathlib.Path,
                    help="text file, one bar per line")
    ap.add_argument("title", help='"Artist - Song Title"')
    ap.add_argument("-o", "--out", type=pathlib.Path,
                    default=pathlib.Path("."))
    ap.add_argument("--ep", help='episode tag, e.g. "ep041"')
    ap.add_argument("--verse", help='bottom-right label, e.g. "VERSE 2"')
    ap.add_argument("--line", type=int, default=3,
                    help="1-based bar shown active (9:16 defaults to 5)")
    ap.add_argument("--vertical", action="store_true",
                    help="export the 9:16 frame instead of the 16:9 set")
    ap.add_argument("--theme", default="light", choices=["light", "dark"],
                    help="9:16 palette: paper or ink (16:9 has one look)")
    ap.add_argument("--scheme",
                    help='band 4 label, e.g. "MULTI, 3 SYLLABLE CHAIN"; '
                         "derived from the analysis when omitted")
    ap.add_argument("--clip-title", action="store_true",
                    help="9:16: put the title in the clip band instead of "
                         "the template's placeholder")
    ap.add_argument("--guides", action="store_true",
                    help="draw the end screen's dashed zones (never export "
                         "a frame with these)")
    o = ap.parse_args()

    from rhyme_engine import analyze

    text = o.lyrics.read_text(encoding="utf-8")
    lines = text.splitlines()
    sync = [(float(i), t) for i, t in enumerate(lines)]
    if o.vertical:
        print(export_vertical(
            analyze(text), sync, o.title, o.out, ep=o.ep,
            line=o.line if o.line != 3 else 5, scheme=o.scheme,
            placeholder=not o.clip_title, theme=o.theme))
    else:
        for p in export_frames(analyze(text), sync, o.title, o.out,
                               ep=o.ep, line=o.line, verse_label=o.verse,
                               guides=o.guides):
            print(p)
