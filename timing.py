"""
timing.py — word-level timing shared by the karaoke, the editor and the
rhyme map: line timestamps from a Whisper pass aligned against the lyrics,
and word start times inside each line.
"""

from __future__ import annotations

import difflib
import functools
import pathlib
import re as _re
import statistics
import subprocess

from palette import assign as _assign, highlighter_pool as _hpool

# Highlighter set — one colour per rhyme sound. palette.assign() reuses
# them adjacency-aware when a song has more families than colours.
HIGHLIGHTERS = ["#FBA6B6", "#29C8DE", "#12A67C",
                "#B79EE9", "#F5893F", "#C9BA3E"]


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
