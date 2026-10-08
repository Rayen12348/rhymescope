"""
confidence.py — how sure the detector is about every highlight.

The detector used to answer one question — rhyme or not — through a stack
of hard bars: this many lines, that many words, never a function word
mid-line. Every bar was a guess about where your ear draws the line, and
no single setting of them is right for every listener or every song. Set
them tight and real rhymes vanish; set them loose and a dense verse turns
into a wall of colour.

So the bars stop deciding. The detector runs twice — once at its normal
bars and once with every recurrence bar lowered a notch — and the union
is the set of CANDIDATES. Each candidate syllable then gets a confidence
from the evidence around it, and one control decides how much you see:

    Clean      the rhymes each bar lands on, and little else
    Balanced   the default: the scheme and every real multi, without the
               single-vowel echoes swept up far from the end of the bar
    Rich       adds the mid-line devices — assonance runs, rhymes inside
               one bar — that some ears hear and others do not
    Everything every candidate either pass found, mid-line devices and
               near-misses included

What counts as evidence was not guessed either. Your own corrections were
the ground truth: across the songs you fixed, the highlights you erased
were overwhelmingly mid-line vowel echoes packed into lines where almost
every syllable was lit, and the ones you added were multi-syllable rhymes
landing on phrase ends. The features below are exactly those signals.

And it keeps learning: every correction you save in the editor is a new
labelled example (see learn.py), so the weights drift toward your ear.
"""

from __future__ import annotations

import math
from collections import Counter

from rhyme_engine import Analysis, Family, STOPWORDS, _clean, analyze

# The three presets, as confidence thresholds. The editor's slider moves
# continuously between them; these are just the named stops.
LEVELS = {"clean": 0.8, "balanced": 0.25, "rich": 0.15, "everything": 0.0}

# The threshold scale changes whenever the weights are refitted, so a
# saved setting records which scale it was saved on. Scale 1 (the first
# weights) mapped to these presets; a file from then is read as the preset
# you picked, not as a raw number that now means something else.
LEVELS_SCALE = 4
_SCALE1 = {0.85: "clean", 0.5: "balanced", 0.08: "rich", 0.0: "everything"}
_SCALE2 = {0.5: "clean", 0.25: "balanced", 0.15: "rich", 0.0: "everything"}
_SCALE3 = _SCALE2


def migrate_level(value: float | None, scale: int | None) -> float | None:
    if value is None or scale == LEVELS_SCALE:
        return value
    table_for = {1: _SCALE1, 2: _SCALE2, 3: _SCALE3}
    for table in ([table_for[scale]] if scale in table_for
                  else [_SCALE1, _SCALE2]):
        for old, name in table.items():
            if abs(value - old) < 1e-6:
                return LEVELS[name]
    return value
# Rich, not Balanced: on the hand-annotated verses it catches 84% of the
# rhyme pairs against 81% (88% against 85% on the held-out verse) while
# random bars with no rhyme in them stay nearly dark (0.72 lit syllables
# against 0.64). Below Rich that noise more than doubles.
DEFAULT_LEVEL = LEVELS["rich"]

FEATURES = ("evidence", "end_route", "mid_route", "tail_zone", "in_default",
            "stress", "content", "depth", "fam_lines", "fam_words",
            "fam_func", "fam_ends", "true_rhyme", "parallel")

# Starting weights for a brand-new install; learn.py refines them from the
# corrections you save. They encode what those corrections showed: a rhyme
# found where the bar LANDS is trusted, and trust falls away with distance
# from the end of the bar — 21 of the first 22 highlights ever erased by
# hand were mid-line echoes, sitting 8 syllables from the end on average,
# while the echoes that were kept sat 4 from it.
# Fitted, not guessed — on two kinds of evidence that pull in different
# directions, so that neither one alone decides the default:
#
#   · a hand-annotated answer key (gold.py): every rhyme group a careful
#     listener marks in dense verses by Prodigy, Eminem and Black Thought.
#     This is the "did it catch what I hear" side.
#   · your own corrections: what you erased, added and kept. This is the
#     "that is not a rhyme to me" side.
#
# Held out one song at a time, it ranks real rhymes above noise at an AUC
# of 0.84-0.87 on songs it was not fitted to. learn.py refines these from
# the corrections you save from here on.
# The detector already answers "how sure" in its own currency — bits of
# log-odds, prior for where the rhyme lands plus evidence for how it
# sounds — so confidence is that number turned into a probability:
#
#     p ≈ sigmoid(0.9 · log-odds − 2.2)
#
# which puts Balanced (0.25) at about +1.2 bits — a little above the
# detector's own bar, because that is where your corrections put the
# line: between 0 and ~1 bit sits most of what you erased, above it most
# of what you painted. Clean is about +4 (only what is plainly a rhyme),
# Rich about +0.5, and Everything reaches every candidate the loose pass
# found. Every other feature starts at
# zero: they are there for learn.py to move, from your corrections, and
# only when that beats this on verses it was not fitted to.
DEFAULT_WEIGHTS = {
    "bias": -6.68,
    "evidence": 18.00,  # (log-odds + 5) / 20, so 18× it = 0.9·log-odds + 4.5
    "in_default": 0.0,
    "end_route": 0.0,
    "mid_route": 0.0,
    "tail_zone": 0.0,
    "depth": 0.0,
    "true_rhyme": 0.0,
    "parallel": 0.0,
    "stress": 0.0,
    "content": 0.0,
    "fam_lines": 0.0,
    "fam_words": 0.0,
    "fam_func": 0.0,
    "fam_ends": 0.0,
}

_PUNCT = set(",.;:!?…—")


def _phrase_end_distance(an: Analysis) -> dict:
    """Syllables from each position to the end of its phrase (0 = last)."""
    word_pos: list = [[] for _ in an.words]
    for i, s in enumerate(an.stream):
        word_pos[s.word_index].append(i)
    out: dict = {}
    cur: list = []
    for w_i, w in enumerate(an.words):
        cur.extend(word_pos[w_i])
        nxt = an.words[w_i + 1] if w_i + 1 < len(an.words) else None
        end = nxt is None or nxt.line != w.line
        if not end:
            end = any(ch in _PUNCT for ch in an.lines[w.line][w.end:nxt.start])
        if end and cur:
            for d, i in enumerate(reversed(cur)):
                out[i] = d
            cur = []
    return out


def _merge(base: Analysis, extra: Analysis) -> tuple[Analysis, set]:
    """Union of two analyses of the SAME text, by position.

    `base` families are the backbone and keep their identity. A candidate
    family from `extra` that shares members with a base family is folded
    into the one it overlaps most; one that shares nothing stands as its
    own (candidate) family. Returns the union and the positions the base
    analysis had on its own."""
    fam_of = dict(base.family_of)
    members = {f.fid: set(f.members) for f in base.families}
    families = [Family(fid=f.fid, label=f.label, members=[], span=f.span)
                for f in base.families]
    for ef in extra.families:
        overlap = Counter(fam_of[m] for m in ef.members if m in fam_of)
        new = [m for m in ef.members if m not in fam_of]
        if overlap:
            target = overlap.most_common(1)[0][0]
        else:
            if len(new) < 2:
                continue
            target = len(families)
            families.append(Family(fid=target, label=ef.label, members=[]))
            members[target] = set()
        for m in new:
            members[target].add(m)
            fam_of[m] = target
    line_of = lambda m: base.words[base.stream[m].word_index].line
    for f in families:
        f.members = sorted(members[f.fid])
        f.span = len({line_of(m) for m in f.members})
    union = Analysis(base.lines, base.words, base.stream, fam_of, families,
                     dict(base.stats))
    route = dict(getattr(extra, "route", {}))
    route.update(getattr(base, "route", {}))
    union.route = route
    odds = dict(getattr(extra, "odds", {}))
    for m, v in getattr(base, "odds", {}).items():
        odds[m] = max(odds.get(m, -99.0), v)
    union.odds = odds
    return union, set(base.family_of)


def _multi_depth(an: Analysis, window: int = 4) -> dict:
    """pos → the length of the longest multi running through it.

    A multi is not "several highlighted syllables in a row" — a dense line
    lights every syllable in a row with unrelated sounds. It is an ORDERED
    SEQUENCE of rhyme sounds that recurs: na-vi-GA-tion against in-for-
    MA-tion is the same run of families twice. So for each highlighted
    syllable, find a partner in its own family within a few lines and walk
    outward in both directions for as long as the two runs keep matching
    family for family. The longest such walk is the depth. A lone vowel
    echo has depth 1 however many neighbours happen to be lit."""
    fo = an.family_of
    line = lambda m: an.words[an.stream[m].word_index].line
    wid = lambda m: an.stream[m].word_index
    txt = lambda m: _clean(an.words[an.stream[m].word_index].text)

    def real_pair(a: int, b: int) -> bool:
        """A step that is actually rhyme: the two syllables share a vowel,
        and they are not both weak syllables on function words. Lining up
        "the" against "to-" family for family is glue, however long the
        run of it goes."""
        sa, sb = an.stream[a], an.stream[b]
        if {v for v, _c in sa.forms}.isdisjoint(v for v, _c in sb.forms):
            return False
        weak = (not sa.stress and not sb.stress
                and (txt(a) in STOPWORDS or txt(b) in STOPWORDS))
        return not weak

    by_fam: dict = {}
    for m, f in fo.items():
        by_fam.setdefault(f, []).append(m)
    out: dict = {}
    for p, f in fo.items():
        best = 1
        lp = line(p)
        for q in by_fam[f]:
            if (q == p or abs(line(q) - lp) > window or wid(q) == wid(p)
                    or txt(q) == txt(p)):
                continue
            n = 1 if real_pair(p, q) else 0
            for step in (-1, 1):
                k = 1
                while True:
                    a, b = p + step * k, q + step * k
                    # the same word twice is repetition, not a longer
                    # multi — "…the dog / …the moon" is one rhyme deep
                    if (a not in fo or b not in fo or fo[a] != fo[b]
                            or a == q or b == p or txt(a) == txt(b)):
                        break
                    n += real_pair(a, b)
                    k += 1
            best = max(best, n)
        out[p] = best
    return out


def _partners(an: Analysis, line_end: dict, window: int = 2) -> dict:
    """pos → (true_rhyme, parallel) from its best partner in the family
    within `window` lines: true_rhyme is 1 when some partner matches the
    vowel AND the coda (a full rhyme, not assonance), parallel is 1 when
    such a partner sits at the same distance from the end of its bar."""
    from rhyme_engine import _coda_score
    word_of = lambda m: an.words[an.stream[m].word_index]
    by_fam: dict = {}
    for m, f in an.family_of.items():
        by_fam.setdefault(f, []).append(m)
    out: dict = {}
    for m, f in an.family_of.items():
        sm = an.stream[m]
        lm, wm = word_of(m).line, _clean(word_of(m).text)
        dm = line_end[lm] - m
        tr = par = 0.0
        for q in by_fam[f]:
            lq = word_of(q).line
            if q == m or abs(lq - lm) > window or _clean(word_of(q).text) == wm:
                continue
            sq = an.stream[q]
            same_v = not {v for v, _c in sm.forms}.isdisjoint(
                v for v, _c in sq.forms)
            coda = max(_coda_score(ca, cb) for _v, ca in sm.forms
                       for _w, cb in sq.forms)
            full = 1.0 if (same_v and coda >= 0.2) else (0.5 if same_v and coda > 0 else 0.0)
            tr = max(tr, full)
            if full and lq != lm and abs((line_end[lq] - q) - dm) <= 1:
                par = 1.0
        out[m] = (tr, par)
    return out


def features(an: Analysis, default_pos: set) -> dict:
    """pos → {feature: value in 0..1} for every highlighted syllable."""
    end_d = _phrase_end_distance(an)
    word_of = lambda m: an.words[an.stream[m].word_index]
    is_func = lambda m: _clean(word_of(m).text) in STOPWORDS

    fam_stats: dict = {}
    for f in an.families:
        ms = f.members
        if not ms:
            continue
        lines = {word_of(m).line for m in ms}
        content = {_clean(word_of(m).text) for m in ms if not is_func(m)}
        fam_stats[f.fid] = {
            "fam_lines": min(len(lines), 6) / 6,
            "fam_words": min(len(content), 6) / 6,
            "fam_func": sum(1 for m in ms if is_func(m)) / len(ms),
            "fam_ends": sum(1 for m in ms if end_d.get(m, 9) <= 2) / len(ms),
        }

    line_end: dict = {}
    for i, s in enumerate(an.stream):
        line_end[an.words[s.word_index].line] = i
    route = getattr(an, "route", {})
    # How strong the detector's own case was, in bits of log-odds. The
    # engine already weighed this highlight against the prior for where
    # it sits; squashing that number into 0..1 hands the confidence model
    # the whole of that reasoning instead of making it guess from the
    # outside. Everything below is what the engine does NOT know: how the
    # highlight sits in the bar, and what kind of company it keeps.
    odds = getattr(an, "odds", {})
    depth = _multi_depth(an)
    partner = _partners(an, line_end)

    out: dict = {}
    for m, fid in an.family_of.items():
        s = an.stream[m]
        ln = word_of(m).line
        r = route.get(m, "glue")
        x = {
            "evidence": max(0.0, min(1.0, (odds.get(m, 0.0) + 5.0) / 20.0)),
            "end_route": 1.0 if r == "end" else 0.0,
            "mid_route": 1.0 if r in ("leonine", "assonance", "glue") else 0.0,
            # distance from the end of the BAR, not the phrase: a comma
            # mid-bar does not make its neighbourhood a rhyme position
            "tail_zone": math.exp(-(line_end[ln] - m) / 4.0),
            "in_default": 1.0 if m in default_pos else 0.0,
            "stress": 1.0 if s.stress == 1 else (0.5 if s.stress else 0.0),
            "content": 0.0 if is_func(m) else 1.0,
            # 0 for a lone echo, 1 for a multi four sounds deep or more
            "depth": (min(depth.get(m, 1), 4) - 1) / 3,
            # a partner nearby that rhymes on the ENDING too, not just the
            # vowel — "dream/theme" is a rhyme, "keep/see" an echo
            "true_rhyme": partner.get(m, (0.0, 0.0))[0],
            # …sitting at the same spot in its bar: parallel bars
            # ("a reoccurin' DREAM / the reoccurin' THEME") are how
            # internal rhyme is built on purpose
            "parallel": partner.get(m, (0.0, 0.0))[1],
        }
        x.update(fam_stats.get(fid, {}))
        out[m] = x
    return out


def _sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, z))))


def predict(x: dict, weights: dict) -> float:
    return _sigmoid(weights["bias"]
                    + sum(weights[k] * x.get(k, 0.0) for k in FEATURES))


def reasons(x: dict, weights: dict) -> list[str]:
    """The evidence behind one highlight, strongest first, in words."""
    say = {
        "evidence": ("the sound itself is strong evidence", "thin evidence for the sound itself"),
        "end_route": ("part of the rhyme the bar lands on", None),
        "mid_route": (None, "swept up by a mid-line vowel pass"),
        "tail_zone": ("close to the end of the bar", None),
        "in_default": ("passed the normal rhyme bars", None),
        "stress": ("on a stressed syllable", "on a weak syllable"),
        "content": ("on a real word", "on a function word"),
        "depth": ("part of a multi-syllable rhyme", None),
        "fam_lines": ("the rhyme comes back across the verse", None),
        "fam_words": ("the rhyme lands on many different words", None),
        "fam_func": (None, "the rhyme is mostly function words"),
        "fam_ends": ("the rhyme lives at the ends of bars", None),
        "true_rhyme": ("rhymes on the ending too, not just the vowel", None),
        "parallel": ("sits at the same spot as its partner in the next bar", None),
    }
    scored = []
    for k in FEATURES:
        v = x.get(k, 0.0)
        contrib = weights[k] * v
        good, bad = say[k]
        if k == "tail_zone" and v < 0.3:
            scored.append((weights[k] * (0.3 - v) * 3, "far from the end of the bar"))
        elif contrib > 0.25 and good:
            scored.append((contrib, good))
        elif contrib < -0.25 and bad:
            scored.append((-contrib, bad))
        elif k in ("stress", "content") and v == 0 and bad:
            scored.append((abs(weights[k]) * 0.6, bad))
    return [t for _c, t in sorted(scored, reverse=True)]


def score(text: str, strict: bool = False,
          weights: dict | None = None, heard: dict | None = None) -> Analysis:
    """Analyse `text` and attach a confidence to every candidate.

    Returns the UNION analysis (every candidate is a family member) with
    `.confidence` {pos: 0..1} and `.evidence` {pos: features}. Use
    `at_level` to cut it down to what a given setting shows."""
    if weights is None:
        from learn import current_weights
        weights = current_weights()
    aligned: dict = {}
    base = analyze(text, strict=strict, heard=heard, align_cache=aligned)
    loose = analyze(text, strict=strict, loose=True, heard=heard,
                    align_cache=aligned)
    union, default_pos = _merge(base, loose)
    ev = features(union, default_pos)
    union.confidence = {m: predict(x, weights) for m, x in ev.items()}
    union.evidence = ev
    union.weights = weights
    return union


def at_level(an: Analysis, level: float, keep: set = frozenset()) -> Analysis:
    """The analysis as it appears at confidence threshold `level`, in place.

    `keep` positions survive regardless — a syllable you painted by hand is
    shown whatever the slider says."""
    conf = getattr(an, "confidence", None)
    if conf is None:
        return an
    line_of = lambda m: an.words[an.stream[m].word_index].line
    kept: list = []
    family_of: dict = {}
    for f in an.families:
        ms = [m for m in f.members if conf.get(m, 1.0) >= level or m in keep]
        if len(ms) < 2:
            continue
        f.members = ms
        f.fid = len(kept)
        f.span = len({line_of(m) for m in ms})
        kept.append(f)
        for m in ms:
            family_of[m] = f.fid
    an.families = kept
    an.family_of = family_of
    n = len(an.stream)
    an.stats = dict(an.stats, families=len(kept),
                    rhymed_syllables=len(family_of),
                    density=(len(family_of) / n) if n else 0.0)
    return an
