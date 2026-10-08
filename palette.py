"""
palette.py — perceptually-separated rhyme colors.

Two problems this solves, both of which a naive hue wheel gets wrong:

1. HUMAN VISION IS NOT LINEAR IN HUE. Evenly-spaced HSL hues cluster badly
   in the yellow/orange/green region — "gold" and "amber" are 30° apart on
   the wheel and nearly identical to the eye, while 30° in the blue range
   is plainly two colors. Colors here are generated in OKLab/OKLCH, where
   Euclidean distance approximates perceived difference, and every color
   is checked against every other for a minimum separation.

2. ADJACENCY IS WHAT ACTUALLY MATTERS. Two similar colors on opposite
   ends of a song are fine; the same two touching on one line are a bug.
   So colors are not handed out in family order — families that appear
   NEAR each other in the text form an adjacency graph, and assignment
   walks that graph most-constrained-first, giving each family the
   remaining color that is perceptually farthest from everything it
   touches.
"""

from __future__ import annotations

import math

# --------------------------------------------------------------------------
# OKLab / OKLCH  (Björn Ottosson's perceptual space)
# --------------------------------------------------------------------------


def _oklch_to_srgb(L: float, C: float, H: float) -> tuple[int, int, int]:
    a = C * math.cos(math.radians(H))
    b = C * math.sin(math.radians(H))
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bl = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

    def enc(v: float) -> int:
        v = max(0.0, min(1.0, v))
        v = 1.055 * (v ** (1 / 2.4)) - 0.055 if v > 0.0031308 else 12.92 * v
        return max(0, min(255, round(v * 255)))

    return enc(r), enc(g), enc(bl)


def _srgb_to_oklab(hex_color: str) -> tuple[float, float, float]:
    def dec(v: int) -> float:
        v /= 255.0
        return ((v + 0.055) / 1.055) ** 2.4 if v > 0.04045 else v / 12.92

    r, g, b = (dec(int(hex_color[k:k + 2], 16)) for k in (1, 3, 5))
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = l ** (1 / 3), m ** (1 / 3), s ** (1 / 3)
    return (0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_)


def _dist(c1: str, c2: str) -> float:
    """Perceived difference between two hex colors (OKLab ΔE)."""
    a, b = _srgb_to_oklab(c1), _srgb_to_oklab(c2)
    # lightness counts less than hue/chroma for block fills: two colors of
    # the same hue at different lightness still read as "the same color"
    return math.sqrt(0.6 * (a[0] - b[0]) ** 2
                     + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


# --------------------------------------------------------------------------
# WCAG contrast — the readability floor
# --------------------------------------------------------------------------
#
# OKLab distance answers "are these two colors different?", which is what
# family separation needs. It does NOT answer "can I read text on this?" —
# that is luminance contrast, a different question with a different metric.
# A fill can be perfectly distinct from its neighbours and still swallow
# the word printed on it.
#
# Naive sRGB luma (0.299R+0.587G+0.114B) gets this wrong in a specific,
# visible way: it under-weights blue and over-weights nothing that
# compensates, so saturated cyans, teals and blues (#00bab3, #009dd4,
# #2193ff) score LOW on luma while being plainly light to the eye. Threshold
# on that number and those colors get white text — 2.4:1, unreadable —
# when dark ink would have given 6.6:1. Every failure of the old rule was
# of exactly this shape. So contrast is computed properly here, and the
# pools carry a hard floor: no color enters a pool that dark ink cannot be
# read on.

INK = "#1D2130"          # the text color painted on top of every fill
_AA = 4.5                # WCAG AA for normal-size text


def _rel_lum(hex_color: str) -> float:
    """WCAG relative luminance (linearized sRGB, D65 coefficients)."""
    def dec(v: int) -> float:
        v /= 255.0
        return ((v + 0.055) / 1.055) ** 2.4 if v > 0.04045 else v / 12.92

    r, g, b = (dec(int(hex_color[k:k + 2], 16)) for k in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(c1: str, c2: str) -> float:
    """WCAG contrast ratio, 1.0 (identical) … 21.0 (black on white)."""
    a, b = _rel_lum(c1), _rel_lum(c2)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def ink_for(fill: str, dark: str = INK, light: str = "#FFFFFF") -> str:
    """The readable text color on `fill`: whichever of dark/light actually
    contrasts more. Measured, never thresholded."""
    return dark if contrast(fill, dark) >= contrast(fill, light) else light


# --------------------------------------------------------------------------
# color pools
# --------------------------------------------------------------------------

def _build_pool(n: int, light: float, chroma: float,
                min_sep: float, bg: str = "", min_bg: float = 0.0,
                min_ink: float = _AA, seeds: tuple = ()) -> list[str]:
    """Generate n colors at a fixed perceptual lightness, spread around the
    hue circle, then greedily keep only those that stay `min_sep` apart —
    refilling with intermediate hues until the count is met.

    `bg`/`min_bg` enforce contrast with the surface the color is painted
    on: a pale highlight on a white card is invisible no matter how well
    it separates from the other highlights, so any candidate closer than
    `min_bg` to the background is rejected outright.

    `min_ink` enforces the other direction — the text painted ON the fill.
    A color dark enough to need white text is rejected, which makes the
    highlighter behave like a real one: the words stay ink-dark on every
    swipe, and no rhyme is ever rendered as white-on-mid-tone mud.

    `seeds` are colors that must come first — the brand chips. They are
    taken as given (a brand is not negotiable against a distance metric);
    generated colors then fill in behind them, keeping their distance from
    the seeds as well as from each other."""
    out: list[str] = list(seeds)[:n]
    # golden-angle hue walk keeps early colors maximally spread, so the
    # first few families (the biggest ones) get the clearest separation
    step, h = 137.507764, 0.0
    tries = 0
    # slight lightness/chroma alternation adds a second axis of difference
    # so hue does not have to carry the whole load once the wheel fills
    # bands descend from `light`: a second axis of difference that always
    # moves AWAY from a pale background, never toward it
    bands = ((light, chroma), (light - 0.06, chroma * 1.12),
             (light - 0.12, chroma * 1.20), (light - 0.03, chroma * 0.78),
             (light - 0.09, chroma * 0.92), (light - 0.15, chroma * 1.05))
    while len(out) < n and tries < n * 240:
        L, C = bands[len(out) % len(bands)]
        cand = _hex(_oklch_to_srgb(L, C, h % 360))
        if ((not bg or _dist(cand, bg) >= min_bg)
                and contrast(cand, INK) >= min_ink
                and all(_dist(cand, c) >= min_sep for c in out)):
            out.append(cand)
        h += step
        tries += 1
        if tries % (n * 12) == 0:        # relax separation if the space is
            min_sep *= 0.9               # exhausted — but never the
            h += 11.0                    # background or ink floors:
    return out                           # unreadable is never a trade


def bright_pool(n: int) -> list[str]:
    """Saturated block colors for dark (ink navy) backgrounds."""
    return _build_pool(n, light=0.78, chroma=0.155, min_sep=0.115,
                       bg="#171B2E", min_bg=0.30)


def pastel_pool(n: int) -> list[str]:
    """Highlighter inks for light backgrounds. Light enough that
    near-black text stays readable on top, but held well clear of the
    white card underneath so every swipe is actually visible."""
    return _build_pool(n, light=0.80, chroma=0.135, min_sep=0.070,
                       bg="#FFFFFF", min_bg=0.155)


def highlighter_pool(n: int, seeds: tuple = ()) -> list[str]:
    """One colour per rhyme family on paper, however many there are.

    A song does not get to have six rhymes because the brand has six
    chips. Past the seed set the pool keeps generating — same lightness
    range, same contrast floors — so a dense verse with twenty families
    gets twenty distinguishable colours instead of the same six handed
    out four times over, which is what makes two unrelated rhymes look
    like the same rhyme."""
    return _build_pool(n, light=0.79, chroma=0.145, min_sep=0.085,
                       bg="#FDFDFC", min_bg=0.155, seeds=seeds)


# --------------------------------------------------------------------------
# adjacency-aware assignment
# --------------------------------------------------------------------------

def sound_colors(analysis, colors: list) -> dict:
    """Freeze a colouring so any later analysis of the same song can
    reproduce it. See `assign`'s `ref` argument.

    Two indexes, because excerpts arrive with two levels of information:

    `by_position` — every highlighted syllable, keyed by where it sits in
    the song (line, character offset, syllable). A clip is a slice of the
    same text, so when the caller knows which song lines it took, this is
    an exact answer, not a guess.

    `by_sound` — every sound the song uses and the colour it wears, for
    callers with no line mapping. Keyed on every member's sound rather
    than each family's modal one, because an excerpt holds a FRAGMENT of a
    family and a fragment's modal is frequently not the whole family's:
    across a song the "-ents" rhyme may be modally (IH, T S) while the one
    verse a clip covers holds the half that is (IH, S)."""
    from collections import Counter
    by_pos: dict = {}
    votes: dict = {}
    for f in analysis.families:
        if f.fid >= len(colors):
            continue
        c = colors[f.fid]
        for m in f.members:
            syl = analysis.stream[m]
            w = analysis.words[syl.word_index]
            by_pos[(w.line, w.start, syl.index_in_word)] = c
            votes.setdefault((syl.bv, syl.cc), Counter())[c] += 1
    return {"by_position": by_pos,
            "by_sound": {s: v.most_common(1)[0][0] for s, v in votes.items()}}


def assign(analysis, pool_fn, ref: dict | None = None,
           line_map: list | None = None) -> list[str]:
    """Give every family a color, guaranteeing that families which appear
    close together in the lyrics get perceptually distant colors.

    Neighbours = families with members on the same line or one line apart
    (that is exactly the span a viewer sees at once). The graph is walked
    most-constrained-first, and each family takes the unused color that
    maximises its minimum distance to already-coloured neighbours.

    `ref` is a whole-song colouring from `sound_colors`, and pins each
    family here to the colour that song already gave it. Family numbering
    is a property of one analysis — analyse a single verse and the
    families are renumbered from scratch — so colouring by number makes
    the same rhyme change colour between one clip and the next. `ref`
    is what keeps every verse, every clip and every format painting one
    rhyme one colour. `line_map[i]` gives the song line that this
    analysis's line `i` came from; with it the lookup is exact, without it
    it falls back to matching on sound."""
    fams = analysis.families
    n = len(fams)
    if n == 0:
        return []
    pool = pool_fn(n)

    pinned: dict = {}
    # A colour you chose yourself outranks anything computed: overrides
    # carry `fixed_color`, and the whole point of setting one is that it
    # is not negotiable against a distance metric (see overrides.py).
    for f in fams:
        c = getattr(f, "fixed_color", None)
        if c:
            pinned[f.fid] = c
    if ref:
        from collections import Counter
        by_pos = ref.get("by_position", {})
        by_sound = ref.get("by_sound", {})
        used = set(pinned.values())

        def song_colour(m):
            """What the whole song painted this syllable."""
            syl = analysis.stream[m]
            w = analysis.words[syl.word_index]
            if line_map is not None and w.line < len(line_map):
                c = by_pos.get((line_map[w.line], w.start, syl.index_in_word))
                if c:
                    return c
            return by_sound.get((syl.bv, syl.cc))

        # biggest families first: they are the ones a viewer tracks across
        # the song, so they get their established colour before a smaller
        # family that shares some of their sounds can claim it
        for f in sorted(fams, key=lambda f: -len(f.members)):
            if f.fid in pinned:
                continue
            tally: Counter = Counter()
            for m in f.members:
                c = song_colour(m)
                if c:
                    tally[c] += 1
            for c, _n in tally.most_common():
                # two families cannot both take one colour, or the excerpt
                # would show one colour where it shows two rhymes
                if c not in used:
                    pinned[f.fid] = c
                    used.add(c)
                    break
        pool = [c for c in pool if c not in used] + list(used)

    line_of_fam: list[set] = []
    for f in fams:
        line_of_fam.append({analysis.words[analysis.stream[m].word_index].line
                            for m in f.members})

    adj: list[set] = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if any(abs(a - b) <= 1 for a in line_of_fam[i]
                   for b in line_of_fam[j]):
                adj[i].add(j)
                adj[j].add(i)

    # most-constrained first, biggest family breaking ties: the families
    # a viewer sees most often are placed while the pool is still wide
    order = sorted(range(n), key=lambda i: (-len(adj[i]), -len(fams[i].members)))
    colors: list[str | None] = [None] * n
    free = list(pool)
    for i, c in pinned.items():           # song-wide colours are fixed
        colors[i] = c
        if c in free:
            free.remove(c)
    for i in order:
        if colors[i]:
            continue
        taken = [colors[j] for j in adj[i] if colors[j]]
        if not taken:
            pick = free[0]
        else:
            pick = max(free, key=lambda c: min(_dist(c, t) for t in taken))
        colors[i] = pick
        free.remove(pick)
        if not free:                     # more families than colors: reuse
            free = list(pool)

    # Greedy is order-dependent and can strand a bad pair. Hill-climb the
    # result: repeatedly take the worst-looking adjacent pair and try
    # swapping one of its colors with every other family's, keeping any
    # swap that raises the global worst case. A few passes converge.
    def worst_pair(cs):
        w, at = 9.0, None
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    d = _dist(cs[i], cs[j])
                    if d < w:
                        w, at = d, (i, j)
        return w, at

    for _ in range(240):
        w, at = worst_pair(colors)
        if at is None:
            break
        improved = False
        for i in at:
            if i in pinned:               # a pinned colour is not ours to move
                continue
            for k in range(n):
                if k == i or k in pinned:
                    continue
                colors[i], colors[k] = colors[k], colors[i]
                w2, _ = worst_pair(colors)
                if w2 > w + 1e-9:
                    improved = True
                    break
                colors[i], colors[k] = colors[k], colors[i]
            if improved:
                break
        if not improved:
            break
    return [c or pool[0] for c in colors]
