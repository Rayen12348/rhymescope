"""
rhyme_engine.py — THE RHYME SCHEME DETECTOR
============================================

Built from how rhyme works in hip-hop, not from how it works in a poetry
textbook — the two are different things, and most of what a rap listener
hears as rhyme a textbook would not call rhyme at all.

WHAT RAP RHYME IS
-----------------
1. It is carried by VOWELS. Poetry asks for the whole rime to match — the
   vowel and every consonant after it. Rap asks for the vowel and treats
   consonants as a bonus: "dog / lost / off" is a rhyme on any record,
   and it is not a rhyme in any sonnet. Studies that measure rhyme in rap
   lyrics (Hirjee & Brown's rhyme analyser; Malmi et al.'s rhyme density)
   both find the same thing: match the vowels and you have found the
   rhymes; demand the consonants too and you miss most of them.

2. Its unit is a SEQUENCE. The signature device of the form is the
   multisyllabic rhyme — a run of vowels repeated in the same order
   ("palms are sweaty / arms are heavy"), across word boundaries, weak
   syllables and small words included. A listener hears the whole run as
   one rhyme, and marks it that way: your own corrections to this map
   paint the weak syllables and the "of"s and "are"s inside a run into
   the run's colour, and a third of your groups mix vowels because a run
   is several vowels, not one.

3. It lands on the BEAT. A run is anchored on a stressed syllable — that
   is where the ear finds it — and the end of the bar is the strongest
   landing point there is, strong enough that a bar ending on a full but
   unstressed vowel still takes the beat ("la-DY / FREE").

4. A lone vowel echo is NOT a rhyme. Two stressed syllables sharing a
   vowel somewhere in the middle of two bars happens constantly by
   accident — English only has about fifteen vowels to go round. Your
   corrections are blunt about it: most of what you erased was exactly
   that, a stressed syllable mid-bar lit because its vowel turned up
   nearby. What turns a shared vowel into a rhyme is structure: it lands
   where the bar lands, or it is part of a longer run, or its whole
   ending matches too, or it keeps coming back.

HOW THE DETECTOR WORKS
----------------------
§1 SOUNDS. Every syllable gets the vowel a rapper actually says: the
   cot–caught merger, pin–pen before nasals, the r-coloured vowels as
   sounds of their own (car, for, air, ear, fire, our, her), and weak
   syllables reduced to schwa — which matches other schwas and the
   stressed "uh" of "us" and "love", because that is what it sounds like.

§2 RUNS OF BEATS. Every pair of stressed syllables sharing a vowel is a
   seed. From each seed the match grows backward and forward BEAT BY
   BEAT — stressed syllable to stressed syllable — for as long as the
   stressed vowels keep agreeing at the same rhythm. The weak syllables
   between beats are free: in a real multi they need not match, and
   there need not be the same number of them. (Growing syllable by
   syllable instead stops at the first weak syllable that differs, and
   cuts nearly every three- and four-beat multi down to one beat.) A run
   that ends a few syllables short of both bar-ends carries on to them
   when those endings agree, so a multi that LANDS on a weak ending is
   found whole.

§3 THE DECISION, in bits.

        log-odds(rhyme) = prior(where the runs land) + evidence(the match)

   The evidence is how improbable the match is by chance: a rare vowel is
   worth more than a common one, every extra syllable of run multiplies
   the improbability, and matching consonants add to it. The prior is
   where the runs end: two bar-endings meeting is almost expected; two
   syllables meeting mid-bar is not, and has to be earned by the match.
   One sentence, one currency, so the two can be traded off honestly.

§4 RETURNS. A sound that keeps coming back inside a bar or two — "king
   … ring … swing … thing" — is a device even when no single pair of it
   would convince. The chance of k of n syllables all carrying the same
   ending by accident is priced exactly, and only an improbable pile-up
   counts.

§5 GROUPS. Runs that were matched to each other form one rhyme group and
   wear one colour, first beat to last, weak syllables and all. A group
   is named by its PATTERN — its stressed vowels in order — so a multi
   has a colour of its own, and the same pattern anywhere in the song
   wears the same one.

§6 VALIDATION (eval.py). Synthetic schemes with known answers, text that
   must stay dark, a corpus of real rap pairs and near-misses, a
   hand-annotated answer key scored for PRECISION as well as recall, and
   your own saved corrections — what you erased must stay erased, what
   you painted must be found.
"""

from __future__ import annotations

import functools
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from lexicon import (ARPABET_VOWELS, CONS_CLASS, STOPWORDS, VOWEL_CLASS,
                     WORD_RE, _clean, phones_for, split_spans, syllabify)

__all__ = ["analyze", "Analysis", "Word", "Syllable", "Family",
           "slice_analysis", "family_sound", "phones_for", "syllabify",
           "split_spans", "STOPWORDS", "WORD_RE", "ARPABET_VOWELS",
           "CONS_CLASS", "VOWEL_CLASS", "MAX_MULTI"]

# the longest run a single rhyme may grow to, in syllables
MAX_MULTI = 8


# ==========================================================================
# data model
# ==========================================================================

@dataclass
class Syllable:
    vowel: str                 # dictionary vowel, e.g. "EY"
    stress: int                # 0, 1, 2 — after flow (see §1)
    coda: tuple                # raw consonants after the nucleus
    word_index: int
    index_in_word: int


@dataclass
class Word:
    text: str                  # as it appears in the lyrics
    line: int
    start: int                 # char offset within the line
    end: int
    syllables: list = field(default_factory=list)   # list[Syllable]
    spans: list = field(default_factory=list)       # [(s,e), ...] per syllable
    oov: bool = False          # not in CMU dict (used heuristic phonemes)


@dataclass
class Family:
    fid: int
    label: str                 # human-readable rhyme sound
    members: list              # list of stream indices
    span: int = 0              # distinct lines touched
    gid: str = ""              # stable identity across runs (overrides.py)
    fixed_color: str = ""      # a colour chosen by hand; palette obeys it


@dataclass
class Analysis:
    lines: list                    # original text lines
    words: list                    # list[Word]
    stream: list                   # list[Syllable] in text order
    family_of: dict                # stream index -> family id
    families: list                 # list[Family]
    stats: dict


# ==========================================================================
# §1  SOUNDS — the vowel a rapper actually says
# ==========================================================================

SCHWA = "Ə"

# Before /r/ a vowel is coloured into a sound of its own, and rap rhymes
# on those sounds as units: car/star/hard, for/more/door/yours, air/there/
# care, here/clear/near, fire/tire, hour/our. NURSE (her/word/bird) is ER.
_R_COLOUR = {
    "AA": "AR", "AO": "OR", "OW": "OR", "UH": "OR", "UW": "OR",
    "EH": "AIR", "AE": "AIR", "EY": "AIR", "IH": "EAR", "IY": "EAR",
    "AY": "IRE", "AW": "OUR", "AH": "ER",
}
_NASAL = ("M", "N", "NG")

# Sounds the matcher treats as the SAME sound (vmatch) must also share a
# colour, or a rhyme it found gets painted in two: "heart" (AR) and a
# non-rhotic "Starter" matched through AR~AA ended up in different colours.
_COLOUR_OF = {"AR": "AA", "EAR": "AIR"}
_REDUCIBLE = ("AH", "IH", "ER", "UH")

# How often each sound carries a stressed syllable in English. Only the
# logarithm is used, and only the shape matters: IH and AA are
# everywhere, OY almost nowhere, so a shared OY is news and a shared IH is
# not. A detector that does not know this cannot tell a rhyme from the
# language simply repeating itself.
_V_FREQ = {
    "IH": .100, "AA": .095, "EH": .085, "AE": .085, "IY": .085, "AH": .080,
    "EY": .070, "AY": .075, "OW": .060, "UW": .050, "ER": .040, "OR": .040,
    "AR": .030, "AW": .020, "UH": .020, "AIR": .020, "EAR": .015,
    "IRE": .006, "OUR": .004, "OY": .005,
}
_V_TOTAL = sum(_V_FREQ.values())


def _stress_shape(phones: str) -> tuple:
    """Which syllables of a pronunciation carry stress: (0, 1, 0) etc."""
    return tuple(1 if st else 0 for _v, st, _c in syllabify(phones))


def _dict_variants(word: str) -> list:
    """Every pronunciation the CMU dictionary lists for `word`."""
    import pronouncing
    return list(pronouncing.phones_for_word(_clean(word)))


@functools.lru_cache(maxsize=None)
def vbits(v: str) -> float:
    """How surprising it is that two stressed syllables share vowel `v`."""
    if v == SCHWA:
        return 0.0
    return -math.log2(_V_FREQ.get(v, 0.01) / _V_TOTAL)


def _coda_classes(coda) -> tuple:
    """The consonants after the vowel, as they are heard: voicing folded,
    final T/D dropped out of a cluster (mind → mine, world → girl), and
    the R of an r-coloured vowel gone into the vowel."""
    c = [p for p in coda]
    if c and c[0] == "R":
        c = c[1:]
    if len(c) >= 2 and c[-1] in ("T", "D"):
        c = c[:-1]
    elif len(c) >= 3 and c[-1] in ("S", "Z") and c[-2] in ("T", "D"):
        c = c[:-2] + c[-1:]
    return tuple(x for x in (CONS_CLASS.get(p, p) for p in c) if x)


def sound_of(vowel: str, stress: int, coda, weak_word: bool,
             strict: bool = False) -> str:
    """The vowel class a syllable is matched on."""
    if stress == 0 and vowel in _REDUCIBLE:
        return SCHWA
    if strict:
        return vowel
    if coda and coda[0] == "R":
        return _R_COLOUR.get(vowel, vowel)
    v = "AA" if vowel == "AO" else vowel
    if v == "EH" and coda and coda[0] in _NASAL:
        v = "IH"                         # pin–pen: again/in, them/him
    return v


# Consonants: (place, manner). Same manner at a neighbouring place blurs
# in flow (unreleased P/T/K, M/N/NG, F/S); nothing else does.
_C_FEAT = {
    "P": (0, "stop"), "T": (1, "stop"), "K": (3, "stop"),
    "F": (0, "fric"), "S": (1, "fric"), "C": (2, "fric"), "J": (2, "affr"),
    "N": (1, "nasal"), "L": (1, "liq"), "R": (1, "liq"),
}


def _csim(x, y) -> float:
    if x == y:
        return 1.0
    fx, fy = _C_FEAT.get(x), _C_FEAT.get(y)
    if not fx or not fy:
        return 0.0
    if fx[1] == fy[1] and abs(fx[0] - fy[0]) <= 1:
        return 0.7
    return 0.0


def coda_sim(a: tuple, b: tuple) -> float:
    """How well two endings agree, 0..1, compared from the last consonant
    backwards — the end of a cluster is what survives fast delivery."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    tot = wsum = 0.0
    for k in range(max(len(a), len(b))):
        w = 0.6 ** k
        wsum += w
        if k < len(a) and k < len(b):
            tot += w * _csim(a[-1 - k], b[-1 - k])
    return tot / wsum


def _coda_score(ca: tuple, cb: tuple) -> float:
    """0..0.35 coda support — the scale confidence.py's features expect."""
    return round(0.35 * coda_sim(ca, cb), 4)


# How often two different vowels meet in REAL rap rhymes against chance,
# in natural-log odds — Hirjee & Brown's matrix, learned from line-final
# rhymes across 40 rap albums ("Automatic detection of internal and
# imperfect rhymes in rap lyrics", ISMIR 2009, table 1). Most pairs are
# -3 to -7: they meet less often than chance. A handful sit near zero or
# above — pen/pin, caught/could, cut/curt, kit/keet — and those are the
# bends rap rhymes through. Only those pairs are listed.
_HB_NEAR = {
    frozenset(p): s for p, s in {
        ("AO", "UH"): 1.1, ("EH", "IH"): 0.2, ("IH", "OY"): 0.2,
        ("OY", "UH"): 0.1, ("AH", "ER"): -0.2, ("AH", "IH"): -0.3,
        ("AO", "OW"): -0.3, ("AW", "AY"): -0.3, ("AO", "OY"): -0.4,
        ("UH", "UW"): -0.5, ("AH", "EH"): -0.6, ("AH", "OY"): -0.6,
        ("AA", "ER"): -0.6, ("AA", "AH"): -0.8, ("AH", "UH"): -0.9,
        ("IH", "IY"): -0.9, ("AA", "OW"): -1.0, ("AH", "OW"): -1.0,
        ("AO", "AW"): -1.0,
    }.items()}
_NEAR_MIN = -1.0          # the weakest bend that still counts as near


def near_vowels(a: str, b: str) -> bool:
    """Two different dictionary vowels that rap rhymes through (above)."""
    return a != b and _HB_NEAR.get(frozenset((a, b)), -9.0) >= _NEAR_MIN


def vmatch(a: str, b: str, strict: bool = False) -> float:
    """Do two syllables share a vowel sound? 1 = the same vowel; 0.5–0.6 =
    weak syllables that sound alike (schwa against schwa, or against the
    stressed "uh" of us/love and the "er" of her); 0 = different sounds."""
    if a == b:
        return 0.5 if a == SCHWA else 1.0
    if strict:
        return 0.0
    if SCHWA in (a, b) and (a in ("AH", "ER") or b in ("AH", "ER")):
        return 0.6
    if {a, b} == {"EAR", "AIR"}:
        # near and square fall together before /r/ in most delivery:
        # here / there, fear / care, criteria / malaria
        return 1.0
    if {a, b} == {"AR", "AA"}:
        # Non-rhotic delivery drops the R of "arms" and leaves "ahms" —
        # which is exactly why palms / arms / mom's / calm is one rhyme.
        # AR is by far the r-vowel rap bends this way most, and the bend
        # is so habitual it counts as the same sound.
        return 1.0
    return 0.0


# ==========================================================================
# §3  the priors — where a rhyme is expected to land (bits of log-odds)
# ==========================================================================

# Both runs end their bars. Rap is built on this: two neighbouring bars
# meeting on the same vowel is the scheme far more often than it is an
# accident, and it gets less likely with every bar between them.
_P_END = {1: 0.5, 2: -0.5, 3: -2.0, 4: -3.0}
_P_PHRASE = -2.5          # both runs end a phrase (comma, bar end)
_P_CROSS = -5.5           # one run ends a bar, the other is mid-bar
_P_INTERNAL = -5.0        # both mid-bar, different bars
_P_SAME_BAR = -7.5        # both inside one bar
_P_FAR = -0.5             # every bar further apart beyond the first
# …and mid-bar rhyme fades much faster with distance than a bar ending
# does: the ear holds the END of the last few bars, not their middles.
# What you erased mid-bar sat almost two bars from its partner on average;
# what you kept sat about one.
_P_FAR_MID = -1.5
_PARALLEL = 1.5           # landing at the same spot in two bars
# A run of ONE shared vowel away from the end of a bar is the lone echo —
# the single thing your corrections erase most. It has to be a perfect
# rhyme (the ending matches too) to count; a longer run does not pay this.
_LONE_ECHO = -2.5
# every mid-bar class together, shifted as one — the single knob between
# "only what is certain" and "everything plausible" for rhyme inside bars
_MID_SHIFT = 0.0

# evidence
_CODA_END = 2.0           # whole ending matches where the run ends
_CODA_MISS = 0.8          # …and when it plainly does not
_PER_SYLLABLE = 1.0       # every further real syllable a run carries
_WEAK_HALF = 0.6          # a pair where only one side is stressed
_WEAK_BOTH = 0.35         # a pair where neither is
_GLUE_BITS = 0.4          # a weak pair (schwa/schwa) inside a run
_MAX_GLUE = 2             # glue pairs in a row before a run is cut
_LANDING_BITS = 1.0       # weak vowels agreeing where both runs land
_MAX_BEATS = 6            # the longest multi, in stressed beats
# Local alignment. Scores are per aligned syllable pair; a
# rhyme must reach _SW_MIN, which takes at least one stressed pair that
# agrees. Skipping a weak syllable is cheap (a rapper swallows them);
# skipping a stressed beat costs more but no longer ends the multi.
_SW_BEAT_HIT = 3.0        # stressed ~ stressed, same vowel
_SW_BEAT_MISS = -3.0      # stressed ~ stressed, different vowel
_SW_HALF_HIT = 1.0        # stressed ~ weak, same vowel
_SW_HALF_NEAR = 0.3       # stressed ~ weak schwa-like
_SW_HALF_MISS = -1.5
_SW_WEAK_HIT = 0.8        # weak ~ weak, same full vowel
_SW_WEAK_NEAR = 0.4       # weak ~ weak, schwa-like
_SW_WEAK_MISS = -0.8      # one mismatched weak vowel: lower, not ended
_SW_SAME = 0.2            # the same small word on both sides (glue)
_SW_GAP_WEAK = -0.6
_SW_GAP_BEAT = -2.0
_SW_MIN = 2.5
_SW_BEAT_BEND = 2.0       # stressed ~ stressed through a merger rule
_SW_PER_BARS = 6          # most rhymes taken from one pair of bars
_SW_BEAT_NEAR = 0.0       # stressed ~ stressed, a bent vowel (_HB_NEAR)
_NEAR_BITS = 0.5          # evidence from a bent beat inside a multi
_MAX_RHYTHM_GAP = 1       # weak syllables' difference between two beats
_MAX_WEAK_RUN = 3         # …and the most weak syllables between them
_RHYTHM_COST = 1.0        # per syllable the two rhythms differ by
_RHYTHM_CLASH = -3.5      # a bar ending on a tail against one without
_TAIL_BITS = 1.0          # per weak tail syllable both rhyme words share
_TAIL_BARS = 4            # tails of rhyming beats this close rhyme too…
_TAIL_SHIFT = 0.0         # …at their beats' log-odds, shifted by this
# Two beats of the commonest vowels lining up happens about once a bar by
# accident: every bar has ~7 stressed beats and is compared with the bars
# around it. So a two-beat run in the middle of a bar, not at a matching
# spot, has to be better than that to count; three beats rarely happen by
# chance and pay nothing.
_SHORT_MULTI_MID = -1.5

# returns (§4)
_P_RETURN = -6.0          # a whole ending returning in a short span
_P_CHAIN = -9.0           # a bare vowel doing it — far easier by chance,
                          # so it has to pile up far higher to count
_RETURN_LINES = 1         # …within this many bars either side

# `loose` is the candidate pass confidence.py scores: every bar relaxed by
# this much, so borderline rhymes can be shown or hidden by the density
# control instead of decided here.
_LOOSE_SLACK = 3.0

# the ear: vowels sung within _EAR_DELTA (log-prob) of the
# clearest one count as heard, when the clearest is at least _EAR_FLOOR
_OPEN_ENDS = ("OW", "AY", "UW", "AW", "OY", "EY")
# Fillers between the words — "Oh, oh, okay", "uh", "hmm", "yeah" — are
# breath and attitude, not rhyme: every one ever corrected was erased.
# Sung vocables ("ooh", "woo") are left in; a hook can rhyme on those.
_FILLERS = frozenset({"oh", "okay", "ok", "uh", "um", "hmm", "mm", "yeah",
                      "yea", "yo", "ay", "ayy", "huh", "ah", "hey"})
_EAR_DELTA = 0.5
_EAR_FLOOR = -2.0
_EAR_MARGIN = 1.0
_EAR_MIN_FIT = -1.0       # mean log-prob per phoneme a bar must align at
_EAR_COLOUR = 0.0         # (bits) handicap of an ear-only rhyme when a
                          # syllable's colour is chosen; 0 = strongest wins
                          # (4 and 20 measured: fewer bends kept, no gain)


def _log2_binom_tail(n: int, k: int, p: float) -> float:
    """−log2 P(X ≥ k) for X ~ Binomial(n, p)."""
    if k <= 0:
        return 0.0
    tail = 0.0
    for j in range(k, n + 1):
        tail += math.comb(n, j) * p ** j * (1 - p) ** (n - j)
    return -math.log2(max(tail, 1e-300))


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


# ==========================================================================
# analyze
# ==========================================================================

def analyze(text: str, strict: bool = False, max_line_gap: int = 4,
            min_carry: int = 2, loose: bool = False,
            heard: dict | None = None,
            align_cache: dict | None = None) -> Analysis:
    """Find the rhymes in `text`.

    `strict` switches off every delivery merger (dictionary vowels only).
    `loose` relaxes every decision by _LOOSE_SLACK bits — the candidate
    pass the density control chooses from; nothing renders it directly.
    `max_line_gap` is how many bars apart two rhymes may land.
    `min_carry` is the fewest bars a group may touch (a group inside one
    bar must instead carry three different words).
    `align_cache` — pass the same empty dict to the normal and the loose
    pass over the same text, strict and heard: the bar alignments do not
    depend on `loose`, so the second pass reuses the first one's."""
    lines = text.splitlines()
    words: list[Word] = []
    stream: list[Syllable] = []
    # ---- pronunciation: the variant that rhymes -------------------------
    # The dictionary often knows several ways to say a word ("bow" as in
    # take a bow / bow and arrow, "live", "read", "route", "either") and
    # the first one listed is only a default. A rapper picks the one that
    # lands the scheme, so every word with variants takes the one whose
    # stressed vowels are heard most in the bars around it — weighing the
    # ends of bars double, because that is where rhyme is expected — and
    # keeps the dictionary default unless another variant clearly fits
    # better.
    # An ad-lib in brackets closing a bar — "(brrt)", "(hold up)" — is the
    # hype man, not the bar: the rhyme lands on the last word BEFORE it.
    adlib_at = {}
    for ln, line in enumerate(lines):
        m = re.search(r"(?:\s*\([^()]*\))+\s*$", line)
        if m and WORD_RE.search(line[:m.start()]):
            adlib_at[ln] = m.start()
    toks = []                                   # (line, match, [variants])
    for ln, line in enumerate(lines):
        for m in WORD_RE.finditer(line):
            phones, oov = phones_for(m.group(0))
            cands = [phones] if phones else []
            # a small function word keeps its default: "a", "the", "to" have
            # variants for emphasis, not for rhyme, and letting them flip
            # to whatever rhymes nearby just manufactures echoes
            if phones and not oov and _clean(m.group(0)) not in STOPWORDS:
                shape = _stress_shape(phones)
                for alt in _dict_variants(m.group(0)):
                    # the same word said differently — same syllables,
                    # stress on the same ones — not a different word ("us"
                    # is also listed as the acronym U.S.) or a different
                    # rhythm (a variant that stresses a weak syllable)
                    if alt not in cands and _stress_shape(alt) == shape:
                        cands.append(alt)
            toks.append((ln, m, cands, oov))

    def _heard(ph):
        """The stressed sounds of one pronunciation."""
        return [sound_of(v, st, coda, False, strict)
                for v, st, coda in syllabify(ph) if st]
    around = defaultdict(Counter)               # line -> sound -> weight
    last_in_line = {}
    for k, (ln, m, cands, oov) in enumerate(toks):
        if m.end() <= adlib_at.get(ln, len(lines[ln])):
            last_in_line[ln] = k
    for k, (ln, m, cands, oov) in enumerate(toks):
        if cands:
            wgt = 2.0 if last_in_line.get(ln) == k else 1.0
            for snd_ in _heard(cands[0]):
                around[ln][snd_] += wgt

    for k, (ln, m, cands, oov) in enumerate(toks):
        w = Word(text=m.group(0), line=ln, start=m.start(), end=m.end())
        w.oov = oov
        phones = cands[0] if cands else None
        if len(cands) > 1 and not strict:
            mine = Counter(_heard(cands[0]))
            wgt = 2.0 if last_in_line.get(ln) == k else 1.0

            def fit(ph):
                return sum(around[l2][x] - (wgt * mine[x] if l2 == ln else 0)
                           for x in _heard(ph)
                           for l2 in range(ln - 2, ln + 3))
            best = max(cands, key=fit)
            if fit(best) >= fit(cands[0]) + 1.0:
                phones = best
        w.pron = phones or ""
        if phones:
            for kk, (v, st, coda) in enumerate(syllabify(phones)):
                s_ = Syllable(v, st, coda, len(words), kk)
                w.syllables.append(s_)
                stream.append(s_)
            w.spans = split_spans(w.text, len(w.syllables))
        words.append(w)
    n = len(stream)
    empty = {"lines": len(lines), "words": len(words), "syllables": n,
             "rhymed_syllables": 0, "families": 0, "density": 0.0}
    if not n:
        return Analysis(lines, words, stream, {}, [], empty)

    wtext = [_clean(words[s.word_index].text) for s in stream]
    line_of = [words[s.word_index].line for s in stream]
    word_of = [s.word_index for s in stream]

    # ---- bars, phrases and sections ------------------------------------
    PUNCT = set(",.;:!?…—")
    bar_end: set = set()          # last syllable of a line
    phrase_end: set = set()       # last syllable before punctuation
    first_of_line: dict = {}
    for i in range(n):
        first_of_line.setdefault(line_of[i], i)
        w = words[word_of[i]]
        last = i == n - 1 or line_of[i + 1] != line_of[i]
        cut = adlib_at.get(w.line)
        if cut is not None:
            if w.end > cut:
                continue
            last = last or (word_of[i + 1] != word_of[i]
                            and words[word_of[i + 1]].end > cut)
        if last:
            bar_end.add(i)
            phrase_end.add(i)
        elif word_of[i + 1] != word_of[i]:
            w, nx = words[word_of[i]], words[word_of[i + 1]]
            if any(ch in PUNCT for ch in lines[w.line][w.end:nx.start]):
                phrase_end.add(i)
    last_of_line = {line_of[i]: i for i in sorted(bar_end)}

    # a blank line or a [header] line ends a section; rhymes do not reach
    # across it (the next verse is a different passage, often a different
    # rapper)
    section = []
    sec = 0
    for ln, line in enumerate(lines):
        s = line.strip()
        if not s or (s.startswith("[") and s.endswith("]")):
            sec += 1
        section.append(sec)
    header = {ln for ln, line in enumerate(lines)
              if line.strip().startswith("[") and line.strip().endswith("]")}

    # ---- §1 flow: which syllables carry the beat -----------------------
    # A one-syllable word's vowel is the word's own stressed vowel — the
    # dictionary marks "in" or "hers" unstressed only because it predicts
    # where they usually fall in a sentence. But a small function word
    # INSIDE a bar is not stressed in flow ("back at 'em" is said
    # back-uh-tem), and only takes the beat where the bar or phrase ends
    # on it ("…all about US").
    pos_of_syl = {id(s): i for i, s in enumerate(stream)}
    for w in words:
        if len(w.syllables) != 1:
            continue
        s = w.syllables[0]
        i = pos_of_syl[id(s)]
        func = _clean(w.text) in STOPWORDS
        if func and i not in phrase_end:
            s.stress = 0
        elif s.stress == 0 and not (func and s.vowel in _REDUCIBLE
                                    and not s.coda):
            s.stress = 1

    snd = []
    for i, s in enumerate(stream):
        weak_word = wtext[i] in STOPWORDS
        snd.append(sound_of(s.vowel, s.stress, s.coda, weak_word, strict))
        s.cc = _coda_classes(s.coda)
        s.bv = snd[-1]
        s.forms = ((snd[-1], s.cc),)

    # ---- what the VOICE said --------------------------------------------
    # `heard` (audio.heard_vowels) scores, for each syllable it placed in
    # the vocal, how strongly each vowel was sung. The vowels sung nearly
    # as clearly as the best one form the syllable's EAR set: a rapper who
    # bends "film" toward "tell" gives it {EH}; a clean "film" gives {IH}.
    # The ear only ever ADDS a way to match — the dictionary vowel still
    # counts — so a muddy recording cannot break a rhyme the lyrics show.
    ear: list = [None] * n
    if heard and not strict:
        syl_heard = heard.get("syllables", heard)
        line_fit = heard.get("line_fit", {})
        for i, s in enumerate(stream):
            if snd[i] == SCHWA:
                continue
            w = words[s.word_index]
            # only where the lyrics lock onto the vocal: a bar the aligner
            # could barely place (a muddy mix, a line sung differently) is
            # heard wrong more often than it is heard bent
            if line_fit.get(str(w.line), 0.0) < _EAR_MIN_FIT:
                continue
            sc = syl_heard.get(f"{w.line}:{w.start}:{s.index_in_word}")
            if not sc:
                continue
            best = max(sc.values())
            if best < _EAR_FLOOR:
                continue
            # a vowel joins only when it was sung BETTER than the
            # dictionary's, by a margin: a clean "film" with EH a close
            # second is the model hesitating, not the rapper bending
            own = max((v for c, v in sc.items()
                       if vmatch(c, snd[i], strict) >= 1.0), default=-99.0)
            # a word the dictionary does not know was pronounced by letter
            # rules — a guess — so there the voice is taken at its word
            margin = 0.0 if getattr(w, "oov", False) else _EAR_MARGIN
            cls = set()
            for c, v in sc.items():
                if (v >= best - _EAR_DELTA and v >= own + margin
                        and c != SCHWA):
                    if c == "EH" and s.coda and s.coda[0] in _NASAL:
                        c = "IH"                 # pin-pen, as sound_of
                    cls.add(c)
            if cls - {snd[i]}:
                ear[i] = cls

    # Prominence: where the beat can land. Lexical stress, or the end of a
    # bar/phrase on a full vowel — "la-DY" takes the beat at the end of a
    # bar, "mo-ther" never does, because its last vowel is a schwa.
    # A content word ending on a weak but FULL open vowel can take the beat
    # anywhere in the bar — "her-O", "Mi-LO", "yel-LOW" rhyme on that -o
    # (not -y: "money", "happy" — far too common to mean anything).
    def open_end(i, s):
        return (i + 1 == n or word_of[i + 1] != word_of[i]) and \
            len(words[word_of[i]].syllables) > 1 and \
            snd[i] in _OPEN_ENDS and wtext[i] not in STOPWORDS
    prom = [bool(s.stress) or (i in phrase_end and snd[i] != SCHWA)
            or open_end(i, s)
            for i, s in enumerate(stream)]
    content = [wtext[i] not in STOPWORDS for i in range(n)]
    # Words in brackets — "(uh, uh)", "(clip on, clip on)", "(I gotta do
    # this)" — are the hype man and the backing vocal, not the bar: of
    # the 47 corrections ever made inside brackets, all 47 were erasures.
    def bracketed(w):
        before = lines[w.line][:w.start]
        return before.count("(") > before.count(")")
    in_brackets = [bracketed(w) for w in words]
    # A line chanted again straight after itself — "Who the fuck is you?"
    # four times over — is a hook being repeated, not bars rhyming: every
    # syllable ever corrected on such a stack was erased (48 of 48). A
    # chorus that comes BACK later is different, and stays in.
    said = [" ".join(re.findall(r"[a-z']+", re.sub(r"\([^()]*\)", "",
                                                  line.lower())))
            for line in lines]
    stacked = {ln for ln, t in enumerate(said) if t and (
        (ln > 0 and said[ln - 1] == t)
        or (ln + 1 < len(said) and said[ln + 1] == t))}
    usable = [line_of[i] not in header and line_of[i] not in stacked
              and not in_brackets[word_of[i]]
              and wtext[i] not in _FILLERS for i in range(n)]

    slack = _LOOSE_SLACK if loose else 0.0

    def dist_to_end(i):
        return last_of_line[line_of[i]] - i

    # ---- §2 runs --------------------------------------------------------
    def vm_at(p, q):
        """vmatch, plus the one merger that depends on what follows the
        vowel: before a nasal the back vowels raise and nasalise together,
        which is why gone meets home and on meets own."""
        vm = vmatch(snd[p], snd[q], strict)
        if not vm and not strict and {snd[p], snd[q]} == {"AA", "OW"} \
                and stream[p].cc[:1] == ("N",) and stream[q].cc[:1] == ("N",):
            return 1.0
        if vm < 1.0 and (ear[p] or ear[q]) and by_ear(p, q):
            return 1.0
        return vm

    def by_ear(p, q):
        """The sound two syllables share once the voice is heard, or None:
        a vowel in one's ear set that matches the other's dictionary vowel
        or ear set."""
        A = (ear[p] or set()) | {snd[p]}
        B = (ear[q] or set()) | {snd[q]}
        for a in sorted(A):
            for b in sorted(B):
                if vmatch(a, b, strict) >= 1.0:
                    return a
        return None

    def ear_only(x, y):
        """Matched only because of what the voice sang?"""
        if not (ear[x] or ear[y]) or SCHWA in (snd[x], snd[y]):
            return False
        return vmatch(snd[x], snd[y], strict) < 1.0 and not (
            {snd[x], snd[y]} == {"AA", "OW"}
            and stream[x].cc[:1] == ("N",) and stream[y].cc[:1] == ("N",))

    def shared_sound(x, y):
        """The sound a matched pair rhymes on — what its colour is."""
        if snd[x] == SCHWA:
            return snd[y]
        if snd[y] == SCHWA or vmatch(snd[x], snd[y], strict) >= 1.0:
            return snd[x]
        return by_ear(x, y) or snd[x]

    def pair_kind(p, q):
        """real: a genuine shared vowel with the beat on at least one side
        · soft: a shared full vowel on two weak syllables · glue: schwa
        against schwa, or the same word twice · None: different sounds.

        One weak pair counts as real: where BOTH runs land on the end of
        a bar or phrase, on the same ending, with the beat on at least one
        side. That is the "…infamous / …us / …murderers" rhyme — weak
        vowels, but the same [əs] landing on the same beat — and it is
        different in kind from "running / sitting", where two unstressed
        suffixes share an ending and nothing takes the beat at all."""
        if wtext[p] == wtext[q] and word_of[p] != word_of[q]:
            return "glue", 0.0
        vm = vm_at(p, q)
        if vm <= 0:
            # A reduced vowel is indistinct, so at the end of a bar the
            # ENDING is what it sounds like: "-ange" of or-ange against
            # "hinge" is the same /ndʒ/ landing. Only an identical,
            # non-empty ending qualifies, and only with the beat on one side.
            if ((snd[p] == SCHWA) != (snd[q] == SCHWA)
                    and p in phrase_end and q in phrase_end
                    and stream[p].cc and stream[p].cc == stream[q].cc
                    and (prom[p] or prom[q])):
                return "real", 0.5
            return None, 0.0
        if vm >= 1.0:
            return ("real" if (prom[p] or prom[q]) else "soft"), 1.0
        if (p in phrase_end and q in phrase_end and stream[p].cc
                and (prom[p] or prom[q])
                and coda_sim(stream[p].cc, stream[q].cc) >= 0.7):
            return "real", vm
        return "glue", vm

    # ---- beats: the stressed syllables a multi is made of ---------------
    # A multisyllabic rhyme is a run of STRESSED vowels repeated in the
    # same order. The weak syllables between them are free — they need
    # not match, and there need not even be the same number of them —
    # which is exactly why growing a match syllable by syllable found
    # almost none: it stopped at the first weak syllable that differed,
    # and cut every three- and four-beat multi down to one beat.
    is_beat = [prom[i] and snd[i] != SCHWA and usable[i] for i in range(n)]
    beats_of = defaultdict(list)
    for i in range(n):
        if is_beat[i]:
            beats_of[line_of[i]].append(i)
    bidx = {}
    for ln, bs in beats_of.items():
        for k, p in enumerate(bs):
            bidx[p] = k
    last_beat_bar = {bs[-1] for bs in beats_of.values()}
    last_beat_phrase = set()
    for ln, bs in beats_of.items():
        ends = sorted(e for e in phrase_end if line_of[e] == ln)
        lo_b = -1
        for e in ends:
            inside = [p for p in bs if lo_b < p <= e]
            if inside:
                last_beat_phrase.add(inside[-1])
            lo_b = e

    def tail(p):
        """The weak syllables after beat p inside its own word — the
        feminine tail of a rhyme word (HEAV-y, KILL-ers)."""
        out = []
        q = p + 1
        while q < n and word_of[q] == word_of[p] and not is_beat[q]:
            out.append(q)
            q += 1
        return out

    def near_at(x, y):
        """A bent vowel: different sounds rap rhymes through (_HB_NEAR)."""
        return not strict and near_vowels(stream[x].vowel, stream[y].vowel)

    def beat_kind(x, y):
        if wtext[x] == wtext[y] and word_of[x] != word_of[y]:
            return "same"             # a repeated word: holds a run, is
        if vm_at(x, y) >= 1.0:        # never the reason for one
            return "real"
        # a bent vowel inside a multi keeps it going — "battery / battle
        # me" — but is never what a rhyme is anchored on
        if near_at(x, y):
            return "near"
        return None

    syls_of_line = defaultdict(list)
    for i in range(n):
        if usable[i]:
            syls_of_line[line_of[i]].append(i)

    # Without an ear set, a syllable's alignment score depends only on its
    # sound, whether its coda opens on N (the ah~oh merger) and whether it
    # takes the beat — a handful of kinds, so each pairing of kinds is
    # scored once per song instead of once per cell.
    sw_kind = [None if ear[i] else (snd[i], stream[i].vowel,
                                    stream[i].cc[:1] == ("N",),
                                    is_beat[i]) for i in range(n)]
    sw_memo: dict = {}

    def sw_score(x, y) -> float:
        if wtext[x] == wtext[y] and word_of[x] != word_of[y]:
            return _SW_SAME
        kx, ky = sw_kind[x], sw_kind[y]
        if kx is None or ky is None:
            return sw_score_of(x, y)
        k = (kx, ky)
        s = sw_memo.get(k)
        if s is None:
            s = sw_memo[k] = sw_score_of(x, y)
        return s

    def sw_score_of(x, y) -> float:
        vm = vm_at(x, y)
        bx, by = is_beat[x], is_beat[y]
        if bx and by:
            if vm < 1.0:
                return _SW_BEAT_NEAR if near_at(x, y) else _SW_BEAT_MISS
            # a match that exists only through a merger rule (ah ~ oh
            # before n, ar ~ ah, ear ~ air) is a bend, not the same vowel,
            # so it should not outbid a true identical-vowel pairing
            # …but a vowel the voice SANG the same is the same vowel
            return (_SW_BEAT_HIT if snd[x] == snd[y] or ear_only(x, y)
                    else _SW_BEAT_BEND)
        if bx or by:
            if vm >= 1.0:
                return _SW_HALF_HIT
            return _SW_HALF_NEAR if vm > 0 else _SW_HALF_MISS
        if vm >= 1.0 and snd[x] != SCHWA:
            return _SW_WEAK_HIT
        return _SW_WEAK_NEAR if vm > 0 else _SW_WEAK_MISS

    def sw_gap(x) -> float:
        return _SW_GAP_BEAT if is_beat[x] else _SW_GAP_WEAK

    def split_far_beats(aligned):
        """One multi, or several rhymes? A multi's stressed beats sit close
        together. When two consecutive matched beats are more than
        _MAX_WEAK_RUN syllables apart on either side, what lies between is
        unrelated material and the alignment is two separate rhymes (an
        internal one and the bar's end rhyme, say) — so it is cut there and
        each piece is judged on its own."""
        segs, cur, last = [], [], None
        for x, y in aligned:
            if is_beat[x] and is_beat[y] and beat_kind(x, y):
                if last is not None and (x - last[0] - 1 > _MAX_WEAK_RUN
                                         or y - last[1] - 1 > _MAX_WEAK_RUN):
                    segs.append(cur)
                    cur = []
                last = (x, y)
            cur.append((x, y))
        if cur:
            segs.append(cur)
        return segs

    def align_bars(la, lb):
        """Smith–Waterman local alignment of two bars, repeated on what is
        left until nothing reaches _SW_MIN (Waterman–Eggert). Each result
        is the list of aligned syllable pairs (x in bar la, y in bar lb).
        A mismatched weak vowel lowers an alignment instead of ending it,
        and a beat present on only one side is a gap, not a wall."""
        A, B = syls_of_line[la], syls_of_line[lb]
        same = la == lb
        na, nb_ = len(A), len(B)
        # Scores and gaps do not change between passes — only which cells
        # are spent does — so they are worked out once per pair of bars.
        # None marks a cell no alignment may use (spent, or the second
        # half of one bar paired back onto its first).
        S = [[None if same and y <= x else sw_score(x, y) for y in B]
             for x in A]
        gA = [sw_gap(x) for x in A]
        gB = [sw_gap(y) for y in B]
        found = []
        for _ in range(_SW_PER_BARS):
            H = [[0.0] * (nb_ + 1) for _ in range(na + 1)]
            P = [[None] * (nb_ + 1) for _ in range(na + 1)]
            best, bi, bj = 0.0, 0, 0
            for i in range(1, na + 1):
                Hu, Hi, Pi, Si, ga = H[i - 1], H[i], P[i], S[i - 1], gA[i - 1]
                for j in range(1, nb_ + 1):
                    s = Si[j - 1]
                    if s is None:
                        continue
                    # max over (score, move) — on a tie "u" beats "l"
                    # beats "d", exactly as comparing the tuples did
                    h, mv = Hu[j - 1] + s, "d"
                    v = Hi[j - 1] + gB[j - 1]
                    if v >= h:
                        h, mv = v, "l"
                    v = Hu[j] + ga
                    if v >= h:
                        h, mv = v, "u"
                    if h > 0:
                        Hi[j], Pi[j] = h, mv
                        if h > best:
                            best, bi, bj = h, i, j
            if best < _SW_MIN:
                break
            i, j = bi, bj
            cells = []
            while i > 0 and j > 0 and P[i][j] is not None and H[i][j] > 0:
                mv = P[i][j]
                if mv == "d":
                    cells.append((i - 1, j - 1))
                    i, j = i - 1, j - 1
                elif mv == "u":
                    i -= 1
                else:
                    j -= 1
            cells.reverse()
            if not cells:
                break
            # A syllable may rhyme with MORE than one syllable of the other
            # bar ("pose" with both "nose" and "bone"), so what an alignment
            # spends is the matched PAIRS, not the syllables — the standard
            # Waterman–Eggert declumping. Passing over a syllable as a
            # mismatch spends nothing at all.
            hit = [(a, b) for a, b in cells if S[a][b] > 0]
            if not hit:
                break
            for a, b in hit:
                S[a][b] = None
            pairs = [(A[a], B[b]) for a, b in cells]
            if same and max(x for x, _ in pairs) >= min(y for _, y in pairs):
                continue                   # the two halves overlap in one bar
            found.append(pairs)
        return found

    def landing(pairs):
        """A multi often LANDS past its last stressed beat, on the bar's
        weak ending: HEARD of US against MUR-der-ERS ends on [əs] both
        times, and that landing is part of the rhyme. When both runs are
        a few syllables from their bar ends and those endings agree,
        the run carries through to them. Returns (end_a, end_b) or None."""
        xk, yk = pairs[-1]
        ex = max(i for i in bar_end if line_of[i] == line_of[xk])
        ey = max(i for i in bar_end if line_of[i] == line_of[yk])
        if (ex == xk and ey == yk) or ex - xk > 3 or ey - yk > 3:
            return None
        if ex == xk or ey == yk:
            return None                  # only one side has anything left
        if pair_kind(ex, ey)[0] != "real":
            return None
        return ex, ey

    def weak_match(a, b, in_word_a, in_word_b):
        """Do two weak syllables inside a run genuinely sound alike?"""
        if wtext[a] == wtext[b] and word_of[a] != word_of[b]:
            return False              # the same word twice is not a rhyme
        vm = vm_at(a, b)
        if vm >= 1.0 and snd[a] != SCHWA:
            return True               # the same full vowel (happ-Y, heav-Y)
        # Schwa against schwa counts only inside a RHYME WORD — the weak
        # tail of a word whose beat rhymes (KILL-ers / BILL-ers) — and only
        # when the endings agree too. A stray small word that happens to
        # have a schwa in it is not part of anybody's rhyme.
        return (vm > 0 and in_word_a and in_word_b
                and coda_sim(stream[a].cc, stream[b].cc) >= 0.7)

    def run_members(pairs):
        """EXACTLY the syllables each side of a run rhymes with — no more.

        Rhyme lives in syllables, not in words. A rapper will cut a word
        at a syllable and rhyme only that piece of it, or rhyme its first
        half with one thing and its second half with another; colouring
        the whole word (or the whole stretch between two beats) hides
        exactly that. So a run covers its matched beats, the weak
        syllables between them that genuinely sound alike on both sides,
        its matching feminine tail, and its landing — and nothing else."""
        A, B = [], []
        wa = {word_of[x] for x, y in pairs if beat_kind(x, y) == "real"}
        wb = {word_of[y] for x, y in pairs if beat_kind(x, y) == "real"}
        for k, (x, y) in enumerate(pairs):
            if beat_kind(x, y) in ("real", "near"):
                A.append(x)
                B.append(y)
            if k + 1 < len(pairs):
                x2, y2 = pairs[k + 1]
                ga, gb = range(x + 1, x2), range(y + 1, y2)
                if len(ga) == len(gb):        # same rhythm: align them
                    for u, v in zip(ga, gb):
                        if weak_match(u, v, word_of[u] in wa,
                                      word_of[v] in wb):
                            A.append(u)
                            B.append(v)
        xk, yk = pairs[-1]
        land = landing(pairs)
        if land:
            A.append(land[0])
            B.append(land[1])
            # what lies between the last beat and the landing, aligned
            # from the landing backwards
            ra = list(range(xk + 1, land[0]))[::-1]
            rb = list(range(yk + 1, land[1]))[::-1]
            for u, v in zip(ra, rb):
                if not weak_match(u, v, word_of[u] in wa or
                                  word_of[u] == word_of[land[0]],
                                  word_of[v] in wb or
                                  word_of[v] == word_of[land[1]]):
                    break
                A.append(u)
                B.append(v)
        else:
            for u, v in zip(tail(xk), tail(yk)):
                if not weak_match(u, v, True, True):
                    break
                A.append(u)
                B.append(v)
        return tuple(sorted(A)), tuple(sorted(B)), list(zip(A, B))

    def run_evidence(pairs):
        bits, nb = 0.0, 0
        for x, y in pairs:
            k = beat_kind(x, y)
            if k == "real":
                bits += vbits(snd[x])
                nb += 1
            elif k == "near":
                bits += _NEAR_BITS
        bits += _PER_SYLLABLE * max(nb - 1, 0)
        for (x1, y1), (x2, y2) in zip(pairs, pairs[1:]):
            bits -= _RHYTHM_COST * abs((x2 - x1) - (y2 - y1))
        xk, yk = pairs[-1]
        ta, tb = tail(xk), tail(yk)
        land = landing(pairs)
        # A feminine rhyme (REP-u-ta-ble / VEG-e-ta-ble, KILL-ers /
        # BILL-ers) is heard on its whole tail, judged from the end of the
        # word: the consonant straight after the beat is just the next
        # syllable's onset, so p against dʒ there is no evidence against
        # it, while every tail syllable that matches is evidence for it.
        same_tail = 0
        if not land:
            for u, v in zip(ta[::-1], tb[::-1]):
                if not weak_match(u, v, True, True):
                    break
                same_tail += 1
        if same_tail:
            bits += _CODA_END * coda_sim(stream[ta[-1]].cc, stream[tb[-1]].cc)
            bits += _TAIL_BITS * same_tail
        else:
            cs = coda_sim(stream[xk].cc, stream[yk].cc)
            bits += _CODA_END * cs
            if cs == 0 and (stream[xk].cc or stream[yk].cc):
                bits -= _CODA_MISS
            if ta and tb and vm_at(ta[0], tb[0]) > 0:
                bits += _TAIL_BITS
        if land:
            bits += _LANDING_BITS + _CODA_END * coda_sim(
                stream[land[0]].cc, stream[land[1]].cc)
        return bits, nb

    def run_prior(pairs, nb):
        xk, yk = pairs[-1]
        gap = line_of[yk] - line_of[xk]
        at_x, at_y = xk in last_beat_bar, yk in last_beat_bar
        ph_x, ph_y = xk in last_beat_phrase, yk in last_beat_phrase
        landed = landing(pairs) is not None
        if landed:
            at_x = at_y = True           # it lands where both bars end
        if at_x and at_y:
            pr, kind = _P_END.get(gap, _P_END[4] + _P_FAR * (gap - 4)), "end"
            # the bar lands on a different rhythm: one rhyme word carries
            # on past the matched beat ("SIL-ver"), the other stops on it
            # ("fish") — not the same rhyme however well the vowels agree
            if not landed and bool(tail(xk)) != bool(tail(yk)):
                pr += _RHYTHM_CLASH
        elif ph_x and ph_y:
            pr, kind = _P_PHRASE + _P_FAR * max(gap - 1, 0), "end"
        elif gap == 0:
            # inside one bar — even when one side is the bar's own ending,
            # there is no other bar to corroborate it
            pr, kind = _P_SAME_BAR, "leonine"
        elif at_x or at_y or ph_x or ph_y:
            pr, kind = _P_CROSS + _P_FAR_MID * max(gap - 1, 0), "echo"
        else:
            pr, kind = _P_INTERNAL + _P_FAR_MID * max(gap - 1, 0), "internal"
        if kind in ("echo", "internal", "leonine"):
            pr += _MID_SHIFT
            # the same spot in two bars, counted in beats from the end
            parallel = gap >= 1 and (len(beats_of[line_of[xk]]) - bidx[xk]
                                     == len(beats_of[line_of[yk]]) - bidx[yk])
            if parallel:
                pr += _PARALLEL
            elif nb == 2:
                pr += _SHORT_MULTI_MID
            # a lone shared vowel mid-bar is an echo, not a rhyme, unless
            # the whole ending is the same too
            if nb <= 1 and not (stream[xk].cc
                                and stream[xk].cc == stream[yk].cc):
                pr += _LONE_ECHO
        return pr, kind

    # The SOUND each rhyming syllable is coloured by. A syllable takes the
    # sound of the rhyme it is part of: two stressed syllables share their
    # vowel, and a weak syllable paired with a stressed one ("-mous" with
    # "us", "-er" with "her") takes the stressed one's sound — it is heard
    # as that rhyme, so it wears that colour. The strongest rhyme decides.
    sound_key: dict = {}
    sound_lo: dict = {}

    def note_sounds(pairs_, lo_):
        for x, y in pairs_:
            k0 = shared_sound(x, y)
            k = _COLOUR_OF.get(k0, k0)
            # a syllable's colour comes from what the lyrics show when
            # they show a rhyme; the ear decides it only where it alone
            # makes the rhyme ("film" sung to meet "tell")
            lo_c = lo_ - (_EAR_COLOUR if ear_only(x, y) else 0.0)
            for z in (x, y):
                if lo_c > sound_lo.get(z, -99.0):
                    sound_key[z], sound_lo[z] = k, lo_c

    seen: dict = {}
    links: list = []           # (lo, kind, syllables A, syllables B, pattern)
    all_beats = sorted(p for bs in beats_of.values() for p in bs)
    bars = sorted(syls_of_line)
    for jb, lb in enumerate(bars):
        for la in reversed(bars[:jb + 1]):
            if lb - la > max_line_gap or section[la] != section[lb]:
                break
            if align_cache is None:
                segs = [seg for al in align_bars(la, lb)
                        for seg in split_far_beats(al)]
            else:
                segs = align_cache.get((la, lb))
                if segs is None:
                    segs = align_cache[la, lb] = [
                        seg for al in align_bars(la, lb)
                        for seg in split_far_beats(al)]
            for aligned in segs:
                # the beats of the alignment, in order: what the rhyme is
                # anchored on and scored by
                pairs = [(x, y) for x, y in aligned
                         if is_beat[x] and is_beat[y] and beat_kind(x, y)]
                while pairs and beat_kind(*pairs[0]) != "real":
                    pairs.pop(0)
                while pairs and beat_kind(*pairs[-1]) != "real":
                    pairs.pop()
                if not pairs:
                    continue               # no stressed anchor: not a rhyme
                key = (pairs[0], pairs[-1])
                if key in seen:
                    continue
                bits, nb = run_evidence(pairs)
                pr, kind = run_prior(pairs, nb)
                # the slack decides what the loose pass KEEPS; the log-odds
                # it records stays honest — confidence.py reads it directly
                lo = pr + bits
                seen[key] = lo
                if lo + slack > 0:
                    sa, sb, matched = run_members(pairs)
                    # weak syllables the alignment itself paired up, when
                    # they genuinely sound alike
                    wa = {word_of[x] for x, _y in pairs}
                    wb = {word_of[y] for _x, y in pairs}
                    lo_x, hi_x = pairs[0][0], pairs[-1][0]
                    extra = [(u, v) for u, v in aligned
                             if lo_x < u < hi_x and not is_beat[u]
                             and not is_beat[v]
                             and weak_match(u, v, word_of[u] in wa,
                                            word_of[v] in wb)]
                    sa = tuple(sorted(set(sa) | {u for u, _v in extra}))
                    sb = tuple(sorted(set(sb) | {v for _u, v in extra}))
                    pat = tuple(snd[x] for x, y in pairs
                                if beat_kind(x, y) == "real")
                    links.append((lo, kind, sa, sb, pat))
                    note_sounds(matched + extra, lo)

    # Weak landings: bars that end on the same weak ending with the beat
    # on one side ("…infamous / …us"). Weak vowels, same landing — see
    # pair_kind. They are single syllables by nature.
    ends = sorted(i for i in phrase_end if usable[i] and stream[i].cc)
    for jj, q in enumerate(ends):
        for p in reversed(ends[:jj]):
            gap = line_of[q] - line_of[p]
            if gap > max_line_gap or section[line_of[p]] != section[line_of[q]]:
                break
            if wtext[p] == wtext[q]:
                continue
            kind, vm = pair_kind(p, q)
            if kind != "real" or vm >= 1.0:
                continue
            both_bar = p in bar_end and q in bar_end
            pr = (_P_END.get(gap, _P_END[4] + _P_FAR * (gap - 4))
                  if both_bar else _P_PHRASE + _P_FAR * max(gap - 1, 0))
            lo = pr + _LANDING_BITS + _CODA_END * coda_sim(stream[p].cc,
                                                            stream[q].cc)
            if lo + slack > 0:
                links.append((lo, "end", (p,), (q,), (SCHWA,)))
                note_sounds([(p, q)], lo)

    # ---- §4 returns: a sound piling up in a short span ------------------
    # One pair of "ring"/"thing" mid-bar is not enough to convince; four of
    # them inside one bar is a device. For every ending (vowel + coda) we
    # ask how improbable its pile-up is among the stressed syllables that
    # could have carried it, bar by bar.
    returns: list = []              # (lo, [positions])
    by_line = defaultdict(list)
    for i in all_beats:
        if content[i] or i in bar_end:
            by_line[line_of[i]].append(i)
    rime_p = {}
    for (v, c) in {(snd[i], stream[i].cc) for i in all_beats}:
        rime_p[(v, c)] = (_V_FREQ.get(v, .01) / _V_TOTAL) * \
            (0.35 if not c else 0.12 / len(c))
    done_windows = set()
    # Two windows per bar: the bar with the next one (a sound carried
    # across a couplet) and the bar alone — three of one vowel in a single
    # bar ("the next rhyme I write might…") is a device, and diluting it
    # with the next bar's beats would hide exactly that.
    for ln, span in sorted((ln, sp) for ln in by_line
                           for sp in (0, _RETURN_LINES)):
        lo_ln, hi_ln = ln, ln + span
        win = [i for l2 in range(lo_ln, hi_ln + 1) for i in by_line.get(l2, [])
               if section[line_of[i]] == section[ln]]
        if len(win) < 3:
            continue
        groups = defaultdict(list)
        for i in win:
            groups[(snd[i], stream[i].cc)].append(i)
        for rime, ps in groups.items():
            if len({wtext[p] for p in ps}) < 3:
                continue
            key = (rime, tuple(ps))
            if key in done_windows:
                continue
            done_windows.add(key)
            surprise = _log2_binom_tail(len(win), len(ps), rime_p[rime])
            lo = _P_RETURN + surprise
            if lo + slack > 0:
                returns.append((lo, ps))
        # The vowel CHAIN — one vowel driven through a bar or two, endings
        # free ("…only · older · cold"). No single pair of it would pass
        # as a rhyme, and your corrections are right to erase a lone one;
        # what makes it a device is how improbably it piles up, so that is
        # the whole test, held to a far higher bar than a whole ending.
        by_v = defaultdict(list)
        for i in win:
            by_v[snd[i]].append(i)
        for v, ps in by_v.items():
            if len({wtext[p] for p in ps}) < 3:
                continue
            key = (v, tuple(ps))
            if key in done_windows:
                continue
            done_windows.add(key)
            pv = _V_FREQ.get(v, .01) / _V_TOTAL
            lo = _P_CHAIN + _log2_binom_tail(len(win), len(ps), pv)
            if lo + slack > 0:
                returns.append((lo, ps))

    # ---- §5 groups ------------------------------------------------------
    # A rhyme joins a colour only by matching that colour's SIGNATURE (its
    # stressed vowels in order), never just by sharing a syllable with one
    # member, and a colour closes after rhyme.grouping.EXPIRY bars unused.
    # See rhyme/grouping.py for why the old union-everything merge had to go.
    from rhyme.grouping import assign_families
    items = [(lo, kind, [sa, sb], tuple(pat))
             for lo, kind, sa, sb, pat in links]
    items += [(lo, "assonance", [tuple(ps)], (snd[ps[0]],))
              for lo, ps in returns]
    # strongest rhyme first decides each syllable's log-odds and route;
    # its COLOUR is then the syllable's own sound (rhyme/grouping.py)
    from rhyme.grouping import sound_families
    _fm, owner, _sigs = assign_families(items, line_of)
    # Feminine tails of beats that rhyme. Two stressed syllables can end
    # up in one rhyme through OTHER links ("FUCK-in'" and "THUG-gin'" each
    # rhyme with the bars around them), and then their weak tails — the
    # "-in'" of both — rhyme as well: a listener hears the whole ending
    # repeat. run_members only adds the tail of the two beats a single
    # alignment paired, so these were left out. Same rhyme, bars close,
    # tails that genuinely sound alike: the tails join, as weak members.
    by_fam = defaultdict(list)
    for p, (f, _lo, _k) in owner.items():
        if is_beat[p]:
            by_fam[sound_key.get(p, _COLOUR_OF.get(snd[p], snd[p]))].append(p)
    tails: dict = {}
    for f, ps in by_fam.items():
        ps.sort()
        for i, p in enumerate(ps):
            for q in ps[i + 1:]:
                if line_of[q] - line_of[p] > _TAIL_BARS:
                    break
                if wtext[p] == wtext[q] or vm_at(p, q) < 1.0:
                    continue
                lo_t = min(owner[p][1], owner[q][1]) + _TAIL_SHIFT
                # from the END of the words back: a feminine rhyme is
                # judged where it lands (REP-u-ta-BLE / VEG-e-ta-BLE)
                for u, v in zip(tail(p)[::-1], tail(q)[::-1]):
                    if not weak_match(u, v, True, True):
                        break
                    for z in (u, v):
                        if z not in owner and lo_t > tails.get(z, (0, -99.0))[1]:
                            tails[z] = (owner[p][0], lo_t, "glue")
    owner.update(tails)
    fam_members = sound_families(
        owner, lambda p: sound_key.get(p, _COLOUR_OF.get(snd[p], snd[p])),
        line_of)
    comp = {k: ms for k, ms in enumerate(fam_members)}
    fam_pat = {k: (sound_key.get(ms[0], snd[ms[0]]),)
               for k, ms in enumerate(fam_members)}

    families: list[Family] = []
    fam_roots: list = []
    family_of: dict = {}
    route: dict = {}
    odds: dict = {}
    for root, members in sorted(comp.items(),
                                key=lambda kv: (-len(kv[1]), min(kv[1]))):
        members = sorted(members)
        mlines = {line_of[m] for m in members}
        # (a family of weak tails only — the "-in'" of fuckin'/thuggin' —
        # is judged by its own words)
        mwords = ({wtext[m] for m in members if snd[m] != SCHWA}
                  or {wtext[m] for m in members})
        if len(mwords) < 2:
            continue                      # one word said twice is not a rhyme
        if len(mlines) < (1 if loose else min_carry) and \
                len({wtext[m] for m in members if content[m]}) < 3:
            continue
        fam = Family(fid=len(families), label="", members=members,
                     span=len(mlines))
        families.append(fam)
        fam_roots.append(root)
        for m in members:
            family_of[m] = fam.fid
            _f, lo_m, k = owner[m]
            odds[m] = lo_m
            real = prom[m] and snd[m] != SCHWA
            route[m] = k if real else "glue"

    for fam, root in zip(families, fam_roots):
        pat = fam_pat.get(root) or ()
        fam.label = " · ".join(v for v in pat if v != SCHWA) or "ə"

    rhymed = len(family_of)
    result = Analysis(lines, words, stream, family_of, families, {
        "lines": len(lines), "words": len(words), "syllables": n,
        "rhymed_syllables": rhymed, "families": len(families),
        "density": rhymed / n if n else 0.0})
    result.route = route
    result.odds = odds
    result.links = links
    result.runs = returns
    return result


# ==========================================================================
# slicing and identity — plumbing the renderers and the editor rely on
# ==========================================================================

def slice_analysis(analysis: Analysis, lines: list,
                   colors: list | None = None) -> Analysis:
    """The part of an analysis covering song `lines` (ascending), renumbered
    to start at line 0 — WITHOUT analysing anything again, so a clip shows
    exactly what the full video shows for those bars, corrections included.

    `colors` are the whole song's family colours; passing them pins every
    family here to the colour it already wears. A family left with one
    member is dropped: its partners are outside the clip."""
    import copy
    keep = {ln: i for i, ln in enumerate(lines)}
    words: list = []
    stream: list = []
    remap: dict = {}
    old_pos = 0
    for w in analysis.words:
        n_syl = len(w.syllables)
        if w.line in keep:
            nw = copy.copy(w)
            nw.line = keep[w.line]
            nw.syllables = []
            for k, syl in enumerate(w.syllables):
                ns = copy.copy(syl)
                ns.word_index = len(words)
                nw.syllables.append(ns)
                remap[old_pos + k] = len(stream)
                stream.append(ns)
            words.append(nw)
        old_pos += n_syl

    families: list = []
    family_of: dict = {}
    for f in analysis.families:
        ms = sorted(remap[m] for m in f.members if m in remap)
        if len(ms) < 2:
            continue
        nf = copy.copy(f)
        nf.fid = len(families)
        nf.members = ms
        nf.span = len({words[stream[m].word_index].line for m in ms})
        if colors is not None and f.fid < len(colors) and not nf.fixed_color:
            nf.fixed_color = colors[f.fid]
        families.append(nf)
        for m in ms:
            family_of[m] = nf.fid

    n = len(stream)
    stats = dict(analysis.stats, lines=len(lines), words=len(words),
                 syllables=n, rhymed_syllables=len(family_of),
                 families=len(families),
                 density=(len(family_of) / n) if n else 0.0)
    return Analysis([analysis.lines[ln] for ln in lines], words, stream,
                    family_of, families, stats)


def family_sound(analysis, fam) -> tuple:
    """A family's identity as a sound: the modal (vowel, ending) across its
    stressed members. `fid` is only a position in one analysis's list; the
    sound is the same whether the rhyme is analysed alone, in its verse or
    in the whole song — which is what lets a colour follow it."""
    ms = [m for m in fam.members if analysis.stream[m].bv != SCHWA] \
        or fam.members
    return Counter((analysis.stream[m].bv, analysis.stream[m].cc)
                   for m in ms).most_common(1)[0][0]
