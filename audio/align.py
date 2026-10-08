"""Which vowel was SUNG in each syllable of the lyrics.

Two steps over the phoneme posteriors of audio/phonemes.py:

1. Forced alignment, one bar at a time. The bar's pronunciation from the
   lexicon is turned into the model's phoneme alphabet and aligned (CTC
   Viterbi) inside the stretch of audio the line sync gives that bar, with
   free "unaligned" audio allowed before and after it (ad-libs, breaths,
   the tail of the last bar). Every vowel slot accepts ANY vowel, so a
   bent vowel does not pull the alignment off — the consonants around it
   pin the timing.
2. In the frames aligned to each vowel, every vowel class is scored by
   the model's best log-probability for it. The class it heard best, and
   by how much it beat the dictionary's, is what the rhyme engine gets.

The classes are the engine's own (rhyme_engine.sound_of): IY IH EY EH AE
AA AH OW UH UW AY AW OY ER, the r-coloured AR OR AIR EAR IRE, and Ə.
"""
from __future__ import annotations

import math

import numpy as np

# model phoneme (espeak IPA) -> engine vowel class
IPA_CLASS = {
    "iː": "IY", "i": "IY",
    "ɪ": "IH", "ᵻ": "IH", "ɨ": "IH",
    "eɪ": "EY", "e": "EY", "eː": "EY", "e̞": "EY",
    "ɛ": "EH", "ɛː": "EH",
    "æ": "AE", "a": "AE",
    "ɑː": "AA", "ɑ": "AA", "aː": "AA", "ɔ": "AA", "ɔː": "AA",
    "ʌ": "AH",
    "ə": "Ə", "ɐ": "Ə",
    "ɚ": "ER", "ɜː": "ER", "ɜ": "ER",
    "oʊ": "OW", "o": "OW", "oː": "OW",
    "ʊ": "UH",
    "uː": "UW", "u": "UW", "ʉ": "UW",
    "aɪ": "AY", "aʊ": "AW", "ɔɪ": "OY", "oɪ": "OY",
    "ɑːɹ": "AR", "ɔːɹ": "OR", "oːɹ": "OR", "ʊɹ": "OR",
    "ɛɹ": "AIR", "ɪɹ": "EAR", "iə": "EAR", "aɪɚ": "IRE",
}

# ARPAbet consonant -> the model phonemes it may come out as. Fast
# delivery flaps T/D, drops TH to F/T, turns -NG into -N.
CONS = {
    "B": ["b"], "CH": ["tʃ"], "D": ["d", "ɾ"], "DH": ["ð", "d", "v"],
    "F": ["f"], "G": ["ɡ"], "HH": ["h"], "JH": ["dʒ"], "K": ["k"],
    "L": ["l", "əl"], "M": ["m"], "N": ["n"], "NG": ["ŋ", "n"],
    "P": ["p"], "R": ["ɹ", "ɚ", "r"], "S": ["s"], "SH": ["ʃ"],
    "T": ["t", "ɾ", "ʔ"], "TH": ["θ", "f", "t"], "V": ["v"], "W": ["w"],
    "Y": ["j"], "Z": ["z", "s"], "ZH": ["ʒ"],
}
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH",
          "IY", "OW", "OY", "UH", "UW"}

_GARBAGE_COST = 4.0     # an unaligned frame costs this much below its best
_SKIP_LINE = -40.0      # leaving out a whole line the vocal does not have


def targets(phones: str, vocab: dict):
    """The model-alphabet target sequence for one pronunciation:
    [(token ids allowed, syllable index or None)]. A vowel followed by R
    is one r-coloured slot, as the model writes it (ɑːɹ)."""
    toks = phones.split()
    out = []
    vowel_ids = [vocab[t] for t in IPA_CLASS if t in vocab]
    k = -1
    i = 0
    while i < len(toks):
        base = toks[i].rstrip("012")
        if base in VOWELS:
            k += 1
            out.append((vowel_ids, k))
            if i + 1 < len(toks) and toks[i + 1] == "R" and base != "ER":
                i += 1                   # the R is inside the vowel slot
        else:
            ids = [vocab[t] for t in CONS.get(base, []) if t in vocab]
            if ids:
                out.append((ids, None))
        i += 1
    return out


def align_song(lp: np.ndarray, lines: list, bands: list):
    """One CTC Viterbi over the whole song: every line's target slots in
    order, free "unaligned" audio between lines (ad-libs, hooks the sheet
    writes once, instrumental), and a costly jump that skips a line the
    vocal does not have. Aligning bar by bar let a line borrow its
    neighbour's audio — "pose" landing on the next bar's "face" — which a
    single pass rules out: each stretch of audio belongs to one line.

    `lines` — per line, its target slots (targets()) or [] ; `bands` — per
    line, (first, last) frame it may occupy (from the line sync, widened).
    Returns per line ([(f0, f1)] per slot, mean log-prob) or None."""
    T = len(lp)
    NEG = np.float32(-1e9)
    blank = 0
    garbage = np.maximum(lp[:, blank], lp.max(1) - _GARBAGE_COST)
    cols, kind, owner = [], [], []        # per state: emission source
    g_state = []                          # garbage state index per line
    layout = []
    for i, tgt in enumerate(lines):
        g_state.append(len(kind))
        kind.append("g"); cols.append(None); owner.append(i)
        slots = []
        if tgt:
            for j, (ids, _k) in enumerate(tgt):
                kind.append("b"); cols.append(None); owner.append(i)
                slots.append(len(kind))
                kind.append("s"); cols.append(ids); owner.append(i)
            kind.append("b"); cols.append(None); owner.append(i)
        layout.append(slots)
    g_state.append(len(kind))
    kind.append("g"); cols.append(None); owner.append(len(lines))
    S = len(kind)
    skip_ok = np.zeros(S, bool)
    for i, slots in enumerate(layout):
        for j, st in enumerate(slots):
            skip_ok[st] = j == 0 or lines[i][j][0] != lines[i][j - 1][0]
        skip_ok[g_state[i + 1]] = bool(slots)
    kind_a = np.array(kind)
    band_lo = np.array([bands[o][0] if o < len(bands) else 0 for o in owner])
    band_hi = np.array([bands[o][1] if o < len(bands) else T for o in owner])
    g_idx = np.array(g_state)
    jump_pen = np.float32(_SKIP_LINE)
    # every slot's emission comes from one of a few token sets (any vowel,
    # T-or-flap, …): compute each set's frame-wise max once
    sets: dict = {}
    slot_idx, slot_set = [], []
    for st in range(S):
        if kind[st] == "s":
            key = tuple(cols[st])
            if key not in sets:
                sets[key] = len(sets)
            slot_idx.append(st)
            slot_set.append(sets[key])
    set_em = np.stack([lp[:, list(k)].max(1) for k in sets], 1) if sets \
        else np.zeros((T, 0), np.float32)
    slot_idx = np.array(slot_idx, int)
    slot_set = np.array(slot_set, int)
    is_g = kind_a == "g"
    is_b = kind_a == "b"
    back = np.zeros((T, S), np.int8)
    score = np.full(S, NEG, np.float32)
    score[0] = garbage[0]
    for t in range(T):
        em = np.full(S, NEG, np.float32)
        em[is_g] = garbage[t]
        em[is_b] = lp[t, blank]
        em[slot_idx] = set_em[t, slot_set]
        em[(t < band_lo) | (t > band_hi)] = NEG
        em[g_idx] = garbage[t]                      # garbage is never banded
        if t == 0:
            score = np.where(np.arange(S) == 0, em, NEG).astype(np.float32)
            continue
        one = np.concatenate(([NEG], score[:-1]))
        two = np.where(skip_ok, np.concatenate(([NEG, NEG], score[:-2])), NEG)
        best = np.maximum(np.maximum(score, one), two)
        code = np.where(best == score, 0, np.where(best == one, 1, 2)).astype(np.int8)
        # skipping a whole line: garbage of line i straight to garbage i+1
        jmp = score[g_idx[:-1]] + jump_pen
        take = jmp > best[g_idx[1:]]
        best[g_idx[1:][take]] = jmp[take]
        code[g_idx[1:][take]] = 3
        back[t] = code
        score = best + em
    s_ = S - 1
    spans = {int(st): [None, None] for st in slot_idx}
    for t in range(T - 1, -1, -1):
        if kind[s_] == "s":
            spans[s_][0] = t
            if spans[s_][1] is None:
                spans[s_][1] = t
        c = back[t, s_]
        if c == 3:
            s_ = g_state[owner[s_] - 1]
        else:
            s_ -= int(c)
    out = []
    for i, slots in enumerate(layout):
        if not slots or any(spans[st][0] is None for st in slots):
            out.append(None)
            continue
        sp = [tuple(spans[st]) for st in slots]
        fit = float(np.mean([lp[a:b + 1][:, cols[st]].max()
                             for st, (a, b) in zip(slots, sp)]))
        out.append((sp, fit))
    return out


def vowel_scores(lp: np.ndarray, f0: int, f1: int, vocab: dict) -> dict:
    """{class: best log-prob of that class} over frames f0..f1 (±1)."""
    a, b = max(f0 - 1, 0), min(f1 + 2, len(lp))
    seg = lp[a:b]
    out: dict = {}
    for tok, cls in IPA_CLASS.items():
        col = vocab.get(tok)
        if col is None:
            continue
        v = float(seg[:, col].max())
        if v > out.get(cls, -1e9):
            out[cls] = v
    return out
