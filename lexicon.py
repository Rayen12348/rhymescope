"""
lexicon.py — words become sounds.

Nothing in here decides what rhymes. It answers one question — how is
this word said? — and hands the answer to rhyme_engine.py as syllables:
a nucleus vowel, its stress, and the consonants after it.

It knows what the CMU dictionary knows, plus the rap lexicon: dropped-G
forms (flowin' → flowing), -a spellings (dolla → dollar), respellings
('em, shawty, thru), numbers as they are said (2019 → "twenty
nineteen"), and coined words built off the stem they were coined from
(killable ← kill + -able) so they carry real vowels and real stress.
"""

from __future__ import annotations
import re
from typing import Optional

import pronouncing
import pyphen

_PYPHEN = pyphen.Pyphen(lang="en_US")

ARPABET_VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER",
    "EY", "IH", "IY", "OW", "OY", "UH", "UW",
}

# Consonant sound classes: voiced/unvoiced pairs and near-sounds merge so
# "cat/mad" (T/D) or "boss/cause" (S/Z) still register as rhyming codas.
# TH/DH fold into F/V (th-fronting: "death/deaf", "smooth/groove") — rappers
# pronounce them that way when bending a rhyme.
CONS_CLASS = {
    "T": "T", "D": "T",
    "S": "S", "Z": "S",
    "K": "K", "G": "K",
    "P": "P", "B": "P",
    "F": "F", "V": "F",
    "M": "N", "N": "N", "NG": "N",
    "CH": "J", "JH": "J", "SH": "C", "ZH": "C",
    "TH": "F", "DH": "F",
    "L": "L", "R": "R",
    "W": "", "Y": "", "HH": "",
}

# Vowel classes for slant mode (looser assonance matching). OY stays its own
# class: "boy" does not rhyme with "go" in any accent.
VOWEL_CLASS = {
    "IY": "I", "IH": "I",
    "EY": "E", "EH": "E", "AE": "E",
    "AA": "A", "AO": "A", "AH": "A",
    "UW": "U", "UH": "U",
    "OW": "O", "OY": "OY",
    "AY": "AY", "AW": "AW", "ER": "ER",
}

# ==========================================================================
# §1  PHONOLOGY — what counts as THE SAME SOUND
# ==========================================================================
#
# Rhyme is not a graded resemblance between vowels. It is IDENTITY of the
# rime — nucleus plus coda — in the phonology the performer is actually

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "to", "of", "in", "on", "at",
    "is", "it", "as", "be", "by", "so", "we", "he", "she", "i", "you",
    "my", "me", "do", "if", "was", "are", "for", "with", "that", "this",
    "them", "then", "than", "his", "her", "its", "our", "your", "they",
    "just", "got", "get", "am", "up", "im", "ill", "id", "ive",
    "what", "when", "why", "how", "who", "all", "can", "cant", "could",
    "would", "should", "will", "wont", "dont", "didnt", "did", "does",
    "done", "has", "have", "had", "aint", "been", "were", "from", "into",
    "out", "over", "about", "cause", "cuz", "while", "where", "there",
    "here", "some", "any", "no", "not", "too", "these", "those", "like",
    "gonna", "wanna", "gotta", "youre", "hes", "shes", "theyre", "ya",
}

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’]*|[0-9]+(?:'?[sS])?")



def _clean(word: str) -> str:
    return word.replace("’", "'").strip("'").lower()


_FALLBACK_VOWELS = [
    ("igh", "AY1"), ("eigh", "EY1"), ("ough", "OW1"), ("augh", "AO1"),
    ("tion", "SH AH0 N"), ("sion", "ZH AH0 N"),
    ("oo", "UW1"), ("ee", "IY1"), ("ea", "IY1"), ("ai", "EY1"),
    ("ay", "EY1"), ("oa", "OW1"), ("ou", "AW1"), ("ow", "AW1"),
    ("oi", "OY1"), ("oy", "OY1"), ("au", "AO1"), ("aw", "AO1"),
    ("a", "AE1"), ("e", "EH1"), ("i", "IH1"), ("o", "AA1"),
    ("u", "AH1"), ("y", "IY0"),
]
_FALLBACK_CONS = {
    "b": "B", "c": "K", "d": "D", "f": "F", "g": "G", "h": "HH", "j": "JH",
    "k": "K", "l": "L", "m": "M", "n": "N", "p": "P", "q": "K", "r": "R",
    "s": "S", "t": "T", "v": "V", "w": "W", "x": "K S", "z": "Z",
}
_FALLBACK_DIGRAPH = {"ch": "CH", "sh": "SH", "th": "TH", "ph": "F", "ng": "NG",
                     "ck": "K", "qu": "K W", "wh": "W", "gh": ""}


def _re_stress(phones: str, mark: str) -> str:
    """Set every vowel in a phoneme string to one stress level."""
    return " ".join(p[:-1] + mark if p and p[-1].isdigit() else p
                    for p in phones.split())


def _fallback_phones(word: str) -> str:
    """Very rough letter-to-ARPABET guess for out-of-vocabulary words
    (slang, names, ad-libs). Enough to give the word a vowel skeleton so
    it can still participate in rhyme matching."""
    w, out, i = _clean(word), [], 0
    # English orthography, two rules deep. Both exist because their absence
    # is destructive rather than merely rough: a doubled consonant invents
    # a syllable boundary that is not there ("killable" → kil-l-able), and
    # a silent final e invents a whole stressed syllable at the very
    # position rhyme is judged from.
    w = re.sub(r"([bcdfglmnprstz])\1", r"\1", w)
    if len(w) > 3 and w.endswith("e") and w[-2] not in "aeiouy":
        w = w[:-1]
    while i < len(w):
        matched = False
        for pat, ph in _FALLBACK_VOWELS:
            if w.startswith(pat, i):
                out.append(ph)
                i += len(pat)
                matched = True
                break
        if matched:
            continue
        two = w[i:i + 2]
        if two in _FALLBACK_DIGRAPH:
            if _FALLBACK_DIGRAPH[two]:
                out.append(_FALLBACK_DIGRAPH[two])
            i += 2
            continue
        ch = w[i]
        if ch in _FALLBACK_CONS:
            out.append(_FALLBACK_CONS[ch])
        i += 1
    # English has ONE primary stress per word. Marking every guessed vowel
    # stressed is not a harmless approximation — stress is half of what an
    # anchor tests, so an all-stressed word cannot align with a real one at
    # any depth. First vowel keeps the stress, the rest reduce.
    # a word-final "a" is a schwa, never the AE of "cat": gotta, bruh-a
    if len(w) > 2 and w.endswith("a") and out and out[-1] == "AE1":
        out[-1] = "AH0"
    seen = False
    for k, ph in enumerate(out):
        if ph[-1].isdigit():
            out[k] = ph[:-1] + ("1" if not seen else "0")
            seen = True
    return " ".join(out)


# Productive suffixes, with the pronunciation they actually carry. Rap
# coins words off real ones without pause — "killable", "acrobatical",
# "unfuckwitable" — and for those the letter-by-letter guess is not rough
# but destructive: it throws away a stem we know exactly, and it puts the
# wrong vowels in the tail, which is the part rhyme is judged on.
#
# Longest match wins, so "-ically" is tried before "-ly" and "-ical".
_SUFFIX = {
    "ically": "IH0 K L IY0", "ical": "IH0 K AH0 L",
    "ability": "AH0 B IH0 L AH0 T IY0",
    "ibility": "AH0 B IH0 L AH0 T IY0",
    "able": "AH0 B AH0 L", "ible": "AH0 B AH0 L",
    "ation": "EY1 SH AH0 N", "ition": "IH1 SH AH0 N",
    "tional": "SH AH0 N AH0 L",
    "tion": "SH AH0 N", "sion": "ZH AH0 N",
    "ously": "AH0 S L IY0", "ous": "AH0 S",
    "fully": "F AH0 L IY0", "ful": "F AH0 L",
    "lessly": "L AH0 S L IY0", "less": "L AH0 S",
    "ness": "N AH0 S", "ment": "M AH0 N T",
    "ary": "EH0 R IY0", "ory": "AO0 R IY0", "ery": "ER0 IY0",
    "ity": "AH0 T IY0", "ify": "AH0 F AY0",
    "ize": "AY0 Z", "ise": "AY0 Z",
    "ish": "IH0 SH", "est": "AH0 S T",
    "ian": "IY0 AH0 N", "ium": "IY0 AH0 M", "ia": "IY0 AH0",
    "ings": "IH0 NG Z", "ing": "IH0 NG",
    "ally": "AH0 L IY0", "ly": "L IY0",
    "al": "AH0 L", "ers": "ER0 Z", "er": "ER0",
    "ors": "ER0 Z", "or": "ER0",
    "y": "IY0", "a": "AH0",
}
_SUFFIX_ORDER = sorted(_SUFFIX, key=len, reverse=True)


def _stem_phones(stem: str) -> Optional[str]:
    """The dictionary pronunciation of a suffix's stem, allowing for the
    spelling changes English makes when it attaches one: a dropped silent
    e (usable→use), a doubled consonant (runnable→run), y→i
    (happiness→happy)."""
    for cand in (stem, stem + "e",
                 stem[:-1] if len(stem) > 2 and stem[-1] == stem[-2] else "",
                 stem[:-1] + "y" if stem.endswith("i") else ""):
        if cand:
            ph = pronouncing.phones_for_word(cand)
            if ph:
                return ph[0]
    return None


def _suffix_phones(word: str) -> Optional[tuple]:
    """Known stem + known suffix, keeping the stem's real vowels and the
    suffix's real reduction ("kill"+"able" → K IH1 L AH0 B AH0 L, which
    lines up with "syllable" syllable for syllable).
    Returns (letters explained by the stem, phones)."""
    for sfx in _SUFFIX_ORDER:
        if not word.endswith(sfx) or len(word) - len(sfx) < 2:
            continue
        stem = _stem_phones(word[:-len(sfx)])
        if stem:
            tail = _SUFFIX[sfx]
            # only one primary stress per word: a suffix that carries its
            # own ("-ATION") demotes the stem's, otherwise the stem keeps it
            if "1" in tail:
                stem = stem.replace("1", "2")
            return len(word) - len(sfx), stem + " " + tail
    return None


def _morph_phones(word: str) -> Optional[tuple]:
    """Longest dictionary word that is a PREFIX of an unknown word, with
    the leftover letters sounded out. Returns (letters explained, phones).

    Rap coins words off real ones constantly — "macadamian" from
    macadamia, "ballistical" from ballistic — and for those the letter-by
    -letter guess is not merely rough, it is destructive: it throws away a
    pronunciation we already know exactly. Guessing "macadamian" gives
    every vowel as a stressed AE, which cannot align with anything;
    macadamia + "n" gives the real vowels and the real stress, which is
    what lets it lock onto "packing stadiums"."""
    for cut in range(len(word) - 1, 3, -1):
        tail = word[cut:]
        if len(tail) > 3:          # too much left to sound out reliably
            break
        cands = pronouncing.phones_for_word(word[:cut])
        if cands:
            # Whatever hangs off a complete stem is an ending, and endings
            # are unstressed — so the tail is pronounced from the suffix
            # table when we know it, and reduced when we have to guess.
            # Sounding it out at full stress puts a second primary stress
            # on the word, at exactly the syllable rhyme is judged from.
            tail_ph = _SUFFIX.get(tail)
            if tail_ph is None:
                tail_ph = _re_stress(_fallback_phones(tail), "0")
            return cut, (cands[0] + " " + tail_ph).strip()
    return None


_ONES = ("zero one two three four five six seven eight nine ten eleven "
         "twelve thirteen fourteen fifteen sixteen seventeen eighteen "
         "nineteen").split()
_TENS = "zero ten twenty thirty forty fifty sixty seventy eighty ninety".split()


def _two_digits(n: int) -> list[str]:
    if n < 20:
        return [_ONES[n]]
    t, o = divmod(n, 10)
    return [_TENS[t]] + ([_ONES[o]] if o else [])


def _three_digits(n: int) -> list[str]:
    h, r = divmod(n, 100)
    out = ([_ONES[h], "hundred"] if h else [])
    if r:
        out += _two_digits(r)
    return out or [_ONES[0]]


def _num_words(tok: str) -> list[str] | None:
    """'2019' → ['twenty','nineteen'], '80s' → ['eighties'], the way the
    number is actually said — so its words can carry rhymes."""
    plural = tok.rstrip("sS") != tok
    digits = tok.rstrip("sS").rstrip("'’")
    if not digits.isdigit():
        return None
    n = int(digits)
    if len(digits) == 4 and digits[0] != "0":       # years read in pairs
        hi, lo = divmod(n, 100)
        if 2000 <= n <= 2009:
            words = ["two", "thousand"] + ([_ONES[lo]] if lo else [])
        elif lo == 0:
            words = _two_digits(hi) + ["hundred"]
        else:
            words = _two_digits(hi) + _two_digits(lo)
    elif n < 1000:
        words = _three_digits(n)
    elif n < 1_000_000 and n % 1000 == 0:
        words = _three_digits(n // 1000) + ["thousand"]
    else:                                           # long strings: digit by digit
        words = [_ONES[int(d)] for d in digits]
    if plural:
        words = words[:-1] + [words[-1] + "s"]
    return words


# rap respellings that the CMU dict knows under another spelling
_RESPELL = {"em": "them", "til": "till", "gon": "gone",
            "tho": "though", "thru": "through", "aint": "ain't",
            "ya": "you", "da": "the", "tha": "the",
            "dat": "that", "dem": "them", "dis": "this", "dese": "these",
            "wit": "with", "nuttin": "nothing", "shawty": "shorty",
            "brotha": "brother", "motha": "mother", "holla": "holler",
            "mic": "mike", "mics": "mikes"}


# words the CMU dict gets WRONG for rap. Some are simply missing the rap
# sense ('mic' is listed as M-IH-K); others are homographs whose dictionary
# order is wrong for this domain — CMU lists 'bass' as the fish (B AE1 S)
# before the low end (B EY1 S), and in a rap lyric it is the low end,
# which is the difference between "the BASS so loud" joining the
# taste/place/face run and dropping out of it.
_OVERRIDE = {"mic": "M AY1 K", "mics": "M AY1 K S",
             "bass": "B EY1 S",
             # the interjection, not the dictionary's first "ow" (as in owe)
             "ow": "AW1",
             # everyday rap words the dictionary does not have, which the
             # letter rules get badly wrong ("homie" as three syllables)
             "homie": "HH OW1 M IY0", "homies": "HH OW1 M IY0 Z",
             "tryna": "T R AY1 N AH0", "nigga": "N IH1 G AH0",
             "niggas": "N IH1 G AH0 Z", "shorty": "SH AO1 R T IY0",
             # the dictionary drops its last syllable
             "swayze": "S W EY1 Z IY0"}

_VOICELESS = {"P", "T", "K", "F", "TH", "S", "SH", "CH"}


def _inflected(w: str) -> Optional[str]:
    """A known word plus a regular ending the dictionary happens to lack:
    "dissed" is diss + T (one syllable, not dis-sed), "niggas" nigga + Z.
    The ending is pronounced by the rule English uses for it, from the
    stem's last sound."""
    tries = []
    if w.endswith("ed") and len(w) > 4:
        tries += [("ed", w[:-2]), ("ed", w[:-1])]
        if w[-3] == w[-4]:
            tries.append(("ed", w[:-3]))
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        tries.append(("s", w[:-1]))
        if w.endswith("es"):
            tries.append(("s", w[:-2]))
    for kind, stem in tries:
        ph, oov = phones_for(stem)
        if not ph or oov:
            continue
        last = ph.split()[-1]
        if kind == "ed":
            end = "IH0 D" if last in ("T", "D") else ("T" if last in _VOICELESS else "D")
        else:
            end = ("IH0 Z" if last in ("S", "Z", "SH", "ZH", "CH", "JH")
                   else ("S" if last in _VOICELESS else "Z"))
        return ph + " " + end
    return None


def phones_for(word: str) -> tuple[Optional[str], bool]:
    """Return (phoneme string, is_oov)."""
    w = _clean(word)
    if not w:
        return None, False
    if w in _OVERRIDE:
        return _OVERRIDE[w], False
    if w[0].isdigit():
        spoken = _num_words(w)
        if not spoken:
            return None, True
        parts = []
        for x in spoken:
            ph = pronouncing.phones_for_word(x)
            if not ph and x.endswith("s"):          # 'eightys' → eighty + Z
                base = pronouncing.phones_for_word(x[:-1])
                ph = [base[0] + " Z"] if base else None
            if not ph:
                return None, True
            parts.append(ph[0])
        return " ".join(parts), False
    cands = pronouncing.phones_for_word(w)
    if not cands and w.endswith("in"):
        # dropped-G spellings: flowin', gettin', runnin' → look up the -ing
        # form (rhyme-wise identical: the nasal class merges N/NG)
        cands = pronouncing.phones_for_word(w + "g")
    if not cands and w.endswith("a") and len(w) > 3:
        # rap -a spellings: dolla → dollar, playa → player, sucka → sucker
        cands = (pronouncing.phones_for_word(w + "r")
                 or pronouncing.phones_for_word(w[:-1] + "er"))
    if not cands and w in _RESPELL:
        cands = pronouncing.phones_for_word(_RESPELL[w])
    if cands:
        return cands[0], False
    infl = _inflected(w)
    if infl:
        return infl, False
    # Built off a real word before guessing: a known stem carries real
    # vowels and real stress, which is what a multi has to align against.
    # Two ways to find one — peel a known suffix, or take the longest known
    # prefix — and they can disagree ("macadamian" is macadam+ian to one
    # and macadamia+n to the other). The one that explains MORE of the word
    # is the better reading of it, which settles the choice without a list
    # of exceptions.
    best = max((c for c in (_suffix_phones(w), _morph_phones(w)) if c),
               key=lambda c: c[0], default=None)
    if best:
        return best[1], False
    guess = _fallback_phones(w)
    return (guess, True) if guess else (None, True)


# --------------------------------------------------------------------------
# Syllabification (phonetic)
# --------------------------------------------------------------------------

def syllabify(phones) -> list[tuple[str, int, tuple]]:
    """Split a phoneme string into (vowel, stress, coda) triples.
    Coda = every consonant after the nucleus up to the next vowel; this makes
    'master' -> [(AE,1,(S,T)), (ER,0,())] which is exactly the unit rap
    rhymes operate on."""
    # a phoneme string ("S P IY1 L") or a list of phonemes — the engine
    # keeps pronunciations as lists so it can show where each came from
    toks = phones.split() if isinstance(phones, str) else list(phones)
    sylls, cur_vowel, cur_stress, cur_coda = [], None, 0, []
    for t in toks:
        base = t.rstrip("012")
        if base in ARPABET_VOWELS:
            if cur_vowel is not None:
                sylls.append((cur_vowel, cur_stress, tuple(cur_coda)))
            cur_vowel = base
            cur_stress = int(t[-1]) if t[-1].isdigit() else 0
            cur_coda = []
        elif cur_vowel is not None:
            cur_coda.append(base)
        # leading onset consonants before the first vowel are ignored:
        # they never matter for rhyme.
    if cur_vowel is not None:
        sylls.append((cur_vowel, cur_stress, tuple(cur_coda)))
    return sylls


# --------------------------------------------------------------------------
# Orthographic syllable spans (for highlighting letters, not phonemes)
# --------------------------------------------------------------------------

_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)


def split_spans(word: str, n: int) -> list[tuple[int, int]]:
    """Split `word` (as written) into n contiguous character spans that
    approximate its n phonetic syllables."""
    if n <= 0:
        return []
    if n == 1:
        return [(0, len(word))]

    # 1) hyphenation dictionary, if it agrees on the count
    parts = _PYPHEN.inserted(word.lower()).split("-")
    if len(parts) == n:
        spans, pos = [], 0
        for p in parts:
            spans.append((pos, pos + len(p)))
            pos += len(p)
        return spans

    # 2) split on spelling vowel groups: each syllable ends right before
    #    the consonant(s) that start the next vowel group
    groups = list(_VOWEL_GROUP_RE.finditer(word))
    if len(groups) == n:
        spans, start = [], 0
        for i in range(n - 1):
            cut = (groups[i].end() + groups[i + 1].start() + 1) // 2
            spans.append((start, cut))
            start = cut
        spans.append((start, len(word)))
        return spans

    # 3) fallback: equal-ish character split
    spans, start = [], 0
    for i in range(n):
        end = round(len(word) * (i + 1) / n)
        end = max(end, start + 1) if start < len(word) else start
        spans.append((start, min(end, len(word))))
        start = spans[-1][1]
    spans[-1] = (spans[-1][0], len(word))
    return spans


# --------------------------------------------------------------------------
# Union-find
# --------------------------------------------------------------------------

