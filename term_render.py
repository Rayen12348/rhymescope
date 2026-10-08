"""
term_render.py — draw an Analysis as a color-coded rhyme map in the terminal.

Same idea as html_render, but with ANSI truecolor escapes: each rhyme family
gets its own color, syllables are highlighted individually (bold + underline),
and a legend maps every color to its sound key and the syllables it rhymes on.
"""

from __future__ import annotations

from html_render import _palette
from rhyme_engine import Analysis

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
UNDER = "\x1b[4m"
DIM = "\x1b[2m"


def _fg(hexcolor: str) -> str:
    r = int(hexcolor[1:3], 16)
    g = int(hexcolor[3:5], 16)
    b = int(hexcolor[5:7], 16)
    return f"\x1b[38;2;{r};{g};{b}m"


def _rule(title: str, width: int = 64) -> str:
    label = f" {title} "
    dash = "─" * max(0, (width - len(label)) // 2)
    return f"{DIM}{dash}{RESET}{BOLD}{label}{RESET}{DIM}{dash}{RESET}"


def syllable_map(analysis: Analysis) -> tuple[dict, dict]:
    """Lookup tables shared by every renderer:
    fam_at[(word_index, syllable_index)] -> family id
    words_by_line[line] -> [(word_index, Word), …]"""
    fam_at: dict = {}
    pos = 0
    for w_i, w in enumerate(analysis.words):
        for s_i in range(len(w.syllables)):
            fid = analysis.family_of.get(pos)
            if fid is not None:
                fam_at[(w_i, s_i)] = fid
            pos += 1

    words_by_line: dict = {}
    for w_i, w in enumerate(analysis.words):
        words_by_line.setdefault(w.line, []).append((w_i, w))
    return fam_at, words_by_line


def rhyme_pieces(analysis: Analysis, fam, limit: int | None = None) -> list:
    """The SYLLABLES a family rhymes on, not the words they sit in.

    One entry per word occurrence, holding only that word's rhyming
    syllables joined by "·" — "tas" from fan-TAS-tic, "as·tic" from
    e-LAS-TIC. Repeats are dropped; `limit` caps how many are returned."""
    by_word: dict = {}
    for m in sorted(fam.members):
        s = analysis.stream[m]
        w = analysis.words[s.word_index]
        if w.spans and s.index_in_word < len(w.spans):
            a, b = w.spans[s.index_in_word]
            piece = w.text[a:b] or w.text
        else:
            piece = w.text
        by_word.setdefault(s.word_index, []).append(piece)
    out: list = []
    for pieces in by_word.values():
        t = "·".join(pieces)
        if t.lower() not in (x.lower() for x in out):
            out.append(t)
        if limit and len(out) == limit:
            break
    return out


def colored_line(analysis: Analysis, colors: list, fam_at: dict,
                 words_by_line: dict, ln: int,
                 active: int | None = None) -> str:
    """One lyrics line with ANSI syllable highlighting. When `active` is
    given, only the first `active` words are revealed (colored); the rest
    stay dim — used by word-by-word karaoke."""
    raw = analysis.lines[ln]
    cursor, buf = 0, []
    for idx, (w_i, w) in enumerate(words_by_line.get(ln, [])):
        buf.append(raw[cursor:w.start])
        if active is not None and idx >= active:
            buf.append(f"{DIM}{w.text}{RESET}")
        elif w.syllables and any((w_i, k) in fam_at
                                 for k in range(len(w.syllables))):
            for k, (s, e) in enumerate(w.spans):
                seg = w.text[s:e]
                fid = fam_at.get((w_i, k))
                if fid is None:
                    buf.append(seg)
                else:
                    buf.append(f"{_fg(colors[fid])}{BOLD}{UNDER}{seg}{RESET}")
        else:
            buf.append(w.text)
        cursor = w.end
    buf.append(raw[cursor:])
    return "".join(buf)


def render_terminal(analysis: Analysis, title: str = "Rhyme map",
                    source: str | None = None) -> str:
    colors = _palette(analysis)
    fam_at, words_by_line = syllable_map(analysis)

    out: list[str] = []

    # header ---------------------------------------------------------------
    out.append("")
    out.append(f"{BOLD}♪ {title}{RESET}")
    st = analysis.stats
    meta = (f"{st['families']} families · "
            f"{st['rhymed_syllables']}/{st['syllables']} syllables rhyming · "
            f"{st['density']:.0%} density · {st['lines']} lines")
    out.append(f"{DIM}{meta}{RESET}")
    if source:
        out.append(f"{DIM}lyrics: {source}{RESET}")
    out.append("")
    out.append(_rule("lyrics"))

    # lyrics ---------------------------------------------------------------
    for ln in range(len(analysis.lines)):
        out.append(colored_line(analysis, colors, fam_at, words_by_line, ln))

    # legend ---------------------------------------------------------------
    out.append("")
    out.append(_rule("legend"))
    for fam in analysis.families:
        example_words = rhyme_pieces(analysis, fam, 4)
        c = _fg(colors[fam.fid])
        dot = f"{c}●{RESET}"
        key = f"{c}{fam.label}{RESET}"
        out.append(f"  {dot} {key}  {DIM}{', '.join(example_words)}{RESET}"
                   f" {DIM}(x{fam.span} lines){RESET}")

    out.append("")
    return "\n".join(out)
