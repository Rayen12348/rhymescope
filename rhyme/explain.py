"""--explain: why a word did or did not rhyme.

For every word matching the query: how it is pronounced and where that
came from, each syllable's vowel class, the family it ended up in (and
at what confidence), the runs that put it there, and the strongest runs
through it that were weighed and turned down — each with its log-odds,
the number the detector decides on."""
from __future__ import annotations


def _syl_text(an, i) -> str:
    s = an.stream[i]
    w = an.words[s.word_index]
    if w.spans and s.index_in_word < len(w.spans):
        a, b = w.spans[s.index_in_word]
        return w.text[a:b] or w.text
    return w.text


def _run(an, members) -> str:
    out, last = [], None
    for i in members:
        wi = an.stream[i].word_index
        piece = _syl_text(an, i)
        if wi == last:
            out[-1] += "·" + piece
        else:
            out.append(piece)
        last = wi
    ln = an.words[an.stream[members[0]].word_index].line
    return f"L{ln + 1} “{' '.join(out)}”"


def explain(text: str, query: str, top: int = 3,
            heard: dict | None = None) -> str:
    import rhyme_engine as R
    from lexicon import phones_for, WORD_RE, _clean
    import rhyme
    shown = rhyme.detect(text, heard=heard)
    raw = R.analyze(text, heard=heard)
    loose = R.analyze(text, loose=True, heard=heard)
    syl_heard = (heard or {}).get("syllables", {})
    line_fit = (heard or {}).get("line_fit", {})
    conf = getattr(shown, "confidence", {})
    fams = {f.fid: f for f in shown.families}
    wanted = [_clean(t) for t in WORD_RE.findall(query)]
    pos: dict = {}
    for i, s in enumerate(shown.stream):
        pos.setdefault(s.word_index, []).append(i)
    out = []
    for wi, w in enumerate(shown.words):
        if _clean(w.text) not in wanted:
            continue
        phones, oov = phones_for(w.text)
        out.append(f"{w.text!r}  line {w.line + 1}  ·  {phones or '(no phones)'}"
                   f"  [{'letter-rule guess' if oov else 'lexicon'}]")
        for i in pos.get(wi, []):
            s = shown.stream[i]
            fam = shown.family_of.get(i)
            head = (f"  syl {s.index_in_word} “{_syl_text(shown, i)}”: "
                    f"{s.vowel}{s.stress} → class {getattr(s, 'bv', '?')}"
                    f"  coda {getattr(s, 'cc', ()) or '–'}")
            sc = syl_heard.get(f"{w.line}:{w.start}:{s.index_in_word}")
            if sc:
                best = sorted(sc.items(), key=lambda kv: -kv[1])[:3]
                head += ("  ·  sung: " + " ".join(f"{c} {v:+.1f}" for c, v in best)
                         + f" (bar fit {line_fit.get(str(w.line), '?')})")
            if fam is None:
                out.append(head + "  ·  not shown")
            else:
                out.append(head + f"  ·  family {fam} “{fams[fam].label}”  "
                           f"{shown.route.get(i, '?')}  "
                           f"conf {conf.get(i, 0):.2f}  "
                           f"{shown.odds.get(i, 0.0):+.1f} bits")
            kept = [(lo, k, a, b) for lo, k, a, b, _p in raw.links
                    if i in a or i in b]
            for lo, k, a, b in sorted(kept, reverse=True)[:top]:
                out.append(f"      kept    {lo:+6.1f} {k:9s} "
                           f"{_run(shown, a)} ~ {_run(shown, b)}")
            refused = [(lo, k, a, b) for lo, k, a, b, _p in loose.links
                       if (i in a or i in b) and lo <= 0]
            for lo, k, a, b in sorted(refused, reverse=True)[:top]:
                out.append(f"      refused {lo:+6.1f} {k:9s} "
                           f"{_run(shown, a)} ~ {_run(shown, b)}")
    return "\n".join(out) or f"no word matching {query!r}"
