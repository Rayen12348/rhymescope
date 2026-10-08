"""
html_render.py — turn an Analysis into a standalone HTML page.

Each rhyme family gets its own color; syllable spans inside words are
highlighted individually, so a single word can carry several families
(e.g. "under-STAND-ing" only lights up the syllables that actually rhyme).
"""

from __future__ import annotations
import html as _html

from rhyme_engine import Analysis


from palette import assign as _assign, bright_pool


def _palette(analysis) -> list[str]:
    """Perceptually-separated colors, assigned so that families appearing
    near each other in the lyrics never look alike (see palette.py)."""
    return _assign(analysis, bright_pool)


_CSS = """
/* RhymeScope brand — reversed (on dark) variant:
   Ink navy surfaces · Signal Cyan #6FCBE0 accent · Space Grotesk / Inter */
:root { color-scheme: dark; }
* { box-sizing: border-box; margin: 0; }
body {
  background: #14172A; color: #D8DBE4;
  font-family: "Inter", system-ui, sans-serif;
  padding: 48px 20px 96px;
}
::selection { background: rgba(111, 203, 224, .25); }
.wrap { max-width: 860px; margin: 0 auto; }
header { margin-bottom: 8px; }
.lockup { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }
.lockup .mark { font-family: "Space Grotesk", sans-serif; font-size: 18px;
  font-weight: 600; color: #F9F9FA; }
.lockup .mark b { color: #6FCBE0; font-weight: 600; }
.lockup .tag { font-size: 13px; color: #7A8095; }
h1 { font-family: "Space Grotesk", sans-serif; font-size: 30px;
  font-weight: 600; color: #F9F9FA; letter-spacing: -.01em; }
.stats {
  display: flex; gap: 22px; flex-wrap: wrap;
  margin: 18px 0 30px; font-size: 13px; color: #A8ADC0;
}
.stats b { font-family: "Space Grotesk", sans-serif; color: #F4F5F8;
  font-weight: 600; }
.lyrics {
  font-size: 19px; line-height: 2.05;
  background: #171B2E; border: 1px solid #262B42; border-radius: 4px;
  padding: 30px 34px; white-space: pre-wrap; word-wrap: break-word;
}
.syl {
  border-radius: 4px; padding: 1px 1px 0;
  color: var(--c); border-bottom: 2px solid var(--c);
  font-weight: 600; cursor: default;
  transition: background .12s ease;
}
.syl:hover, .syl.hot { background: color-mix(in srgb, var(--c) 22%, transparent); }
.legend {
  margin-top: 28px; display: flex; flex-wrap: wrap; gap: 8px;
}
.chip {
  display: inline-flex; align-items: center; gap: 8px;
  font-size: 12.5px; padding: 5px 11px; border-radius: 999px;
  background: #171B2E; border: 1px solid #262B42; color: #B9BEd0;
  cursor: pointer; user-select: none;
}
.chip .dot { width: 10px; height: 10px; border-radius: 50%; background: var(--c); }
.chip .k { font-family: ui-monospace, "Cascadia Mono", monospace; color: var(--c); }
.chip.active { border-color: var(--c); }
.note { margin-top: 26px; font-size: 12.5px; color: #7A8095; line-height: 1.7; }
@media (max-width: 560px) { .lyrics { font-size: 16px; padding: 20px 16px; } }
"""

_LOGO_SVG = """<svg width="34" height="34" viewBox="0 0 64 64" fill="none">
<circle cx="32" cy="32" r="24" stroke="#F9F9FA" stroke-width="3"/>
<line x1="32" y1="2" x2="32" y2="9" stroke="#6FCBE0" stroke-width="3" stroke-linecap="round"/>
<line x1="32" y1="55" x2="32" y2="62" stroke="#6FCBE0" stroke-width="3" stroke-linecap="round"/>
<line x1="2" y1="32" x2="9" y2="32" stroke="#6FCBE0" stroke-width="3" stroke-linecap="round"/>
<line x1="55" y1="32" x2="62" y2="32" stroke="#6FCBE0" stroke-width="3" stroke-linecap="round"/>
<rect x="20" y="26" width="4" height="12" rx="2" fill="#F9F9FA"/>
<rect x="27" y="20" width="4" height="24" rx="2" fill="#6FCBE0"/>
<rect x="34" y="24" width="4" height="16" rx="2" fill="#F9F9FA"/>
<rect x="41" y="18" width="4" height="28" rx="2" fill="#6FCBE0"/>
</svg>"""

_JS = """
document.querySelectorAll('.chip').forEach(chip => {
  chip.addEventListener('click', () => {
    const f = chip.dataset.f;
    const on = chip.classList.toggle('active');
    document.querySelectorAll('.syl[data-f="'+f+'"]')
      .forEach(s => s.classList.toggle('hot', on));
  });
});
"""


def render(analysis: Analysis, title: str = "Rhyme map") -> str:
    colors = _palette(analysis)

    # stream index lookup: (word_index, syll_index) -> family id
    fam_at = {}
    pos = 0
    for w_i, w in enumerate(analysis.words):
        for s_i in range(len(w.syllables)):
            fid = analysis.family_of.get(pos)
            if fid is not None:
                fam_at[(w_i, s_i)] = fid
            pos += 1

    # rebuild each line with highlighted syllable spans
    out_lines = []
    words_by_line: dict = {}
    for w_i, w in enumerate(analysis.words):
        words_by_line.setdefault(w.line, []).append((w_i, w))

    for ln, raw in enumerate(analysis.lines):
        cursor, buf = 0, []
        for w_i, w in words_by_line.get(ln, []):
            buf.append(_html.escape(raw[cursor:w.start]))
            if w.syllables and any((w_i, k) in fam_at for k in range(len(w.syllables))):
                for k, (s, e) in enumerate(w.spans):
                    seg = _html.escape(w.text[s:e])
                    fid = fam_at.get((w_i, k))
                    if fid is None:
                        buf.append(seg)
                    else:
                        fam = analysis.families[fid]
                        cls = "syl"
                        buf.append(
                            f'<span class="{cls}" data-f="{fid}" '
                            f'style="--c:{colors[fid]}" '
                            f'title="family {fid + 1}: {_html.escape(fam.label)}">'
                            f"{seg}</span>"
                        )
            else:
                buf.append(_html.escape(w.text))
            cursor = w.end
        buf.append(_html.escape(raw[cursor:]))
        out_lines.append("".join(buf))

    chips = []
    for fam in analysis.families:
        from term_render import rhyme_pieces
        example_words = rhyme_pieces(analysis, fam, 3)
        ccls = "chip"
        chips.append(
            f'<span class="{ccls}" data-f="{fam.fid}" style="--c:{colors[fam.fid]}">'
            f'<span class="dot"></span>'
            f'<span class="k">{_html.escape(fam.label)}</span>'
            f'<span>{_html.escape(", ".join(example_words))}</span></span>'
        )

    st = analysis.stats
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html.escape(title)} — RhymeScope</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>{_CSS}</style></head>
<body><div class="wrap">
<header>
  <div class="lockup">{_LOGO_SVG}
    <span class="mark">Rhyme<b>Scope</b></span>
    <span class="tag">· syllable-level rhyme map</span>
  </div>
  <h1>{_html.escape(title)}</h1>
</header>
<div class="stats">
  <span><b>{st['families']}</b> rhyme families</span>
  <span><b>{st['rhymed_syllables']}</b>/<b>{st['syllables']}</b> syllables rhyming</span>
  <span><b>{st['density']:.0%}</b> rhyme density</span>
  <span><b>{st['lines']}</b> lines</span>
</div>
<div class="lyrics">{chr(10).join(out_lines)}</div>
<div class="legend">{''.join(chips)}</div>
<p class="note">Each color is one rhyme family. Highlighting is per syllable —
a word can belong to several families at once. Click a legend chip to light up
its family in the text; hover any syllable to see its sound key.</p>
</div><script>{_JS}</script></body></html>
"""
