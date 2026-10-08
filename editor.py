"""
editor.py — fix the rhyme map by hand, in the terminal.

The detector gets most of a song right and some of it wrong, and only you
can hear which is which. This opens the map you are about to render right
where you already are — the same terminal the rest of RhymeScope runs in
— lets you paint over the mistakes, and writes the result next to the
audio as a corrections file that every later render reads.

WHY IT LIVES HERE AND NOT IN A BROWSER. Correcting a rhyme is a step in
the middle of a session: you look at the map, fix the two syllables that
are wrong, and re-render. Sending that step to a browser window makes it
a context switch — a second window to find, a page to remember to save,
and an editing surface that knows nothing about the terminal you came
from. The map already renders in the terminal (menu 5); this is that map
with a cursor in it.

NOTHING IS KEYS-ONLY. Arrows move, space paints, and that is enough to
correct a whole song. The mouse works too where the terminal reports it
(virtually all of them do): click paints, drag paints a run, right-click
erases, double-click takes a whole word, the wheel scrolls. Every key is
listed in the footer, and `?` opens the full list.

WHAT IT CANNOT BREAK. The editor produces exactly one kind of output: a
mapping from syllable positions to rhyme families (see overrides.py). It
has no access to the timing path at all — line times, word times and the
Whisper alignment are computed elsewhere and cached on the song before
you ever open this. Correct a highlight, re-render, and every word lands
on the same frame it did before. That is the whole design constraint.

Corrections land in a file beside the audio on this machine and nothing
leaves the computer.
"""

from __future__ import annotations

import os
import pathlib
import re
import select
import shutil
import signal
import sys
import termios
import time
import tty
from collections import Counter

import confidence as C
import overrides as O
from palette import ink_for
from term_render import BOLD, DIM, RESET, UNDER, _fg

# --------------------------------------------------------------------------
# terminal plumbing
# --------------------------------------------------------------------------

LIME = "#CBF135"          # RhymeScope's one accent colour
PLAIN = "#DCDCD6"         # a rhyme with no colour of its own yet
REV = "\x1b[7m"
HEAD = _fg(LIME) + BOLD

_ALT_ON = "\x1b[?1049h\x1b[?25l"
_ALT_OFF = "\x1b[?25h\x1b[?1049l"
# 1000 = clicks, 1002 = clicks + drag motion, 1006 = SGR coordinates (the
# only encoding that survives past column 95, which any real song needs)
_MOUSE_ON = "\x1b[?1000h\x1b[?1002h\x1b[?1006h"
_MOUSE_OFF = "\x1b[?1006l\x1b[?1002l\x1b[?1000l"

_SGR_MOUSE = re.compile(r"\x1b\[<(\d+);(\d+);(\d+)([Mm])")
_CSI = re.compile(r"\x1b\[([0-9;]*)([A-Za-z~])")
_SS3 = re.compile(r"\x1bO([A-Za-z])")

_CSI_KEYS = {"A": "up", "B": "down", "C": "right", "D": "left",
             "H": "home", "F": "end", "Z": "btab"}
_TILDE_KEYS = {"1": "home", "3": "del", "4": "end", "5": "pgup",
               "6": "pgdn", "7": "home", "8": "end"}
_CTRL_KEYS = {"\r": "enter", "\n": "enter", "\t": "tab", "\x7f": "bs",
              "\x08": "bs", "\x1b": "esc", "\x03": "ctrl-c",
              "\x12": "ctrl-r", "\x13": "ctrl-s"}


def _bg(hexcolor: str) -> str:
    r, g, b = (int(hexcolor[i:i + 2], 16) for i in (1, 3, 5))
    return f"\x1b[48;2;{r};{g};{b}m"


def _chip(fill: str, text: str, bold: bool = False) -> str:
    """Text painted on a rhyme's colour, in ink that can be read on it."""
    return (f"{_bg(fill)}{_fg(ink_for(fill))}{BOLD if bold else ''}"
            f"{text}{RESET}")


class _Input:
    """Keys and mouse reports off a raw-mode terminal.

    Escape sequences arrive in pieces, so a partial one is held in the
    buffer and completed on the next read rather than being reported as a
    lone Esc — which would otherwise make a fast arrow-key repeat look
    like someone hammering Escape."""

    def __init__(self, fd: int):
        self.fd = fd
        self.buf = ""

    def _fill(self, timeout: float) -> bool:
        try:
            ready, _, _ = select.select([self.fd], [], [], timeout)
        except (OSError, ValueError):
            return False
        if not ready:
            return False
        chunk = os.read(self.fd, 4096)
        if not chunk:
            return False
        self.buf += chunk.decode("utf-8", "replace")
        return True

    def next(self, timeout: float = 0.5):
        if not self.buf and not self._fill(timeout):
            return None
        return self._parse()

    def _parse(self):
        b = self.buf
        if b[0] != "\x1b":
            self.buf = b[1:]
            return ("key", _CTRL_KEYS.get(b[0], b[0]))

        if len(b) == 1 and not self._fill(0.03):
            self.buf = ""
            return ("key", "esc")
        b = self.buf

        m = _SGR_MOUSE.match(b)
        if m:
            self.buf = b[m.end():]
            code, x, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return ("mouse", code, x, y, m.group(4) == "M")

        m = _SS3.match(b)
        if m:
            self.buf = b[m.end():]
            return ("key", _CSI_KEYS.get(m.group(1), m.group(1)))

        m = _CSI.match(b)
        if m:
            self.buf = b[m.end():]
            params, final = m.group(1), m.group(2)
            if final == "~":
                return ("key", _TILDE_KEYS.get(params.split(";")[0], "?"))
            return ("key", _CSI_KEYS.get(final, "?"))

        if b.startswith("\x1b[<") or b.startswith("\x1b["):
            # an escape sequence still arriving — wait for the rest
            if self._fill(0.03):
                return self._parse()
        self.buf = b[1:]
        return ("key", "esc")


# --------------------------------------------------------------------------
# the model: lyrics broken into paintable syllables
# --------------------------------------------------------------------------

_CUSTOM = re.compile(r"new\d+")      # id of a rhyme you invented


class Syl:
    """One paintable syllable: where it sits, what it says, and what the
    detector made of it."""

    __slots__ = ("idx", "key", "wid", "text", "orig", "line", "conf", "pos",
                 "word")

    def __init__(self, idx: int, key: str, wid: str, text: str,
                 orig: str | None, line: int, conf: float = 1.0,
                 pos: int = -1, word: str = ""):
        self.idx = idx          # position in reading order
        self.key = key          # the stable address overrides.py stores
        self.wid = wid          # which word it belongs to
        self.text = text
        self.orig = orig        # the rhyme the detector heard, or None
        self.line = line
        self.conf = conf        # how sure it is (confidence.py)
        self.pos = pos          # stream position, for the "why" readout
        self.word = word


def build_model(analysis, color_by_gid: dict) -> dict:
    """Everything the editor draws: the lyrics as syllables, and the
    rhymes available to paint with.

    `analysis` is the SCORED one (confidence.score): every candidate the
    detector found, each with a confidence. What is actually highlighted
    depends on the density setting, which the editor applies live — so
    moving it shows exactly what a render at that setting would show."""
    O.assign_gids(analysis)
    conf = getattr(analysis, "confidence", {})
    gid_of: dict = {}
    fams: list = []
    for f in analysis.families:
        # Words that identify this rhyme at a glance: rank by how often a
        # word appears in the family, preferring stressed syllables — the
        # first members alone put one long word's syllables in several
        # swatches and three rhymes end up with the same label.
        tally: Counter = Counter()
        for m in f.members:
            w = analysis.words[analysis.stream[m].word_index]
            tally[w.text] += 2 if analysis.stream[m].stress else 1
        fams.append({"sound": f.gid, "label": f.label,
                     "examples": [w for w, _n in tally.most_common(3)],
                     "words": " ".join(tally).lower()})
        for m in f.members:
            gid_of[m] = f.gid

    seen: set = set()
    for f in fams:
        key = " · ".join(f["examples"])
        if key in seen:
            f["examples"] = f["examples"][:2] + [f"({f['label']})"]
        seen.add(key)

    lines: list = []
    syls: list = []
    pos = 0
    word_start: dict = {}
    for w_i, w in enumerate(analysis.words):
        word_start[w_i] = pos
        pos += len(w.syllables)
    for ln, text in enumerate(analysis.lines):
        chunks: list = []
        cursor = 0
        for w_i, w in enumerate(analysis.words):
            if w.line != ln or not w.syllables:
                continue
            items = []
            for k, (s, e) in enumerate(w.spans):
                p = word_start[w_i] + k
                sy = Syl(len(syls), f"{w.line}:{w.start}:{k}",
                         f"{ln}-{w.start}", w.text[s:e], gid_of.get(p), ln,
                         conf.get(p, 1.0), p, w.text)
                syls.append(sy)
                items.append(sy)
            chunks.append({"pre": text[cursor:w.start], "syls": items})
            cursor = w.end
        lines.append({"chunks": chunks, "tail": text[cursor:]})

    from palette import highlighter_pool
    from video import HIGHLIGHTERS
    pool = highlighter_pool(len(fams) + 10, tuple(HIGHLIGHTERS))
    spare = [c for c in pool if c not in color_by_gid.values()][-8:]

    return {"lines": lines, "syls": syls, "families": fams,
            "color_by_sound": dict(color_by_gid), "spare": spare or [LIME],
            "words_by_sound": {f["sound"]: f["words"] for f in fams},
            "label_by_sound": {f["sound"]: " · ".join(f["examples"])
                               for f in fams}}


# --------------------------------------------------------------------------
# the editor
# --------------------------------------------------------------------------

_GONE = object()          # "there was no edit here before"

_HELP = [
    ("move",    "← →  or  h l      syllable"),
    ("",        "↑ ↓  or  k j      line"),
    ("",        "home / end        start / end of line"),
    ("",        "pgup / pgdn       a page   ·   g / G   top / bottom"),
    ("",        "[  ]              prev / next syllable of the rhyme you hold"),
    ("paint",   "space             paint this syllable"),
    ("",        "enter             paint the whole word"),
    ("",        "x  or  del        erase — this isn't a rhyme"),
    ("",        "v                 select a run, then space or x  (esc cancels)"),
    ("rhymes",  "p  or  /         every rhyme, in a list you can type at"),
    ("",        "click             a rhyme in the rack holds it"),
    ("",        "tab / shift-tab   next / previous rhyme"),
    ("",        "1 … 9             the first nine, straight off the rack"),
    ("",        "e                 dropper — hold the rhyme under the cursor"),
    ("",        "n                 start a rhyme the detector missed"),
    ("",        "c                 recolour the rhyme you're holding"),
    ("density", "−  /  +           show fewer / more rhymes (Clean … Everything)"),
    ("",        "f                 review the next borderline highlight"),
    ("",        "i                 why is this highlighted? (and how it's said)"),
    ("undo",    "u  or  z          undo        ·   y  or  ctrl-r   redo"),
    ("",        "o                 show the detector's original reading"),
    ("",        "r                 revert every correction"),
    ("mouse",   "click             paint   ·   drag   paint a run"),
    ("",        "right-click       erase   ·   double-click   the whole word"),
    ("",        "wheel             scroll   ·   click the density line to change it"),
    ("finish",  "s                 save       ·   q   done (asks first)"),
]


class Editor:
    """The editing session: state, key handling, and the screen it draws.

    Deliberately free of terminal I/O — `frame()` returns the screen as a
    string and `handle()` takes one event — so the whole thing can be
    driven by a test without a terminal attached."""

    def __init__(self, src: pathlib.Path, analysis, color_by_gid: dict,
                 title: str, level: float | None = None):
        self.src = src
        self.title = title
        self.analysis = analysis
        state = O.resolve_legacy(O.load(src), analysis)
        self.edits: dict = dict(state["edits"])
        # colours you chose or saw — pinned on save so a render wears
        # exactly the colours this screen showed
        self.colors: dict = dict(state["colors"])
        self.names: dict = dict(state["names"])
        self.level: float = (state["sensitivity"]
                             if state["sensitivity"] is not None
                             else level if level is not None
                             else C.DEFAULT_LEVEL)
        self.reviewed: set = set()          # borderline syllables visited

        m = build_model(analysis, color_by_gid)
        self.lines = m["lines"]
        self.syls: list = m["syls"]
        self.fams: list = m["families"]
        self.color_by_sound: dict = m["color_by_sound"]
        self.label_by_sound: dict = m["label_by_sound"]
        self.words_by_sound: dict = m["words_by_sound"]
        self.spare: list = m["spare"]

        # Rhymes you invented in an earlier session come back as "newN".
        # One that still has syllables is already a family in the rack
        # (build_model keys it by that id), so only the empty ones — a
        # rhyme started and never painted — are listed separately. The
        # counter still spans both, so a new id is never reused.
        mine = sorted({k for k in [*self.colors, *self.edits.values()]
                       if _CUSTOM.fullmatch(k)}, key=lambda s: int(s[3:]))
        self.customs: list = list(mine)
        known = {f["sound"] for f in self.fams}
        for v in dict.fromkeys(self.edits.values()):
            if v != O.NONE and v not in known and v not in self.customs:
                self.customs.append(v)
                self.names.setdefault(v, f"rhyme {v}")
        self.custom_n = max((int(s[3:]) for s in mine), default=0)
        self.spare_at = 0

        # palette() asks which rhyme you are holding, so there has to be an
        # answer before the first time it is called
        self.tool: str = O.NONE
        pal = self.palette()
        self.tool = pal[0][0] if pal else O.NONE
        self.cur = 0
        self.top = 0
        self.anchor: int | None = None      # visual selection start
        self.show_orig = False
        self.help = False
        self.picker: dict | None = None     # {"sel", "top", "filter"}
        self.rack_hits: list = []           # (row, c0, c1, sound) — clickable
        self.density_hits: list = []        # (row, c0, c1, level or "review")
        self.picker_hits: list = []         # (row, palette index)
        self.msg = ("click or drag to paint · right-click erases · "
                    "−/+ changes how much is shown · ? every key")
        self.dirty = False
        self.quit = False
        self.confirm: tuple | None = None   # (question, action)
        self.undo: list = []
        self.redo: list = []

        self._w = 0
        self.rows: list = []
        self.pos: dict = {}
        self.body_x, self.body_y, self.body_h = 2, 6, 10
        self.drag: list | None = None       # the batch a mouse drag is filling
        self.drag_sound = O.NONE
        self.last_click = (0.0, -1)
        self.saved_n = 0                    # corrections on disk

    # -- rhymes -----------------------------------------------------------

    def palette(self) -> list:
        """Every rhyme you can paint with, rack order: the ones the current
        density shows (or that your corrections use), then yours."""
        live = {sy.orig for sy in self.syls if self.reading(sy)}
        live |= {v for v in self.edits.values() if v != O.NONE}
        return ([(f["sound"], " · ".join(f["examples"])) for f in self.fams
                 if f["sound"] in live or f["sound"] == self.tool]
                + [(s, self.names.get(s, "your rhyme " + s[3:]))
                   for s in self.customs])

    def color(self, sound: str | None) -> str:
        if not sound:
            return PLAIN
        return self.colors.get(sound) or self.color_by_sound.get(sound, PLAIN)

    def label(self, sound: str | None) -> str:
        if not sound or sound == O.NONE:
            return "erase"
        if sound in self.names:
            return self.names[sound]
        return self.label_by_sound.get(sound, "your rhyme " + sound[3:]
                                       if sound.startswith("new") else sound)

    def reading(self, sy: Syl) -> str | None:
        """What the detector shows on this syllable at the current density."""
        return sy.orig if (sy.orig and sy.conf >= self.level) else None

    def shown(self, sy: Syl) -> str | None:
        """The rhyme this syllable is wearing right now."""
        if self.show_orig:
            return self.reading(sy)
        v = self.edits.get(sy.key, _GONE)
        if v is _GONE:
            return self.reading(sy)
        return None if v == O.NONE else v

    # -- density and the evidence behind a highlight ----------------------

    _ORDER = ("clean", "balanced", "rich", "everything")   # sparse → dense

    def level_name(self) -> str:
        for name, v in C.LEVELS.items():
            if abs(v - self.level) < 1e-6:
                return name.capitalize()
        return f"custom {self.level:.0%}"

    def set_level(self, level: float) -> None:
        self.level = max(0.0, min(0.99, level))
        self.dirty = True
        lit = sum(1 for sy in self.syls if self.shown(sy))
        self.msg = (f"density {self.level_name()} · {lit} syllables "
                    f"highlighted · your corrections still apply")

    def step_level(self, d: int) -> None:
        """d = +1 shows more rhymes, −1 fewer."""
        vals = [C.LEVELS[n] for n in self._ORDER]
        if d > 0:
            nxt = [v for v in vals if v < self.level - 1e-9]
            self.set_level(nxt[0] if nxt else vals[-1])
        else:
            nxt = [v for v in reversed(vals) if v > self.level + 1e-9]
            self.set_level(nxt[0] if nxt else vals[0])

    def why(self, sy: Syl, n: int = 1) -> list:
        ev = getattr(self.analysis, "evidence", {}).get(sy.pos)
        w = getattr(self.analysis, "weights", None)
        if not ev or not w:
            return []
        return C.reasons(ev, w)[:n]

    def unsure(self) -> list:
        """Borderline highlights: the detector could go either way."""
        return [sy for sy in self.syls
                if sy.orig and abs(sy.conf - self.level) <= 0.25
                and sy.key not in self.edits and sy.key not in self.reviewed]

    def next_unsure(self) -> None:
        todo = self.unsure()
        if not todo:
            self.msg = "nothing borderline left at this density"
            return
        after = [sy for sy in todo if sy.idx > self.cur] or todo
        sy = after[0]
        self.reviewed.add(sy.key)
        self.cur = sy.idx
        self.anchor = None
        self.tool = sy.orig
        on = self.reading(sy) is not None
        reason = (self.why(sy) or ["no strong evidence either way"])[0]
        self.msg = (f"{'shown' if on else 'hidden'} · {sy.conf:.0%} sure · "
                    f"{reason} — space {'keeps' if on else 'adds'} it · "
                    f"x erases · f next ({len(todo) - 1} left)")

    def info(self) -> None:
        if not self.syls:
            return
        from rhyme_engine import phones_for
        sy = self.syls[self.cur]
        ph = phones_for(sy.word)[0] or "?"
        parts = [f"“{sy.word}” is said {ph}"]
        if sy.orig:
            parts.append(f"{sy.conf:.0%} sure")
            parts += self.why(sy, 3)
        else:
            parts.append("the detector heard no rhyme here")
        self.msg = " · ".join(parts)

    # -- editing ----------------------------------------------------------

    def _put(self, idx: int, sound: str, batch: list) -> None:
        """Paint one syllable, recording what was there for undo.

        Painting a syllable back to what the detector already said is not
        a correction — it drops the override instead of freezing the
        detector's current reading into the file. That is what keeps the
        correction count honest, and what lets an engine improvement
        still reach a syllable you once touched and then put back."""
        sy = self.syls[idx]
        want = None if sound == (self.reading(sy) or O.NONE) else sound
        prev = self.edits.get(sy.key, _GONE)
        if (prev if prev is not _GONE else None) == want:
            return
        batch.append((sy.key, prev))
        if want is None:
            self.edits.pop(sy.key, None)
        else:
            self.edits[sy.key] = want

    def _commit(self, batch: list) -> int:
        if batch:
            self.undo.append(batch)
            self.redo.clear()
            self.dirty = True
        return len(batch)

    def _step(self, frm: list, to: list) -> bool:
        b = frm.pop() if frm else None
        if not b:
            return False
        inv = []
        for key, prev in b:
            inv.append((key, self.edits.get(key, _GONE)))
            if prev is _GONE:
                self.edits.pop(key, None)
            else:
                self.edits[key] = prev
        to.append(inv)
        self.dirty = True
        return True

    def _targets(self) -> list:
        """What a paint key applies to: the selection, or the cursor."""
        if self.anchor is None:
            return [self.cur]
        lo, hi = sorted((self.anchor, self.cur))
        return list(range(lo, hi + 1))

    def paint(self, sound: str) -> None:
        if not self.syls:
            return
        batch: list = []
        for i in self._targets():
            self._put(i, sound, batch)
        n = self._commit(batch)
        if self.anchor is not None:
            self.msg = (f"{n} syllable{'' if n == 1 else 's'} → "
                        f"{self.label(sound)}")
            self.anchor = None
        else:
            self.msg = ("erased" if sound == O.NONE
                        else f"painted {self.label(sound)}")

    def paint_word(self, idx: int | None = None) -> None:
        if not self.syls:
            return
        wid = self.syls[self.cur if idx is None else idx].wid
        batch: list = []
        for sy in self.syls:
            if sy.wid == wid:
                self._put(sy.idx, self.tool, batch)
        self._commit(batch)
        self.anchor = None
        self.msg = "painted the whole word"

    def new_rhyme(self) -> None:
        self.custom_n += 1
        sound = f"new{self.custom_n}"
        self.colors[sound] = self.spare[self.spare_at % len(self.spare)]
        self.names[sound] = f"your rhyme {self.custom_n}"
        self.spare_at += 1
        self.customs.append(sound)
        self.tool = sound
        self.dirty = True
        self.msg = ("new rhyme ready — paint the syllables it covers "
                    "(it needs at least two)")

    def recolour(self) -> None:
        if self.tool == O.NONE:
            return
        self.colors[self.tool] = self.spare[self.spare_at % len(self.spare)]
        self.spare_at += 1
        self.dirty = True
        self.msg = f"recoloured {self.label(self.tool)}"

    def revert(self) -> None:
        batch = [(k, v) for k, v in self.edits.items()]
        self.edits.clear()
        self._commit(batch)
        self.msg = "reverted every correction — press s to keep it that way"

    def save(self) -> int:
        """Write the corrections beside the audio. Returns how many.

        Everything that decides what a render looks like goes in: your
        edits, the density, and the colour of every rhyme on screen —
        pinned, so the video cannot recolour a rhyme behind your back."""
        on_screen = {self.shown(sy) for sy in self.syls} - {None}
        used = on_screen | {v for v in self.edits.values() if v != O.NONE}
        state = {
            "edits": dict(self.edits),
            "colors": {g: self.color(g) for g in used},
            "names": {g: n for g, n in self.names.items() if g in used},
            "sensitivity": self.level,
        }
        try:
            O.save(self.src, state, self.analysis.lines)
        except OSError as exc:
            # never let a failed write look like a successful one
            self.msg = f"COULD NOT SAVE — {exc.strerror or exc}"
            return -1
        self.dirty = False
        n = self.saved_n = len(self.edits)
        self.msg = (f"saved · density {self.level_name()}"
                    + (f" · {n} correction{'' if n == 1 else 's'}" if n else ""))
        # A rhyme of one syllable is dropped on the way into a render;
        # say so now rather than letting it quietly vanish later.
        worn: Counter = Counter(self.shown(sy) for sy in self.syls)
        lonely = [g for g in used if worn.get(g, 0) == 1]
        if lonely:
            self.msg += (f" · heads up: {self.label(lonely[0])} has only "
                         f"one syllable, and a rhyme needs two")
        return n

    # -- layout -----------------------------------------------------------

    def layout(self, width: int) -> None:
        """Wrap the lyrics to the terminal, word by word, and remember
        where every syllable landed — the same map answers both arrow
        keys and mouse clicks."""
        self._w = width
        body_w = max(24, width - self.body_x - 2)
        rows: list = []
        pos: dict = {}
        for ln, line in enumerate(self.lines):
            row = {"ln": ln, "parts": [], "syl": []}
            col = 0
            for ch in line["chunks"]:
                need = len(ch["pre"]) + sum(len(s.text) for s in ch["syls"])
                if col and col + need > body_w:
                    rows.append(row)
                    row = {"ln": ln, "parts": [], "syl": []}
                    col = 0
                    pre = ""        # a wrapped row never opens with a gap
                else:
                    pre = ch["pre"]
                if pre:
                    row["parts"].append((pre, None))
                    col += len(pre)
                for sy in ch["syls"]:
                    row["parts"].append((sy.text, sy))
                    pos[sy.idx] = (len(rows), col)
                    row["syl"].append((col, col + len(sy.text), sy.idx))
                    col += len(sy.text)
            if line["tail"].strip() and col + len(line["tail"]) <= body_w:
                row["parts"].append((line["tail"], None))
            rows.append(row)
        self.rows, self.pos = rows, pos

    def _ensure_visible(self) -> None:
        r = self.pos.get(self.cur, (0, 0))[0]
        if r < self.top:
            self.top = r
        elif r >= self.top + self.body_h:
            self.top = r - self.body_h + 1
        self.top = max(0, min(self.top, max(0, len(self.rows) - self.body_h)))

    def _move(self, d: int) -> None:
        if self.syls:
            self.cur = max(0, min(len(self.syls) - 1, self.cur + d))

    def _move_row(self, d: int) -> None:
        """Up/down moves to the nearest syllable in the row above or
        below — blank rows are stepped over, not landed on."""
        if not self.syls:
            return
        r, c = self.pos.get(self.cur, (0, 0))
        k = r + d
        while 0 <= k < len(self.rows):
            cells = self.rows[k]["syl"]
            if cells:
                self.cur = min(cells, key=lambda t: abs(t[0] - c))[2]
                return
            k += d

    def _row_edge(self, last: bool) -> None:
        if not self.rows:
            return
        cells = self.rows[self.pos.get(self.cur, (0, 0))[0]]["syl"]
        if cells:
            self.cur = cells[-1][2] if last else cells[0][2]

    def _page(self, d: int) -> None:
        if not self.rows:
            return
        r = self.pos.get(self.cur, (0, 0))[0]
        self.top = max(0, min(r + d * self.body_h,
                              max(0, len(self.rows) - self.body_h)))
        k = max(0, min(len(self.rows) - 1, r + d * self.body_h))
        while 0 <= k < len(self.rows) and not self.rows[k]["syl"]:
            k += 1 if d > 0 else -1
        if 0 <= k < len(self.rows) and self.rows[k]["syl"]:
            self.cur = self.rows[k]["syl"][0][2]

    def _family_jump(self, d: int) -> None:
        """Walk the rhyme you're holding — the fast way to check whether
        a family really belongs together."""
        n = len(self.syls)
        for step in range(1, n + 1):
            i = (self.cur + d * step) % n
            if self.shown(self.syls[i]) == self.tool:
                self.cur = i
                self.msg = f"{self.label(self.tool)}"
                return
        self.msg = "nothing else wears that rhyme yet"

    def _pick(self, i: int) -> None:
        pal = self.palette()
        if 0 <= i < len(pal):
            self.tool = pal[i][0]
            self.msg = f"holding {self.label(self.tool)}"

    def _cycle(self, d: int) -> None:
        pal = [s for s, _l in self.palette()]
        if not pal:
            return
        i = pal.index(self.tool) if self.tool in pal else -d % len(pal)
        self.tool = pal[(i + d) % len(pal)]
        self.msg = f"holding {self.label(self.tool)}"

    # -- the rhyme picker -------------------------------------------------
    #
    # Nine rhymes fit on the number row and a real song has far more than
    # nine — twenty-odd is ordinary. Everything past the ninth used to be
    # reachable only by pressing tab until it came round, which is why
    # rhyme 14 was, in practice, unfixable. This lists every rhyme at
    # once, filtered by typing, and is the same list whether a song has
    # three families or fifty.

    def _picker_open(self) -> None:
        pal = self.palette()
        held = next((i for i, (s, _l) in enumerate(pal) if s == self.tool), 0)
        self.picker = {"sel": held, "top": 0, "filter": ""}
        self.msg = "type to narrow · ↑↓ choose · enter picks · esc closes"

    def _picker_rows(self) -> list:
        """Every rhyme that matches what you've typed, with its number,
        its words and how many syllables currently wear it."""
        worn: Counter = Counter()
        for sy in self.syls:
            sound = self.shown(sy)
            if sound:
                worn[sound] += 1
        q = (self.picker or {}).get("filter", "").lower()
        out = []
        for i, (sound, lab) in enumerate(self.palette()):
            hay = lab.lower() + " " + self.words_by_sound.get(sound, "")
            if q and q not in hay and q != str(i + 1):
                continue
            out.append((i, sound, lab, worn.get(sound, 0)))
        return out

    def _picker_body(self, body_w: int) -> list:
        rows = self._picker_rows()
        pk = self.picker
        pk["sel"] = max(0, min(pk["sel"], len(rows) - 1)) if rows else 0
        h = max(3, self.body_h - 2)
        if pk["sel"] < pk["top"]:
            pk["top"] = pk["sel"]
        elif pk["sel"] >= pk["top"] + h:
            pk["top"] = pk["sel"] - h + 1
        pk["top"] = max(0, min(pk["top"], max(0, len(rows) - h)))

        typed = pk["filter"]
        head = (f"  {DIM}pick a rhyme{RESET}  {BOLD}{typed}{RESET}"
                f"{DIM}▏{RESET}" if typed else
                f"  {DIM}pick a rhyme — type to narrow the list{RESET}")
        out = [head, ""]
        self.picker_hits = []
        if not rows:
            out.append(f"  {DIM}nothing matches “{typed}”{RESET}")
            return out[:self.body_h] + [""] * max(0, self.body_h - len(out))

        for k in range(pk["top"], min(len(rows), pk["top"] + h)):
            i, sound, lab, n = rows[k]
            on = k == pk["sel"]
            num = f"{i + 1:>3}"
            swatch = _chip(self.color(sound), "  ")
            count = f"{n} syllable{'' if n == 1 else 's'}" if n else "unused"
            text = self._fit(lab, max(8, body_w - 22))
            line = (f"{BOLD if on else ''}{'▸' if on else ' '} {num}{RESET} "
                    f"{swatch} {BOLD if on else ''}{text}{RESET}"
                    f"{DIM}{' ' * max(1, body_w - 10 - len(text) - len(count))}"
                    f"{count}{RESET}")
            self.picker_hits.append((len(out), i))
            out.append("  " + line)
        if len(rows) > h:
            out.append(f"  {DIM}{len(rows)} rhymes — keep typing to "
                       f"narrow{RESET}")
        return out[:self.body_h] + [""] * max(0, self.body_h - len(out))

    def _picker_key(self, k: str) -> None:
        pk = self.picker
        rows = self._picker_rows()
        if k in ("esc", "ctrl-c"):
            self.picker = None
            self.msg = ""
        elif k in ("up", "btab"):
            pk["sel"] -= 1
        elif k in ("down", "tab"):
            pk["sel"] += 1
        elif k == "pgup":
            pk["sel"] -= max(1, self.body_h - 3)
        elif k == "pgdn":
            pk["sel"] += max(1, self.body_h - 3)
        elif k == "home":
            pk["sel"] = 0
        elif k == "end":
            pk["sel"] = len(rows) - 1
        elif k in ("enter", " "):
            if rows:
                self._pick(rows[max(0, min(pk["sel"], len(rows) - 1))][0])
            self.picker = None
        elif k == "bs":
            pk["filter"] = pk["filter"][:-1]
            pk["sel"] = 0
        elif len(k) == 1 and k.isprintable():
            pk["filter"] += k
            pk["sel"] = 0
        if self.picker:
            pk["sel"] = max(0, min(pk["sel"], max(0, len(rows) - 1)))

    # -- drawing ----------------------------------------------------------

    def _cell(self, sy: Syl, text: str) -> str:
        """One syllable, wearing its rhyme.

        Three things have to be legible at once and they use three
        different channels, so they never collide: the rhyme is the
        COLOUR, a correction you made is the UNDERLINE, and where you are
        is the BLOCK — the cursor inverts the cell, which keeps the
        rhyme's colour visible instead of painting over it."""
        sound = self.shown(sy)
        fill = self.color(sound) if sound else None
        lo, hi = ((-1, -2) if self.anchor is None
                  else sorted((self.anchor, self.cur)))
        under = UNDER if (not self.show_orig and sy.key in self.edits) else ""
        if sy.idx == self.cur or lo <= sy.idx <= hi:
            head = (f"{_fg(fill)}{_bg(ink_for(fill))}" if fill
                    else f"{_fg('#0A0A0A')}{_bg(LIME)}")
            bold = BOLD if sy.idx == self.cur else ""
            return f"{head}{bold}{under}{text}{RESET}"
        if fill:
            return f"{_bg(fill)}{_fg(ink_for(fill))}{under}{text}{RESET}"
        return f"{under}{text}{RESET}" if under else text

    def _rack(self, width: int) -> list:
        """The rhymes you can paint with. Long songs have more families
        than fit, so the rack scrolls with the one you're holding rather
        than silently hiding it."""
        pal = self.palette()
        if not pal:
            return [f"  {DIM}no rhymes detected yet — press "
                    f"{RESET}n{DIM} to start one{RESET}"]
        held = next((i for i, (s, _l) in enumerate(pal) if s == self.tool), -1)
        chips = []
        for i, (sound, lab) in enumerate(pal):
            lab = lab if len(lab) <= 24 else lab[:23] + "…"
            # every rhyme shows its real number, including past the ninth.
            # It used to show "·" there, which hid the one thing that
            # identifies a rhyme you cannot reach from the number row —
            # the number you then type into the picker (p).
            num = str(i + 1)
            mark = "▸" if i == held else " "
            chips.append((sound, f"{mark}{num} {lab} ", i == held))

        packed: list = []
        cur: list = []
        w = 0
        for chip in chips:
            need = len(chip[1]) + 1
            if cur and w + need > width - 4:
                packed.append(cur)
                cur, w = [], 0
            cur.append(chip)
            w += need
        if cur:
            packed.append(cur)

        limit = 3
        more = ""
        if len(packed) > limit:
            hrow = next((k for k, row in enumerate(packed)
                         if any(c[2] for c in row)), 0)
            start = min(max(0, hrow - 1), len(packed) - limit)
            more = (f" {DIM}(+{len(packed) - limit} more — "
                    f"{RESET}p{DIM} lists them all){RESET}")
            packed = packed[start:start + limit]

        # Remember where every chip landed so the mouse can pick one.
        # frame() always draws the rack as the third line onward, so a
        # chip on rack line k sits on screen row k + 3 (1-based).
        self.rack_hits = []
        out = []
        for k, row in enumerate(packed):
            col = 2
            for sound, t, _on in row:
                self.rack_hits.append((k + 2, col, col + len(t), sound))
                col += len(t) + 1
            line = " ".join(_chip(self.color(s), t, bold=on)
                            for s, t, on in row)
            out.append("  " + line + (more if k == len(packed) - 1 else ""))
        return out

    def _fit(self, text: str, width: int) -> str:
        return text if len(text) <= width else text[:max(0, width - 1)] + "…"

    def _split(self, left: str, right: str, width: int,
               lstyle: str = "", rstyle: str = DIM) -> str:
        """A line with something at each end, sized in plain characters
        so the colour codes never throw the arithmetic off."""
        left = self._fit(left, max(4, width - len(right) - 3))
        gap = max(1, width - len(left) - len(right) - 2)
        return (f"  {lstyle}{left}{RESET if lstyle else ''}"
                f"{' ' * gap}{rstyle}{right}{RESET}")

    def _here(self) -> str:
        if not self.syls:
            return "no lyrics"
        sy = self.syls[self.cur]
        sound = self.shown(sy)
        out = f"“{sy.text}” · {self.label(sound) if sound else 'no rhyme'}"
        if sy.orig:
            out += f" · {sy.conf:.0%} sure"
            why = self.why(sy)
            if why:
                out += f" — {why[0]}"
        return out

    def frame(self, width: int, height: int) -> str:
        if width != self._w:
            self.layout(width)
        width = max(40, width)
        rack = self._rack(width)
        self.body_y = 4 + len(rack)
        self.body_h = max(3, height - (6 + len(rack)))
        self._ensure_visible()

        n = len(self.edits)
        state = (f"{n} correction{'' if n == 1 else 's'}" if n
                 else "no corrections yet")
        # "unsaved" alone left the saved case looking identical to a
        # session that had never saved at all, which is exactly when you
        # go and press q wondering whether it took
        state += (" · unsaved" if self.dirty
                  else " · saved" if self.saved_n or n else "")
        rule = f"{DIM}{'─' * (width - 4)}{RESET}"

        # the rhyme you're holding shows as a colour chip, whose escape
        # codes no plain-width calculation can see — so this one line is
        # sized by hand rather than through _split
        text = f" {self.label(self.tool)} " if self.tool != O.NONE else " erase "
        chip = (_chip(self.color(self.tool), text) if self.tool != O.NONE
                else f"{REV}{text}{RESET}")
        lead, right = "painting  ", "? keys   s save   q done"
        gap = max(1, width - len(lead) - len(text) - len(right) - 2)
        out = [self._split(f"♪ {self.title}", state, width, lstyle=BOLD),
               f"  {lead}{chip}{' ' * gap}{DIM}{right}{RESET}"]
        if self.show_orig:
            out[1] = (f"  {_chip(LIME, ' showing the detector’s original ')}"
                      f"  {DIM}press o to come back to your edits{RESET}")

        out += rack
        out.append(self._density_line(len(out), width))
        out.append("  " + rule)

        body_w = max(24, width - self.body_x - 2)
        if self.help:
            out += self._help_body(body_w)
        elif self.picker:
            out += self._picker_body(body_w)
        else:
            for k in range(self.body_h):
                ri = self.top + k
                out.append("  " + (self._row(ri, body_w)
                                   if ri < len(self.rows) else ""))
        out.append("  " + rule)
        if self.confirm:
            out.append(self._confirm_bar(width))
        else:
            out.append(self._split(self.msg or self._here(),
                                   self._footer_keys(), width,
                                   lstyle="" if self.msg else DIM))

        buf = ["\x1b[H"]
        for k, line in enumerate(out[:height]):
            buf.append("\x1b[K" + line)
            if k < min(len(out), height) - 1:
                buf.append("\r\n")
        buf.append("\x1b[J")
        return "".join(buf)

    def _density_line(self, row: int, width: int) -> str:
        """How much the detector shows, as four clickable stops, plus the
        review queue. The one control that decides how busy the map is —
        so it sits in plain sight rather than behind a key."""
        self.density_hits = []
        col = 2
        text = "density "
        parts = [f"{DIM}{text}{RESET}"]
        col += len(text)
        for name in self._ORDER:
            lab = f" {name.capitalize()} "
            on = abs(C.LEVELS[name] - self.level) < 1e-6
            parts.append(_chip(LIME, lab, bold=True) if on
                         else f"{DIM}{lab}{RESET}")
            self.density_hits.append((row, col, col + len(lab),
                                      C.LEVELS[name]))
            col += len(lab)
        if not any(abs(v - self.level) < 1e-6 for v in C.LEVELS.values()):
            lab = f" {self.level_name()} "
            parts.append(_chip(LIME, lab, bold=True))
            col += len(lab)
        hint = "  −/+ "
        parts.append(f"{DIM}{hint}{RESET}")
        col += len(hint)
        n = len(self.unsure())
        if n:
            lab = f" review {n} borderline ▸ "
            parts.append("  " + f"{REV}{lab}{RESET}")
            col += 2
            self.density_hits.append((row, col, col + len(lab), "review"))
        return "  " + "".join(parts)

    def _confirm_bar(self, width: int) -> str:
        """The question, spelled out, with what each answer does.

        This used to be stored and never drawn: the footer said only
        "y / n" and the question itself was invisible, so the one prompt
        standing between you and losing a session's corrections was
        unreadable. Answers are spelled out for the same reason."""
        q, _act, hint = self.confirm[0], self.confirm[1], self.confirm[2]
        q = self._fit(q, max(10, width - len(hint) - 6))
        gap = max(1, width - len(q) - len(hint) - 4)
        return (f"  {_chip(LIME, f' {q} ', bold=True)}"
                f"{' ' * gap}{BOLD}{hint}{RESET}  ")

    def _footer_keys(self) -> str:
        if self.picker is not None:
            return "↑↓ choose · enter picks · esc closes"
        if self.anchor is not None:
            return "space paint · x erase · esc cancel"
        return "−/+ density · f review · i why · p rhymes · u undo"

    def _row(self, ri: int, body_w: int) -> str:
        parts = []
        col = 0
        for text, sy in self.rows[ri]["parts"]:
            if col >= body_w:
                break
            t = text[:body_w - col]
            parts.append(self._cell(sy, t) if sy is not None else t)
            col += len(t)
        return "".join(parts)

    def _help_body(self, body_w: int) -> list:
        out = []
        for head, text in _HELP:
            out.append(f"  {HEAD}{head:<7}{RESET}{DIM}│{RESET} {text}"
                       if head else f"  {'':<7}{DIM}│{RESET} {text}")
        out.append("")
        out.append(f"  {DIM}any key closes this{RESET}")
        return out[:self.body_h] + [""] * max(0, self.body_h - len(out))


    # -- events -----------------------------------------------------------

    def handle(self, ev: tuple) -> None:
        if ev[0] == "key":
            self._key(ev[1])
        elif ev[0] == "mouse":
            self._mouse(*ev[1:])

    def _key(self, k: str) -> None:
        if self.confirm:
            _q, act, _hint, default = self.confirm
            # `enter` takes the safe answer, which differs per question:
            # saving on the way out, keeping your work on a revert.
            if k in ("y", "Y") or (default and k in ("enter", "s")):
                self.confirm = None
                act(True)
            elif k in ("n", "N") or (not default and k == "enter"):
                self.confirm = None
                act(False)
            elif k in ("esc", "ctrl-c"):
                self.confirm = None
                self.msg = "never mind — nothing was changed"
            return
        if self.help:
            self.help = False
            return
        if self.picker is not None:
            self._picker_key(k)
            return

        if k in ("left", "h"):
            self._move(-1)
        elif k in ("right", "l"):
            self._move(1)
        elif k in ("up", "k"):
            self._move_row(-1)
        elif k in ("down", "j"):
            self._move_row(1)
        elif k == "home":
            self._row_edge(False)
        elif k == "end":
            self._row_edge(True)
        elif k == "pgup":
            self._page(-1)
        elif k == "pgdn":
            self._page(1)
        elif k == "g":
            self.cur = 0
        elif k == "G":
            self.cur = max(0, len(self.syls) - 1)
        elif k == "[":
            self._family_jump(-1)
        elif k == "]":
            self._family_jump(1)

        elif k == " ":
            self.paint(self.tool)
        elif k == "enter":
            self.paint_word()
        elif k in ("x", "del", "bs"):
            self.paint(O.NONE)
        elif k == "v":
            if self.anchor is None:
                self.anchor = self.cur
                self.msg = "selecting — move, then space paints or x erases"
            else:
                self.anchor = None
                self.msg = "selection dropped"
        elif k == "esc":
            if self.anchor is not None:
                self.anchor = None
                self.msg = "selection dropped"

        elif k == "tab":
            self._cycle(1)
        elif k == "btab":
            self._cycle(-1)
        elif k in "123456789":
            self._pick(int(k) - 1)
        elif k in ("p", "/"):
            self._picker_open()
        elif k == "e":
            sound = self.shown(self.syls[self.cur]) if self.syls else None
            if sound:
                self.tool = sound
                self.msg = f"holding {self.label(sound)}"
            else:
                self.msg = "that syllable has no rhyme to copy"
        elif k in ("-", "_"):
            self.step_level(-1)
        elif k in ("+", "="):
            self.step_level(1)
        elif k == "f":
            self.next_unsure()
        elif k == "i":
            self.info()
        elif k == "n":
            self.new_rhyme()
        elif k == "c":
            self.recolour()

        elif k in ("u", "z"):
            self.msg = ("undone" if self._step(self.undo, self.redo)
                        else "nothing to undo")
        elif k in ("y", "ctrl-r"):
            self.msg = ("redone" if self._step(self.redo, self.undo)
                        else "nothing to redo")
        elif k == "o":
            self.show_orig = not self.show_orig
            self.msg = ("this is the detector's reading — your edits are "
                        "safe" if self.show_orig else "")
        elif k == "r":
            if self.edits:
                self.confirm = (
                    f"throw away all {len(self.edits)} corrections?",
                    lambda yes: self.revert() if yes else None,
                    "y  throw away   ·   n  keep them", False)
            else:
                self.msg = "nothing to revert"

        elif k in ("s", "ctrl-s"):
            self.save()
        elif k == "?":
            self.help = True
        elif k in ("q", "ctrl-c"):
            self._leave()

        if k not in ("v", "esc") and k in ("left", "right", "up", "down",
                                           "h", "l", "j", "k", "home",
                                           "end", "pgup", "pgdn", "g", "G"):
            self.msg = ""
        self._ensure_visible()

    def _leave(self) -> None:
        if not self.dirty:
            self.quit = True
            return

        def answer(yes: bool) -> None:
            if yes:
                self.save()
            self.quit = True

        n = len(self.edits)
        self.confirm = (
            f"save your {n} correction{'' if n == 1 else 's'} before leaving?",
            answer, "y / enter  save & close   ·   n  discard   ·   esc  stay",
            True)

    # -- mouse ------------------------------------------------------------

    def hit(self, x: int, y: int) -> int | None:
        """Which syllable is under this cell, if any."""
        r = (y - 1) - self.body_y + self.top
        c = (x - 1) - self.body_x
        if (self.help or self.picker is not None or self.confirm
                or r < self.top or r < 0):
            return None
        if r >= len(self.rows) or r >= self.top + self.body_h:
            return None
        for c0, c1, idx in self.rows[r]["syl"]:
            if c0 <= c < c1:
                return idx
        return None

    def _rack_click(self, x: int, y: int) -> bool:
        """Clicking a rhyme in the rack holds it — the shortest route to
        the ones the number row cannot reach."""
        pal = [sd for sd, _l in self.palette()]
        for row, c0, c1, sound in self.rack_hits:
            if y - 1 == row and c0 <= x - 1 < c1:
                if sound in pal:
                    self._pick(pal.index(sound))
                return True
        return False

    def _mouse(self, code: int, x: int, y: int, press: bool) -> None:
        if code & 64:                                   # wheel
            step = -3 if (code & 1) == 0 else 3
            if self.picker is not None:
                self.picker["sel"] = max(0, self.picker["sel"] + step)
                return
            self.top = max(0, min(self.top + step,
                                  max(0, len(self.rows) - self.body_h)))
            return
        if not press:                                   # button released
            if self.drag is not None:
                self._commit(self.drag)
                self.drag = None
            return
        if self.help:
            self.help = False
            return
        if self.picker is not None:                     # click a rhyme
            for row, i in self.picker_hits:
                if y - 1 - self.body_y == row:
                    self._pick(i)
                    self.picker = None
                    return
            return
        if self._rack_click(x, y):                      # click a swatch
            return
        for row, c0, c1, what in self.density_hits:     # click a density
            if y - 1 == row and c0 <= x - 1 < c1:
                if what == "review":
                    self.next_unsure()
                else:
                    self.set_level(what)
                return
        idx = self.hit(x, y)
        if idx is None:
            return
        if code & 32:                                   # dragging
            if self.drag is not None:
                self.cur = idx
                self._put(idx, self.drag_sound, self.drag)
            return

        now = time.monotonic()
        double = (now - self.last_click[0] < 0.45
                  and self.syls[self.last_click[1]].wid == self.syls[idx].wid)
        self.last_click = (now, idx)
        self.cur = idx
        self.anchor = None
        if double:
            self.paint_word(idx)
            return
        self.drag_sound = O.NONE if (code & 3) == 2 else self.tool
        self.drag = []
        self._put(idx, self.drag_sound, self.drag)
        self.msg = ("erased" if self.drag_sound == O.NONE
                    else f"painted {self.label(self.drag_sound)}"
                         " — drag to keep painting")


# --------------------------------------------------------------------------
# running it on a real terminal
# --------------------------------------------------------------------------

def open_session(src: pathlib.Path, text: str, title: str,
                 strict: bool = False,
                 level: float | None = None,
                 heard: dict | None = None) -> Editor:
    """Build an editing session for a song, with the colours a render
    would use right now — the ones on screen are the ones you get."""
    from palette import highlighter_pool
    from video import HIGHLIGHTERS, _palette

    state = O.load(src)
    if state["sensitivity"] is not None:
        level = state["sensitivity"]
    elif level is None:
        level = C.DEFAULT_LEVEL
    union = C.score(text, strict=strict, heard=heard)
    O.assign_gids(union)
    shown = O.apply(C.score(text, strict=strict, heard=heard), state, level)
    now = _palette(shown)
    color_by_gid = {f.gid: now[f.fid] for f in shown.families}
    # rhymes hidden at this density still need a colour of their own for
    # when you turn it up — never one a visible rhyme is already wearing
    pool = highlighter_pool(len(union.families) + 12, tuple(HIGHLIGHTERS))
    free = [c for c in pool if c not in set(color_by_gid.values())]
    for f in union.families:
        if f.gid not in color_by_gid:
            color_by_gid[f.gid] = free.pop(0) if free else PLAIN
    return Editor(src, union, color_by_gid, title, level)


def edit(src: pathlib.Path, text: str, title: str, strict: bool = False,
         level: float | None = None, heard: dict | None = None) -> int:
    """Open the rhyme editor on this song and run it until you're done.

    Returns how many corrections are saved on disk when it closes (0 if
    you saved nothing), and leaves the terminal exactly as it found it —
    alternate screen off, cursor back, mouse reporting off — including
    when something raises on the way out."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise RuntimeError("the rhyme editor needs an interactive terminal")

    ed = open_session(src, text, title, strict=strict, level=level,
                      heard=heard)
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    resized = [False]

    def winch(*_a):
        resized[0] = True

    prev_winch = signal.signal(signal.SIGWINCH, winch)
    write = sys.stdout.write
    try:
        tty.setraw(fd)
        write(_ALT_ON + _MOUSE_ON)
        sys.stdout.flush()
        keys = _Input(fd)
        last = None
        while not ed.quit:
            size = shutil.get_terminal_size((80, 24))
            if resized[0]:
                resized[0] = False
                ed._w = 0                     # force a re-wrap
                last = None
            screen = ed.frame(size.columns, size.lines)
            if screen != last:                # only paint what changed:
                write(screen)                 # a still screen never flickers
                sys.stdout.flush()
                last = screen
            ev = keys.next(0.25)
            if ev:
                ed.handle(ev)
    finally:
        write(_MOUSE_OFF + _ALT_OFF)
        sys.stdout.flush()
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        signal.signal(signal.SIGWINCH, prev_winch)
    return ed.saved_n
