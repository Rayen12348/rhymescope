"""
overrides.py — your corrections to a detected rhyme scheme.

The detector is good and it is not perfect. On any song there will be a
syllable it lit that you hear as nothing, or a rhyme it walked past. This
module is how you say so, and how that decision reaches every render.

THREE PROMISES, each of which an earlier version broke:

1. EVERY RENDER HONOURS YOUR FIXES. Corrections are folded into the one
   analysis every output is built from — the whole-song video, and every
   verse clip and time range, which are slices of it (see
   rhyme_engine.slice_analysis). Clips used to analyse their lines afresh,
   and so silently threw your corrections away.

2. WHAT YOU PAINT IS WHAT RENDERS. The colours you saw in the editor are
   saved and pinned, so a re-render cannot recolour a rhyme behind your
   back.

3. NOTHING HERE TOUCHES TIMING. A correction changes which rhyme a
   syllable belongs to. Line times, word times and the Whisper alignment
   live on a separate path this module cannot reach.

HOW THINGS ARE ADDRESSED. A syllable is its position in the lyrics —
"line:char:syllable" — which survives re-analysis and engine upgrades. A
rhyme is its GROUP ID: "f:<the position of its first member>" for one the
detector found, "newN" for one you started. (Rhymes used to be addressed
by their sound, but a rhyme's sound shifts as you move syllables in and
out of it, so a second editing session could split one rhyme in two.)
The lyrics themselves are snapshotted, so if a song's lyrics come back
slightly different next time, corrections follow their lines rather than
their line numbers.
"""

from __future__ import annotations

import difflib
import json
import pathlib

VERSION = 2
NONE = "none"          # "this is not a rhyme"


def path_for(src: pathlib.Path) -> pathlib.Path:
    """Where a song's corrections live: beside the audio, named after it."""
    return src.with_suffix(".rhymes.json")


def empty_state() -> dict:
    return {"version": VERSION, "lines": None, "edits": {}, "colors": {},
            "names": {}, "sensitivity": None}


def load(src: pathlib.Path) -> dict:
    """This song's corrections (an empty state when there are none).

    Version-1 files — edits addressed to rhyme SOUNDS — are read too; their
    sound keys are resolved against the song when applied."""
    p = path_for(src)
    state = empty_state()
    if not p.exists():
        return state
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return state
    if data.get("version") == 1:
        state["edits"] = dict(data.get("edits") or {})
        state["colors"] = dict(data.get("groups") or {})
        return state
    if data.get("version") != VERSION:
        return state
    for k in ("lines", "sensitivity"):
        state[k] = data.get(k)
    try:
        from confidence import migrate_level
        state["sensitivity"] = migrate_level(state["sensitivity"],
                                             data.get("scale"))
    except ImportError:
        pass
    for k in ("edits", "colors", "names"):
        state[k] = dict(data.get(k) or {})
    return state


def save(src: pathlib.Path, state: dict, lines: list) -> pathlib.Path:
    """Write corrections beside the audio, with the lyrics they refer to."""
    p = path_for(src)
    out = empty_state()
    out.update({k: state.get(k) for k in ("edits", "colors", "names",
                                         "sensitivity")})
    out["lines"] = list(lines)
    try:
        from confidence import LEVELS_SCALE
        out["scale"] = LEVELS_SCALE
    except ImportError:
        pass
    out["edits"] = out["edits"] or {}
    out["colors"] = out["colors"] or {}
    out["names"] = out["names"] or {}
    p.write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    try:                      # so learning finds it wherever the song lives
        import learn
        learn.register(src)
    except Exception:
        pass
    return p


def text_for(src: pathlib.Path, state: dict | None = None) -> str | None:
    """The lyrics a corrections file was made against: the snapshot saved
    inside it when there is one, else the .lyrics.txt beside the audio."""
    state = state if state is not None else load(src)
    if state.get("lines"):
        return "\n".join(state["lines"])
    p = src.with_suffix(".lyrics.txt")
    return p.read_text(encoding="utf-8") if p.exists() else None


def key_for(analysis, pos: int) -> str:
    """The stable address of stream position `pos`."""
    syl = analysis.stream[pos]
    w = analysis.words[syl.word_index]
    return f"{w.line}:{w.start}:{syl.index_in_word}"


def sound_key(sound) -> str:
    vowel, coda = sound
    return vowel + ("+" + "".join(coda) if coda else "")


def assign_gids(analysis) -> None:
    """Give every family a stable group id, if it has none yet."""
    for f in analysis.families:
        if not f.gid and f.members:
            f.gid = "f:" + key_for(analysis, min(f.members))


def resolve_legacy(state: dict, analysis) -> dict:
    """Rewrite version-1 edits (addressed to a rhyme SOUND, like "AH+N")
    to the group ids of this analysis, so an editor shows them in the
    rhyme they actually join. Unresolvable ones are left as they are and
    behave as a rhyme of their own."""
    from rhyme_engine import family_sound
    assign_gids(analysis)
    by_sound = {sound_key(family_sound(analysis, f)): f.gid
                for f in analysis.families if f.members}
    by_gid = {f.gid: f for f in analysis.families}
    keymap = {key_for(analysis, p): p for p in range(len(analysis.stream))}

    def now(v):
        if v == NONE or v in by_gid or v.startswith("new"):
            return v
        if v.startswith("f:"):             # a group id that went stale
            f = _family_now(v, analysis, keymap, by_gid)
            return f.gid if f is not None else v
        return by_sound.get(v, v)
    out = dict(state)
    out["edits"] = {k: now(v) for k, v in (state.get("edits") or {}).items()}
    colors = dict(state.get("colors") or {})
    for v_old, v_new in {(v, out["edits"][k])
                         for k, v in (state.get("edits") or {}).items()}:
        if v_new != v_old and v_old in colors:
            colors.setdefault(v_new, colors[v_old])
    out["colors"] = colors
    return out


def _line_map(saved: list | None, current: list) -> dict | None:
    """old line number → new, when the lyrics changed since the save.
    None means identical (nothing to remap)."""
    if not saved or list(saved) == list(current):
        return None
    sm = difflib.SequenceMatcher(None, list(saved), list(current),
                                 autojunk=False)
    out: dict = {}
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            out[a + k] = b + k
    return out


def _remap(key: str, lm: dict | None) -> str | None:
    if lm is None:
        return key
    prefix = "f:" if key.startswith("f:") else ""
    body = key[len(prefix):]
    parts = body.split(":")
    if len(parts) != 3 or not parts[0].isdigit():
        return key                     # "none", "newN", legacy sound keys
    new = lm.get(int(parts[0]))
    return None if new is None else f"{prefix}{new}:{parts[1]}:{parts[2]}"


def _anchor(target: str, keymap: dict) -> int | None:
    """The syllable a detector group id is named after ("f:<its key>")."""
    return keymap.get(target[2:]) if target.startswith("f:") else None


def _family_now(target: str, analysis, keymap: dict, by_gid: dict):
    """The rhyme a stale group id means today.

    A group id names a rhyme by its first syllable at the moment you
    painted. Whenever the analysis changes — a detector upgrade, or the
    vocal being heard minutes after you saved — the rhyme can start on a
    different syllable and carry a different id. It is still the rhyme
    that syllable is in, so that is where the paint goes; without this, a
    syllable painted into it became a rhyme of one and vanished."""
    if target in by_gid:
        return by_gid[target]
    pos = _anchor(target, keymap)
    if pos is None:
        return None
    for f in analysis.families:
        if pos in f.members:
            return f
    return None


def apply(analysis, state: dict, level: float | None = None):
    """Fold the slider level and your corrections into `analysis`, in place.

    Order matters: the slider first decides what the detector shows, then
    your edits overrule it syllable by syllable — a syllable you painted
    stays painted at any slider setting, and one you erased stays erased.
    Returns the analysis."""
    from rhyme_engine import Family, family_sound

    assign_gids(analysis)
    lm = _line_map(state.get("lines"), analysis.lines)
    edits: dict = {}
    for k, v in (state.get("edits") or {}).items():
        nk = _remap(k, lm)
        nv = _remap(v, lm) if v.startswith("f:") else v
        if nk is not None and nv is not None:
            edits[nk] = nv
    colors = {(_remap(g, lm) or g): c
              for g, c in (state.get("colors") or {}).items()}
    names = state.get("names") or {}

    conf = getattr(analysis, "confidence", None)
    by_gid = {f.gid: f for f in analysis.families}
    by_sound = {sound_key(family_sound(analysis, f)): f
                for f in analysis.families if f.members}
    members = {f.gid: {m for m in f.members
                       if level is None or conf is None
                       or conf.get(m, 1.0) >= level}
               for f in analysis.families}

    keymap = {key_for(analysis, p): p for p in range(len(analysis.stream))}
    for k, target in edits.items():
        pos = keymap.get(k)
        if pos is None:
            continue
        for ms in members.values():        # one syllable, one rhyme
            ms.discard(pos)
        if target == NONE:
            continue
        fam = (by_gid.get(target) or by_sound.get(target)
               or _family_now(target, analysis, keymap, by_gid))
        if fam is None:
            fam = Family(fid=-1, label=names.get(target, target),
                         members=[], gid=target)
            analysis.families.append(fam)
            by_gid[target] = fam
            members[target] = set()
            # the rhyme you painted into is gone from the detector's view,
            # but you saw it: its first syllable comes along, so a single
            # painted syllable is not left a "rhyme" of one and dropped
            anchor = _anchor(target, keymap)
            if anchor is not None and target[2:] not in edits:
                members[target].add(anchor)
        elif fam.gid != target:
            if target in colors:
                colors.setdefault(fam.gid, colors[target])
            anchor = _anchor(target, keymap)
            if anchor is not None and target[2:] not in edits:
                members[fam.gid].add(anchor)
        members[fam.gid].add(pos)

    line_of = lambda m: analysis.words[analysis.stream[m].word_index].line
    kept: list = []
    family_of: dict = {}
    for f in analysis.families:
        ms = sorted(members.get(f.gid, ()))
        if len(ms) < 2:
            continue
        f.fid = len(kept)
        f.members = ms
        f.span = len({line_of(m) for m in ms})
        c = colors.get(f.gid)
        if not c and f.gid in edits.values():
            # a legacy (version-1) file pinned colours by sound
            c = colors.get(sound_key(family_sound(analysis, f)))
        if c:
            f.fixed_color = c
        kept.append(f)
        for m in ms:
            family_of[m] = f.fid
    analysis.families = kept
    analysis.family_of = family_of
    n = len(analysis.stream)
    analysis.stats = dict(analysis.stats, families=len(kept),
                          rhymed_syllables=len(family_of),
                          density=(len(family_of) / n) if n else 0.0)
    return analysis
