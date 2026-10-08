"""Grouping rhymes into colour families without chaining.

The legacy grouping merged through ANY shared link: A~B and B~C put A, B
and C in one colour even when A and C have nothing in common, plus two
more merges on top (runs landing on the same beat, and one song-wide merge
per sound). On a real song that welded up to 18 different rhyme patterns
into a single colour, which is what made different rhymes share a colour
and made multis invisible.

Two rules replace it:

* A rhyme joins a family only when it matches the family's SIGNATURE — the
  sequence of stressed vowel classes the family is built on (e.g. OW·EY·AE)
  — not merely one of its members. A one-beat AE rhyme touching a
  three-beat OW·EY·AE multi does not recolour it.
* Families EXPIRE: a signature not used for EXPIRY bars starts a new family
  (a new colour) when it comes back.

Rhymes are placed strongest first, so when two claim the same syllable the
more certain one keeps it.
"""
from __future__ import annotations

EXPIRY = 8          # bars a family may go unused before it closes


def assign_families(items, line_of, expiry: int = EXPIRY):
    """Place rhymes into families.

    `items` — (log_odds, kind, spans, signature) per accepted rhyme, where
    `spans` is a list of syllable-position tuples (the two sides of a pair,
    or the members of a returning sound) and `signature` a tuple of vowel
    classes. `line_of[p]` is the bar of syllable p.

    Returns (families, owner, signatures): member lists, {position:
    (family index, log_odds, kind)}, and each family's signature."""
    fams: list = []
    owner: dict = {}
    for lo, kind, spans, sig in sorted(items, key=lambda it: -it[0]):
        pos = [p for sp in spans for p in sp]
        free = [p for p in pos if p not in owner]
        if not free:
            continue
        lines = [line_of[p] for p in pos]
        lo_l, hi_l = min(lines), max(lines)
        target = None
        # a family this rhyme already touches, with the SAME signature
        for p in pos:
            if p in owner and fams[owner[p][0]]["sig"] == sig:
                target = owner[p][0]
                break
        if target is None:
            # the nearest still-open family built on the same signature
            best = None
            for k, f in enumerate(fams):
                if f["sig"] != sig:
                    continue
                dist = max(0, lo_l - f["last"], f["first"] - hi_l)
                if dist <= expiry and (best is None or dist < best[0]):
                    best = (dist, k)
            if best is not None:
                target = best[1]
        if target is None:
            fams.append({"sig": sig, "members": set(),
                         "first": lo_l, "last": hi_l})
            target = len(fams) - 1
        f = fams[target]
        for p in free:
            f["members"].add(p)
            owner[p] = (target, lo, kind)
        f["first"] = min(f["first"], lo_l)
        f["last"] = max(f["last"], hi_l)
    return ([sorted(f["members"]) for f in fams], owner,
            [f["sig"] for f in fams])


def form_families(nodes, scores: dict, threshold: float = 0.6) -> list:
    """Average-linkage clustering over pairwise rhyme scores (0..1).

    Two clusters merge only when the AVERAGE score between all their
    members clears `threshold` — so C, strong with B but weak with A,
    does not ride into A's family on B's back."""
    def s(a, b):
        return scores.get((a, b), scores.get((b, a), 0.0))

    clusters = [[x] for x in nodes]
    while True:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                avg = (sum(s(a, b) for a in clusters[i] for b in clusters[j])
                       / (len(clusters[i]) * len(clusters[j])))
                if avg >= threshold and (best is None or avg > best[0]):
                    best = (avg, i, j)
        if best is None:
            return clusters
        _avg, i, j = best
        clusters[i] += clusters[j]
        del clusters[j]


def sound_families(members, sound_of, line_of, expiry: int = EXPIRY):
    """Colour every rhyming SYLLABLE by its own sound (the Spited way).

    `members` — {position: (log_odds, kind)} for every syllable some
    accepted rhyme covers. A family is one sound (vowel class) over one
    stretch of the song: members of the same sound belong together until
    that sound goes unused for more than `expiry` bars, when it starts a
    new family (a new colour).

    Why syllables and not whole patterns: a multi is a SEQUENCE of sounds,
    and a listener hears each syllable of it rhyme with the same sound
    elsewhere. Colouring whole patterns split the same "oo" end
    rhyme into one colour inside a two-beat multi and another when it
    stood alone — 77 of 258 hand-annotated rhyme pairs came out in two
    different colours that way. One sound, one colour, locally: a multi
    shows up as the same sequence of colours repeating, and two different
    sounds can never share a colour, so unrelated rhymes cannot merge.

    Returns a list of member lists."""
    by_sound: dict = {}
    for p in members:
        by_sound.setdefault(sound_of(p), []).append(p)
    fams = []
    for _sound, ps in by_sound.items():
        ps.sort(key=lambda p: (line_of[p], p))
        cur = [ps[0]]
        for p in ps[1:]:
            if line_of[p] - line_of[cur[-1]] > expiry:
                fams.append(cur)
                cur = []
            cur.append(p)
        fams.append(cur)
    return fams
