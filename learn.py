"""
learn.py — the detector learns your ear from the corrections you save.

Every fix you make in the rhyme editor is a labelled example of what YOU
hear. Erasing a highlight says "that is not a rhyme to me"; painting one
says "that is"; and the highlights you look straight at and leave alone
say "that one's right". Across enough of them a pattern appears — for the
first songs corrected by hand it was stark: echoes swept up at the front
of the bar get erased, rhymes where the bar lands get kept.

So confidence.py's weights are not fixed. After every save, the model is
refitted on everything you have ever corrected, starting from the default
weights and pulled toward your corrections only as far as the evidence
supports (an L2 prior: a handful of fixes nudges it, hundreds reshape
it). The next song you open is scored with your weights.

Nothing is uploaded; the model is one small JSON file in your config dir,
and deleting it resets the detector to its defaults.
"""

from __future__ import annotations

import json
import math
import pathlib

CONFIG = pathlib.Path.home() / ".config" / "rhymescope"
WEIGHTS_FILE = CONFIG / "weights.json"
REGISTRY = CONFIG / "corrections.json"

# below this many labelled examples the defaults are used as they are —
# a model fitted to a dozen clicks would be fitting the clicks, not the ear
MIN_EXAMPLES = 20
PRIOR_STRENGTH = 2.0      # L2 pull back toward the default weights

# Which detector the weights were learned against. Weights fitted to one
# detector's features mean nothing for another's — applying them anyway
# is how a rebuilt detector ends up lighting 80% of a song — so a weights
# file only counts for the model it was fitted to.
MODEL_ID = "runs-v2"     # v2: ad-libs, fillers and stacked hooks never
                          # candidates; feminine tails scored per syllable
IMPLICIT_KEEP = 0.35      # weight of "you saw it and left it alone"


def current_weights() -> dict:
    from confidence import DEFAULT_WEIGHTS
    try:
        data = json.loads(WEIGHTS_FILE.read_text())
        w = data.get("weights") or {}
        if data.get("model") == MODEL_ID and \
                data.get("examples", 0) >= MIN_EXAMPLES and \
                set(DEFAULT_WEIGHTS) <= set(w):
            return {k: float(w[k]) for k in DEFAULT_WEIGHTS}
    except (OSError, ValueError):
        pass
    return dict(DEFAULT_WEIGHTS)


def register(src: pathlib.Path) -> None:
    """Remember where a corrections file lives, so learning can find it
    even when the song is not in the downloads folder."""
    CONFIG.mkdir(parents=True, exist_ok=True)
    try:
        known = json.loads(REGISTRY.read_text())
    except (OSError, ValueError):
        known = []
    s = str(pathlib.Path(src).resolve())
    if s.startswith(("/tmp/", "/var/tmp/")):
        return                # a scratch copy is not something you corrected
    if s not in known:
        known.append(s)
        REGISTRY.write_text(json.dumps(known, indent=1))


def _sources() -> list[pathlib.Path]:
    out: list = []
    try:
        out += [pathlib.Path(p) for p in json.loads(REGISTRY.read_text())]
    except (OSError, ValueError):
        pass
    downloads = pathlib.Path(__file__).resolve().parent / "downloads"
    for f in downloads.glob("*.rhymes.json"):
        out.append(f.with_name(f.name[:-len(".rhymes.json")] + ".x"))
    seen, uniq = set(), []
    for p in out:
        key = str(p.with_suffix(""))
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def examples(weights: dict | None = None) -> list:
    """(features, label, weight, group) for every correction on disk.
    `group` is (song, verse block), used to hold data out when checking
    that learning generalises instead of memorising."""
    import confidence as C
    import overrides as O
    from confidence import DEFAULT_WEIGHTS
    rows: list = []
    for src in _sources():
        state = O.load(src)
        edits = state.get("edits") or {}
        if not edits:
            continue
        text = O.text_for(src, state)
        if not text:
            continue
        u = C.score(text, weights=weights or DEFAULT_WEIGHTS)
        keymap = {O.key_for(u, p): p for p in range(len(u.stream))}
        lines_touched = [int(k.split(":")[0]) for k in edits]
        lo, hi = min(lines_touched), max(lines_touched)
        touched = set()
        for k, v in edits.items():
            p = keymap.get(k)
            if p is None:
                continue
            touched.add(p)
            if p in u.evidence:
                ln = u.words[u.stream[p].word_index].line
                rows.append((u.evidence[p], 0 if v == O.NONE else 1, 1.0,
                             (src.stem, ln // 8)))
        for p, x in u.evidence.items():
            ln = u.words[u.stream[p].word_index].line
            if p not in touched and lo <= ln <= hi and x["in_default"]:
                rows.append((x, 1, IMPLICIT_KEEP, (src.stem, ln // 8)))
    return rows


def _fit(rows: list, prior: dict, steps: int = 1500, lr: float = 0.5) -> dict:
    """Weighted logistic regression, pulled toward `prior` (L2)."""
    import numpy as np
    from confidence import FEATURES
    if not rows:
        return dict(prior)
    X = np.array([[1.0] + [x.get(k, 0.0) for k in FEATURES]
                  for x, _y, _w, _g in rows])
    y = np.array([r[1] for r in rows], dtype=float)
    wt = np.array([r[2] for r in rows], dtype=float)
    keys = ["bias", *FEATURES]
    w0 = np.array([prior[k] for k in keys], dtype=float)
    w = w0.copy()
    total = wt.sum()
    for _ in range(steps):
        z = np.clip(X @ w, -30, 30)
        pr = 1 / (1 + np.exp(-z))
        grad = X.T @ ((pr - y) * wt) / total + PRIOR_STRENGTH * (w - w0) / total
        w -= lr * grad
    return {k: float(v) for k, v in zip(keys, w)}


def auc(rows: list, weights: dict) -> float:
    from confidence import predict
    pos = [(predict(x, weights), wt) for x, y, wt, _g in rows if y == 1]
    neg = [(predict(x, weights), wt) for x, y, wt, _g in rows if y == 0]
    if not pos or not neg:
        return float("nan")
    num = den = 0.0
    for p, pw in pos:
        for n, nw in neg:
            num += pw * nw * ((p > n) + 0.5 * (p == n))
            den += pw * nw
    return num / den


def retrain() -> dict:
    """Refit on every correction saved so far. Returns a report."""
    from confidence import DEFAULT_WEIGHTS
    rows = examples()
    n_explicit = sum(1 for r in rows if r[2] == 1.0)
    report = {"examples": n_explicit, "used": False}
    if n_explicit < MIN_EXAMPLES:
        return report
    # Adopt learned weights ONLY when they beat the defaults on verses they
    # were not fitted to. A model that fits your corrections perfectly and
    # then does worse on the next verse has learned your clicks, not your
    # ear — and on the first real corrections (29 fixes, one song) that is
    # exactly what happened: 0.81 held-out against the defaults' 0.88.
    # Until the evidence is there, the detector does not change.
    cv = cross_validate(rows)
    learned = _fit(rows, DEFAULT_WEIGHTS)
    better = (cv["folds"] >= 3 and cv["auc_learned"] == cv["auc_learned"]
              and cv["auc_learned"] > cv["auc_default"] + 0.01)
    report.update(cv, used=better)
    CONFIG.mkdir(parents=True, exist_ok=True)
    if better:
        WEIGHTS_FILE.write_text(json.dumps(
            {"model": MODEL_ID, "examples": n_explicit, "weights": learned,
             "cv": cv}, indent=1))
    elif WEIGHTS_FILE.exists():
        WEIGHTS_FILE.unlink()          # never keep weights that stopped helping
    return report


def cross_validate(rows: list | None = None) -> dict:
    """Hold each verse block out in turn, fit on the rest, score the held-
    out block. The honest test of whether learning helps on lyrics it has
    never seen — in-sample accuracy would flatter any model."""
    from confidence import DEFAULT_WEIGHTS, predict
    rows = rows if rows is not None else examples()
    groups = sorted({r[3] for r in rows})
    scored_def, scored_learn = [], []
    for g in groups:
        train = [r for r in rows if r[3] != g]
        test = [r for r in rows if r[3] == g]
        if not any(r[1] == 0 for r in test) or not any(r[1] == 1 for r in test):
            continue
        w = _fit(train, DEFAULT_WEIGHTS)
        scored_def += [(predict(x, DEFAULT_WEIGHTS), y, wt) for x, y, wt, _ in test]
        scored_learn += [(predict(x, w), y, wt) for x, y, wt, _ in test]

    def _auc(sc):
        pos = [(s, w) for s, y, w in sc if y == 1]
        neg = [(s, w) for s, y, w in sc if y == 0]
        if not pos or not neg:
            return float("nan")
        num = sum(pw * nw * ((p > n) + 0.5 * (p == n))
                  for p, pw in pos for n, nw in neg)
        return num / sum(pw * nw for _p, pw in pos for _n, nw in neg)
    return {"folds": len(groups), "auc_default": _auc(scored_def),
            "auc_learned": _auc(scored_learn)}
