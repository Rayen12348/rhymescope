"""rhyme — the rhyme scheme detector.

    import rhyme
    analysis = rhyme.detect(text)              # what the app shows
    print(rhyme.explain(text, "sick with the"))  # why
"""
from __future__ import annotations


def detect(text: str, prons: list | None = None, level: float | None = None,
           heard: dict | None = None):
    """Analyse `text` and return the Analysis the app would show.

    `prons` — pronunciation overrides (gold Spans with `.pron`) — are
    accepted but not used yet. `heard` is audio.heard_vowels(audio_path,
    text): what the voice actually sang, used beside the dictionary."""
    import confidence as C
    lv = C.DEFAULT_LEVEL if level is None else level
    return C.at_level(C.score(text, weights=C.DEFAULT_WEIGHTS, heard=heard), lv)


from rhyme.explain import explain  # noqa: E402
