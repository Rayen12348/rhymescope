"""What the voice actually says, phoneme by phoneme.

A wav2vec2 model fine-tuned to write down PHONEMES rather than words
(facebook/wav2vec2-lv-60-espeak-cv-ft). A speech-to-text model like Whisper
hears "film" and writes "film" however it was sung; this one writes what
it heard — "f ɛ l m" when the rapper bends the vowel to meet "tell". That
difference is the whole point: rappers change their accent to make things
rhyme, and the lyrics sheet cannot show it.

The output is a frame-by-frame log-probability over the model's phoneme
alphabet (50 frames a second), computed once per song on the separated
vocal and kept in the rhymescope cache next to it.
"""
from __future__ import annotations

import json
import pathlib

MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"
FRAME = 0.02            # seconds per output frame
_WINDOW = 20            # seconds of audio per forward pass (memory bound)
VERSION = 1             # bump when the posteriors change

_model = None
_vocab = None


def _load():
    global _model, _vocab
    if _model is None:
        from huggingface_hub import snapshot_download
        from transformers import Wav2Vec2ForCTC
        d = pathlib.Path(snapshot_download(
            MODEL, allow_patterns=["*.json", "*.bin", "*.safetensors"]))
        _vocab = json.loads((d / "vocab.json").read_text())
        _model = Wav2Vec2ForCTC.from_pretrained(str(d)).eval()
    return _model, _vocab


def vocab() -> dict:
    """phoneme token -> column index in the posteriors."""
    return _load()[1]


def posteriors(audio: str, progress=None):
    """(frames × tokens) log-probabilities for the separated vocal of
    `audio`, as float32. Cached on disk; separating the vocal and running
    the model take a few minutes each on a CPU the first time."""
    import numpy as np
    import stt
    disk = stt.cached("phonemes.npy", audio, "lv60", VERSION)
    if disk.exists():
        return np.load(disk).astype(np.float32)
    vocal, _sep = stt.isolate_vocal(audio, progress)
    import torch
    from faster_whisper.audio import decode_audio
    model, _v = _load()
    if progress:
        progress("hearing the phonemes …")
    a = decode_audio(vocal, sampling_rate=16000)
    step = 16000 * _WINDOW
    out = []
    with torch.inference_mode():
        for s in range(0, len(a), step):
            x = a[s:s + step]
            if len(x) < 400:
                break
            x = (x - x.mean()) / (x.std() + 1e-7)
            lg = model(torch.from_numpy(x)[None]).logits[0]
            out.append(torch.log_softmax(lg, -1).numpy())
            if progress:
                progress(min((s + step) / len(a), 1.0))
    lp = np.concatenate(out) if out else np.zeros((0, len(_vocab)), np.float32)
    np.save(disk, lp.astype(np.float16))
    return lp
