# RhymeScope

![python](https://img.shields.io/badge/python-3.11%2B-blue)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Type a song name, get its rhyme scheme mapped syllable by syllable, with every
rhyme family in its own color. Output it as:

- a **rhyme map** in the terminal or as HTML
- **karaoke**: the song plays while the words light up in sync

## Install

Requires Python 3.11+ and `ffmpeg`.

```bash
git clone https://github.com/Rayen12348/rhymescope.git
cd rhymescope
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Optional: `pip install demucs` separates vocals from the beat, which makes
transcription much more accurate when a song has no lyrics online.

## Usage

```bash
python tui.py                              # interactive menu
python tui.py "Eminem - Lose Yourself"     # start with a song
python tui.py song.mp3                     # or a local file
```

```
    4  play along           music + synced lyrics
    5  show rhyme map       here in the terminal
    6  fix the rhymes       paint the map, right here
```

Run `python tui.py --help` for all options (Whisper model, density, offset…).

## How it works

1. **Audio**: downloaded with yt-dlp.
2. **Lyrics**: LRCLIB → Genius → AZLyrics/Musixmatch, with every result checked
   against the song. Whisper transcribes the audio as a last resort.
3. **Rhymes**: each word is turned into phonemes (CMU dictionary plus rap
   spellings), and rhymes are matched on stressed vowel runs across nearby bars.
4. **Sync**: every word is aligned to the vocal.

Corrections you make in the editor (menu `6`) are saved and applied to every
map of that song.

## License

[MIT](LICENSE).
Respect the terms of the sites RhymeScope reads from and the rights of the artists.
