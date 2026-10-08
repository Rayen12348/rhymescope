#!/usr/bin/env python3
"""
RhymeScope — interactive terminal app.

    python tui.py                              # menu: type a song, pick outputs
    python tui.py "Eminem - Lose Yourself"     # jump straight to that song

Type a song name (or a local audio/lyrics file). RhymeScope downloads the
audio (yt-dlp), fetches the real lyrics (Genius → LRCLIB → fallbacks),
detects the rhyme scheme, syncs every word to the vocal with a Whisper
alignment pass, then shows whatever you pick from the menu:

    · karaoke       — plays the song, words reveal in sync
    · rhyme map     — color-coded, right in the terminal

The Whisper pass runs once per song and is reused by every output.
"""

from __future__ import annotations

import argparse
import re
import pathlib
import shutil
import subprocess
import sys
import time

import fetch as fetch_mod
from html_render import render as render_html, _palette
from stt import AUDIO_EXT
from rhyme_engine import analyze
from term_render import (BOLD, DIM, RESET, _fg, colored_line,
                         render_terminal, syllable_map)

ROOT = pathlib.Path(__file__).resolve().parent
DOWNLOADS = ROOT / "downloads"

try:
    import studio               # optional local extras
    studio.ui = sys.modules[__name__]
except ImportError:
    studio = None

# RhymeScope brand: lime accent (terminals are dark, lime is the brand's
# one accent colour)
ACCENT = _fg("#CBF135")
GREEN = _fg("#57d17a")
RED = _fg("#ff7676")

_BANNER = f"""
  {BOLD}╦═╗┬ ┬┬ ┬┌┬┐┌─┐{RESET}{ACCENT}{BOLD}┌─┐┌─┐┌─┐┌─┐┌─┐{RESET}
  {BOLD}╠╦╝├─┤└┬┘│││├┤ {RESET}{ACCENT}{BOLD}└─┐│  │ │├─┤├┤{RESET}
  {BOLD}╩╚═┴ ┴ ┴ ┴ ┴└─┘{RESET}{ACCENT}{BOLD}└─┘└─┘└─┘┴ ┴└─┘{RESET}
  {DIM}◎ rhyme maps · karaoke{RESET}
"""


def _bar(pct: float, width: int = 34) -> str:
    filled = int(pct * width)
    return f"[{'█' * filled}{'·' * (width - filled)}] {pct * 100:4.0f}%"


def _step(n, msg: str) -> None:
    print(f"{ACCENT}[{n}]{RESET} {msg}")


def _prog_bar(p) -> None:
    sys.stdout.write(f"\r    {_bar(p)}")
    sys.stdout.flush()


def _clear_bar() -> None:
    sys.stdout.write("\r" + " " * 50 + "\r")


# --------------------------------------------------------------------------
# pipeline: song name / file  →  analysis (+ sync when available)
# --------------------------------------------------------------------------

class Song:
    """Everything the menu actions need, computed lazily and cached."""

    def __init__(self, opts):
        self.opts = opts
        self.src: pathlib.Path | None = None
        self.title = ""
        self.source = ""
        self.analysis = None
        self.sync = None            # [(seconds, line), …] or None until timed
        self.schedule = None        # per-line word times (Whisper-aligned)
        self._aligned = False
        self.rendered = False       # a video was produced this session
        self.html_saved = False     # user explicitly kept the HTML map
        self.text = ""              # the lyrics the analysis was built from
        self.edited = False         # corrections are in play
        self.level = None           # density the map is shown at
        self._listener = None       # background listening pass (thread)
        self.listen_state: dict = {}
        self.heard = None           # vowels heard in the vocal (audio/)
        self._ear_applied = False

    # -- load -------------------------------------------------------------

    def load(self, user_input: str) -> None:
        p = pathlib.Path(user_input)
        duration = None
        artist_hint = ""
        typed = user_input          # what you asked for, as words
        if p.exists() and p.is_file():
            self.src = p
            self.title = p.stem
            # a path is not a search: its folder names are not words the
            # lyrics must contain, and passing them as if they were made
            # every local file come back "no lyrics online"
            typed = p.stem
            # How long the track is, asked of the file itself. This is not
            # a detail: the lyrics sources rank by duration, so without it
            # a bare title like "Business" or "Kim" comes back as whatever
            # OTHER song of that name the search happens to rank first,
            # and everything downstream is confidently about the wrong
            # song. A download knows its duration; a local file has to be
            # asked, and nothing ever asked.
            if p.suffix.lower() in AUDIO_EXT:
                try:
                    from timing import _audio_duration
                    duration = _audio_duration(p) or None
                except Exception:
                    duration = None
            _step(1, f"Using local file {DIM}{p.name}{RESET}"
                  + (f" {DIM}({duration/60:.0f}:{duration%60:02.0f}){RESET}"
                     if duration else ""))
        else:
            _step(1, f"Searching YouTube for {BOLD}{user_input}{RESET} …")
            self.src, meta = fetch_mod.download_audio(
                user_input, DOWNLOADS, progress=_prog_bar)
            _clear_bar()
            self.title = meta["title"]
            duration = meta["duration"]
            # what the user typed + the channel name disambiguate songs
            # that share a title across artists (e.g. two "Deja Vu"s)
            artist_hint = f"{meta.get('uploader') or ''} {user_input}"
            print(f"    {GREEN}✓{RESET} {self.title}")

        text = None
        if self.src.suffix.lower() in AUDIO_EXT:
            saved = self.src.with_suffix(".lyrics.txt")
            if self.opts.no_online and saved.exists():
                # offline: the lyrics already saved beside the audio are
                # the lyrics — never re-transcribe over them
                text = saved.read_text(encoding="utf-8").strip() or None
                if text:
                    self.source = "saved lyrics file"
                    print(f"    {GREEN}✓{RESET} {self.source}")
            if text is None and not self.opts.no_online:
                _step(2, "Fetching lyrics …")
                hit = fetch_mod.fetch_lyrics(self.title, duration,
                                             artist=artist_hint,
                                             typed=typed)
                if hit:
                    text, self.source, self.sync, _dur = hit
                    self.src.with_suffix(".lyrics.txt").write_text(
                        text + "\n", encoding="utf-8")
                    note = " (timestamped)" if self.sync else ""
                    print(f"    {GREEN}✓{RESET} {self.source}{note}")
            if text is None:
                import stt
                sep = (" (separating the vocal first — this takes a few "
                       "minutes)" if stt.have_separator() else "")
                if (not self.opts.no_online
                        and fetch_mod.last_lookup_failed_on_network()):
                    # not "no lyrics" — the lyrics sites could not be
                    # reached. Say that, so a real song on Genius is not
                    # mistaken for one that has no lyrics anywhere.
                    print(f"    {RED}✗ couldn't reach the lyrics sites "
                          f"(network){RESET} {DIM}— transcribing for now; "
                          f"load the song again later for the real "
                          f"lyrics{RESET}")
                    _step(2, f"Transcribing with Whisper{sep} …")
                else:
                    _step(2, f"No lyrics online — transcribing with Whisper{sep} …")
                text, self.sync, report = stt.transcribe_with_timing(
                    str(self.src), self.opts.model, self.opts.language,
                    progress=lambda p: (_prog_bar(p)
                                        if isinstance(p, float) else None))
                _clear_bar()
                self.source = "Whisper transcription"
                # A transcript can be confidently wrong, and everything
                # downstream is built on it — say so rather than letting a
                # video of the wrong words look like a finished result.
                print(f"    {GREEN}✓{RESET} transcribed "
                      f"{report['coverage']:.0%} of the track"
                      f"{' from the isolated vocal' if report['separated'] else ''}")
                if not report["ok"]:
                    print(f"    {DIM}⚠ {report['note']}{RESET}")
                if not report["separated"]:
                    print(f"    {DIM}install demucs for much better "
                          f"transcription:  pip install demucs{RESET}")
                # A rough transcript is not good enough to ANALYSE a song
                # but it is easily good enough to RECOGNISE one. Search
                # the lyrics databases with what we heard: if the real
                # page turns up, its words are the right words — properly
                # spelled, ad-libs included — and the alignment pass below
                # lines them up against this recording, so the timing
                # still comes from the audio and not from a lyrics site.
                found = None
                if not self.opts.no_online:
                    _step(2, "Looking the song up from what was heard …")
                    found = fetch_mod.identify_by_lyrics(
                        text, hint=fetch_mod._tokens(artist_hint))
                if found:
                    text, self.source, self.sync, _dur = found
                    self.src.with_suffix(".lyrics.txt").write_text(
                        text + "\n", encoding="utf-8")
                    print(f"    {GREEN}✓{RESET} {self.source}")
                    print(f"    {DIM}using the real lyrics, lined up to "
                          f"the audio{RESET}")
                else:
                    print(f"    {DIM}no match online — keeping the "
                          f"transcript{RESET}")
        else:
            text = self.src.read_text(encoding="utf-8")
            self.source = "local lyrics file"

        if text and self.src.suffix.lower() in AUDIO_EXT:
            # your fixes were made on other lyrics than this load found (a
            # re-edited page, another source, a fresh transcript, even
            # another song with the same title): they are addressed by
            # line and letter, so they only survive on the lyrics they
            # were made on
            kept = corrected_lyrics(self.src, text)
            if kept is not None:
                text, self.sync = kept, None
                self.source = "the lyrics your corrections were made on"
                self.src.with_suffix(".lyrics.txt").write_text(
                    text + "\n", encoding="utf-8")
                print(f"    {GREEN}✓{RESET} kept {self.source} "
                      f"{DIM}(today's lyrics would have lost some of "
                      f"them){RESET}")

        if not text or not text.strip():
            raise RuntimeError("no lyrics found — nothing to analyze")

        self.text = text
        _step(3, "Detecting the rhyme scheme …")
        self.reanalyze()
        s = self.analysis.stats
        print(f"    {GREEN}✓{RESET} {s['families']} families · "
              f"{s['rhymed_syllables']}/{s['syllables']} syllables rhyme "
              f"({s['density']:.0%})")

        # the slow part — separating and listening to the vocal — starts
        # now and runs while you use the menu (see start_listening)
        self.start_listening()

        if not self.opts.no_save:
            out = self.src.with_suffix(".rhymes.html")
            out.write_text(render_html(self.analysis, title=self.title),
                           encoding="utf-8")

    def reanalyze(self) -> None:
        """Detect the scheme, then apply your density and corrections.

        Always from the pristine text, in three steps that every output
        shares: the detector scores every candidate rhyme (confidence.py),
        the density setting decides which of them show, and your
        corrections overrule both, syllable by syllable. Nothing here
        reads or writes timing — see overrides.py."""
        import confidence
        import overrides
        state = overrides.load(self.src)
        level = state["sensitivity"]
        if level is None:
            level = confidence.LEVELS.get(getattr(self.opts, "density",
                                                  "rich"),
                                          confidence.DEFAULT_LEVEL)
        self.level = level
        # what the voice sang, when it has been heard already (instant
        # from the cache); otherwise the background pass fills it in
        if self.heard is None and not getattr(self.opts, "no_ear", False):
            from audio import cached_heard, find_audio
            self.heard = cached_heard(find_audio(self.src), self.text)
        scored = confidence.score(self.text, strict=self.opts.strict,
                                  heard=self.heard)
        self.analysis = overrides.apply(scored, state, level)
        self.edited = bool(state["edits"])

    # -- timing (lazy, Whisper transcription shared via timing._listen) -----

    def start_listening(self) -> None:
        """Start the slow part NOW, in the background.

        Syncing words to the vocal means separating the voice from the
        beat and listening to it — minutes on a laptop CPU. It used to wait
        until you asked for a video, so you sat through all of it at the
        moment you wanted a result. Now it starts as soon as the song is
        loaded and runs while you look at the map, fix rhymes or pick a
        format; by the time you render, it is usually done. (And it is
        cached on disk, so a song is only ever listened to once.)"""
        if (self._listener is not None or self.src is None
                or self.src.suffix.lower() not in AUDIO_EXT):
            return
        import threading
        st = self.listen_state = {"stage": "starting", "pct": 0.0,
                                  "done": False, "error": None}

        def progress(p):
            if isinstance(p, float):
                st["pct"] = p
            elif isinstance(p, str):
                st["stage"] = ("separating the vocal"
                               if p.startswith("separat") else p)

        def work():
            try:
                from timing import _listen
                _listen(self.src, self.opts.model, self.opts.language,
                        progress=progress)
                # …then hear WHICH vowels were sung, so rhymes the rapper
                # makes by bending his accent are found (audio/)
                if not getattr(self.opts, "no_ear", False) and self.heard is None:
                    import audio
                    a = audio.find_audio(self.src)
                    if a:
                        st["stage"] = "hearing the vowels"
                        st["pct"] = 0.0
                        audio.heard_vowels(a, self.text, progress=progress)
                        st["ear_ready"] = True
            except Exception as exc:          # never take the app down
                st["error"] = str(exc)
            finally:
                st["done"] = True

        self._listener = threading.Thread(target=work, daemon=True)
        self._listener.start()

    def apply_ear(self) -> str:
        """Once the background pass has heard the vowels, re-check the
        rhymes with them. Returns a note for the menu, or ""."""
        if self._ear_applied or not self.listen_state.get("ear_ready"):
            return ""
        self._ear_applied = True
        before = self.analysis.stats["rhymed_syllables"]
        self.heard = None                    # picked up from the cache
        self.reanalyze()
        if self.heard is None:
            return ""
        gain = self.analysis.stats["rhymed_syllables"] - before
        return (f"heard the vocal — rhymes re-checked by ear "
                f"({gain:+d} syllables)")

    def listening_note(self) -> str:
        """One line for the menu about the background pass."""
        st = self.listen_state
        if not st:
            return ""
        if st.get("done"):
            return "" if st.get("error") else "synced to the vocal — renders start straight away"
        if st.get("stage") == "hearing the vowels":
            return ("hearing which vowels are sung, in the background"
                    "  (a few minutes, once per song — accent bends)")
        if st.get("stage") == "listening":
            pct = st.get("pct", 0.0)
            return ("listening to the vocal in the background"
                    + (f" … {pct:.0%}" if pct > 0.01 else " …")
                    + "  (about a minute — keep going, it won't block you)")
        return f"{st.get('stage', 'preparing')} in the background …"

    def _wait_listening(self) -> None:
        """If the background pass is still running, show it and wait."""
        t = self._listener
        if t is None or not t.is_alive():
            return
        _step("~", "Finishing the listening pass started in the background …")
        while t.is_alive():
            st = self.listen_state
            if st.get("stage") == "listening":
                _prog_bar(st.get("pct", 0.0))
            else:
                sys.stdout.write(f"\r    {DIM}{st.get('stage', '')} …{RESET}   ")
                sys.stdout.flush()
            t.join(0.5)
        _clear_bar()


    def _aprog(self, p) -> None:
        if isinstance(p, tuple):
            _, m, total, drift = p
            _clear_bar()
            print(f"    {GREEN}✓{RESET} {m}/{total} anchored to the vocal"
                  f"{DIM} (drift {drift:+.2f}s){RESET}" if drift else
                  f"    {GREEN}✓{RESET} {m}/{total} anchored to the vocal")
        else:
            _prog_bar(p)

    def ensure_sync(self) -> None:
        """Line timing: from LRC when we have it, else built by listening.

        This does not fail any more. Every song that has lyrics and audio
        gets a timing — anchored to the vocal where the two line up, and
        spread across the track where they do not — because a song whose
        words could not be matched used to get no karaoke and no video at
        all, which is the worst possible answer to "sync this song".

        A very poor match is also the clearest evidence there is that the
        lyrics belong to a DIFFERENT song: search engines answer a bare
        title like "Business" or "Kim" with whatever they rank first. So
        when almost nothing lines up, the song is looked up again using
        what was actually heard, and the better lyrics win."""
        if self.sync is not None:
            return
        self._wait_listening()
        from timing import sync_from_alignment
        _step("~", "Timing the lyrics from the audio (Whisper) …")
        rep: dict = {}
        self.sync = sync_from_alignment(
            self.src, list(self.analysis.lines),
            model_size=self.opts.model, language=self.opts.language,
            progress=self._aprog, report=rep)
        _clear_bar()
        if self.sync is None:
            raise RuntimeError("no lyrics to time")
        if rep.get("mode") == "even" and not self.opts.no_online:
            if self._relookup():
                self.sync = None
                return self.ensure_sync()
        if rep.get("mode") == "even":
            print(f"    {DIM}⚠ these words don't line up with this audio — "
                  f"timing is spread across the track, so it will drift. "
                  f"Nudge it with --offset, or try a different source.{RESET}")
        elif rep.get("mode") == "partial":
            print(f"    {DIM}only {rep['anchors']} of {rep['lines']} bars "
                  f"could be pinned to the vocal — the rest are "
                  f"interpolated{RESET}")
        self._apply_offset()

    def _relookup(self) -> bool:
        """The lyrics don't match the audio: ask the databases what this
        recording actually is, using what Whisper heard. Returns True when
        better lyrics were found and adopted."""
        try:
            from timing import _listen
            heard = _listen(self.src, self.opts.model, self.opts.language)
        except Exception:
            return False
        text = " ".join(t for t, _ts in heard)
        if len(text.split()) < 25:
            return False
        print(f"    {DIM}these lyrics don't match the recording — looking "
              f"the song up from what was heard …{RESET}")
        found = fetch_mod.identify_by_lyrics(text)
        if not found or found[0].strip() == (self.text or "").strip():
            print(f"    {DIM}no better match found — keeping these "
                  f"lyrics{RESET}")
            return False
        self.text, self.source, self.sync, _dur = found
        self.src.with_suffix(".lyrics.txt").write_text(
            self.text + "\n", encoding="utf-8")
        print(f"    {GREEN}✓{RESET} {self.source} {DIM}(matched to what "
              f"the track actually sings){RESET}")
        self.reanalyze()
        return True

    def ensure_schedule(self) -> None:
        """Word timing: Whisper word timestamps snapped to the lyrics."""
        self.ensure_sync()
        if self._aligned:
            return
        self._wait_listening()
        from timing import align_word_times
        _fam, wbl = syllable_map(self.analysis)
        _step("~", "Aligning every word to the vocal …")
        sched = align_word_times(self.src, self.sync, wbl,
                                 model_size=self.opts.model,
                                 language=self.opts.language,
                                 progress=self._aprog)
        _clear_bar()
        if sched is None:
            print(f"    {DIM}alignment unreliable — using interpolated "
                  f"timing{RESET}")
        self.schedule = sched
        self._aligned = True

    def _apply_offset(self) -> None:
        if self.opts.offset and self.sync:
            self.sync = [(max(t + self.opts.offset, 0.0), s)
                         for t, s in self.sync]


# --------------------------------------------------------------------------
# menu actions
# --------------------------------------------------------------------------

def show_map(song: Song) -> None:
    print(render_terminal(song.analysis, title=song.title, source=song.source))


def corrected_lyrics(src, fetched: str) -> str | None:
    """The lyrics saved with your corrections, when the freshly fetched
    ones would strand any of them — else None (use what was fetched).

    A correction is addressed by line and letter. Lines that come back
    unchanged are followed wherever they moved; a line that comes back
    different takes its corrections with it. So when today's lyrics would
    lose even one fix, the lyrics they were made on win."""
    import overrides
    state = overrides.load(src)
    snap = state.get("lines")
    if not state.get("edits") or not snap:
        return None
    lm = overrides._line_map(snap, fetched.splitlines())
    if all(overrides._remap(k, lm) is not None for k in state["edits"]):
        return None
    return "\n".join(snap)


def glob_escape(text: str) -> str:
    return re.sub(r"([\[\]*?])", r"[\1]", text)


def density_name(song: Song) -> str:
    import confidence
    lv = song.level if song.level is not None else confidence.DEFAULT_LEVEL
    for name, v in confidence.LEVELS.items():
        if abs(v - lv) < 1e-6:
            return name
    return f"custom {lv:.0%}"


def corrections(song: Song) -> int:
    import overrides
    return len(overrides.load(song.src)["edits"])


def change_density(song: Song, d: int) -> None:
    """Show fewer (−) or more (+) rhymes, and remember it for this song so
    every render — now and later — uses the same setting."""
    import confidence
    import overrides
    order = [confidence.LEVELS[n] for n in ("clean", "balanced", "rich",
                                            "everything")]
    lv = song.level if song.level is not None else confidence.DEFAULT_LEVEL
    if d > 0:
        nxt = [v for v in order if v < lv - 1e-9]
        lv = nxt[0] if nxt else order[-1]
    else:
        nxt = [v for v in reversed(order) if v > lv + 1e-9]
        lv = nxt[0] if nxt else order[0]
    state = overrides.load(song.src)
    state["sensitivity"] = lv
    overrides.save(song.src, state, song.analysis.lines)
    song.reanalyze()
    st = song.analysis.stats
    print(f"  {GREEN}✓{RESET} density {density_name(song)} · "
          f"{st['density']:.0%} of syllables lit · {st['families']} rhymes"
          f"{DIM}  (saved for this song){RESET}")
    if studio and studio.made_videos(song):
        print(f"  {DIM}videos you already made keep the old setting until "
              f"you re-render them{RESET}")


def edit_rhymes(song: Song, theme: str = "light") -> None:
    """Fix the rhyme map by hand, here in the terminal. A correction is
    saved beside the audio and every later analysis of the song uses it."""
    import editor
    import overrides
    before = overrides.load(song.src)
    try:
        editor.edit(song.src, song.text, song.title,
                    strict=song.opts.strict, level=getattr(song, "level", None),
                    heard=song.heard)
    except RuntimeError as exc:          # no terminal to draw on
        print(f"  {RED}✗ {exc}{RESET}")
        return

    after = overrides.load(song.src)
    if after == before:
        print(f"  {DIM}nothing saved — the map is unchanged{RESET}")
        return
    # every fix is a lesson: refit the detector's confidence on all the
    # corrections you have ever saved (it only switches to the new weights
    # when they beat the defaults on verses they were not fitted to)
    try:
        import learn
        rep = learn.retrain()
        if rep.get("used"):
            print(f"  {GREEN}✓{RESET} the detector learned from your "
                  f"{rep['examples']} corrections "
                  f"{DIM}(held-out accuracy {rep['auc_default']:.2f} → "
                  f"{rep['auc_learned']:.2f}){RESET}")
    except Exception:
        pass
    song.reanalyze()
    st = song.analysis.stats
    n = len(after["edits"])
    print(f"  {GREEN}✓{RESET} saved · {n} correction{'' if n == 1 else 's'} · "
          f"{st['families']} rhymes · {st['density']:.0%} of syllables lit")

    if studio:
        studio.after_edit(song, theme)


def karaoke(song: Song) -> None:
    """Play the audio; lyrics reveal word by word at their aligned times."""
    if shutil.which("ffplay") is None:
        print(f"{RED}✗ ffplay not found (install ffmpeg){RESET}")
        return
    song.ensure_schedule()
    from timing import word_schedule

    colors = _palette(song.analysis)
    fam_at, wbl = syllable_map(song.analysis)
    schedule = song.schedule or word_schedule(song.sync, wbl)

    print(f"\n{BOLD}♪ {song.title}{RESET}  {DIM}— playing · Ctrl-C to "
          f"stop{RESET}\n")
    player = subprocess.Popen(
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
         str(song.src)])
    t0 = time.monotonic()

    def wait_until(ts: float) -> bool:
        wait = ts - (time.monotonic() - t0)
        if wait > 0:
            time.sleep(wait)
        return player.poll() is None

    try:
        for ln, (ts, txt) in enumerate(song.sync):
            # the aligned first word may start before the LRC line stamp —
            # follow the vocal, not the file
            if schedule[ln]:
                ts = min(ts, schedule[ln][0])
            if not wait_until(ts):
                break
            stamp = f"{DIM}{int(ts // 60)}:{ts % 60:05.2f}{RESET}"
            if not txt.strip():
                print(f" {stamp}  {DIM}· · ·{RESET}")
                continue
            for k, wts in enumerate(schedule[ln]):
                if not wait_until(wts):
                    break
                line = colored_line(song.analysis, colors, fam_at, wbl,
                                    ln, active=k + 1)
                sys.stdout.write(f"\r\x1b[2K {stamp}  {line}")
                sys.stdout.flush()
            print()
        player.wait()
    except KeyboardInterrupt:
        print(f"\n{DIM}stopped.{RESET}")
    finally:
        if player.poll() is None:
            player.terminate()


def save_html(song: Song) -> None:
    out = song.src.with_suffix(".rhymes.html")
    out.write_text(render_html(song.analysis, title=song.title),
                   encoding="utf-8")
    song.html_saved = True
    print(f"{GREEN}✓{RESET} HTML map → {BOLD}{out}{RESET}")


def shift_offset(song: Song) -> None:
    try:
        raw = input(f"{ACCENT}offset seconds (+later / -earlier) >{RESET} ")
        delta = float(raw.strip())
    except (ValueError, EOFError, KeyboardInterrupt):
        print(f"{DIM}unchanged.{RESET}")
        return
    if song.sync:
        song.sync = [(max(t + delta, 0.0), x) for t, x in song.sync]
    if song.schedule:
        song.schedule = [[max(t + delta, 0.0) for t in row]
                         for row in song.schedule]
    print(f"{GREEN}✓{RESET} lyrics shifted {delta:+.2f}s")


def cleanup(song: Song) -> None:
    """Once a video exists, only the .mp4 files stay: the downloaded
    audio, the .lyrics.txt and the auto-written HTML map are removed.
    A local file the user provided themselves is never touched, and an
    HTML map they explicitly saved (settings → h) is kept.

    Corrections (.rhymes.json) are never cleaned up. They are the one
    thing here a person made by hand rather than a machine generated, and
    they outlive the audio deliberately: load the same song again and
    your fixes come straight back."""
    if not song.rendered or song.src is None:
        return
    targets = [song.src.with_suffix(".lyrics.txt")]
    if not song.html_saved:
        targets.append(song.src.with_suffix(".rhymes.html"))
    if song.src.parent == DOWNLOADS and song.src.suffix.lower() != ".mp4":
        targets.append(song.src)
    gone = 0
    for t in targets:
        try:
            if t.suffix == ".json":        # never a person's corrections
                continue
            if t.exists() and ".rhymes." not in t.name.replace(
                    ".rhymes.html", "") and not t.name.endswith(".mp4"):
                t.unlink()
                gone += 1
        except OSError:
            pass
    if gone:
        print(f"{DIM}cleaned {gone} working file(s) — only the videos "
              f"remain{RESET}")


def toggle_strict(song: Song) -> None:
    song.opts.strict = not song.opts.strict
    song.reanalyze()          # keeps your density and corrections
    st = song.analysis.stats
    print(f"{GREEN}✓{RESET} strict {'ON' if song.opts.strict else 'off'} — "
          f"{st['families']} families, {st['density']:.0%} density")


# --------------------------------------------------------------------------
# the menu
# --------------------------------------------------------------------------

_MODELS = ["small", "base", "medium"]


def _settings(song: Song, theme: list) -> None:
    """Everything you rarely change, one level down so the main menu can
    stay four choices wide."""
    while True:
        o = song.opts
        print(f"""
  {DIM}settings{RESET}
{studio.settings_top(theme[0], o) if studio else ''}    {ACCENT}d{RESET}  density          {getattr(o, 'density', 'rich')}  {DIM}(clean · balanced · rich · everything — per song in the editor){RESET}
    {ACCENT}s{RESET}  strict rhymes    {'on' if o.strict else 'off'}
    {ACCENT}w{RESET}  whisper model    {o.model}
{studio.settings_mid(o) if studio else ''}    {ACCENT}o{RESET}  nudge lyric timing
    {ACCENT}h{RESET}  save HTML map
{studio.settings_end() if studio else ''}    {ACCENT}b{RESET}  back""")
        try:
            c = input(f"\n{ACCENT}set >{RESET} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        try:
            if studio and studio.setting(c, song, theme):
                pass
            elif c == "d":
                order = ["clean", "balanced", "rich", "everything"]
                cur = getattr(o, "density", "rich")
                o.density = order[(order.index(cur) + 1) % 4 if cur in order else 1]
                song.reanalyze()
                st = song.analysis.stats
                print(f"{GREEN}✓{RESET} density: {o.density} · "
                      f"{st['density']:.0%} of syllables lit"
                      f"{DIM} (a song you saved in the editor keeps its own){RESET}")
            elif c == "s":
                toggle_strict(song)
            elif c == "w":
                o.model = (_MODELS[(_MODELS.index(o.model) + 1) % len(_MODELS)]
                           if o.model in _MODELS else "small")
                print(f"{GREEN}✓{RESET} whisper model: {o.model}"
                      f"{DIM} (next listening pass){RESET}")
            elif c == "o":
                shift_offset(song)
            elif c == "h":
                save_html(song)
            elif c in ("b", "", "q"):
                return
        except Exception as exc:
            print(f"  {RED}✗ {exc}{RESET}")


def menu(song: Song, theme: list) -> bool:
    """Action loop for a loaded song. Returns False to quit the app."""
    while True:
        ear = song.apply_ear()
        if ear:
            print(f"  {GREEN}✓{RESET} {ear}")
        st = song.analysis.stats
        print(f"""
  {BOLD}{song.title}{RESET}
  {DIM}{song.source} · {st['families']} rhymes · """
              f"""{st['density']:.0%} of syllables lit · density """
              f"""{RESET}{ACCENT}{density_name(song)}{RESET}{DIM}"""
              f"""{f' · {corrections(song)} of your fixes' if song.edited else ''}{RESET}"""
              f"""{chr(10) + '  ' + DIM + '◷ ' + song.listening_note() + RESET if song.listening_note() else ''}

{studio.menu_rows() if studio else ''}    {ACCENT}4{RESET}  play along           {DIM}music + synced lyrics{RESET}
    {ACCENT}5{RESET}  show rhyme map       {DIM}here in the terminal{RESET}
    {ACCENT}6{RESET}  fix the rhymes       {DIM}paint · erase · review, right here{RESET}

    {ACCENT}−{RESET} / {ACCENT}+{RESET} fewer / more rhymes    {ACCENT}n{RESET} new song    {ACCENT},{RESET} settings    {ACCENT}q{RESET} quit""")
        try:
            c = input(f"\n{ACCENT}rhymescope >{RESET} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            cleanup(song)
            return False
        try:
            if studio and studio.action(c, song, theme):
                pass
            elif c == "4":
                karaoke(song)
            elif c == "5":
                show_map(song)
            elif c == "6":
                edit_rhymes(song, theme[0])
            elif c in ("-", "−", "_", "+", "="):
                change_density(song, +1 if c in ("+", "=") else -1)
            elif c in (",", ".", "s"):
                _settings(song, theme)
            elif c in ("n", ""):
                cleanup(song)
                return True
            elif c in ("q", "quit", "exit"):
                cleanup(song)
                return False
        except Exception as exc:
            print(f"  {RED}✗ {exc}{RESET}")


def main() -> None:
    ap = argparse.ArgumentParser(description="RhymeScope terminal app")
    ap.add_argument("input", nargs="?", default=None,
                    help="song name or audio/.txt file (optional)")
    ap.add_argument("--model", default="small",
                    help="whisper model size (default small)")
    ap.add_argument("--language", default=None,
                    help="force language code, e.g. en")
    ap.add_argument("--density", default="rich",
                    choices=["clean", "balanced", "rich", "everything"],
                    help="how much of the rhyme map to show (default rich)")
    ap.add_argument("--no-ear", action="store_true",
                    help="rhymes from the lyrics only — don't listen for "
                         "accent bends in the vocal")
    ap.add_argument("--strict", action="store_true",
                    help="exact vowel matching only (no slant/bend rhymes)")
    ap.add_argument("--no-online", action="store_true",
                    help="skip online lyrics; always transcribe")
    ap.add_argument("--no-save", action="store_true",
                    help="don't write the .rhymes.html map")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="shift lyric timing by N seconds (positive = later)")
    if studio:
        studio.add_args(ap)
    opts = ap.parse_args()
    theme = [getattr(opts, "theme", "light")]

    print(_BANNER)
    pending = opts.input
    while True:
        if pending is None:
            print(f"  {DIM}Type a song name (e.g. “Kendrick Lamar - HUMBLE”),"
                  f" a file path,"
                  + (" “album <name>” for a whole album," if studio else "") +
                  f" or 'q' to quit.{RESET}")
            try:
                pending = input(f"\n{ACCENT}♪ >{RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not pending:
                continue
            if pending.lower() in ("q", "quit", "exit"):
                break
        low = pending.lower()
        if studio and low.startswith(("album ", "album:")):
            studio.album_mode(pending[6:].strip(" :"), opts, theme)
            pending = None
            continue
        song = Song(opts)
        try:
            song.load(pending)
        except Exception as exc:
            print(f"  {RED}✗ {exc}{RESET}")
            pending = None
            continue
        pending = None
        if studio and studio.one_shot(song, opts, theme):
            break
        if not menu(song, theme):
            break
    print(f"{DIM}bye.{RESET}")


if __name__ == "__main__":
    main()
