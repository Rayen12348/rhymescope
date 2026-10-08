"""
fetch.py — turn a song name into (audio file, real lyrics).

Audio:  yt-dlp, best-audio stream of the top YouTube search hit.
Lyrics: LRCLIB (free, no key) first; Genius page scrape as fallback.
        Whisper transcription stays as the caller's last resort.
"""

from __future__ import annotations

import pathlib
import re

import requests

_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Rhymescope/1.0"}


# --------------------------------------------------------------------------
# network: retry the transient, remember the failure
# --------------------------------------------------------------------------
#
# Every lyrics source used to swallow its own errors, so a dropped
# connection or a slow LRCLIB looked exactly like "this song has no lyrics"
# — and the app said so, for songs sitting right there on Genius. Requests
# now retry what is worth retrying, and a lookup that failed because the
# network did is reported as that, not as a missing song.

_NET_FAILURES: list = []          # this lookup's unrecoverable errors
_NET_OK = [0]                     # …and how many requests did get through


def _get(url: str, **kw):
    import time as _time
    kw.setdefault("headers", _UA)
    kw.setdefault("timeout", 20)
    last = None
    for attempt in range(2):
        try:
            r = requests.get(url, **kw)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 1:
                _time.sleep(1.5)
                continue
            _NET_OK[0] += 1
            return r
        except (requests.ConnectionError, requests.Timeout) as exc:
            last = exc
            if attempt < 1:
                _time.sleep(1.0)
    _NET_FAILURES.append(f"{url.split('/')[2]}: {last}")
    raise last


def last_lookup_failed_on_network() -> bool:
    """True when the most recent lookup found nothing AND could not reach
    at least one lyrics site — i.e. "not found" may really be "not asked"."""
    return bool(_NET_FAILURES)


# --------------------------------------------------------------------------
# audio download
# --------------------------------------------------------------------------

class _Blocked(Exception):
    """A download that came back empty — worth another rung of the ladder."""


class _Hush:
    """yt-dlp logs every error and retry to stderr on top of raising it,
    and those lines start with a \\r that eats our own prefix. We report
    the failure ourselves once the whole ladder is out of rungs."""

    def debug(self, msg): pass
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass


# transient/blocked signatures worth retrying: throttling, a refused
# stream URL, a dropped connection. Matched on the HTTP code so a song
# title that happens to contain "503" can't look like a failure.
_HTTP_FAIL = re.compile(r"HTTP Error (40[39]|429|5\d\d)")
_SOFT_FAIL = ("unable to download", "got error", "giving up", "timed out",
              "connection reset", "connection aborted", "temporarily",
              "no downloadable result", "nothing came back")


def _clean(exc) -> str:
    """yt-dlp writes \r into its messages to overwrite its progress bar;
    printed as-is it erases whatever prefix the caller put in front."""
    msg = re.sub(r"\s*\r\s*", " ", str(exc)).strip()
    return re.sub(r"^(ERROR:\s*)+", "", msg).strip() or exc.__class__.__name__


def _transient(exc) -> bool:
    msg = _clean(exc).lower()
    return bool(_HTTP_FAIL.search(str(exc))) or any(
        s in msg for s in _SOFT_FAIL)


def download_audio(query: str, dest_dir: pathlib.Path,
                   progress=None) -> tuple[pathlib.Path, dict]:
    """Download the best audio of the top YouTube hit for `query`.
    Returns (file path, info dict with title/uploader/duration).
    `progress(pct)` is called with 0..1 during the download."""
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError

    dest_dir.mkdir(exist_ok=True)

    def hook(d):
        if progress and d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                progress(d.get("downloaded_bytes", 0) / total)

    # bias plain-text searches toward audio uploads: music videos carry
    # intros/skits that shift every lyric timestamp (sync drift)
    if not query.startswith(("http://", "https://")):
        query = f"{query} audio"

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(dest_dir / "%(title)s.%(ext)s"),
        "noplaylist": True,
        "default_search": "ytsearch1",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,       # silence yt-dlp's own bar; we render our own
        "logger": _Hush(),
        "progress_hooks": [hook],
        # yt-dlp's library defaults are 0 retries (its CLI uses 10), so a
        # single 503 from one googlevideo host would kill the download
        "retries": 10,
        "fragment_retries": 10,
        "extractor_retries": 3,
    }
    def _dl(o):
        with YoutubeDL(o) as ydl:
            info = ydl.extract_info(query, download=True)
            if info and "entries" in info:         # search result wrapper
                # a failed download leaves the entry empty rather than
                # raising, so an empty list means "nothing came back"
                hits = [e for e in (info["entries"] or []) if e]
                if not hits:
                    raise _Blocked(f"no downloadable result for “{query}”")
                info = hits[0]
            if not info:
                raise _Blocked(f"nothing came back for “{query}”")
            return info, pathlib.Path(ydl.prepare_filename(info))

    # YouTube failures come in waves: stale cached client tokens, one
    # blocked player client (403), or a host refusing the stream URL the
    # default clients hand out (503). The ladder: 1) normal attempt ·
    # 2) drop the cache, rotate through fallback player clients · 3) the
    # android client alone — the defaults sometimes return an HLS URL that
    # 503s while android still serves a plain progressive one ·
    # 4) authenticate with the local Firefox cookies, which survives blocks
    # that stop every anonymous client.
    def _clients(*names):
        return dict(opts, extractor_args={
            "youtube": {"player_client": list(names)}})

    retry = _clients("default", "android", "tv")
    attempts = [opts, retry, _clients("android")]
    if (pathlib.Path.home() / ".mozilla" / "firefox").exists():
        attempts.append(dict(retry, cookiesfrombrowser=("firefox",)))

    info = path = last = None
    for k, o in enumerate(attempts):
        try:
            info, path = _dl(o)
            break
        except (DownloadError, _Blocked) as exc:
            if not _transient(exc):
                raise RuntimeError(_clean(exc)) from exc
            last = exc
            if k == 0:
                with YoutubeDL(opts) as ydl:
                    ydl.cache.remove()
    if info is None:
        raise RuntimeError(
            f"YouTube wouldn’t serve this one — {_clean(last)}. "
            f"It is usually temporary; try again, or pass a local audio file.")

    meta = {
        "title": info.get("title") or query,
        "uploader": info.get("uploader") or "",
        "duration": info.get("duration"),
    }
    return path, meta


# --------------------------------------------------------------------------
# albums
# --------------------------------------------------------------------------

def album_tracks(query: str) -> tuple[str, str, list[str]] | None:
    """Resolve an album name → (album_title, artist, [track titles]).
    iTunes Search API first, Deezer as fallback — both keyless."""
    try:
        r = requests.get("https://itunes.apple.com/search",
                         params={"term": query, "entity": "album",
                                 "limit": 1},
                         headers=_UA, timeout=30)
        r.raise_for_status()
        hits = r.json().get("results") or []
        if hits:
            alb = hits[0]
            r2 = requests.get("https://itunes.apple.com/lookup",
                              params={"id": alb["collectionId"],
                                      "entity": "song"},
                              headers=_UA, timeout=30)
            r2.raise_for_status()
            tracks = [x["trackName"] for x in r2.json().get("results", [])
                      if x.get("wrapperType") == "track"]
            if tracks:
                return alb["collectionName"], alb["artistName"], tracks
    except Exception:
        pass
    try:
        r = requests.get("https://api.deezer.com/search/album",
                         params={"q": query}, headers=_UA, timeout=30)
        r.raise_for_status()
        data = r.json().get("data") or []
        if data:
            alb = data[0]
            r2 = requests.get(
                f"https://api.deezer.com/album/{alb['id']}/tracks",
                headers=_UA, timeout=30)
            r2.raise_for_status()
            tracks = [t["title"] for t in r2.json().get("data", [])]
            if tracks:
                return alb["title"], alb["artist"]["name"], tracks
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------
# lyrics
# --------------------------------------------------------------------------

def _norm_ascii(s: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


_NOISE_TOKENS = {"official", "video", "audio", "lyrics", "lyric", "topic",
                 "vevo", "music", "records", "feat", "prod", "remix",
                 "explicit", "clean", "the", "featuring", "ft", "with",
                 "hd", "hq", "full", "album", "song", "track", "visualizer",
                 "live", "session", "performance", "cover", "version",
                 "edit", "mix", "mixtape", "single", "release", "premiere",
                 "remaster", "remastered", "deluxe", "bonus", "extended",
                 "instrumental", "slowed", "reverb", "sped", "loop",
                 "new", "out", "now", "watch", "subscribe", "channel",
                 "uploads", "uploaded", "provided", "auto", "generated"}


def _tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", _norm_ascii(s or "").lower())
            if len(t) > 2 and t not in _NOISE_TOKENS}


def _artist_ok(candidate: str | None, hint: set[str]) -> bool:
    """Does the candidate artist overlap the query/uploader tokens?
    Substring both ways, so 'Eminem' matches an 'EminemVEVO' channel."""
    for a in _tokens(candidate):
        for h in hint:
            if a in h or h in a:
                return True
    return False


# --------------------------------------------------------------------------
# reading a YouTube title the way a person does
# --------------------------------------------------------------------------
#
# "Rakim x Kurupt x Masta Killa - Rabbit (feat. MAC SHAWN 100) [Official
# Music Video]" is not eight search words. It is ARTISTS - TITLE (EXTRAS),
# and each part answers a different question: the title says which song,
# the artists say whose, the extras are mostly noise. Treating it as one
# bag of words broke lookups three ways — the junk went into the search
# (so sites returned nothing at all), the right result could not cover
# "half the words" of a bloated title, and titles written in symbols
# ("Pu$$y") tokenised to nothing and could never match. So a title is
# parsed into its parts first, and each part is used for what it is.

_DASH = re.compile(r"\s+[-–—|｜~]\s+")
_ARTIST_SEP = re.compile(r"\s+(?:x|&|and|feat\.?|ft\.?|featuring|with|vs\.?)\s+"
                         r"|\s*[,/]\s*|\s+×\s+", re.I)
_FEAT = re.compile(r"\b(?:feat\.?|ft\.?|featuring)\s+", re.I)
_TITLE_NOISE = re.compile(
    r"\b(official|music|video|audio|lyrics?|lyric video|hd|hq|4k|visualizer|"
    r"explicit|new|\d{4}|v\d|remaster(ed)?|topic|mv)\b|\+", re.I)


def _norm_title(s: str | None) -> str:
    """Unicode look-alikes to plain characters, quotes out, $ as S —
    "＂Microphone Fiend＂" and "Pu$$y" become text that can be compared."""
    import unicodedata
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("$", "s")
    s = re.sub(r"[\"“”„‘’'`´]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _brackets(s: str) -> tuple[str, str]:
    """(text outside any brackets, the text inside them)."""
    inside = " ".join(re.findall(r"[(\[{]([^)\]}]*)[)\]}]", s))
    outside = re.sub(r"[(\[{][^)\]}]*[)\]}]", " ", s)
    return re.sub(r"\s+", " ", outside).strip(), inside


def _squash(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _norm_ascii(_norm_title(s)).lower())


def _words(s: str) -> list:
    return re.findall(r"[a-z0-9]+", _norm_ascii(_norm_title(s)).lower())


class _Ask:
    """What a search is actually asking for: a title, maybe artists, and
    a few words that should tip a tie (the extras, what you typed)."""

    def __init__(self, raw: str, typed: str | None = None):
        t = _norm_title(raw)
        body, extras = _brackets(t)
        feats = [_TITLE_NOISE.sub(" ", f)
                 for grp in re.findall(r"[(\[{]([^)\]}]*)[)\]}]", t)
                 for f in re.findall(r"(?:feat\.?|ft\.?|featuring)\s+(.+)",
                                     grp, flags=re.I)]
        parts = _DASH.split(body)
        if len(parts) >= 2:
            left, right = parts[0], " ".join(parts[1:])
        else:
            left, right = "", body
        # a "feat." on the title side belongs to the artists
        m = _FEAT.search(right)
        if m:
            feats.append(right[m.end():])
            right = right[:m.start()]
        m = _FEAT.search(left)
        if m:
            feats.append(left[m.end():])
            left = left[:m.start()]
        self.title = re.sub(r"\s+", " ", _TITLE_NOISE.sub(" ", right)).strip(" -")
        artists = [a for a in _ARTIST_SEP.split(_TITLE_NOISE.sub(" ", left))
                   if _squash(a)]
        for f in feats:
            artists += [a for a in _ARTIST_SEP.split(f) if _squash(a)]
        self.artists = [a.strip() for a in artists]
        # the other reading of "A - B", for uploads written title-first
        self.swapped = None
        if len(parts) >= 2 and self.artists:
            self.swapped = (re.sub(r"\s+", " ", _TITLE_NOISE.sub(" ", left)).strip(),
                            [right.strip()])
        prefer = set(_words(extras)) | set(_words(typed or ""))
        self.prefer = {w for w in prefer
                       if len(w) > 2 and w not in _NOISE_TOKENS
                       and w not in set(_words(self.title))}

    def queries(self, raw: str, typed: str | None) -> list:
        """Searches to try, MOST SPECIFIC FIRST — a source may stop at the
        first query that returns anything, and "Alright" alone returns a
        German rock band before it returns Kendrick Lamar."""
        out = []
        lead = self.artists[0] if self.artists else ""
        if self.title and lead:
            out.append(f"{lead} {self.title}")
        if typed and not lead and self.prefer & set(_words(typed)):
            out.append(_clean_query(_norm_title(typed)))    # "kendrick lamar alright"
        if self.title and not lead:
            # no artist in the title — but uploads often park it in
            # brackets: "Much Too Much (the church)"
            for grp in re.findall(r"[(\[{]([^)\]}]*)[)\]}]", _norm_title(raw)):
                g = re.sub(r"\s+", " ", _TITLE_NOISE.sub(" ", grp)).strip()
                if g and not _FEAT.match(g):
                    out.append(f"{self.title} {g}")
        if self.title and len(self.artists) > 1:
            out.append(f"{self.title} {' '.join(self.artists[:3])}")
        if typed:
            out.append(_clean_query(_norm_title(typed)))
        out.append(_clean_query(_norm_title(raw)))
        if self.title:
            out.append(self.title)
        seen, uniq = set(), []
        for q in out:
            k = q.lower().strip(" +-")
            if k and k not in seen:
                seen.add(k)
                uniq.append(q.strip(" +-"))
        return uniq


def _contains_run(long: list, short: list) -> bool:
    n = len(short)
    return any(long[i:i + n] == short for i in range(len(long) - n + 1))


def _title_match(candidate: str | None, title: str) -> bool:
    """Same song title, allowing for how differently sites write one:
    "N.Y. State of Mind" / "NY State of Mind", "Pu$$y (RNT)" / "Pu$$y",
    "Smack You (Suge Knight & Ja Rule Diss)" / "Smack You"."""
    import difflib
    cand, _x = _brackets(_norm_title(candidate))
    q, c = _squash(title), _squash(cand)
    if not q or not c:
        return False
    if q == c:
        return True
    qw, cw = _words(title), _words(cand)
    short, long = sorted((qw, cw), key=len)
    if short and _contains_run(long, short) and len(short) / len(long) >= 0.5:
        return True
    return difflib.SequenceMatcher(None, q, c).ratio() >= 0.88


def _artist_match(candidate: str | None, artists: list) -> bool:
    """Does any artist we were given appear in the candidate's credits?
    Credits are compared squashed, so "A$AP Rocky" meets "ASAP Rocky" and
    "JAY-Z" meets "Jay Z"; one artist is enough — a feature credit that one
    site lists and another drops must not sink the match."""
    cand = _squash(candidate)
    cand_w = set(_words(candidate))
    for a in artists:
        sq = _squash(a)
        if len(sq) >= 3 and sq in cand:
            return True
        aw = [w for w in _words(a) if len(w) > 1 and w not in _NOISE_TOKENS]
        if aw and all(w in cand_w for w in aw):
            return True
    return False


_NOT_A_SONG = re.compile(r"translation|traducci|übersetzung|перевод|tradu[cç]|"
                         r"çeviri|tłumacz|vertaling|listening log|playlist|"
                         r"\blist\b|annotated|tracklist", re.I)


def _is_song_page(title: str | None, artist: str | None) -> bool:
    """Genius hosts translations, reading lists and user playlists next to
    the songs, titled with the song's own name — never lyrics we want."""
    a = artist or ""
    return not (a.lower().startswith("genius ") or _NOT_A_SONG.search(title or "")
                or _NOT_A_SONG.search(a) or a.lower() in ("community lists",
                                                          "spotify"))


def _relevant(title: str | None, artist: str | None, want) -> bool:
    """Is this result plausibly the song we asked for?

    Every source ranks by popularity, so a search that matched nothing
    relevant still returns something — and returning SOMETHING is the
    worst possible answer: a whole video built on a different song. So a
    result has to carry the TITLE we asked for, and when the request named
    artists, at least one of them in its credits. Nothing else counts
    against it — album names, feature lists and upload tags are not part
    of what the song is called (see _Ask)."""
    if not want:
        return True
    if isinstance(want, _Ask):
        if not _is_song_page(title, artist):
            return False
        readings = [(want.title, want.artists)]
        if want.swapped:
            readings.append(want.swapped)
        if any(t and _title_match(title, t)
               and (not a or _artist_match(artist, a))
               for t, a in readings):
            return True
        # Typed without a dash — "Notorious BIG Juicy", "Kendrick Lamar
        # HUMBLE" — the artist and the title run together, so the whole
        # phrase is not the title. It still names this song if the
        # result's title appears inside it and what is left over is the
        # result's artist.
        if not want.artists:
            qw = _words(want.title)
            cw = _words(_brackets(_norm_title(title))[0])
            if cw and len(cw) < len(qw) and _contains_run(qw, cw):
                rest = " ".join(w for w in qw if w not in cw)
                return bool(rest) and _artist_match(artist, [rest])
        return False
    # (a bare word set — the old interface, kept for any other caller)
    covered = lambda got: {w for w in want
                           if any(w in g or g in w for g in got)}
    in_title = covered(_tokens(title))
    if not in_title:
        return False
    hits = len(in_title | covered(_tokens(artist)))
    return hits >= min(len(want), max(2, (len(want) + 1) // 2))


_VERSION_WORDS = {"remix", "live", "cover", "instrumental", "acapella",
                  "acappella", "slowed", "sped", "reverb", "demo", "karaoke",
                  "remastered", "extended", "radio", "clean", "edited",
                  "freestyle", "mashup", "bootleg", "flip", "vip"}


def _version_penalty(title: str | None, queries: list) -> float:
    """A remix, a live take or a cover that you did not ask for.

    Its title matches as well as the original's — often better, it is the
    original's title plus a tag — and its lyrics can differ. The words that
    mark a version are stripped from titles as noise, which is right for
    matching and wrong for ranking, so they are checked here on the raw
    title against what was actually asked for."""
    asked = set(re.findall(r"[a-z]+", " ".join(queries).lower()))
    got = set(re.findall(r"[a-z]+", (title or "").lower()))
    return 1.0 if (got & _VERSION_WORDS) - asked else 0.0


def _title_score(title: str | None, want) -> float:
    """How precisely a track title matches what we asked for, 0..1.
    Extra words cost ("Juicy" beats "Juicy J - Riley"). Orders candidates
    that already passed `_relevant`; never admits or rejects one."""
    if not want:
        return 0.0
    if isinstance(want, _Ask):
        q = set(_words(want.title))
        got = set(_words(_brackets(_norm_title(title))[0]))
        if not q:
            return 0.0
        return len(q & got) / len(q) - 0.15 * len(got - q)
    got = _tokens(title)
    hit = {w for w in want if any(w in g or g in w for g in got)}
    extra = len(got - {g for g in got
                       if any(w in g or g in w for w in want)})
    return len(hit) / len(want) - 0.15 * extra


def fetch_lyrics(query: str, duration: float | None = None,
                 artist: str | None = None, typed: str | None = None
                 ) -> tuple[str, str, list | None, float | None] | None:
    """Look up real lyrics for `query`.
    Returns (lyrics, source, sync, track_duration) or None. `sync` is a
    list of (seconds, line_text) — one entry per lyrics line, in order —
    when timestamped lyrics are available (LRCLIB), else None.
    If `duration` (seconds) is given, the hit with the closest track
    length wins — timestamps from a different edit drift by seconds.

    `typed` is what the user actually asked for, which is stronger
    evidence of the song they mean than the uploader's title for it.
    `artist` is a free-text artist hint (uploader channel, the user's
    typed query …). Songs by different artists share titles — 'Deja Vu'
    is Eminem AND Olivia Rodrigo — so the sources search WITH the artist
    tokens and PREFER hits whose artist matches. A mismatch demotes a hit
    but never vetoes it, because the hint is usually a YouTube channel
    name rather than an actual artist, and there is nothing for it to
    match. What protects against analyzing the wrong song is the title
    check (`_relevant`), which every source applies.

    Priority: LRCLIB with real timestamps that match this audio edit
    (exact sync beats everything), then Genius text (timed from the audio
    by Whisper), then the remaining sources, then plain LRCLIB."""
    ask = _Ask(query, typed)
    a_tokens = _tokens(artist or "") | ask.prefer
    q_tokens = _tokens(query)
    hint = q_tokens | a_tokens
    # WHAT THE SONG IS CALLED vs WHO MIGHT HAVE MADE IT — two different
    # questions, and they need different evidence.
    #
    # `want` decides whether a result is the right song, so it is built
    # from the track title alone. Folding the uploader in there poisons
    # it: searching "Juicy" from a channel called "HipHopClassics" made
    # `hiphopclassics` a word the result had to contain, so "Juicy by The
    # Notorious B.I.G." scored as irrelevant to a search for Juicy.
    #
    # `hint` keeps everything, because for choosing BETWEEN plausible
    # results any scrap of artist signal helps — it just never gets to
    # veto a result on its own.
    # `typed` is what the person actually asked for, and it belongs in
    # `want`: a YouTube title is often just "Alright", while they typed
    # "Kendrick Lamar Alright" — drop that and a German rock band's
    # "Alright" is every bit as good an answer as Kendrick's.
    want = ask
    # several searches, cleanest first: sites that return nothing at all
    # for a title full of upload tags find the song at once without them
    queries = ask.queries(query, typed)
    extra = a_tokens - q_tokens
    extra = {t for t in extra
             if not any(o != t and o in t for o in hint)}
    if extra and queries:
        queries.append(f"{queries[0]} {' '.join(sorted(extra))}")

    import time as _time
    for attempt in range(2):
        _NET_FAILURES.clear()
        _NET_OK[0] = 0
        hit = _lookup(queries, duration, hint, want)
        if hit or not _NET_FAILURES:
            return hit
        if not _NET_OK[0]:
            return None       # nothing answered at all: offline, don't stall
        # found nothing, but only because SOME site could not be reached:
        # that is not an answer yet — give it one more go
        _time.sleep(2)
    return None


def _lookup(queries, duration, hint, want):
    lrc = None
    try:
        lrc = _lrclib(queries, duration, hint, want)
    except Exception:
        pass
    if lrc and lrc[2] is not None:
        track_dur = lrc[3]
        if duration is None or track_dur is None \
                or abs(duration - track_dur) <= 3:
            return lrc
    for fn in (_genius, _azlyrics, _musixmatch):
        try:
            hit = fn(queries, duration, hint, want)
        except Exception:
            hit = None
        if hit:
            return hit
    return lrc


_LRC_TAG = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]")


def parse_lrc(lrc: str) -> list[tuple[float, str]]:
    """Parse LRC text into [(seconds, line_text), …] sorted by time."""
    events: list[tuple[float, str]] = []
    for raw in lrc.splitlines():
        tags = _LRC_TAG.findall(raw)
        if not tags:
            continue
        text = _LRC_TAG.sub("", raw).strip()
        for mm, ss in tags:
            events.append((int(mm) * 60 + float(ss), text))
    events.sort(key=lambda e: e[0])
    return events


def _clean_query(q: str) -> str:
    """Strip YouTube-title noise so lyric search matches the actual song."""
    q = re.sub(r"[([].*?(official|video|audio|lyric|lyrics|hd|4k|remaster"
               r"|explicit|clean|visualizer|mv)[^)\]]*[)\]]", " ", q, flags=re.I)
    q = re.sub(r"\b(official|video|audio|lyrics?|hd|4k|visualizer)\b", " ", q, flags=re.I)
    return re.sub(r"\s+", " ", q).strip()


def _lrclib(queries: list[str], duration: float | None,
            hint: set | None = None,
            want: set | None = None) -> tuple[str, str] | None:
    want = want or set(hint or ()) or _tokens(queries[-1])
    first: list = []
    hits: list = []
    for q in queries[:5]:
        r = _get("https://lrclib.net/api/search",
                 params={"q": _clean_query(q)}, headers=_UA, timeout=30)
        r.raise_for_status()
        found = [h for h in r.json() if (h.get("plainLyrics") or "").strip()
                 and _relevant(h.get("trackName"), h.get("artistName"), want)]
        if not found:
            continue
        # Prefer a hit whose artist matches what we know. Not matching is
        # NOT disqualifying (a channel name is usually not an artist) —
        # but it is a reason to try the next query before settling, since
        # a bare title finds every song of that name before the one meant.
        good = ([h for h in found if _artist_ok(h.get("artistName"), hint)]
                if hint else [])
        if good:
            hits = good
            break
        first = first or found
    hits = hits or first
    if not hits:
        return None

    # prefer synced (timestamped) hits, then the closest track duration:
    # a 10s duration gap means every lyric drifts by seconds
    def rank(h):
        has_sync = bool((h.get("syncedLyrics") or "").strip())
        dur_gap = (abs(h["duration"] - duration)
                   if duration and h.get("duration") else 999)
        # when the audio's length is unknown, nothing separates two hits
        # that both contain the word we searched for — so the tighter
        # title wins ("Juicy" over "Juicy J - Riley")
        return (_version_penalty(h.get("trackName"), queries), not has_sync,
                dur_gap, -_title_score(h.get("trackName"), want))

    h = min(hits, key=rank)
    src = f"LRCLIB — {h.get('artistName', '?')} · {h.get('trackName', '?')}"
    track_dur = h.get("duration")

    sync = None
    if (h.get("syncedLyrics") or "").strip():
        sync = parse_lrc(h["syncedLyrics"])
        if sync:
            # derive the plain text FROM the synced lines so analysis line
            # numbers map 1:1 onto timestamps (karaoke + video rely on this)
            return "\n".join(t for _, t in sync), src, sync, track_dur
    return h["plainLyrics"].strip(), src, sync, track_dur


def _genius_search(q: str) -> list[dict]:
    """Song results for `q`. Genius runs two search endpoints and they
    disagree: /api/search returns one flat, well-ranked hit list, while
    /api/search/multi buckets hits by type — and its "song" bucket comes
    back EMPTY for plenty of real songs (soundtrack and battle cuts, say)
    that /api/search ranks first, so reading only that bucket dropped them
    on the floor and sent the caller to Whisper. Pool both, flat list
    first, deduped by song id."""
    out: list[dict] = []
    seen: set = set()
    for url in ("https://genius.com/api/search",
                "https://genius.com/api/search/multi"):
        try:
            r = _get(url, params={"q": q}, headers=_UA, timeout=30)
            r.raise_for_status()
            resp = r.json()["response"]
        except Exception:
            continue        # one endpoint down is not the end of the search
        hits = list(resp.get("hits") or [])
        for section in resp.get("sections") or []:
            # "song" is the typed bucket; "top_hit" carries the same
            # shape and is the only bucket filled for some queries
            if section.get("type") in ("song", "top_hit"):
                hits += section.get("hits") or []
        for h in hits:
            res = h.get("result") or {}
            if res.get("_type") != "song":
                continue
            key = res.get("id") or res.get("url")
            if key in seen:
                continue
            seen.add(key)
            out.append(res)
    return out


def _genius(queries: list[str], duration: float | None,
            hint: set | None = None,
            want: set | None = None) -> tuple[str, str] | None:
    # first hit whose artist matches the hint wins; a bare popularity-first
    # pick grabs the wrong song whenever two artists share a title
    want = want or set(hint or ()) or _tokens(queries[-1])
    # Rank, don't refuse. Every candidate has to look like the song we
    # asked for (_relevant); among those, one whose ARTIST also matches
    # what we know wins, and otherwise the search engine's own ranking
    # decides. The old rule required the artist to match and returned
    # nothing when none did — but the artist hint is built from the
    # uploader's channel name, which for "Juicy" uploaded by
    # "HipHopClassics" matches no artist alive. Songs that are sitting
    # right there on Genius came back as "no lyrics online" because of it.
    best = None
    for qi, q in enumerate(queries[:4]):
        try:
            cands = _genius_search(_clean_query(q))
        except Exception:
            continue
        for i, c in enumerate(cands):
            name = ((c.get("primary_artist") or {}).get("name")
                    or c.get("artist_names") or "")
            if not _relevant(c.get("title") or c.get("full_title"),
                             name, want):
                continue
            matched = bool(hint) and _artist_ok(name, hint)
            key = (_version_penalty(c.get("title"), queries), not matched,
                   qi, -round(_title_score(c.get("title"), want), 3), i)
            if best is None or key < best[0]:
                best = (key, c)
        if best is not None and not best[0][0] and not best[0][1]:
            break              # an unversioned artist match: nothing beats it
    top = best[1] if best else None
    if not top:
        return None
    url, title = top["url"], top.get("full_title", "")
    text = _genius_lyrics(url)
    return (text, f"Genius — {title}", None, None) if text else None


def _genius_lyrics(url: str) -> str | None:
    """The lyrics text of one Genius song page, page furniture stripped."""
    try:
        page = _get(url, headers=_UA, timeout=30).text
    except Exception:
        return None
    blocks = _div_blocks(page, 'data-lyrics-container="true"')
    if not blocks:
        return None
    # the first container embeds a page header (contributors, translation
    # language list) — remove those blocks before stripping tags
    text = "\n".join(_strip_div_block(b, "data-exclude-from-selection")
                     for b in blocks)
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = _html_unescape(text)
    # drop [Verse 1], [Chorus] … section markers
    text = re.sub(r"^\[.*?\]$", "", text, flags=re.M)
    # residual Genius page furniture that survives tag stripping
    text = text.replace("You might also like", "\n")
    text = re.sub(r"^\d+\s*Contributors.*$", "", text, flags=re.M)
    text = re.sub(r"^Translations\b.*$", "", text, flags=re.M)
    text = re.sub(r"^See .+ Live\b.*$|^Get tickets as low as.*$", "",
                  text, flags=re.M)
    text = re.sub(r"\d*Embed\s*$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text or None


def _div_blocks(page: str, attr_marker: str) -> list[str]:
    """Inner HTML of every <div …marker…>…</div>, nested divs handled
    (a lazy .*?</div> regex truncates at the first nested close)."""
    open_re = re.compile(r"<div[^>]*" + re.escape(attr_marker) + r"[^>]*>")
    tag_re = re.compile(r"<div\b|</div>")
    out, i = [], 0
    while (m := open_re.search(page, i)):
        depth, end, i = 0, len(page), len(page)
        for m2 in tag_re.finditer(page, m.start()):
            depth += 1 if m2.group(0) != "</div>" else -1
            if depth == 0:
                end, i = m2.start(), m2.end()
                break
        out.append(page[m.end():end])
    return out


def _strip_div_block(html_s: str, attr_marker: str) -> str:
    """Remove every <div …marker…>…</div> block, nested divs included."""
    out, i = [], 0
    open_re = re.compile(r"<div[^>]*" + re.escape(attr_marker) + r"[^>]*>")
    tag_re = re.compile(r"<div\b|</div>")
    while True:
        m = open_re.search(html_s, i)
        if not m:
            out.append(html_s[i:])
            return "".join(out)
        out.append(html_s[i:m.start()])
        depth, j = 0, len(html_s)
        for m2 in tag_re.finditer(html_s, m.start()):
            depth += 1 if m2.group(0) != "</div>" else -1
            if depth == 0:
                j = m2.end()
                break
        i = j


def _azlyrics(queries: list[str], duration: float | None,
              hint: set | None = None,
              want: set | None = None) -> tuple[str, str] | None:
    """Scrape lyrics from AZLyrics.com (fallback source)."""
    q = _clean_query(queries[-1]).lower()
    parts = q.split(" - ", 1) if " - " in q else [q, ""]
    if len(parts) < 2 or not parts[1]:
        return None

    artist, song = parts[0].replace(" ", "-"), parts[1].replace(" ", "-")
    url = f"https://www.azlyrics.com/lyrics/{artist}/{song}.html"
    try:
        r = _get(url, headers=_UA, timeout=30)
        r.raise_for_status()
    except Exception:
        return None

    # AZLyrics wraps lyrics in a specific div pattern
    match = re.search(r'<div class="col-xs-12 col-lg-8 col-md-8 col-sm-12">(.*?)<div class="page-nav"',
                      r.text, re.S)
    if not match:
        match = re.search(r'<!-- start of lyrics -->(.*?)<!-- end of lyrics -->', r.text, re.S | re.I)
    if not match:
        return None

    text = match.group(1)
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = _html_unescape(text)
    text = re.sub(r"^\[.*?\]$", "", text, flags=re.M)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return (text, f"AZLyrics — {artist.replace('-', ' ')} · {song.replace('-', ' ')}", None, None) \
           if text and len(text) > 50 else None


def _musixmatch(queries: list[str], duration: float | None,
                hint: set | None = None,
                want: set | None = None) -> tuple[str, str] | None:
    """Scrape lyrics from Musixmatch.com (fallback source)."""
    q = _clean_query(queries[0])
    try:
        # Direct Musixmatch API-like request
        r = _get("https://www.musixmatch.com/search",
                        params={"q": q}, headers=_UA, timeout=30)
        r.raise_for_status()
    except Exception:
        return None

    # Look for song links in search results. The FIRST link is the search
    # engine's best guess, which for a query it matched nothing in is just
    # its most popular song — so it has to look like what we asked for
    # before we take it (see _relevant).
    want = want or set(hint or ()) or _tokens(q)
    match = next((m for m in
                  re.finditer(r'href="(/lyrics/[^"]+)"[^>]*>([^<]+)<', r.text)
                  if _relevant(m.group(2), m.group(1), want)), None)
    if not match:
        return None

    path, title = match.group(1, 2)
    url = "https://www.musixmatch.com" + path if not path.startswith("http") else path

    try:
        page = _get(url, headers=_UA, timeout=30)
        page.raise_for_status()
    except Exception:
        return None

    # Musixmatch uses span elements for lyrics
    blocks = re.findall(r'<span[^>]*?>((?:[^<]|<br\s*/?>\n)+?)</span>', page.text)
    if not blocks:
        # Alternative: look for div-based lyrics structure
        match = re.search(r'<div[^>]+class="[^"]*lyrics[^"]*"[^>]*>(.*?)</div>', page.text, re.S)
        if match:
            blocks = [match.group(1)]

    if not blocks:
        return None

    text = "\n".join(blocks)
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = _html_unescape(text)
    text = re.sub(r"^\[.*?\]$", "", text, flags=re.M)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return (text, f"Musixmatch — {title}", None, None) if text and len(text) > 50 else None


def _html_unescape(s: str) -> str:
    import html
    return html.unescape(s)


# --------------------------------------------------------------------------
# identify a song from what Whisper heard
# --------------------------------------------------------------------------
#
# When the title search comes up empty, the pipeline falls back to
# Whisper's transcript — and that transcript is the weakest input in the
# whole system. It mishears names, invents spellings, and drops ad-libs,
# so every rhyme judged on it is judged on approximately the right words.
#
# But a rough transcript is still enough to RECOGNISE a song, even when it
# is not good enough to analyse one. Searching a lyrics database with a
# few distinctive lines finds the real page, and the real page has the
# real words — correctly spelled, correctly punctuated, with the ad-libs
# the vocal buried. Those get lined up to the audio by the ordinary
# alignment pass, so the timing still comes from the recording rather than
# from the lyrics site.
#
# The danger is obvious and is the same one _relevant exists to stop: a
# search that matches nothing still returns its most popular song. So a
# candidate is only accepted if its lyrics actually look like what we
# heard, measured two ways — see _lyric_agreement.

_WORD = re.compile(r"[a-z']+")


def _lyric_words(text: str) -> list[str]:
    return _WORD.findall(_norm_ascii(text or "").lower())


def _lyric_agreement(heard: str, found: str) -> float:
    """How much two sets of lyrics agree, 0..1.

    Measured on adjacent word PAIRS, and measured in BOTH directions.

    Pairs, because shared vocabulary is cheap and shared word order is
    not: two unrelated songs use most of the same few hundred English
    words, but they do not put them in the same sequence.

    Both directions, because recall alone is trivially gamed by length.
    Genius hosts more than songs — books, scripts, speeches — and a
    40,000-word memoir contains a fair sample of every bigram in English
    by sheer volume. Scored on recall it looks like a perfect match for
    anything; scored on precision it collapses, because almost none of
    ITS pairs appear in a three-minute rap. So the score is the harmonic
    mean of the two, and a length sanity check rejects the mismatch
    outright before it gets that far.

    (Learned the hard way: an early version of this identified an
    unlisted track as `Angela's Ashes`.)"""
    a, b = _lyric_words(heard), _lyric_words(found)
    if len(a) < 20 or len(b) < 20:
        return 0.0
    # a song is not a novel, and it is not a fragment either
    ratio = len(b) / len(a)
    if not 0.3 <= ratio <= 3.0:
        return 0.0
    pa = {(a[i], a[i + 1]) for i in range(len(a) - 1)}
    pb = {(b[i], b[i + 1]) for i in range(len(b) - 1)}
    inter = len(pa & pb)
    if not inter:
        return 0.0
    recall, precision = inter / len(pa), inter / len(pb)
    return 2 * recall * precision / (recall + precision)


def _lyric_probes(transcript: str, n: int = 5) -> list[str]:
    """Lines worth searching a lyrics database with.

    Long lines beat short ones (more to match on), and lines that appear
    only once beat the hook — a chorus is the most repeated thing in the
    song and often the most generic, so it is the worst query. Probes are
    drawn from across the track so one badly-heard passage can't sink the
    identification."""
    seen: dict = {}
    for raw in transcript.splitlines():
        line = " ".join(_lyric_words(raw))
        if len(line.split()) >= 5:
            seen[line] = seen.get(line, 0) + 1
    uniq = [ln for ln, c in seen.items() if c == 1]
    uniq.sort(key=lambda ln: -len(ln))
    return uniq[:n]


def identify_by_lyrics(transcript: str, hint: set | None = None
                       ) -> tuple[str, str, list | None, float | None] | None:
    """Find the real lyrics for a song we could only transcribe.

    Returns the same shape as `fetch_lyrics` (with sync None — the caller
    lines them up against the audio), or None when nothing convincing
    turns up, in which case the transcript stands as the best we have."""
    probes = _lyric_probes(transcript)
    if not probes:
        return None
    best = None
    seen: set = set()
    for probe in probes:
        try:
            cands = _genius_search(probe)
        except Exception:
            continue
        for c in cands[:4]:
            url = c.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            text = _genius_lyrics(url)
            if not text:
                continue
            score = _lyric_agreement(transcript, text)
            # Covers and re-uploads carry the same words as the original,
            # so agreement alone cannot separate them. When we know who
            # we were looking for, nudge their version ahead — it decides
            # nothing on its own, it only breaks a tie the lyrics can't.
            artist = ((c.get("primary_artist") or {}).get("name")
                      or c.get("artist_names") or "")
            if hint and _artist_ok(artist, hint):
                score += 0.05
            if best is None or score > best[0]:
                best = (score, text, c.get("full_title", ""))
        # a strong hit early means the rest of the probes are wasted work
        if best and best[0] >= 0.45:
            break
    # Calibrated against both the right song and wrong ones. The right
    # song scores 0.37 even with 45% of its words mangled; unrelated
    # songs sit at 0.02-0.04. This sits in that gap, nearer the wrong
    # side of it, because the cost is asymmetric: a missed match just
    # leaves the transcript in place, while a false one silently swaps in
    # a different song's words.
    if best and best[0] >= 0.22:
        return (best[1], f"Genius — {best[2]} (found from the audio)",
                None, None)
    return None
