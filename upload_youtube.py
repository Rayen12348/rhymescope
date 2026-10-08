#!/usr/bin/env python3
"""
upload_youtube.py — upload a rhyme-map video to YouTube.

One-time setup (Google makes everyone do this, there's no way around it):
  1. https://console.cloud.google.com → create a project
  2. Enable "YouTube Data API v3"
  3. OAuth consent screen → External → add yourself as a test user
  4. Credentials → Create credentials → OAuth client ID → Desktop app
  5. Download the JSON, save it here as  client_secret.json

First run opens your browser to authorize; the token is cached in
token.json so every later upload is fully automatic:

    python upload_youtube.py "song.rhymes.mp4" --title "Lose Yourself — rhyme map"
    python upload_youtube.py video.mp4 --privacy public
"""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
CLIENT_SECRET = ROOT / "client_secret.json"
TOKEN = ROOT / "token.json"
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def _credentials():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = None
    if TOKEN.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN), SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    elif not creds or not creds.valid:
        if not CLIENT_SECRET.exists():
            sys.exit(
                "client_secret.json not found — see the setup steps at the "
                "top of upload_youtube.py"
            )
        flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET),
                                                         SCOPES)
        creds = flow.run_local_server(port=0)
    TOKEN.write_text(creds.to_json(), encoding="utf-8")
    return creds


def upload(video: pathlib.Path, title: str, description: str,
           tags: list[str], privacy: str) -> str:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    yt = build("youtube", "v3", credentials=_credentials())
    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": "10",          # Music
        },
        "status": {"privacyStatus": privacy},
    }
    media = MediaFileUpload(str(video), chunksize=8 * 1024 * 1024,
                            resumable=True)
    req = yt.videos().insert(part="snippet,status", body=body,
                             media_body=media)
    print(f"uploading {video.name} …")
    resp = None
    while resp is None:
        status, resp = req.next_chunk()
        if status:
            print(f"\r    {status.progress():5.0%}", end="", flush=True)
    print()
    return f"https://youtu.be/{resp['id']}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Upload a video to YouTube")
    ap.add_argument("video", help="path to the .mp4")
    ap.add_argument("--title", default=None,
                    help="video title (default: filename)")
    ap.add_argument("--description",
                    default="Color-coded rhyme scheme, one color per rhyme "
                            "family — generated with Rhymescope.")
    ap.add_argument("--tags", default="rhyme scheme,rap,lyrics,rhymes",
                    help="comma-separated tags")
    ap.add_argument("--privacy", default="unlisted",
                    choices=["private", "unlisted", "public"])
    args = ap.parse_args()

    video = pathlib.Path(args.video)
    if not video.is_file():
        sys.exit(f"not found: {video}")
    title = args.title or video.stem.replace(".rhymes", " — rhyme map")

    url = upload(video, title, args.description,
                 [t.strip() for t in args.tags.split(",") if t.strip()],
                 args.privacy)
    print(f"✓ {url}")


if __name__ == "__main__":
    main()
