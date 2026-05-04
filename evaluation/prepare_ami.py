"""Download a small AMI video subset and convert it into our benchmark layout.

The AMI Meeting Corpus (Carletta et al., 2005) provides multi-camera
recordings of scenario and non-scenario meetings, distributed by the
University of Edinburgh:

    https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/<id>/video/
    https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/<id>/audio/

⚠️  AMI distributes audio and video as **separate files**: the camera
AVIs (Overhead, Overview*, Corner, Closeup*) are *silent*, and the
audio lives under ``audio/<id>.Mix-Headset.wav`` (one mixed track of
all four headset mics). To get a single video-with-audio file that our
ASR + diarization pipeline can consume, we mux them locally with
ffmpeg into ``data/raw/ami/<id>.mp4``.

We materialize each selected meeting as:
  data/raw/ami/<id>.mp4          (video + Mix-Headset audio, muxed)
  data/gold/ami/<id>.json        ({"summary": ..., "action_items": []})

CLI: ``python -m evaluation.prepare_ami`` (default four ``…a`` meetings);
``python -m evaluation.prepare_ami --extra-six`` appends six more QMSum-test
ids with gold; or pass explicit ``--meetings …`` (see argparse help).

Gold summaries are *reused* from the QMSum benchmark when available
(QMSum's test split happens to include several AMI scenario meetings),
so you don't need to re-annotate. If a meeting isn't covered by QMSum,
we write a stub gold file with an empty summary that you can fill in
manually — ``evaluation.evaluate`` will skip it until the summary is
populated.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

from config import GOLD_DIR, RAW_DIR

# Default subset: 4 AMI scenario meetings spanning all three series in
# QMSum's test split (ES = Edinburgh, IS = ICSI-style, TS = TNO). Picking
# the "a" (kickoff) sessions keeps each meeting at ~15-20 min so the
# eval finishes in a reasonable wall time. All four have QMSum gold
# summaries we can reuse without re-annotating.
DEFAULT_MEETINGS = ["ES2004a", "ES2011a", "IS1003a", "TS3004a"]

# Six more meetings from QMSum's *test* split (same ``general_query`` gold as
# the default ids).  Complements the four ``…a`` kickoffs with ``b`` sessions
# plus one extra series (``TS3011a``) — all have JSON at
# ``Yale-LILY/QMSum/.../data/ALL/test/<id>.json``.
EXTRA_MEETINGS = [
    "ES2004b",
    "ES2004c",
    "ES2011b",
    "IS1003b",
    "TS3004b",
    "TS3011a",
]

_QMSUM_GOLD_URL = (
    "https://raw.githubusercontent.com/Yale-LILY/QMSum/main/"
    "data/ALL/test/{name}.json"
)

_AMI_BASE = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
# Camera priority for the "table view" we want — the AMI series differ:
#   - ES (Edinburgh):  has Overhead + Corner
#   - IS (ICSI-style): has Overhead + Corner
#   - TS (TNO):        has Overview1/Overview2 (no Overhead)
# We try in order and use the first camera that exists for each meeting.
WIDE_ANGLE_CAMERAS = ["Overhead", "Overview1", "Corner", "Closeup1"]
_AUDIO_TRACK = "Mix-Headset"  # single mixed track of all 4 headset mics


def _video_url(meeting_id: str, camera: str) -> str:
    return f"{_AMI_BASE}/{meeting_id}/video/{meeting_id}.{camera}.avi"


def _audio_url(meeting_id: str) -> str:
    return f"{_AMI_BASE}/{meeting_id}/audio/{meeting_id}.{_AUDIO_TRACK}.wav"


def _url_exists(url: str) -> bool:
    """Cheap HEAD check so we don't 404-and-retry mid-stream."""
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status == 200
    except Exception:
        return False


def _resolve_camera(meeting_id: str, preferred: list[str]) -> str | None:
    for cam in preferred:
        if _url_exists(_video_url(meeting_id, cam)):
            return cam
    return None


def _stream_download(url: str, out_path: Path, label: str) -> None:
    """Stream a remote file to disk in 1 MiB chunks (safe for >100MB files)."""
    print(f"[ami] downloading {label}")
    print(f"      {url}")
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp_path, "wb") as f:
        shutil.copyfileobj(resp, f, length=1 << 20)
    tmp_path.rename(out_path)
    size_mb = out_path.stat().st_size / 1e6
    print(f"[ami] saved {out_path.name} ({size_mb:.1f} MB)")


def _mux_video_audio(video_path: Path, audio_path: Path, out_path: Path) -> None:
    """Combine silent AMI video + Mix-Headset audio into a single .mp4.

    Uses ``-c copy`` for the video stream (no re-encoding — fast) and
    AAC for the audio (mp4 doesn't accept raw PCM). ``-shortest`` clips
    to the shorter of the two, which guards against any duration drift.
    """
    print(f"[ami] muxing → {out_path.name}")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    size_mb = out_path.stat().st_size / 1e6
    print(f"[ami] muxed {out_path.name} ({size_mb:.1f} MB)")


def _fetch_qmsum_gold(meeting_id: str, gold_dir: Path) -> bool:
    """Get a gold summary for this meeting from QMSum.

    Resolution order:
      1) Local cache at ``data/gold/qmsum/<id>.json`` (already prepared)
      2) Fetch from the QMSum GitHub repo and convert to our gold schema
         (keep only the general meeting-level summary; action items are
         not annotated in QMSum, so the list is empty).

    Returns True iff a gold file was made available at ``gold_dir``.
    """
    dst = gold_dir / f"{meeting_id}.json"
    if dst.exists():
        return True

    cached = GOLD_DIR / "qmsum" / f"{meeting_id}.json"
    if cached.exists():
        dst.write_text(cached.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[ami] reused QMSum gold → {dst.name}")
        return True

    url = _QMSUM_GOLD_URL.format(name=meeting_id)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            qmsum_meeting = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"[ami] no QMSum gold for {meeting_id} ({e})")
        return False

    gq = qmsum_meeting.get("general_query_list") or []
    summary = (gq[0].get("answer") if gq else "") or ""
    if not summary.strip():
        print(f"[ami] QMSum has no general summary for {meeting_id}")
        return False

    payload = {"summary": summary.strip(), "action_items": []}
    dst.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[ami] fetched QMSum gold → {dst.name}")
    return True


def _write_stub_gold(meeting_id: str, gold_dir: Path) -> None:
    """Drop a placeholder gold so the user can fill it in by hand."""
    dst = gold_dir / f"{meeting_id}.json"
    if dst.exists():
        return
    dst.write_text(
        json.dumps({"summary": "", "action_items": []}, indent=2),
        encoding="utf-8",
    )
    print(f"[ami] wrote stub gold (empty summary) → {dst.name}")


def _prepare_one(
    meeting_id: str,
    raw_dir: Path,
    gold_dir: Path,
    preferred_cameras: list[str],
    keep_intermediate: bool,
) -> bool:
    final_path = raw_dir / f"{meeting_id}.mp4"
    if final_path.exists():
        print(f"[ami] {meeting_id}: muxed video cached, skipping download")
    else:
        cam = _resolve_camera(meeting_id, preferred_cameras)
        if cam is None:
            print(
                f"[ami] ⚠️  no usable camera for {meeting_id} "
                f"(tried {', '.join(preferred_cameras)})"
            )
            return False

        silent_video = raw_dir / f"{meeting_id}.{cam}.silent.avi"
        audio_path = raw_dir / f"{meeting_id}.{_AUDIO_TRACK}.wav"
        try:
            if not silent_video.exists():
                _stream_download(
                    _video_url(meeting_id, cam),
                    silent_video,
                    f"{meeting_id}.{cam}.avi (silent video)",
                )
            if not audio_path.exists():
                _stream_download(
                    _audio_url(meeting_id),
                    audio_path,
                    f"{meeting_id}.{_AUDIO_TRACK}.wav",
                )
            _mux_video_audio(silent_video, audio_path, final_path)
        except (subprocess.CalledProcessError, OSError) as e:
            print(f"[ami] ⚠️  failed to prepare {meeting_id}: {e}")
            return False
        finally:
            if not keep_intermediate:
                # Remove the silent .avi + standalone .wav once muxed —
                # they're 2-3× the size of the final .mp4 combined.
                if final_path.exists():
                    silent_video.unlink(missing_ok=True)
                    audio_path.unlink(missing_ok=True)

    if not _fetch_qmsum_gold(meeting_id, gold_dir):
        _write_stub_gold(meeting_id, gold_dir)
    return True


def prepare(
    meetings: list[str] = DEFAULT_MEETINGS,
    benchmark: str = "ami",
    camera: str | None = None,
    keep_intermediate: bool = False,
) -> list[str]:
    raw_dir = RAW_DIR / benchmark
    gold_dir = GOLD_DIR / benchmark
    raw_dir.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)

    # If user pinned a specific camera, only try that. Otherwise, fall back
    # through the wide-angle priority list per meeting.
    preferred_cameras = [camera] if camera else WIDE_ANGLE_CAMERAS

    # One-time cleanup: an earlier version of this script wrote silent
    # `<id>.avi` files directly. They have no audio stream and break the
    # downstream pipeline. Drop them so we re-prepare with mux.
    for stale in raw_dir.glob("*.avi"):
        if stale.name.endswith(".silent.avi"):
            continue
        print(f"[ami] removing stale silent video: {stale.name}")
        stale.unlink()

    prepared: list[str] = []
    for name in meetings:
        if _prepare_one(name, raw_dir, gold_dir, preferred_cameras, keep_intermediate):
            prepared.append(name)

    return prepared


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meetings",
        nargs="*",
        default=None,
        help=(
            "AMI meeting ids to download (default: DEFAULT_MEETINGS — four "
            "``…a`` sessions). Pass explicit ids to override."
        ),
    )
    parser.add_argument(
        "--extra-six",
        action="store_true",
        help=(
            "Append EXTRA_MEETINGS (six more QMSum-test AMI sessions with "
            "gold summaries) after resolving ``--meetings``."
        ),
    )
    parser.add_argument("--benchmark", default="ami")
    parser.add_argument(
        "--camera",
        default=None,
        choices=[
            "Overhead", "Overview1", "Overview2", "Corner",
            "Closeup1", "Closeup2", "Closeup3", "Closeup4",
        ],
        help=(
            "Pin a specific AMI camera angle. By default we try "
            "Overhead → Overview1 → Corner → Closeup1 per meeting "
            "(different AMI series ship different cameras)."
        ),
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep the silent .avi and standalone .wav after muxing.",
    )
    args = parser.parse_args()

    meetings = list(args.meetings) if args.meetings is not None else list(
        DEFAULT_MEETINGS
    )
    if args.extra_six:
        seen = set(meetings)
        for m in EXTRA_MEETINGS:
            if m not in seen:
                meetings.append(m)
                seen.add(m)

    names = prepare(
        meetings, args.benchmark, args.camera, args.keep_intermediate
    )
    print(f"[ami] prepared {len(names)} meetings under benchmark={args.benchmark!r}")
