"""Download a small AMI video subset and convert it into our benchmark layout.

The AMI Meeting Corpus (Carletta et al., 2005) provides multi-camera
recordings of scenario and non-scenario meetings, distributed by the
University of Edinburgh:

    https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/<id>/video/

Each meeting has several camera angles. We download the ``Overhead``
camera by default — it captures all four participants around the table,
which is the most informative single angle for meeting summarization
(and it's only ~40-50 MB compressed per meeting).

We materialize each selected meeting as:
  data/raw/ami/<id>.avi          (downloaded video)
  data/gold/ami/<id>.json        ({"summary": ..., "action_items": []})

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
import urllib.request
from pathlib import Path

from config import GOLD_DIR, RAW_DIR

# Default subset: AMI scenario meetings that overlap with the QMSum test
# split (so gold summaries already exist). Both are ~15-min "kickoff"
# sessions of the remote-control design scenario, with 4 participants.
DEFAULT_MEETINGS = ["ES2004a", "TS3004a"]

_AMI_BASE = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
# Camera priority for the "table view" we want — the AMI series differ:
#   - ES (Edinburgh):  has Overhead + Corner
#   - IS (ICSI-style): has Overhead + Corner
#   - TS (TNO):        has Overview1/Overview2 (no Overhead)
# We try in order and use the first camera that exists for each meeting.
WIDE_ANGLE_CAMERAS = ["Overhead", "Overview1", "Corner", "Closeup1"]


def _video_url(meeting_id: str, camera: str) -> str:
    return f"{_AMI_BASE}/{meeting_id}/video/{meeting_id}.{camera}.avi"


def _camera_exists(meeting_id: str, camera: str) -> bool:
    """Cheap HEAD check so we don't 404-and-retry mid-stream."""
    req = urllib.request.Request(_video_url(meeting_id, camera), method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status == 200
    except Exception:
        return False


def _resolve_camera(meeting_id: str, preferred: list[str]) -> str | None:
    for cam in preferred:
        if _camera_exists(meeting_id, cam):
            return cam
    return None


def _download_video(meeting_id: str, camera: str, out_path: Path) -> None:
    """Stream the AMI video to disk so we don't buffer 40MB+ in RAM."""
    url = _video_url(meeting_id, camera)
    print(f"[ami] downloading {meeting_id}.{camera}.avi")
    print(f"      {url}")
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp_path, "wb") as f:
        shutil.copyfileobj(resp, f, length=1 << 20)  # 1 MiB chunks
    tmp_path.rename(out_path)
    size_mb = out_path.stat().st_size / 1e6
    print(f"[ami] saved {out_path.name} ({size_mb:.1f} MB)")


def _copy_qmsum_gold(meeting_id: str, gold_dir: Path) -> bool:
    """Reuse a QMSum gold file for this meeting if we've already prepared it."""
    src = GOLD_DIR / "qmsum" / f"{meeting_id}.json"
    if not src.exists():
        return False
    dst = gold_dir / f"{meeting_id}.json"
    if dst.exists():
        return True
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[ami] reused QMSum gold → {dst.name}")
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


def prepare(
    meetings: list[str] = DEFAULT_MEETINGS,
    benchmark: str = "ami",
    camera: str | None = None,
) -> list[str]:
    raw_dir = RAW_DIR / benchmark
    gold_dir = GOLD_DIR / benchmark
    raw_dir.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)

    # If user pinned a specific camera, only try that. Otherwise, fall back
    # through the wide-angle priority list per meeting.
    preferred_cameras = [camera] if camera else WIDE_ANGLE_CAMERAS

    prepared: list[str] = []
    for name in meetings:
        video_path = raw_dir / f"{name}.avi"

        if video_path.exists():
            print(f"[ami] {name}: video cached, skipping download")
        else:
            cam = _resolve_camera(name, preferred_cameras)
            if cam is None:
                print(
                    f"[ami] ⚠️  no usable camera for {name} "
                    f"(tried {', '.join(preferred_cameras)})"
                )
                continue
            try:
                _download_video(name, cam, video_path)
            except Exception as e:  # network failures, mid-stream errors
                print(f"[ami] ⚠️  failed to download {name}: {e}")
                continue

        if not _copy_qmsum_gold(name, gold_dir):
            _write_stub_gold(name, gold_dir)

        prepared.append(name)

    return prepared


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meetings",
        nargs="*",
        default=DEFAULT_MEETINGS,
        help="AMI meeting ids to download (default: a 2-meeting subset)",
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
    args = parser.parse_args()

    names = prepare(args.meetings, args.benchmark, args.camera)
    print(f"[ami] prepared {len(names)} meetings under benchmark={args.benchmark!r}")
