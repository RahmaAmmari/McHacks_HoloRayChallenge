from pathlib import Path

from tracker import RealTimeDrawingTracker

DATASET_ROOT = Path("D:/Projects/McHacks/Dataset/Dataset")
FOLDERS = ["Echo", "Intrapartum", "Lapchole", "POCUS"]
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


def list_videos(folder: Path):
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXTS])


def pick_folder():
    print("\nChoose dataset folder:")
    for i, f in enumerate(FOLDERS, 1):
        print(f"  {i}) {f}")
    s = input("Folder #: ").strip()
    if not s.isdigit():
        return None
    idx = int(s)
    if not (1 <= idx <= len(FOLDERS)):
        return None
    return DATASET_ROOT / FOLDERS[idx - 1]


def pick_video(videos, limit=50):
    if not videos:
        return None
    print("\nChoose video:")
    show = videos[:limit]
    for i, p in enumerate(show, 1):
        print(f"  {i}) {p.name}")
    s = input("Video #: ").strip()
    if not s.isdigit():
        return None
    idx = int(s)
    if not (1 <= idx <= len(show)):
        return None
    return show[idx - 1]


if __name__ == "__main__":
    folder = pick_folder()
    if folder is None:
        print("Cancelled.")
        raise SystemExit(0)

    videos = list_videos(folder)
    if not videos:
        print("No videos found in:", folder)
        raise SystemExit(0)

    video_path = pick_video(videos)
    if video_path is None:
        print("Cancelled.")
        raise SystemExit(0)

    print("\nOpening:", video_path)
    tracker = RealTimeDrawingTracker(dataset_root=str(DATASET_ROOT))
    tracker.start(str(video_path))
