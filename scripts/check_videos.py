"""Verify that every MELD annotation row has a matching video clip on disk.

MELD clips are named dia{Dialogue_ID}_utt{Utterance_ID}.mp4; each split
lives in its own folder inside data/raw/MELD.Raw.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MELD_DIR = RAW_DIR / "MELD.Raw"

# split name -> (annotation csv, video folder)
SPLITS = {
    "train": ("train_sent_emo.csv", "train_splits"),
    "validation": ("validation_sent_emo.csv", "dev_splits_complete"),
    "test": ("test_sent_emo.csv", "output_repeated_splits_test"),
}


def clip_name(dialogue_id: int, utterance_id: int) -> str:
    return f"dia{dialogue_id}_utt{utterance_id}.mp4"


def check_split(split: str, csv_name: str, video_dir_name: str) -> list[str]:
    video_dir = MELD_DIR / video_dir_name
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Video folder not found: {video_dir}")

    df = pd.read_csv(RAW_DIR / csv_name)
    on_disk = {p.name for p in video_dir.glob("*.mp4") if not p.name.startswith("._")}
    expected = [clip_name(d, u) for d, u in zip(df["Dialogue_ID"], df["Utterance_ID"])]
    missing = [name for name in expected if name not in on_disk]

    print(f"{split:<10} rows={len(df):>5}  videos={len(on_disk):>5}  missing={len(missing)}")
    return missing


def main() -> None:
    for split, (csv_name, video_dir_name) in SPLITS.items():
        missing = check_split(split, csv_name, video_dir_name)
        if missing:
            print(f"  first missing: {missing[:5]}")


if __name__ == "__main__":
    main()
