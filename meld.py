"""
Exploratory Data Analysis (EDA) for the MELD (Multimodal EmotionLines Dataset).
Downloads lightweight CSV annotations directly without loading raw video files into RAM.
Part of the Bachelor's Thesis: "Analýza textových, zvukových a vizuálnych modalít k rozpoznávaniu emócií"
Author: Ruslan Rozlutskyi
"""

import os
import urllib.request
import torch
import pandas as pd

MELD_CSV_URLS = {
    "train": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv",
    "validation": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/dev_sent_emo.csv",
    "test": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/test_sent_emo.csv",
}


def check_hardware() -> None:
    """Verify CUDA availability and print GPU specifications."""
    print("=== Hardware Environment ===")
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"CUDA Available: Yes | Device: {device_name} ({vram_gb:.2f} GB VRAM)")
    else:
        print("CUDA Available: No | Running on CPU")


def download_annotations(data_dir: str = "data/raw") -> dict[str, pd.DataFrame]:
    """Download lightweight MELD CSV annotation files if not already cached locally."""
    os.makedirs(data_dir, exist_ok=True)
    dfs = {}
    for split, url in MELD_CSV_URLS.items():
        local_path = os.path.join(data_dir, f"{split}_sent_emo.csv")
        if not os.path.exists(local_path):
            print(f"Downloading {split} annotations...")
            urllib.request.urlretrieve(url, local_path)
        dfs[split] = pd.read_csv(local_path)
    return dfs


def analyze_split(df: pd.DataFrame, split_name: str) -> pd.Series:
    """Print summary statistics and emotion distribution for a dataset split."""
    print(f"\n--- Split: {split_name.upper()} (Total utterances: {len(df)}) ---")
    emotion_counts = df["Emotion"].value_counts()
    emotion_percentages = df["Emotion"].value_counts(normalize=True) * 100

    summary = pd.DataFrame({
        "Count": emotion_counts,
        "Percentage (%)": emotion_percentages.round(2),
    })
    print(summary)
    return emotion_counts


def main() -> None:
    check_hardware()

    print("\n=== Loading MELD Text & Emotion Annotations ===")
    splits = download_annotations()
    train_df, val_df, test_df = splits["train"], splits["validation"], splits["test"]

    sample = train_df.iloc[42]
    print("\n=== Sample Utterance (Train Index 42) ===")
    print(f"Dialogue ID : {sample['Dialogue_ID']}")
    print(f"Utterance ID: {sample['Utterance_ID']}")
    print(f"Speaker     : {sample['Speaker']}")
    print(f"Utterance   : \"{sample['Utterance']}\"")
    print(f"Emotion     : {sample['Emotion']}")
    print(f"Sentiment   : {sample['Sentiment']}")

    print("\n=== Class Distribution Analysis ===")
    train_counts = analyze_split(train_df, "train")
    val_counts = analyze_split(val_df, "validation")
    test_counts = analyze_split(test_df, "test")

    os.makedirs("results", exist_ok=True)
    dist_summary = pd.DataFrame({
        "Train": train_counts,
        "Validation": val_counts,
        "Test": test_counts,
    }).fillna(0).astype(int)

    output_path = os.path.join("results", "meld_class_distribution.csv")
    dist_summary.to_csv(output_path)
    print(f"\nSaved class distribution summary to: {output_path}")


if __name__ == "__main__":
    main()