from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

childes_path = PROJECT_ROOT / "raw_data" / "childes.csv"
age_histogram_path = PROJECT_ROOT / "figures" / "target_child_age_histogram.png"

# every read is restricted to rows for children 24 months old or younger,
# matching the filter applied in childes_preprocess.py

childes_df = pd.read_csv(childes_path, usecols=["target_child_id", "target_child_age"])
childes_df = childes_df[childes_df["target_child_age"] <= 24.0]
n_unique_children = childes_df["target_child_id"].nunique()

childes_df = pd.read_csv(childes_path, usecols=["speaker_id", "target_child_age"])
childes_df = childes_df[childes_df["target_child_age"] <= 24.0]
n_unique_speaker = childes_df["speaker_id"].nunique()

childes_df = pd.read_csv(childes_path, usecols=["transcript_id", "target_child_age"])
childes_df = childes_df[childes_df["target_child_age"] <= 24.0]
n_unique_transcripts = childes_df["transcript_id"].nunique()

childes_df = pd.read_csv(childes_path, usecols=["target_child_age"])
childes_df = childes_df[childes_df["target_child_age"] <= 24.0]
age_counts, age_bin_edges = np.histogram(childes_df["target_child_age"].dropna(), bins=10)

childes_df = pd.read_csv(childes_path, usecols=["target_child_id", "speaker_id", "target_child_age"])
childes_df = childes_df[childes_df["target_child_age"] <= 24.0]
n_unique_child_speaker_pairs = childes_df[["target_child_id", "speaker_id"]].drop_duplicates().shape[0]

print(f"unique target_child_ids: {n_unique_children}")
print(f"unique speakers: {n_unique_speaker}")
print(f"unique transcripts: {n_unique_transcripts}")
print(f"unique (target_child_id, speaker_id) pairs: {n_unique_child_speaker_pairs}")

print("target_child_age histogram:")
for count, lo, hi in zip(age_counts, age_bin_edges[:-1], age_bin_edges[1:]):
    print(f"  [{lo:.1f}, {hi:.1f}): {count}")

fig, ax = plt.subplots()
ax.bar(age_bin_edges[:-1], age_counts, width=np.diff(age_bin_edges), align="edge", edgecolor="black")
ax.set_xlabel("target_child_age")
ax.set_ylabel("count")
ax.set_title("Distribution of target_child_age (<= 24 months)")
fig.savefig(age_histogram_path)
print(f"wrote histogram to {age_histogram_path}")
