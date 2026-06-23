from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

childes_path = PROJECT_ROOT / "raw_data" / "childes.csv"
childes_txt_path = PROJECT_ROOT / "data" / "childes.txt"
glossary_path = PROJECT_ROOT / "data" / "glossary.csv"
counts_path = PROJECT_ROOT / "data" / "counts.csv"
counts_all_incl_path = PROJECT_ROOT / "data" / "counts_all_incl.csv"

# 0. only return the id, stem, speaker_id, target_child_id, transcript_id cols,
# restricted to rows for children 24 months old or younger
childes_df = pd.read_csv(childes_path)[
    ["id", "stem", "speaker_id", "target_child_id", "transcript_id", "target_child_age"]
]
childes_df = childes_df[childes_df["target_child_age"] <= 24.0]
glossary_df = pd.read_csv(glossary_path)

# 1. take the "stem" column, convert all to string type and lowercase
stems = childes_df["stem"].dropna().astype(str).str.lower()
with open(childes_txt_path, "w") as f:
    f.write("\n".join(stems))

# 2. split each stem row on whitespace
rows = stems.str.split()

# 3. count using these methods:
# a. inclusive: count exact matches of each alt. e.g. for sentence "ice and ice cream", ice = 2, cream = 1
# b. exclusive: within each row, look ahead for potential compound matches (two words separated
#    by a space that are in the glossary as an alt). if one is found, add +1 to the count of the compound,
#    and do not allow the individual tokens in that compound to add to the individual count of those tokens
#    for example, for sentence "ice and ice cream", ice = 1, ice cream = 1, cream = 0
# c. traditional: count exact matches of each unique base.


def by_word_count(strings):
    by_len = {}
    for s in strings:
        by_len.setdefault(len(s.split()), set()).add(s)
    return by_len


alts = glossary_df["alt"].dropna().astype(str).str.lower().unique()
bases = glossary_df["base"].dropna().astype(str).str.lower().unique()

alts_by_len = by_word_count(alts)
bases_by_len = by_word_count(bases)

single_word_alts = alts_by_len.get(1, set())
multiword_alts = {n: words for n, words in alts_by_len.items() if n > 1}
single_word_bases = bases_by_len.get(1, set())
multiword_bases = {n: words for n, words in bases_by_len.items() if n > 1}

two_word_alts = multiword_alts.get(2, set())
two_word_bases = multiword_bases.get(2, set())

global_token_counts = Counter()
incl_multiword_alt_counts = Counter()
incl_multiword_base_counts = Counter()
excl_alt_counts = Counter()
excl_base_counts = Counter()


def count_multiword_matches(tokens, n_tokens, multiword_sets, counts):
    for n, target_set in multiword_sets.items():
        for j in range(n_tokens - n + 1):
            gram = " ".join(tokens[j:j + n])
            if gram in target_set:
                counts[gram] += 1


def count_exclusive_matches(tokens, n_tokens, single_word_set, two_word_set, counts):
    i = 0
    while i < n_tokens:
        if i + 1 < n_tokens:
            bigram = tokens[i] + " " + tokens[i + 1]
            if bigram in two_word_set:
                counts[bigram] += 1
                i += 2
                continue
        if tokens[i] in single_word_set:
            counts[tokens[i]] += 1
        i += 1


for tokens in rows:
    n_tokens = len(tokens)
    global_token_counts.update(tokens)

    # inclusive: every n-gram matching a multi-word alt/base is counted, regardless of overlap
    count_multiword_matches(tokens, n_tokens, multiword_alts, incl_multiword_alt_counts)
    count_multiword_matches(tokens, n_tokens, multiword_bases, incl_multiword_base_counts)

    # exclusive: greedily consume two-word compound matches before single tokens
    count_exclusive_matches(tokens, n_tokens, single_word_alts, two_word_alts, excl_alt_counts)
    count_exclusive_matches(tokens, n_tokens, single_word_bases, two_word_bases, excl_base_counts)


def incl_count_for(string, multiword_counts):
    return global_token_counts[string] if len(string.split()) == 1 else multiword_counts[string]


lower_alts = glossary_df["alt"].astype(str).str.lower()
lower_bases = glossary_df["base"].astype(str).str.lower()

glossary_df["alt_incl_count"] = lower_alts.map(lambda a: incl_count_for(a, incl_multiword_alt_counts))
glossary_df["alt_excl_count"] = lower_alts.map(excl_alt_counts)

# return data/counts.csv with the mcdi glossary df but with columns added called
# alt_incl_count for inclusive count of that alt
# alt_excl_count for exclusive count of that alt
# total_alt_incl_count, which is sum of incl_count col for a unique mcdi word
# total_alt_excl_count, which is the sum of excl_count col for a unique mcdi word
# base_incl_count, which is the count of the base for a unique mcdi word
# base_excl_count, which is count of the base for a unique mcdi word
# alt_diff = total_alt_incl_count - total_alt_excl_count, should never be negative if done right
# base_diff = base_incl_count - base_excl_count, should never be negative if done right
# alt_base_excl_diff = total_alt_excl_count - base_excl_count
# alt_base_incl_diff = total_alt_incl_count - base_incl_count
# return .csv with col order sorted as:
# mcdi, base, alt, then those columns above in order,then leftover coluns
glossary_df["total_alt_incl_count"] = glossary_df.groupby("mcdi")["alt_incl_count"].transform("sum")
glossary_df["total_alt_excl_count"] = glossary_df.groupby("mcdi")["alt_excl_count"].transform("sum")

# base counts are computed once per unique (mcdi, base) pair, then summed per mcdi,
# so a base shared by several alt rows isn't counted once per alt expansion
unique_bases = glossary_df[["mcdi", "base"]].drop_duplicates().copy()
unique_lower_base = unique_bases["base"].astype(str).str.lower()
unique_bases["base_incl_count"] = unique_lower_base.map(lambda b: incl_count_for(b, incl_multiword_base_counts))
unique_bases["base_excl_count"] = unique_lower_base.map(excl_base_counts)
base_totals = unique_bases.groupby("mcdi")[["base_incl_count", "base_excl_count"]].sum()
glossary_df = glossary_df.merge(base_totals, on="mcdi", how="left")

glossary_df["alt_diff"] = glossary_df["total_alt_incl_count"] - glossary_df["total_alt_excl_count"]
glossary_df["base_diff"] = glossary_df["base_incl_count"] - glossary_df["base_excl_count"]
glossary_df["alt_base_excl_diff"] = glossary_df["total_alt_excl_count"] - glossary_df["base_excl_count"]
glossary_df["alt_base_incl_diff"] = glossary_df["total_alt_incl_count"] - glossary_df["base_incl_count"]

# also compute log transformed counts; log1p avoids -inf for zero counts
glossary_df["total_alt_incl_logcount"] = np.log1p(glossary_df["total_alt_incl_count"])
glossary_df["total_alt_excl_logcount"] = np.log1p(glossary_df["total_alt_excl_count"])
glossary_df["base_incl_logcount"] = np.log1p(glossary_df["base_incl_count"])
glossary_df["base_excl_logcount"] = np.log1p(glossary_df["base_excl_count"])

# log differences, mirroring alt_diff/base_diff/alt_base_excl_diff/alt_base_incl_diff but on log counts
glossary_df["alt_logdiff"] = glossary_df["total_alt_incl_logcount"] - glossary_df["total_alt_excl_logcount"]
glossary_df["base_logdiff"] = glossary_df["base_incl_logcount"] - glossary_df["base_excl_logcount"]
glossary_df["alt_base_excl_logdiff"] = glossary_df["total_alt_excl_logcount"] - glossary_df["base_excl_logcount"]
glossary_df["alt_base_incl_logdiff"] = glossary_df["total_alt_incl_logcount"] - glossary_df["base_incl_logcount"]


ordered_cols = [
    "mcdi", "base", "alt",
    "alt_incl_count", "alt_excl_count",
    "total_alt_incl_count", "total_alt_excl_count",
    "base_incl_count", "base_excl_count",
    "alt_diff", "base_diff", "alt_base_excl_diff", "alt_base_incl_diff",
    "total_alt_incl_logcount", "total_alt_excl_logcount",
    "base_incl_logcount", "base_excl_logcount",
    "alt_logdiff", "base_logdiff", "alt_base_excl_logdiff", "alt_base_incl_logdiff",
]
leftover_cols = [c for c in glossary_df.columns if c not in ordered_cols]
glossary_df = glossary_df[ordered_cols + leftover_cols]

glossary_df.to_csv(counts_path, index=False)

# also return data/counts_all_incl.csv with every single token and an inclusive count
counts_all_incl_df = pd.DataFrame(
    global_token_counts.items(), columns=["token", "alt_incl_count"]
).sort_values("alt_incl_count", ascending=False)
counts_all_incl_df.to_csv(counts_all_incl_path, index=False)

