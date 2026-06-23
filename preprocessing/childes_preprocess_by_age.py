from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

childes_path = PROJECT_ROOT / "raw_data" / "childes.csv"
glossary_path = PROJECT_ROOT / "data" / "glossary.csv"
avgs_path = PROJECT_ROOT / "raw_data" / "mcdi_prod_avgs_16-30.csv"
counts_by_age_path = PROJECT_ROOT / "data" / "counts_by_age.csv"

AGES = range(16, 31)

glossary_df = pd.read_csv(glossary_path)
avgs_df = pd.read_csv(avgs_path)

# load every stem once, paired with its target_child_age, so each age cutoff
# below can just take a subset rather than re-reading/re-tokenizing childes.csv
childes_df = pd.read_csv(childes_path, usecols=["stem", "target_child_age"])
stems = childes_df["stem"].dropna()
ages = childes_df.loc[stems.index, "target_child_age"]
all_tokens = stems.astype(str).str.lower().str.split()


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


def incl_count_for(string, global_token_counts, multiword_counts):
    return global_token_counts[string] if len(string.split()) == 1 else multiword_counts[string]


lower_alts = glossary_df["alt"].astype(str).str.lower()

age_dfs = []
for age in AGES:
    rows = all_tokens[ages <= age]

    global_token_counts = Counter()
    incl_multiword_alt_counts = Counter()
    incl_multiword_base_counts = Counter()
    excl_alt_counts = Counter()
    excl_base_counts = Counter()

    for tokens in rows:
        n_tokens = len(tokens)
        global_token_counts.update(tokens)
        count_multiword_matches(tokens, n_tokens, multiword_alts, incl_multiword_alt_counts)
        count_multiword_matches(tokens, n_tokens, multiword_bases, incl_multiword_base_counts)
        count_exclusive_matches(tokens, n_tokens, single_word_alts, two_word_alts, excl_alt_counts)
        count_exclusive_matches(tokens, n_tokens, single_word_bases, two_word_bases, excl_base_counts)

    age_df = glossary_df.copy()
    age_df["alt_incl_count"] = lower_alts.map(
        lambda a: incl_count_for(a, global_token_counts, incl_multiword_alt_counts)
    )
    age_df["alt_excl_count"] = lower_alts.map(excl_alt_counts)
    age_df["total_alt_incl_count"] = age_df.groupby("mcdi")["alt_incl_count"].transform("sum")
    age_df["total_alt_excl_count"] = age_df.groupby("mcdi")["alt_excl_count"].transform("sum")

    # base counts are computed once per unique (mcdi, base) pair, then summed per mcdi,
    # so a base shared by several alt rows isn't counted once per alt expansion
    unique_bases = age_df[["mcdi", "base"]].drop_duplicates().copy()
    unique_lower_base = unique_bases["base"].astype(str).str.lower()
    unique_bases["base_incl_count"] = unique_lower_base.map(
        lambda b: incl_count_for(b, global_token_counts, incl_multiword_base_counts)
    )
    unique_bases["base_excl_count"] = unique_lower_base.map(excl_base_counts)
    base_totals = unique_bases.groupby("mcdi")[["base_incl_count", "base_excl_count"]].sum()
    age_df = age_df.merge(base_totals, on="mcdi", how="left")

    age_df["alt_diff"] = age_df["total_alt_incl_count"] - age_df["total_alt_excl_count"]
    age_df["base_diff"] = age_df["base_incl_count"] - age_df["base_excl_count"]
    age_df["alt_base_excl_diff"] = age_df["total_alt_excl_count"] - age_df["base_excl_count"]
    age_df["alt_base_incl_diff"] = age_df["total_alt_incl_count"] - age_df["base_incl_count"]

    age_df["total_alt_incl_logcount"] = np.log1p(age_df["total_alt_incl_count"])
    age_df["total_alt_excl_logcount"] = np.log1p(age_df["total_alt_excl_count"])
    age_df["base_incl_logcount"] = np.log1p(age_df["base_incl_count"])
    age_df["base_excl_logcount"] = np.log1p(age_df["base_excl_count"])

    age_df["alt_logdiff"] = age_df["total_alt_incl_logcount"] - age_df["total_alt_excl_logcount"]
    age_df["base_logdiff"] = age_df["base_incl_logcount"] - age_df["base_excl_logcount"]
    age_df["alt_base_excl_logdiff"] = age_df["total_alt_excl_logcount"] - age_df["base_excl_logcount"]
    age_df["alt_base_incl_logdiff"] = age_df["total_alt_incl_logcount"] - age_df["base_incl_logcount"]

    # avg_production comes from mcdi_prod_avgs_16-30.csv for this age, not glossary.csv;
    # item_id there lines up with mcdi_id (verified: e.g. "bird" is item_id/mcdi_id 18)
    age_avg = avgs_df[["item_id", str(age)]].rename(
        columns={"item_id": "mcdi_id", str(age): "avg_production"}
    )
    age_df = age_df.drop(columns="avg_production").merge(age_avg, on="mcdi_id", how="left")

    age_df["age at count"] = age

    age_dfs.append(age_df)

counts_by_age_df = pd.concat(age_dfs, ignore_index=True)

ordered_cols = [
    "mcdi", "base", "alt", "age at count",
    "alt_incl_count", "alt_excl_count",
    "total_alt_incl_count", "total_alt_excl_count",
    "base_incl_count", "base_excl_count",
    "alt_diff", "base_diff", "alt_base_excl_diff", "alt_base_incl_diff",
    "total_alt_incl_logcount", "total_alt_excl_logcount",
    "base_incl_logcount", "base_excl_logcount",
    "alt_logdiff", "base_logdiff", "alt_base_excl_logdiff", "alt_base_incl_logdiff",
]
leftover_cols = [c for c in counts_by_age_df.columns if c not in ordered_cols]
counts_by_age_df = counts_by_age_df[ordered_cols + leftover_cols]

counts_by_age_df.to_csv(counts_by_age_path, index=False)
print(f"wrote {len(counts_by_age_df)} rows to {counts_by_age_path}")
