from collections import Counter
import pandas as pd

childes_path = "/Users/se/Projects/Baby-Token-Counts/raw_data/childes.csv"
childes_txt_path = "/Users/se/Projects/Baby-Token-Counts/data/childes.txt"
glossary_path = "/Users/se/Projects/Baby-Token-Counts/data/glossary.csv"
counts_path = "/Users/se/Projects/Baby-Token-Counts/data/counts.csv"
counts_all_incl_path = "/Users/se/Projects/Baby-Token-Counts/data/counts_all_incl.csv"

# 0. only return the id, stem, speaker_id, target_child_id, transcript_id cols
childes_df = pd.read_csv(childes_path)[
    ["id", "stem", "speaker_id", "target_child_id", "transcript_id"]
]
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

alts = glossary_df["alt"].dropna().astype(str).str.lower().unique()
alts_by_len = {}
for alt in alts:
    alts_by_len.setdefault(len(alt.split()), set()).add(alt)
single_word_alts = alts_by_len.get(1, set())
multiword_alts = {n: words for n, words in alts_by_len.items() if n > 1}

global_token_counts = Counter()
incl_multiword_counts = Counter()
excl_counts = Counter()

for tokens in rows:
    global_token_counts.update(tokens)

    # inclusive: every n-gram matching a multi-word alt is counted, regardless of overlap
    for n, alt_set in multiword_alts.items():
        for j in range(len(tokens) - n + 1):
            gram = " ".join(tokens[j:j + n])
            if gram in alt_set:
                incl_multiword_counts[gram] += 1

    # exclusive: greedily consume two-word compound matches before single tokens
    two_word_alts = multiword_alts.get(2, set())
    i = 0
    n_tokens = len(tokens)
    while i < n_tokens:
        if i + 1 < n_tokens:
            bigram = tokens[i] + " " + tokens[i + 1]
            if bigram in two_word_alts:
                excl_counts[bigram] += 1
                i += 2
                continue
        if tokens[i] in single_word_alts:
            excl_counts[tokens[i]] += 1
        i += 1


def incl_count_for(alt):
    n = len(alt.split())
    return global_token_counts[alt] if n == 1 else incl_multiword_counts[alt]


lower_alts = glossary_df["alt"].astype(str).str.lower()
glossary_df["incl_count"] = lower_alts.map(incl_count_for)
glossary_df["excl_count"] = lower_alts.map(excl_counts)

# return data/counts.csv with the mcdi glossary df but with columns added called
# incl_count for inclusive count
# excl_count for exclusive count
# total_incl_count, which is sum of incl_count col for a unique mcdi word
# and total_excl_count, which is the sum of excl_count col for a unique mcdi word
glossary_df["total_incl_count"] = glossary_df.groupby("mcdi")["incl_count"].transform("sum")
glossary_df["total_excl_count"] = glossary_df.groupby("mcdi")["excl_count"].transform("sum")
glossary_df.to_csv(counts_path, index=False)

# also return data/counts_all_incl.csv with every single token and an inclusive count
counts_all_incl_df = pd.DataFrame(
    global_token_counts.items(), columns=["token", "incl_count"]
).sort_values("incl_count", ascending=False)
counts_all_incl_df.to_csv(counts_all_incl_path, index=False)
