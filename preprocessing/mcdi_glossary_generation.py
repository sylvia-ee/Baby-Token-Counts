import re
from pathlib import Path

from string_ops import *
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GRAMMAR_GENERATORS = {
    "singular_noun": singular_generator,
    "plural_noun": plural_generator,
    "possessive": possessive_generator,
    "plural_possessive": plural_possessive_generator,
    "dumb_plural": dumb_plural_generator,
    "dumb_plural_possessive": dumb_plural_poss_generator,
}

def generate_bases(mcdi_string):
    """
    generates all syntax-cleaned forms for given mcdi word
    """
    
    # if it's an empty cell or not a string, 
    # return [whatever_it_is]
    if not isinstance(mcdi_string, str):
        return [mcdi_string]
        
    cleaned = lowercase_and_strip(mcdi_string)
    cleaned = strip_asterisk(cleaned)
    cleaned = strip_sense(cleaned)
    
    bases_list = split_variants(cleaned)
    return bases_list

def apply_cat_excl(path_to_c_excl, df, filter=True):
    cat_excl = pd.read_csv(path_to_c_excl)[["category", "excl_reason"]]
    df = df.drop(columns="excl_reason").merge(cat_excl, on="category", how="left")
    if filter:
        df = df[df["excl_reason"].isna()]
    return df

def apply_word_excl(path_to_w_excl, df, filter=True):
    word_excl = pd.read_csv(path_to_w_excl)[["mcdi", "base", "excl_reason"]]
    df = df.drop(columns="excl_reason").merge(word_excl, on=["mcdi", "base"], how="left")

    if filter:
        df = df[df["excl_reason"].isna()]
    return df

def apply_manual_grammar(path_to_manual_grammar, df):
    manual_grammar = pd.read_csv(path_to_manual_grammar)[["mcdi_id", "base", "grammar"]]
    df = df.merge(manual_grammar, on=["mcdi_id", "base"], how="left", suffixes=("", "_manual"))
    df["grammar"] = df["grammar_manual"].combine_first(df["grammar"])
    df = df.drop(columns="grammar_manual")
    return df

def generate_alt_forms(base, existing_grammar):

    if not isinstance(base, str):
        return [(base, existing_grammar)]

    # for compound words, grammatization is applied only to the second
    # (head noun) word, e.g. "high chair" -> "high chairs", not "highs chair"
    *prefix_words, head = base.split(" ")
    prefix = " ".join(prefix_words)

    forms = {}
    for tag in str(existing_grammar).split(", "):
        forms.setdefault(base, set()).add(tag)

    for tag, generator in GRAMMAR_GENERATORS.items():
        alt_head = generator(head)
        alt = f"{prefix} {alt_head}" if prefix else alt_head
        forms.setdefault(alt, set()).add(tag)

    return [(alt, ", ".join(sorted(tags))) for alt, tags in forms.items()]

def apply_word_incl(path_to_w_incl, df):
    word_incl = pd.read_csv(path_to_w_incl)[["mcdi", "base", "alt", "grammar"]]
    included = df.merge(word_incl, on=["mcdi", "base"], how="inner", suffixes=("", "_incl"))
    included["alt"] = included["alt_incl"]
    included["grammar"] = included["grammar_incl"]
    included = included.drop(columns=["alt_incl", "grammar_incl"])
    return pd.concat([df, included], ignore_index=True)

def merge_grammar_tags(grammars):
    tags = set()
    for grammar in grammars:
        tags.update(str(grammar).split(", "))
    return ", ".join(sorted(tags))

def generate_alts(df):

    # 1. default alt to base for rows without an inclusion-provided alt;
    # base itself must never change after the initial base-creation step
    df["alt"] = df["alt"].fillna(df["base"])

    # 2. generate grammatical forms for alts
    df["alt_grammar"] = df.apply(
        lambda row: generate_alt_forms(row["alt"], row["grammar"]), axis=1
    )
    df = df.explode("alt_grammar", ignore_index=True)
    df[["alt", "grammar"]] = pd.DataFrame(df["alt_grammar"].tolist(), index=df.index)
    df = df.drop(columns="alt_grammar")

    # 3. add a plus-joined alt for each compound-word alt, e.g. "ice cream" -> "ice+cream"
    compound_alts = df[df["alt"].str.contains(" ", na=False)].copy()
    compound_alts["alt"] = compound_alts["alt"].str.replace(" ", "+")
    df = pd.concat([df, compound_alts], ignore_index=True)

    # 4. different paths (e.g. an explicit "+"-joined inclusion and the plus-joining
    # of a space-joined inclusion) can land on the same (mcdi, base, alt); collapse
    # those into one row, unioning their grammar tags so none are lost
    other_cols = [c for c in df.columns if c not in ("mcdi", "base", "alt", "grammar")]
    df = df.groupby(["mcdi", "base", "alt"], as_index=False).agg(
        grammar=("grammar", merge_grammar_tags), **{c: (c, "first") for c in other_cols}
    )

    return df


# 0. load in dfs from paths
mcdi_ibi_path = PROJECT_ROOT / "raw_data" / "mcdi_ibi.csv"
cat_excl_path = PROJECT_ROOT / "inclusions_and_exclusions" / "category-exclusions_set1.csv"
word_excl_path = PROJECT_ROOT / "inclusions_and_exclusions" / "word-exclusions_set1.csv"
word_incl_path = PROJECT_ROOT / "inclusions_and_exclusions" / "word-inclusions_set1.csv"
manual_grammar_path = PROJECT_ROOT / "inclusions_and_exclusions" / "manual_grammar_set1.csv"

mcdi_ibi_df = pd.read_csv(mcdi_ibi_path)
cat_excl_df = pd.read_csv(cat_excl_path)
word_excl_df = pd.read_csv(word_excl_path)
word_incl_df = pd.read_csv(word_incl_path)

# 1. modify mcdi df col names and add cols
mcdi_ibi_df = mcdi_ibi_df[["downloaded", "item_id", "english_gloss", "category", "24"]]

mcdi_ibi_df.columns = ["download_date", "mcdi_id", "mcdi", "category", "avg_production"]

mcdi_ibi_df = mcdi_ibi_df.assign(
    base=None,
    alt=None,
    grammar=None,
    incl_reason=None,
    excl_reason=None
)

# 2. generate bases for mcdi df
mcdi_ibi_df["base"] = mcdi_ibi_df["mcdi"].apply(generate_bases)
mcdi_ibi_df = mcdi_ibi_df.explode("base", ignore_index=True)

# 3. apply category exclusions and word exclusions
mcdi_ibi_df = apply_cat_excl(cat_excl_path, mcdi_ibi_df)
mcdi_ibi_df = apply_word_excl(word_excl_path, mcdi_ibi_df)

# 4. classify grammar for each base, then apply manual overrides
# note this happens because my classification tool was a helper, not definitive
mcdi_ibi_df["grammar"] = mcdi_ibi_df["base"].apply(get_grammatical_profile)
mcdi_ibi_df = apply_manual_grammar(manual_grammar_path, mcdi_ibi_df)

# 5. apply inclusions
mcdi_ibi_df = apply_word_incl(word_incl_path, mcdi_ibi_df)

# 6. generate alts
mcdi_ibi_df = generate_alts(mcdi_ibi_df)

mcdi_ibi_df = mcdi_ibi_df.sort_values(by="base")
mcdi_ibi_df.to_csv(PROJECT_ROOT / "data" / "glossary.csv", index=False)

