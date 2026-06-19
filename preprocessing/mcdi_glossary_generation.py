import re
from string_ops import *
import pandas as pd

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

    if not isinstance(base, str) or re.search(r'\s', base):
        return [(base, existing_grammar)]

    forms = {}
    for tag in str(existing_grammar).split(", "):
        forms.setdefault(base, set()).add(tag)

    for tag, generator in GRAMMAR_GENERATORS.items():
        alt = generator(base)
        forms.setdefault(alt, set()).add(tag)

    return [(alt, ", ".join(sorted(tags))) for alt, tags in forms.items()]

def generate_alts(df, inclusions_df=False):

    # 1. copy base vals to alt
    df["alt"] = df["base"]

    # 2. include manual inclusions
    if inclusions_df is not False:
        included = df.merge(
            inclusions_df, on=["mcdi", "base"], how="inner", suffixes=("", "_incl")
        )
        included["base"] = included["alt_incl"]
        included["alt"] = included["alt_incl"]
        included["grammar"] = included["grammar_incl"]
        included = included.drop(columns=["alt_incl", "grammar_incl"])
        df = pd.concat([df, included], ignore_index=True)

    # 3. generate grammatical forms for alts
    df["alt_grammar"] = df.apply(
        lambda row: generate_alt_forms(row["base"], row["grammar"]), axis=1
    )
    df = df.explode("alt_grammar", ignore_index=True)
    df[["alt", "grammar"]] = pd.DataFrame(df["alt_grammar"].tolist(), index=df.index)
    df = df.drop(columns="alt_grammar")

    return df


# 0. load in dfs from paths 
mcdi_ibi_path = "/Users/se/Projects/Baby-Token-Counts/raw_data/mcdi_ibi.csv"
cat_excl_path = "/Users/se/Projects/Baby-Token-Counts/inclusions_and_exclusions/category-exclusions_set1.csv"
word_excl_path = "/Users/se/Projects/Baby-Token-Counts/inclusions_and_exclusions/word-exclusions_set1.csv"
word_incl_path = "/Users/se/Projects/Baby-Token-Counts/inclusions_and_exclusions/word-inclusions_set1.csv"
manual_grammar_path = "/Users/se/Projects/Baby-Token-Counts/inclusions_and_exclusions/manual_grammar_set1.csv"

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

# 5. generate alts
mcdi_ibi_df = generate_alts(mcdi_ibi_df, word_incl_df)

mcdi_ibi_df = mcdi_ibi_df.sort_values(by="base")
mcdi_ibi_df.to_csv("test.csv", index=False)