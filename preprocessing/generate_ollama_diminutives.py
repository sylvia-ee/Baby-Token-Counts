import json
import urllib.request
import pandas as pd
from string_ops import lowercase_and_strip, strip_asterisk, strip_sense, split_variants, get_grammatical_profile

RAW_PATH = "/Users/se/Projects/Baby-Token-Counts/raw_data/mcdi_ibi.csv"
CAT_EXCL_PATH = "/Users/se/Projects/Baby-Token-Counts/inclusions_and_exclusions/category-exclusions_set1.csv"
WORD_EXCL_PATH = "/Users/se/Projects/Baby-Token-Counts/inclusions_and_exclusions/word-exclusions_set1.csv"
OUTPUT_PATH = "/Users/se/Projects/Baby-Token-Counts/inclusions_and_exclusions/ollama_suggestions_inclusions.csv"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:latest"
SEED = 42

PROMPT_TEMPLATE = (
    "You are an expert in child-directed speech (CDS) and English morphology. "
    "Generate natural diminutives for a given noun using these rules, applied "
    "in priority order. "
    "1. If the word is 3+ syllables, truncate to the first stressed syllable "
    "and add \"-ie\"/\"-y\" (e.g., banana -> nanie, stomach -> tummy). "
    "2. If the word ends in /s z sh ch j/ (s, z, sh, ch, j), add \"-y\" not "
    "\"-ie\", and double the final consonant only if the preceding vowel is "
    "short (e.g., bus -> bussy, fish -> fishy, bridge -> no change). "
    "3. If the word ends in a written vowel (a, e, o, u), drop it before "
    "adding \"-ie\"/\"-y\" (e.g., nose -> nosy, potato -> potatie). "
    "4. For CVC or CVCC words, double the final consonant if the preceding "
    "vowel is short, then add \"-y\"/\"-ie\" (e.g., dog -> doggy, cat -> "
    "catty, bug -> buggy). "
    "5. If none of the above produce a natural English form, do not include "
    "that form in your answer. "
    "Do not produce forms with awkward consonant clusters (e.g. busie, "
    "churchy, bridgie). Do not add a suffix to a word already ending in "
    "\"-y\" (e.g. happy stays unchanged). Do not truncate monosyllables. The "
    "noun may be a single word or a multi-word compound (e.g. \"ice cream\", "
    "\"belly button\") — if it's a compound, apply the rules to the head "
    "noun (the last/main word) and leave the rest unchanged, e.g. \"teddy "
    "bear\" -> \"teddy bearie\". "
    "Apply these rules to the noun \"{word}\" and return every plausible "
    "diminutive, including spelling variants. Respond with ONLY a JSON "
    "object: {{\"diminutives\": [\"<word or phrase>\", ...]}}. Each "
    "diminutive must be lowercase and different from the original word. If "
    "no natural diminutive exists for this word, use an empty list []."
)


def clean_to_bases(mcdi_string):
    if not isinstance(mcdi_string, str):
        return [mcdi_string]
    cleaned = strip_sense(strip_asterisk(lowercase_and_strip(mcdi_string)))
    return split_variants(cleaned)


def get_diminutives(word):
    """
    queries the local ollama model for all plausible singular_noun
    diminutives of word. seed + temperature=0 make this deterministic
    across runs on this machine.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": PROMPT_TEMPLATE.format(word=word),
        "format": "json",
        "stream": False,
        "options": {"temperature": 0, "seed": SEED},
    }
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read())

    diminutives = json.loads(result["response"]).get("diminutives", [])
    return {d.strip().lower() for d in diminutives if isinstance(d, str) and d.strip()}


if __name__ == "__main__":
    # 1. load raw mcdi data and clean into (mcdi_id, base) pairs
    raw_df = pd.read_csv(RAW_PATH)[["item_id", "english_gloss", "category"]]
    raw_df.columns = ["mcdi_id", "mcdi", "category"]

    raw_df["base"] = raw_df["mcdi"].apply(clean_to_bases)
    raw_df = raw_df.explode("base", ignore_index=True)

    # 2. apply category and word exclusions, same as the main glossary pipeline
    cat_excl = pd.read_csv(CAT_EXCL_PATH)
    raw_df = raw_df[~raw_df["category"].isin(cat_excl["category"])]

    word_excl = pd.read_csv(WORD_EXCL_PATH)
    if not word_excl.empty:
        excluded_pairs = set(zip(word_excl["mcdi"], word_excl["base"]))
        raw_df = raw_df[~raw_df.apply(lambda r: (r["mcdi"], r["base"]) in excluded_pairs, axis=1)]

    # 3. keep only bases classified as singular_noun (compounds included)
    is_singular_noun = raw_df["base"].apply(
        lambda b: isinstance(b, str) and "singular_noun" in get_grammatical_profile(b).split(", ")
    )
    raw_df = raw_df[is_singular_noun]

    # 4. dedupe to one row per unique base, keeping its lowest mcdi_id for traceability
    unique_bases = (
        raw_df.sort_values("mcdi_id")
        .drop_duplicates(subset="base", keep="first")
        .sort_values("base")
        [["mcdi_id", "base"]]
    )

    # 5. query ollama for all plausible diminutives of each unique base,
    # adding one row per suggestion (sets aren't iteration-order stable
    # across runs, so the final sort below is what keeps output deterministic)
    rows = []
    for _, row in unique_bases.iterrows():
        diminutives = get_diminutives(row["base"]) - {row["base"]}
        for diminutive in diminutives:
            rows.append({
                "mcdi_id": row["mcdi_id"],
                "mcdi": row["base"],
                "alt": diminutive,
                "grammar": "singular_noun, diminutive",
                "incl_reason": "ollama-suggested diminutive, pending manual review",
                "source": OLLAMA_MODEL,
            })

    suggestions_df = pd.DataFrame(rows).sort_values(["mcdi_id", "alt"])
    suggestions_df.to_csv(OUTPUT_PATH, index=False)
    print(f"wrote {len(suggestions_df)} suggestions to {OUTPUT_PATH}")
