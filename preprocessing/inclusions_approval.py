
# after manual annotation, run this script to merge ollama_suggestions with word-inclusions_set1.csv
# overwriting the original .csv. 
# ideally, you'll do something like this to generate your glossary:
# 1. manually figure out your inclusions/exclusions
# 2. run ollama to get diminutive suggestions
# 3. annotate that manually to accept or reject suggestions
# 4. merge .csv
# 5. run mcdi_glossary_generation pipeline (which will implement suggestions) 

# afterwards, you'll start counting. 