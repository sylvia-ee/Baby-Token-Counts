import re
import inflect

grammar_machine = inflect.engine()

# syntax

def lowercase_and_strip(string):
    """
    lowercases string and removes all trailing whitespace
    """
    if not isinstance(string, str): 
        return string
    
    cleaned_string = string.lower().strip()

    return cleaned_string

def strip_sense(string):
    """
    removes (sense) from end of string
    """
    sense_stripped = re.sub(r'\s*\(.*?\)', '', string).strip()
    return sense_stripped

def strip_asterisk(string):
    """
    removes * from end of string
    """
    asterisk_stripped = string.replace('*', '')
    return asterisk_stripped

def split_variants(string):
    """
    splits string on / and returns as list
    """
    split_list = [word.strip() for word in string.split('/')]
    return split_list

# grammar

## classifier
def get_grammatical_profile(token):
    """
    Analyzes the token to determine its morphological state.
    Words with identical singular/plural forms (e.g. "fish", "sheep")
    are classified as both singular_noun and plural_noun.
    """
    is_poss = token.endswith("'s") or token.endswith("'")
    if is_poss:
        return "possessive"

    singular_form = grammar_machine.singular_noun(token)
    is_plural = bool(singular_form)
    is_singular = not is_plural or singular_form == token

    profiles = []
    if is_singular:
        profiles.append("singular_noun")
    if is_plural:
        profiles.append("plural_noun")
    return ", ".join(profiles)

## generators
def singular_generator(token):
    base_val = grammar_machine.singular_noun(token)
    final_token = base_val if base_val else token
    return final_token

def plural_generator(token):
    plu_val = grammar_machine.plural_noun(token)
    final_token = plu_val if plu_val else token
    return final_token

def possessive_generator(token):
    base_val = singular_generator(token)
    poss_token = f"{base_val}'s"
    return poss_token

def plural_possessive_generator(token):
    plu_val = plural_generator(token)
    suffix = "'" if plu_val.endswith("s") else "s'"
    final_token = f"{plu_val}{suffix}"
    return final_token

def dumb_plural_generator(token):
    sing_val = singular_generator(token)
    dumb_plu_token = f"{sing_val}s"
    return dumb_plu_token

def dumb_plural_poss_generator(token):
    sing_val = singular_generator(token)
    dumb_plu = f"{sing_val}s"
    
    suffix = "'" if dumb_plu.endswith("s") else "'s"
    dumb_plu_poss = f"{dumb_plu}{suffix}"
    return dumb_plu_poss