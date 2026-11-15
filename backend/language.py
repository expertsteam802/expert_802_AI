"""Small, explicit English language-processing foundation."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "knowledge" / "english_dictionary.json"

def tokens(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+", (text or "").lower())

def load_dictionary():
    import json
    try:
        with DATA.open(encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}

def correct_common_grammar(text):
    value = text.strip()
    notes = []
    rules = [
        (r"\bI is\b", "I am", "Use 'am' with the subject 'I'."),
        (r"\b(he|she|it) are\b", r"\1 is", "Use 'is' with he, she, or it."),
        (r"\b(you|we|they) is\b", r"\1 are", "Use 'are' with you, we, or they."),
        (r"\b(he|she|it) don't\b", r"\1 doesn't", "Use 'doesn't' with he, she, or it."),
        (r"\b(he|she|it) doesn't ([a-z]+)s\b", r"\1 doesn't \2", "After 'doesn't', use the base form of the verb."),
        (r"\bI has\b", "I have", "Use 'have' with 'I'."),
        (r"\b(he|she|it) have\b", r"\1 has", "Use 'has' with he, she, or it."),
        (r"\b(a) ([aeiou][a-z]+)\b", r"an \2", "Use 'an' before a vowel sound (this spelling rule is only an approximation)."),
    ]
    for pattern, replacement, explanation in rules:
        changed = re.sub(pattern, replacement, value, flags=re.I)
        if changed != value:
            value = changed
            notes.append(explanation)
    return value, notes

def dictionary_lookup(word):
    return load_dictionary().get(word.lower().strip(".,!?;:'\""), None)
