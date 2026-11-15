"""English spelling, tokenization, morphology, and grammar helpers."""
import json
import re
from pathlib import Path
from functools import lru_cache

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "knowledge" / "english_dictionary.json"
WORDNET_DATA = ROOT / "knowledge" / "wordnet_dictionary.json"

def tokens(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+", (text or "").lower())

def load_dictionary():
    try:
        with DATA.open(encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}

@lru_cache(maxsize=1)
def load_wordnet():
    try:
        with WORDNET_DATA.open(encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}

def dictionary_words():
    words = set(load_dictionary())
    words.update(load_wordnet())
    words.update({"i", "a", "an", "the", "and", "or", "but", "to", "of", "in", "on",
                  "at", "for", "is", "am", "are", "was", "were", "be", "been", "being",
                  "my", "your", "his", "her", "its", "our", "their", "right", "now",
                  "need", "want", "game", "please", "what", "where", "when", "why", "how"})
    return words

def _distance(a, b, limit=2):
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        row_min = i
        for j, cb in enumerate(b, 1):
            value = min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb))
            current.append(value)
            row_min = min(row_min, value)
        if row_min > limit:
            return limit + 1
        previous = current
    return previous[-1]

def _split_joined(token, vocabulary):
    if token in vocabulary or len(token) < 7:
        return [token]
    # Dynamic segmentation prefers a complete multiword split over one unknown token.
    best = [None] * (len(token) + 1)
    best[0] = []
    for end in range(1, len(token) + 1):
        for start in range(max(0, end - 18), end):
            piece = token[start:end]
            if best[start] is not None and piece in vocabulary and len(piece) >= 2:
                candidate = best[start] + [piece]
                if best[end] is None or len(candidate) < len(best[end]):
                    best[end] = candidate
    if best[-1] and len(best[-1]) > 1:
        return best[-1]
    # Keep a short unknown tail as a candidate typo after recognized words.
    valid = [i for i, parts in enumerate(best[:-1]) if parts and len(token) - i <= 15]
    if valid:
        end = max(valid)
        tail = token[end:]\n        if len(tail) == 1 and end > 0 and tail == token[end - 1]:\n            return best[end]\n        return best[end] + [tail]
    # A repeated final keystroke is a common source of merged-word failures.
    if len(token) > 7 and token[-1] == token[-2]:
        return _split_joined(token[:-1], vocabulary)
    return [token]

def correct_spelling(text):
    vocab = dictionary_words()
    output = []
    changes = []
    for original in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[^A-Za-z]+", text):
        if not original or not original[0].isalpha():
            output.append(original)
            continue
        lower = original.lower()
        pieces = _split_joined(lower, vocab)
        fixed = []
        for piece in pieces:
            if piece in vocab:
                fixed.append(piece)
                continue
            # Collapse accidental repeated keystrokes: nowww -> now.
            squeezed = re.sub(r"([a-z])\1{2,}", r"\1\1", piece)
            if squeezed in vocab:
                replacement = squeezed
            else:
                max_distance = 1 if len(piece) < 7 else 2
                candidates = [w for w in vocab if abs(len(w) - len(piece)) <= max_distance]
                ranked = sorted(((_distance(piece, w, max_distance), w) for w in candidates))
                replacement = ranked[0][1] if ranked and ranked[0][0] <= max_distance else piece
            fixed.append(replacement)
        corrected = " ".join(fixed)
        if corrected != lower:
            changes.append((lower, corrected))
        output.append(corrected)
    result = "".join(output)
    if result:
        result = result[0].upper() + result[1:]
    return result, changes

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
    key = word.lower().strip(".,!?;:'\"")
    basic = load_dictionary().get(key)
    if basic:
        return basic
    return load_wordnet().get(key)
