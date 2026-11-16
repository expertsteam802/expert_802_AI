"""Minimal structural parsing; open-ended meaning is handled by the language model."""
import re
from dataclasses import dataclass, asdict
from backend.language import tokens

@dataclass
class IntentFrame:
    kind: str
    normalized: str
    target: str = ""
    action: str = ""
    confidence: float = 0.0
    slots: dict = None

    def to_dict(self):
        value = asdict(self)
        value["slots"] = value["slots"] or {}
        return value

STOP = {
    "a","an","the","i","you","he","she","it","we","they","me","my","your",
    "his","her","our","their","is","am","are","was","were","be","been",
    "do","does","did","to","of","in","on","at","for","with","and","or",
    "but","what","who","where","when","why","how","please","can","could",
    "would","should","will","now","just","really","very","about","this",
    "that","these","those","there","here","have","has","had","right",
}

def normalize(text):
    return re.sub(r"\s+", " ", str(text or "").strip())

def parse_intent(message, history=None):
    text = normalize(message)
    if not text:
        return IntentFrame("empty", "", confidence=1.0, slots={})

    # Safety boundary: identify explicit intent to physically harm a person.
    low = text.lower()
    harm = re.search(
        r"\b(?:kill|murder|stab|shoot|poison|hurt|attack)\b.{0,80}\b(?:someone|somebody|a person|him|her|them)\b"
        r"|\b(?:someone|somebody|a person)\b.{0,80}\b(?:kill|murder|stab|shoot|poison|hurt|attack)\b",
        low,
    )
    if harm:
        return IntentFrame("harm_risk", text, confidence=0.99, slots={"safety": "potential_physical_harm"})

    # Arithmetic is handled by the bounded calculator, not by generated prose.
    has_numbers = bool(re.search(r"\d", low))
    has_operator = bool(re.search(r"[+*/%^]|(?<!^)-|\b(?:plus|minus|times|multiplied by|divided by|over|modulo|mod)\b", low))
    if has_numbers and has_operator:
        expression = re.sub(r"^(?:what is|calculate|compute|solve)\s+", "", text, flags=re.I).strip(" ?!.")
        return IntentFrame("calculation", text, target=expression, confidence=0.98, slots={"operation": "arithmetic"})

    return IntentFrame("message", text, target=text, confidence=1.0, slots={})
