"""Intent and slot analysis. This module builds structured meaning, not canned replies."""
import re
from dataclasses import dataclass, asdict
from backend.language import correct_spelling, tokens

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

ALIASES = {
    "ur": "your", "u": "you", "r": "are", "wht": "what",
    "pls": "please", "plz": "please", "thx": "thanks",
    "bc": "because", "bcz": "because", "idk": "i do not know",
    "im": "i am", "ive": "i have", "dont": "do not",
    "cant": "cannot", "wanna": "want to", "gonna": "going to",
}
STOP = {
    "a","an","the","i","you","he","she","it","we","they","me","my","your",
    "his","her","our","their","is","am","are","was","were","be","been",
    "do","does","did","to","of","in","on","at","for","with","and","or",
    "but","what","who","where","when","why","how","please","can","could",
    "would","should","will","now","just","really","very","about","this",
    "that","these","those","there","here","have","has","had","right"
}

def normalize(text):
    raw = re.sub(r"\s+", " ", (text or "").strip())
    if not raw:
        return ""
    # A one-word turn may be a topic, acronym, command, or name. Never
    # autocorrect it in isolation ("gta" must not become "eta").
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_+#.-]*", raw):
        return raw
    # Expand explicit chat shorthand only in multi-word messages.
    raw = re.sub(r"\b([A-Za-z]+)\b", lambda m: ALIASES.get(m.group(1).lower(), m.group(1)), raw)
    fixed, _ = correct_spelling(raw)
    return re.sub(r"\s+", " ", fixed).strip()

def _clean_target(value):
    value = re.sub(r"[?!.]+$", "", value.strip())
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"^(?:a|an|the)\s+", "", value, flags=re.I)
    return value.strip(" '""“”")

def parse_intent(message, history=None):
    text = normalize(message)
    low = re.sub(r"[?!.]+$", "", text.lower().strip())
    words = tokens(low)
    if not low:
        return IntentFrame("empty", "", confidence=1.0, slots={})

    last_assistant = ""
    for item in history if isinstance(history, list) else []:
        if isinstance(item, dict) and item.get("role") == "assistant":
            candidate = item.get("text", item.get("content", ""))
            if isinstance(candidate, str) and candidate.strip():
                last_assistant = candidate.strip()
    if low in {"yes", "yeah", "yep", "sure", "okay", "ok", "no", "nope", "nah"}:
        pending = bool(last_assistant and ("?" in last_assistant or re.search(r"\b(shall|would you|do you|can you|should i)\b", last_assistant, re.I)))
        kind = "confirmation" if low in {"yes", "yeah", "yep", "sure", "okay", "ok"} else "decline"
        return IntentFrame(kind if pending else "acknowledgement", text, target=last_assistant if pending else "",
                           confidence=0.94 if pending else 0.7, slots={"pending_question": pending})

    topic_terms = {
        "html", "css", "javascript", "typescript", "python", "java", "c",
        "c++", "c#", "code", "coding", "programming", "gui", "ui", "api",
        "sql", "database", "website", "web", "app", "android", "ios",
        "gta", "fifa", "minecraft", "fbi", "ai", "machine learning"
    }
    if low in topic_terms:
        return IntentFrame("topic", text, target=text, action="discuss",
                           confidence=0.9, slots={"topic": text})

    if re.fullmatch(r"(hi|hello|hey|good morning|good afternoon|good evening|howdy)[!. ]*", low):
        return IntentFrame("greeting", text, confidence=0.99, slots={})
    if re.fullmatch(r"(thanks|thank you|thx|many thanks)[!. ]*", low):
        return IntentFrame("gratitude", text, confidence=0.99, slots={})
    if re.search(r"\b(who are you|what are you|your name)\b", low):
        return IntentFrame("identity", text, confidence=0.99, slots={})

    math_match = re.match(r"^(?:what is|calculate|compute|solve)\s+(.+?)[?!.]*$", low)
    if math_match:
        expression = math_match.group(1).strip()
        has_number = bool(re.search(r"\d", expression))
        has_operator = bool(re.search(r"[+\-*/%^]|\b(plus|minus|times|divided by|over|percent of|x)\b", expression))
        if has_number and has_operator:
            return IntentFrame("calculation", text, target=expression, confidence=0.95, slots={})

    patterns = [
        r"^(?:what is|what's|define|meaning of|definition of)\s+(.+?)(?:\s+in full(?: words)?)?$",
        r"^what does\s+(.+?)\s+mean$",
        r"^what is the meaning of\s+(.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, low)
        if match:
            return IntentFrame("definition", text, target=_clean_target(match.group(1)), confidence=0.97, slots={})

    if re.search(r"\b(what about it|what about that|and it|why is that|how about it|tell me more)\b", low):
        return IntentFrame("followup", text, confidence=0.82, slots={})

    request_match = re.search(
        r"\b(?:i need|i want|i would like|help me|can you help me|find me|show me|make me|build me|create me)\s+(.+)$",
        low,
    )
    if request_match:
        target = _clean_target(request_match.group(1))
        return IntentFrame("request", text, target=target, action="help", confidence=0.88, slots={"object": target})

    if re.search(r"\b(explain|describe|how does|how do|why does|why do|teach me)\b", low):
        return IntentFrame("explanation", text, target=text, confidence=0.78, slots={})

    if low.endswith("?") or re.match(r"^(what|who|where|when|why|how|which|is|are|can|could|does|do|did|will|would)\b", low):
        return IntentFrame("question", text, target=text, confidence=0.68, slots={})

    return IntentFrame("statement", text, target=text, confidence=0.55, slots={})
