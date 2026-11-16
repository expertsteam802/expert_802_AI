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

    # Social turns must be classified before the generic topic fallback.
    # Otherwise "hi" is misrouted into local retrieval.
    if re.fullmatch(r"(hi|hello|hey|good morning|good afternoon|good evening|howdy|bro+|bruh|yo)[!. ,]*", low):
        return IntentFrame("greeting", text, confidence=0.99, slots={})
    if re.fullmatch(r"(thanks|thank you|thx|many thanks)[!. ]*", low):
        return IntentFrame("gratitude", text, confidence=0.99, slots={})


    # Recognize knowledge questions before lexical verb/object parsing.
    if re.match(r"^(?:do you know|have you heard of|are you familiar with)\s+", low):
        target = re.sub(r"^(?:do you know|have you heard of|are you familiar with)\s+", "", low).strip(" ?!.")
        return IntentFrame("knowledge_check", text, target=_clean_target(target),
                           confidence=0.94, slots={"question_type": "familiarity"})
    if re.search(r"\b(?:i want to|i am going to|i'm going to|help me)\s+(?:kill|murder|stab|shoot|poison|hurt)\s+(?:someone|somebody|him|her|them|a person)\b", low):
        return IntentFrame("harm_risk", text, confidence=0.99, slots={"target": "person"})
    if re.search(r"\b(?:materials|weapons|tools|ways)\s+(?:to|used to)\s+(?:kill|murder|hurt)\s+(?:someone|somebody|a person)\b", low):
        return IntentFrame("harm_risk", text, confidence=0.99, slots={"target": "person"})

    math_match = re.match(r"^(?:what is|calculate|compute|solve)\s+(.+?)[?!.]*$", low)
    if math_match:
        expression = math_match.group(1).strip()
        if re.search(r"\d", expression) and re.search(r"[+\-*/%^]|\b(plus|minus|times|divided by|over|percent of|x)\b", expression):
            return IntentFrame("calculation", text, target=expression, confidence=0.95, slots={})
    # Extract a verb/object frame compositionally from the user's wording.
    # Verb status comes from the local lexical database, not a hand-written
    # list of supported topics or canned examples.
    def lexical_is_verb(word):
        from backend.language import dictionary_lookup
        entry = dictionary_lookup(word)
        if isinstance(entry, str):
            return entry.strip().lower().startswith("to ")
        if isinstance(entry, dict):
            pos = str(entry.get("part_of_speech", "")).lower()
            definitions = entry.get("definitions", [])
            return pos.startswith("verb") or any(
                isinstance(value, str) and value.strip().lower().startswith("to ")
                for value in (definitions if isinstance(definitions, list) else [definitions])
            )
        return False

    words_only = re.findall(r"[a-z][a-z'-]*", low)
    start = 0
    if words_only[:3] in (["i", "want", "to"], ["i", "need", "to"], ["i", "would", "like"]):
        start = 3 if words_only[2] == "to" else 2
    elif words_only[:2] in (["can", "you"], ["could", "you"], ["please", "help"]):
        start = 2
    elif words_only[:1] in (["please"],):
        start = 1
    if words_only:
        candidate_indexes = [start] if start < len(words_only) else []
        if not candidate_indexes and start == 0:
            candidate_indexes = [0]
        for verb_index in candidate_indexes:
            verb = words_only[verb_index]
            if lexical_is_verb(verb):
                object_words = words_only[verb_index + 1:]
                object_text = " ".join(object_words).strip()
                if not object_text:
                    continue
                return IntentFrame(
                    "action_request", text, target=object_text, action=verb,
                    confidence=0.82 if object_text else 0.68,
                    slots={"verb": verb, "object": object_text, "syntax": "imperative_or_infinitive"}
                )

    if re.fullmatch(r"(hi|hello|hey|good morning|good afternoon|good evening|howdy)[!. ]*", low):
        return IntentFrame("greeting", text, confidence=0.99, slots={})
    if re.fullmatch(r"(thanks|thank you|thx|many thanks)[!. ]*", low):
        return IntentFrame("gratitude", text, confidence=0.99, slots={})
    if re.search(r"\b(who are you|what are you|your name)\b", low):
        return IntentFrame("identity", text, confidence=0.99, slots={})

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

    # Fall back to a bare topic only after full question and action forms.
    if (words_only and len(words_only) <= 3 and not re.search(r"[?]", text)
            and words_only[0] not in {"what", "who", "where", "when", "why", "how", "is", "are", "do", "does", "did", "can", "could", "would", "will", "should", "i", "you", "he", "she", "it", "we", "they"}):
        return IntentFrame("topic", text, target=text, action="explore",
                           confidence=0.55, slots={"topic_candidate": text})

    return IntentFrame("statement", text, target=text, confidence=0.55, slots={})
