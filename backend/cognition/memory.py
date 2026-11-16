"""Conversation-state extraction and lightweight reference resolution."""
from dataclasses import dataclass, asdict
from .intent import STOP
import re
from backend.language import tokens

@dataclass
class DialogueState:
    topic: str = ""
    previous_user_message: str = ""
    previous_assistant_message: str = ""
    facts: list = None

    def to_dict(self):
        value = asdict(self)
        value["facts"] = value["facts"] or []
        return value

def _history_pairs(history):
    pairs = []
    for item in history if isinstance(history, list) else []:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        text = item.get("text", item.get("content", ""))
        if role in ("user", "assistant") and isinstance(text, str) and text.strip():
            pairs.append((role, text.strip()))
    return pairs[-30:]

def _topic_from(text):
    clean = (text or "").strip().rstrip("?.!")
    patterns = [
        r"^(?:what\s+is|what's|define|meaning\s+of|definition\s+of)\s+(.+?)(?:\s+in\s+full(?:\s+words)?)?$",
        r"^(?:do\s+you\s+know|have\s+you\s+heard\s+of|are\s+you\s+familiar\s+with)\s+(.+)$",
        r"^(?:tell\s+me\s+about|explain|describe)\s+(.+)$",
        r"^(?:i\s+need\s+help\s+with|help\s+me\s+with|i\s+want\s+to\s+know\s+about)\s+(.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, clean, re.I)
        if match:
            return re.sub(r"^(?:a|an|the)\s+", "", match.group(1).strip(), flags=re.I)
    # Preserve meaningful multiword names and abbreviations rather than
    # selecting the final grammatical token (e.g. "in full" -> "full").
    candidates = [w for w in tokens(clean) if w not in STOP and len(w) > 1]
    return candidates[-1] if candidates else ""

def build_state(history):
    pairs = _history_pairs(history)
    user_messages = [text for role, text in pairs if role == "user"]
    assistant_messages = [text for role, text in pairs if role == "assistant"]
    topic = ""
    for message in reversed(user_messages):
        candidate = _topic_from(message)
        if candidate:
            topic = candidate
            break
    facts = []
    for role, message in pairs:
        if role != "user":
            continue
        patterns = [
            r"\b([A-Za-z][\w -]{1,40}?)\s+(?:is|are|was|were)\s+([^.!?]{2,100})",
            r"\bi\s+(?:like|love|prefer|hate|use|own|work on)\s+([^.!?]{2,100})",
        ]
        for index, pattern in enumerate(patterns):
            match = re.search(pattern, message, re.I)
            if match:
                facts.append({
                    "kind": "relation" if index == 0 else "preference",
                    "subject": match.group(1).strip() if index == 0 else "user",
                    "predicate": "is" if index == 0 else "preference",
                    "text": message,
                    "value": match.group(2 if index == 0 else 1).strip(),
                })
    return DialogueState(
        topic=topic,
        previous_user_message=user_messages[-1] if user_messages else "",
        previous_assistant_message=assistant_messages[-1] if assistant_messages else "",
        facts=facts[-12:],
    )

def resolve_reference(text, state):
    low = (text or "").lower()
    if re.search(r"\b(it|that|this|they|them|those|he|she)\b", low) and state and state.topic:
        return state.topic
    return ""
