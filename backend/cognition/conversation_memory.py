"""Conversation-grounded semantic memory with explicit provenance.

This module extracts only user-stated meanings and relations. It does not treat
assistant guesses or retrieved dictionary passages as facts about the user.
"""
import re
from backend.language import tokens

_TOPIC_PATTERNS = (
    r"^(?:what\s+is|what's|define|meaning\s+of|definition\s+of)\s+(.+?)(?:\s+in\s+full(?:\s+words)?)?$",
    r"^(?:do\s+you\s+know|have\s+you\s+heard\s+of|are\s+you\s+familiar\s+with)\s+(.+?)\??$",
    r"^(?:tell\s+me\s+about|explain|describe)\s+(.+)$",
    r"^(?:i\s+need\s+help\s+with|help\s+me\s+with|i\s+want\s+to\s+know\s+about)\s+(.+)$",
)
_PRONOUN_CORRECTION = re.compile(
    r"^(?:it|this|that|also\s+it)\s+(?:means|refers\s+to|is\s+called)\s+(.+)$", re.I
)
_COPULA_CORRECTION = re.compile(r"^(?:it\s+is|it's|its)\s+(.+)$", re.I)
_RELATION = re.compile(
    r"^(.{1,60}?)\s+(?:means|refers\s+to|is\s+called|stands\s+for)\s+(.{2,240})$",
    re.I,
)


def canonical(value):
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def _strip_punctuation(value):
    return re.sub(r"^[\s:=-]+|[\s.!?;]+$", "", (value or "")).strip()


def topic_from_turn(text):
    clean = _strip_punctuation(text)
    for pattern in _TOPIC_PATTERNS:
        match = re.match(pattern, clean, re.I)
        if match:
            return _strip_punctuation(match.group(1))
    # A bare noun, acronym or product name is a valid topic candidate.
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9+#._-]{1,40}", clean):
        return clean
    return ""


class ConversationMemory:
    def __init__(self, history=None):
        self.turns = [
            (item.get("role"), item.get("text", item.get("content", "")).strip())
            for item in (history if isinstance(history, list) else [])
            if isinstance(item, dict)
            and item.get("role") in {"user", "assistant", "memory"}
            and isinstance(item.get("text", item.get("content", "")), str)
            and item.get("text", item.get("content", "")).strip()
        ][-120:]
        self.facts = []
        self.topic = ""
        self._build()

    def _build(self):
        for role, text in self.turns:
            clean = _strip_punctuation(text)
            if role == "memory":
                remembered = _RELATION.match(clean)
                if remembered:
                    subject, value = map(_strip_punctuation, remembered.groups())
                    if subject and value and subject.lower() not in {"it", "this", "that", "also it"}:
                        self._remember(subject, value, text, update_topic=False)
                continue
            if role != "user":
                continue
            relation = _RELATION.match(clean)
            if relation:
                subject, value = map(_strip_punctuation, relation.groups())
                if subject.lower() in {"it", "this", "that", "also it"} and self.topic:
                    subject = self.topic
                if subject and value:
                    self._remember(subject, value, text)
                    continue

            pronoun = _PRONOUN_CORRECTION.match(clean)
            if pronoun and self.topic:
                value = _strip_punctuation(pronoun.group(1))
                if value:
                    self._remember(self.topic, value, text)
                    continue

            copula = _COPULA_CORRECTION.match(clean)
            if copula and self.topic:
                value = _strip_punctuation(copula.group(1))
                if value:
                    self._remember(self.topic, value, text)
                    continue

            candidate = topic_from_turn(clean)
            if candidate:
                self.topic = candidate

    def _remember(self, subject, value, source, update_topic=True):
        key = canonical(subject)
        if not key:
            return
        self.facts = [fact for fact in self.facts if fact["key"] != key]
        self.facts.append({
            "key": key, "subject": subject, "value": value,
            "source": source, "kind": "user_defined_meaning", "relation": "refers_to",
        })
        self.topic = subject

    def meaning_for(self, term):
        key = canonical(term)
        for fact in reversed(self.facts):
            if fact["key"] == key:
                return fact
        return None

    def resolve_topic(self, text=""):
        candidate = topic_from_turn(text)
        if candidate:
            return candidate
        low = (text or "").lower()
        if re.search(r"\b(it|that|this|they|them|those|he|she)\b", low):
            return self.topic
        return self.topic

    def export_facts(self):
        return [dict(fact) for fact in self.facts]
