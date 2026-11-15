"""Provider-independent cognition pipeline: normalize -> parse -> remember -> retrieve -> reason."""
import re
from .intent import parse_intent, normalize
from .memory import build_state, resolve_reference
from .retrieval import LocalRetriever
from .reasoning import calculate, derive_simple_facts

class CognitiveEngine:
    def __init__(self):
        self.retriever = LocalRetriever()

    def reply(self, message, history=None):
        frame = parse_intent(message)
        state = build_state(history)
        base = {
            "engine": "expert802-cognitive-core",
            "language": "English",
            "intent": frame.to_dict(),
            "dialogue_state": state.to_dict(),
            "model_configured": False,
        }
        if frame.kind == "empty":
            return {**base, "reply": "Send me a message and I'll work with it."}
        if frame.kind == "greeting":
            return {**base, "reply": "Hello. What are you working on or thinking about?"}
        if frame.kind == "gratitude":
            return {**base, "reply": "You're welcome. What would you like to work on next?"}
        if frame.kind == "identity":
            return {**base, "reply": "I'm Expert 802 AI. My current core combines language normalization, intent analysis, conversation-state tracking, local knowledge retrieval, and deterministic reasoning. I'm still being developed; I won't claim abilities that aren't implemented."}

        if frame.kind == "calculation":
            try:
                value = calculate(frame.target)
                rendered = int(value) if isinstance(value, float) and value.is_integer() else value
                return {**base, "reply": f"{frame.target.strip()} = {rendered}", "reasoning": {"operation": "arithmetic", "result": rendered}}
            except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
                return {**base, "reply": "I recognized this as a calculation, but couldn't safely parse the expression. Write it with numbers and operators, for example: 18 * (4 + 2)."}

        if frame.kind == "definition":
            entry = self.retriever.define(frame.target)
            if entry and entry["definition"]:
                related = ", ".join(entry["synonyms"])
                answer = f"{frame.target} ({entry['part_of_speech']}): {entry['definition']}"
                if related:
                    answer += f" Related terms: {related}."
                return {**base, "reply": answer, "evidence": [entry], "confidence": 0.97}
            return {**base, "reply": f"I couldn't find a reliable local definition for “{frame.target}”. If you share the sentence where you saw it, I can analyze its use in context.", "confidence": 0.3}

        if frame.kind == "request":
            target = frame.target
            topic = resolve_reference(target, state) or target
            return {**base, "reply": self._respond_to_request(topic), "resolved_topic": topic, "confidence": frame.confidence}

        if frame.kind == "followup":
            topic = resolve_reference(frame.normalized, state)
            if topic:
                hits = self.retriever.retrieve(topic, limit=2)
                if hits:
                    return {**base, "reply": f"You're referring to {topic}. Here's the most relevant information I have: {hits[0]['text']}", "evidence": hits, "resolved_topic": topic}
                return {**base, "reply": f"You're referring to {topic}. I have the topic from our earlier messages, but not enough evidence to answer the follow-up yet. What part of it do you mean?", "resolved_topic": topic}
            return {**base, "reply": "I need the subject of that follow-up. Which earlier point are you referring to?"}

        if frame.kind == "explanation":
            query = frame.target
            hits = self.retriever.retrieve(query, limit=3)
            if hits:
                evidence = " ".join(item["text"] for item in hits[:2])
                return {**base, "reply": evidence, "evidence": hits, "confidence": min(0.9, hits[0]["score"] / 3)}
            return {**base, "reply": "I can analyze the question, but my current local knowledge does not contain enough evidence for a dependable explanation. Give me the specific term, system, or example you want examined.", "confidence": 0.25}

        if frame.kind == "question":
            topic = resolve_reference(frame.normalized, state)
            query = frame.normalized
            if topic and re.search(r"\b(it|that|this|they|them)\b", frame.normalized.lower()):
                query = topic + " " + frame.normalized
            hits = self.retriever.retrieve(query, limit=3)
            if hits and hits[0]["score"] >= 0.55:
                return {**base, "reply": "The strongest match in my local knowledge is: " + hits[0]["text"], "evidence": hits, "resolved_topic": topic, "confidence": min(0.85, hits[0]["score"] / 3)}
            if topic:
                return {**base, "reply": f"I resolved “it” to “{topic}”, but my local knowledge doesn't contain enough evidence to answer. Can you narrow down what you want to know about {topic}?", "resolved_topic": topic, "confidence": 0.35}
            return {**base, "reply": "I can identify this as a question, but I don't yet have enough grounded information to answer it accurately. Add context or a source, and I'll analyze it rather than inventing an answer.", "confidence": 0.2}

        # Statements are retained as context for later turns; do not mislabel them as questions.
        return {**base, "reply": self._interpret_statement(frame.normalized), "confidence": frame.confidence}

    @staticmethod
    def _respond_to_request(target):
        low = target.lower()
        if re.search(r"\b(game|gta|fifa|minecraft)\b", low):
            return f"You need help with {target}. I can help plan, choose, build, or troubleshoot it. What outcome do you want?"
        return f"You want help with {target}. Tell me the result you want, and I'll break the task into steps."

    @staticmethod
    def _interpret_statement(text):
        cleaned = text.strip()
        if not cleaned:
            return "I couldn't extract a usable statement."
        return f"I've parsed that as a statement: “{cleaned}”. What would you like me to do with it—explain it, check it, compare it, or use it as context?"
