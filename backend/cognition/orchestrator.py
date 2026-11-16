"""Provider-independent cognition pipeline: normalize -> parse -> remember -> retrieve -> reason."""
import re
from .intent import parse_intent, normalize
from .memory import build_state, resolve_reference
from .retrieval import LocalRetriever
from .reasoning import calculate, derive_simple_facts
from backend.language import tokens
from .knowledge_graph import KnowledgeGraph
from .programming import is_programming_request, build_work_plan
from .neural_runtime import NeuralRuntime

class CognitiveEngine:
    def __init__(self):
        self.retriever = LocalRetriever()
        self.neural = NeuralRuntime()

    def reply(self, message, history=None):
        original = (message or "").strip()
        parts = [
            part.strip(" \t\r\n")
            for part in re.split(
                r"\s{2,}|(?<=[?!])\s+(?=(?:what|who|where|when|why|how)\b)",
                original,
                flags=re.I,
            )
            if part.strip(" \t\r\n")
        ]
        if len(parts) > 1:
            running = list(history) if isinstance(history, list) else []
            rendered = []
            for part in parts:
                result = self._reply_single(part, running)
                answer = result.get("reply", "")
                rendered.append("You: " + part + "\nExpert 802 AI: " + answer)
                running.extend([{"role": "user", "text": part}, {"role": "assistant", "text": answer}])
            return {"engine": "expert802-cognitive-core", "reply": "\n\n".join(rendered),
                    "language": "English", "batch_processed": True, "batch_items": len(parts)}
        return self._reply_single(original, history)

    def _reply_single(self, message, history=None):
        frame = parse_intent(message, history)
        state = build_state(history)
        knowledge_graph = KnowledgeGraph.from_history(history)
        base = {
            "engine": "expert802-cognitive-core",
            "language": "English",
            "intent": frame.to_dict(),
            "dialogue_state": state.to_dict(),
            "cognition_trace": ["normalize", "parse_intent", "build_dialogue_state", "build_knowledge_graph", "retrieve_evidence", "reason", "compose_response"],
        }
        if frame.kind == "empty":
            return {**base, "reply": "Send me a message and I'll work with it."}
        if frame.kind == "greeting":
            return {**base, "reply": "Hello. What are you working on or thinking about?"}
        if frame.kind == "gratitude":
            return {**base, "reply": "You're welcome. What would you like to work on next?"}
        if frame.kind == "topic":
            topic = frame.target
            evidence = self._retrieve_topic_evidence(topic)
            if evidence:
                return {**base, "reply": "I found this in my local references:\n" + "\n".join(item["text"] for item in evidence[:2]) +
                        "\n\nIs this the aspect of the topic you meant?", "topic": topic,
                        "evidence": evidence, "reasoning": {"method": "lexical_and_corpus_retrieval"}}
            return {**base, "reply": f"You've introduced the topic “{topic}”. I don't have enough matching reference material to explain it reliably yet. What are you trying to find out about it?", "topic": topic,
                    "reasoning": {"method": "topic_detection_without_supported_evidence"}}
        if frame.kind == "action_request":
            action, target = frame.action, frame.target
            evidence = self._retrieve_topic_evidence(target) if target else []
            if evidence:
                evidence_text = "\n".join(item["text"] for item in evidence[:2])
            else:
                evidence_text = ""
            capability = (
                "I can explain the steps, help troubleshoot it, or work with files and tools that are connected to this session. "
                "I cannot directly control your computer or launch a game from this chat."
            )
            reply = f"I understand this as an instruction to {action}" + (f" {target}." if target else ".")
            if evidence_text:
                reply += "\n\nRelevant local reference material:\n" + evidence_text
            reply += "\n\n" + capability
            return {**base, "reply": reply, "action_frame": {"verb": action, "object": target},
                    "evidence": evidence, "reasoning": {"method": "compositional_action_object_analysis"}}
        if frame.kind in {"confirmation", "decline"}:
            if frame.kind == "decline":
                return {**base, "reply": "No problem. What would you prefer to do instead?", "pending_question": frame.target}
            return {**base, "reply": "Got it. What would you like me to do next?", "pending_question": frame.target}
        if frame.kind == "acknowledgement":
            return {**base, "reply": "I heard your acknowledgement, but I don't have a specific pending question to attach it to. What should I apply it to?", "confidence": frame.confidence}
        if frame.kind == "identity":
            neural_status = "My private neural runtime is ready." if self.neural.health()["ready"] else "My private neural generator is not connected yet."
            return {**base, "reply": "I'm Expert 802 AI, a provider-independent conversational and programming assistant. My implemented core includes intent analysis, dialogue context, explicit-fact reasoning, local retrieval, engineering task planning, static code inspection and registered tools. " + neural_status + " Code execution is restricted to an optional isolated test runner; I do not claim to have run or changed code without a tool result."}

        if is_programming_request(message):
            plan = build_work_plan(message)
            plan_text = "\n".join(str(index + 1) + ". " + step for index, step in enumerate(plan["steps"]))
            neural = self.neural.generate(message, history, "Engineering workflow:\n" + plan_text)
            if neural["ok"]:
                return {**base, "engine": neural["engine"], "intent": {**frame.to_dict(), "kind": "programming_task"},
                        "engineering_plan": plan, "reply": neural["reply"], "neural_generated": True}
            next_input = plan.get("next_input", "Send the relevant source files or project snapshot so I can inspect the actual code.")
            answer = "I classified this as a " + plan["kind"] + " engineering task.\n\n" + plan_text + "\n\n" + next_input + "\n\nCode Lab can inspect Python, JavaScript, TypeScript, HTML, CSS and JSON project files. A trained private neural generator is not connected yet; I will not pretend to generate or execute arbitrary code."
            return {**base, "intent": {**frame.to_dict(), "kind": "programming_task"},
                    "engineering_plan": plan, "reply": answer}

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
            neural = self.neural.generate(message, history)
            if neural["ok"]:
                return {**base, "engine": neural["engine"], "reply": neural["reply"], "neural_generated": True}
            return {**base, "reply": "I can analyze the question, but my current local knowledge does not contain enough evidence for a dependable explanation. Give me the specific term, system, or example you want examined.", "confidence": 0.25}

        if frame.kind == "question":
            graph_answer = knowledge_graph.answer(frame.normalized)
            if graph_answer:
                return {**base, **graph_answer, "memory_used": True, "reasoning": {"method": "explicit_fact_graph"}}
            remembered = self._match_fact(frame.normalized, state.facts)
            if remembered:
                return {**base, "reply": "From your earlier message: " + remembered["text"],
                        "evidence": [remembered], "confidence": 0.99, "memory_used": True}
            topic = resolve_reference(frame.normalized, state)
            query = frame.normalized
            if topic and re.search(r"\b(it|that|this|they|them)\b", frame.normalized.lower()):
                query = topic + " " + frame.normalized
            hits = self.retriever.retrieve(query, limit=3)
            if hits and hits[0]["score"] >= 0.55:
                return {**base, "reply": "The strongest match in my local knowledge is: " + hits[0]["text"], "evidence": hits, "resolved_topic": topic, "confidence": min(0.85, hits[0]["score"] / 3)}
            if topic:
                return {**base, "reply": f"I resolved “it” to “{topic}”, but my local knowledge doesn't contain enough evidence to answer. Can you narrow down what you want to know about {topic}?", "resolved_topic": topic, "confidence": 0.35}
            neural = self.neural.generate(message, history)
            if neural["ok"]:
                return {**base, "engine": neural["engine"], "reply": neural["reply"], "neural_generated": True}
            return {**base, "reply": "I can identify this as a question, but I don't yet have enough grounded information to answer it accurately. Add context or a source, and I'll analyze it rather than inventing an answer.", "confidence": 0.2}

        # Treat declarative input as information to inspect and retain, not as a
        # request to choose from a menu of canned operations.
        relation = re.search(r"\b(.+?)\s+(?:means|refers to|describes|is called)\s+(.+)$", message, re.I)
        if relation:
            subject, meaning = relation.group(1).strip(), relation.group(2).strip().rstrip(".!?")
            return {**base, "reply": f"Understood: {subject} refers to {meaning}. I'll use that meaning in this conversation.",
                    "asserted_fact": {"subject": subject, "relation": "refers_to", "object": meaning},
                    "reasoning": {"method": "relation_extraction"}}

        neural = self.neural.generate(message, history)
        if neural["ok"]:
            return {**base, "engine": neural["engine"], "reply": neural["reply"], "neural_generated": True}
        return {**base, "reply": self._interpret_statement(frame.normalized), "confidence": frame.confidence}

    def _retrieve_topic_evidence(self, topic):
        if not topic:
            return []
        terms = tokens(topic)
        queries = [topic]
        # Try simple English inflection recovery so "games" can retrieve
        # references indexed under "game", without rewriting the user's text.
        for term in terms:
            if len(term) > 4 and term.endswith("s"):
                queries.append(term[:-1])
            if len(term) > 5 and term.endswith("ies"):
                queries.append(term[:-3] + "y")
        results = []
        seen = set()
        for query in queries:
            for item in self.retriever.retrieve(query, limit=3):
                key = item["text"]
                if key not in seen:
                    seen.add(key)
                    results.append(item)
        for term in terms[:4]:
            entry = self.retriever.define(term)
            if entry and entry.get("definition"):
                text = f"{term} ({entry.get('part_of_speech', 'word')}): {entry['definition']}"
                if text not in seen:
                    seen.add(text)
                    results.append({"text": text, "score": 1.0, "source": "local_dictionary"})
        return sorted(results, key=lambda item: item.get("score", 0), reverse=True)[:4]

    @staticmethod
    def _match_fact(query, facts):
        query_terms = set(tokens(query))
        for fact in reversed(facts or []):
            subject_terms = set(tokens(fact.get("subject", "")))
            if subject_terms and subject_terms.intersection(query_terms):
                return fact
            if fact.get("kind") == "preference" and set(tokens(fact.get("value", ""))).intersection(query_terms):
                return fact
        return None

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
