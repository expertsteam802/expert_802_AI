"""Explicit-fact graph with conservative transitive reasoning and provenance."""
from collections import defaultdict, deque
import re

class KnowledgeGraph:
    def __init__(self):
        self.edges = defaultdict(list)

    @staticmethod
    def canonical(value):
        value = re.sub(r"[^a-z0-9' -]", " ", (value or "").lower())
        value = re.sub(r"\s+", " ", value).strip()
        value = re.sub(r"^(?:the|a|an)\s+", "", value)
        return value

    @classmethod
    def from_history(cls, history):
        graph = cls()
        for item in history if isinstance(history, list) else []:
            if not isinstance(item, dict) or item.get("role") != "user":
                continue
            text = item.get("text", item.get("content", ""))
            if not isinstance(text, str):
                continue
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
                sentence = sentence.strip()
                patterns = [
                    (r"^(.+?)\s+(?:is|are|was|were)\s+in\s+(.+)$", "in"),
                    (r"^(.+?)\s+(?:is|are)\s+(?:a|an)\s+(.+)$", "type"),
                    (r"^(.+?)\s+has\s+(.+)$", "has"),
                    (r"^(.+?)\s+(?:is|are)\s+(.+)$", "is"),
                ]
                for pattern, predicate in patterns:
                    match = re.match(pattern, sentence, re.I)
                    if not match:
                        continue
                    subject = cls.canonical(match.group(1))
                    obj = cls.canonical(match.group(2))
                    if subject and obj and subject not in {"i", "you", "we", "they", "he", "she", "it"}:
                        graph.edges[(subject, predicate)].append({"object": obj, "source": sentence})
                    break
        return graph

    def _paths(self, subject, predicate="in"):
        start = self.canonical(subject)
        queue = deque([(start, [start], [])])
        visited = {start}
        while queue:
            node, nodes, sources = queue.popleft()
            for edge in self.edges.get((node, predicate), []):
                target = edge["object"]
                next_sources = sources + [edge["source"]]
                if target not in visited:
                    yield target, nodes + [target], next_sources
                    visited.add(target)
                    queue.append((target, nodes + [target], next_sources))

    def answer(self, question):
        text = self.canonical(question)
        where = re.match(r"^where\s+(?:is|are|was|were)\s+(.+)$", text)
        if where:
            subject = self.canonical(where.group(1))
            direct = self.edges.get((subject, "in"), [])
            objects = sorted({edge["object"] for edge in direct})
            if len(objects) == 1:
                return {"reply": f"According to what you told me, {subject.title()} is in {objects[0].title()}.",
                        "evidence": [edge["source"] for edge in direct], "confidence": 0.99}
            if len(objects) > 1:
                return {"reply": f"Your earlier notes give conflicting locations for {subject.title()}: {', '.join(value.title() for value in objects)}. Which one should I use?",
                        "evidence": [edge["source"] for edge in direct], "confidence": 0.99}

        relation = re.match(r"^(?:is|are|was|were)\s+(.+?)\s+in\s+(.+)$", text)
        if relation:
            subject, target = self.canonical(relation.group(1)), self.canonical(relation.group(2))
            direct = self.edges.get((subject, "in"), [])
            for edge in direct:
                if edge["object"] == target:
                    return {"reply": f"Yes. Your earlier note says {subject.title()} is in {target.title()}.",
                            "evidence": [edge["source"]], "confidence": 0.99}
            for found, nodes, sources in self._paths(subject, "in"):
                if found == target:
                    chain = " -> ".join(node.title() for node in nodes)
                    return {"reply": f"Yes. This follows from the facts you gave me: {chain}.",
                            "evidence": sources, "confidence": 0.95}

        return None
