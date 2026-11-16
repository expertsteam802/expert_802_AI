"""Explicit-fact graph with provenance, type rules and conservative inference."""
from collections import defaultdict, deque
import re


class KnowledgeGraph:
    def __init__(self):
        self.edges = defaultdict(list)
        self.type_rules = defaultdict(list)
        self.negative_types = defaultdict(list)
        self.meanings = defaultdict(list)

    @staticmethod
    def canonical(value):
        value = re.sub(r"[^a-z0-9' -]", " ", (value or "").lower())
        value = re.sub(r"\s+", " ", value).strip()
        return re.sub(r"^(?:the|a|an)\s+", "", value)

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
                sentence = sentence.strip().rstrip(".!?")
                if not sentence:
                    continue

                meaning = re.match(r"^(.+?)\s+(?:means|refers to|is called|stands for)\s+(.+)$", sentence, re.I)
                if meaning:
                    subject, value = map(cls.canonical, meaning.groups())
                    if subject and value:
                        graph.meanings[subject].append({"value": value, "source": sentence})
                    continue

                universal = re.match(r"^(?:all|every|each)\s+(.+?)\s+(?:are|is)\s+(.+)$", sentence, re.I)
                if universal:
                    subject, target = map(cls.canonical, universal.groups())
                    if subject and target:
                        graph.type_rules[subject].append({"type": target, "source": sentence})
                    continue

                location = re.match(r"^(.+?)\s+(?:is|are|was|were)\s+in\s+(.+)$", sentence, re.I)
                if location:
                    subject, target = map(cls.canonical, location.groups())
                    if subject and target:
                        graph.edges[(subject, "in")].append({"object": target, "source": sentence})
                    continue

                typed = re.match(r"^(.+?)\s+(?:is|are|was|were)\s+(not\s+)?(?:a|an)\s+(.+)$", sentence, re.I)
                if typed:
                    subject, negated, target = typed.groups()
                    subject, target = cls.canonical(subject), cls.canonical(target)
                    if subject and target:
                        destination = graph.negative_types if negated else graph.type_rules
                        destination[subject].append({"type": target, "source": sentence})
                    continue

                relation = re.match(r"^(.+?)\s+(?:has|have)\s+(.+)$", sentence, re.I)
                if relation:
                    subject, target = map(cls.canonical, relation.groups())
                    if subject and target:
                        graph.edges[(subject, "has")].append({"object": target, "source": sentence})
                    continue
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

    def _inferred_types(self, subject):
        start = self.canonical(subject)
        queue = deque([(start, [])])
        visited = {start}
        found = []
        while queue:
            node, sources = queue.popleft()
            for edge in self.type_rules.get(node, []):
                target = edge["type"]
                path_sources = sources + [edge["source"]]
                if target not in visited:
                    found.append((target, path_sources))
                    visited.add(target)
                    queue.append((target, path_sources))
            # An instance's type inherits every universal rule for that class.
            if not node.startswith("class:"):
                for rule in self.type_rules.get("class:" + node, []):
                    target = rule["type"]
                    path_sources = sources + [rule["source"]]
                    if target not in visited:
                        found.append((target, path_sources))
                        visited.add(target)
                        queue.append((target, path_sources))
        return found

    def _type_answer(self, subject, target):
        subject, target = self.canonical(subject), self.canonical(target)
        positive = self._inferred_types(subject)
        negative = self.negative_types.get(subject, [])
        yes = [(kind, sources) for kind, sources in positive if kind == target]
        no = [edge for edge in negative if edge["type"] == target]
        if yes and no:
            return {"reply": f"I have conflicting facts about whether {subject.title()} is a {target}: one supports it and another denies it.",
                    "evidence": [source for _, sources in yes for source in sources] + [x["source"] for x in no],
                    "confidence": 0.45}
        if yes:
            sources = yes[0][1]
            return {"reply": f"Yes. Based on the facts you gave me, {subject.title()} is a {target.title()}.",
                    "evidence": sources, "confidence": 0.96,
                    "reasoning": {"method": "type_inheritance", "inference_chain": sources}}
        if no:
            return {"reply": f"No. Your earlier statement says {subject.title()} is not a {target.title()}.",
                    "evidence": [item["source"] for item in no], "confidence": 0.96}
        return None

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
                            "evidence": sources, "confidence": 0.95,
                            "reasoning": {"method": "transitive_location_inference", "inference_chain": sources}}

        typed = re.match(r"^(?:is|are)\s+(?:a|an|the\s+)?(.+?)\s+(?:a|an)\s+(.+)$", text)
        if typed:
            return self._type_answer(typed.group(1), typed.group(2))

        for subject, facts in self.meanings.items():
            if re.search(r"\b" + re.escape(subject) + r"\b", text) and re.match(r"^(?:what|which)\s+", text):
                values = facts[-1]
                return {"reply": f"According to what you told me, {subject.title()} means {values['value']}.",
                        "evidence": [values["source"]], "confidence": 0.99}

        return None
