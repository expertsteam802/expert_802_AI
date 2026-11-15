"""Local lexical retrieval over licensed project knowledge and the English corpus."""
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from backend.language import dictionary_lookup, tokens

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "knowledge" / "english_corpus.txt"

class LocalRetriever:
    def __init__(self):
        self.documents = self._load_documents()
        self.index = defaultdict(list)
        self.lengths = []
        for doc_id, text in enumerate(self.documents):
            terms = tokens(text)
            counts = Counter(terms)
            self.lengths.append(len(terms))
            for term, count in counts.items():
                self.index[term].append((doc_id, count))
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))

    @staticmethod
    def _load_documents():
        try:
            text = CORPUS.read_text(encoding="utf-8")
        except OSError:
            return []
        chunks = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if len(part.strip()) > 35]
        return chunks

    def retrieve(self, query, limit=3):
        terms = list(dict.fromkeys(tokens(query)))
        if not terms or not self.documents:
            return []
        scores = defaultdict(float)
        total = len(self.documents)
        for term in terms:
            postings = self.index.get(term, [])
            if not postings:
                continue
            df = len(postings)
            idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
            for doc_id, frequency in postings:
                length = self.lengths[doc_id]
                denominator = frequency + 1.2 * (1 - 0.75 + 0.75 * length / max(1, self.average_length))
                scores[doc_id] += idf * frequency * 2.2 / denominator
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        return [{"text": self.documents[i], "score": round(score, 4)} for i, score in ranked if score > 0]

    @staticmethod
    def define(term):
        entry = dictionary_lookup(term)
        if isinstance(entry, dict):
            definitions = entry.get("definitions", [])
            return {
                "term": term,
                "definition": definitions[0] if definitions else "",
                "part_of_speech": entry.get("part_of_speech", "word"),
                "synonyms": entry.get("synonyms", [])[:8],
            }
        if isinstance(entry, str) and entry:
            return {"term": term, "definition": entry, "part_of_speech": "word", "synonyms": []}
        return None
