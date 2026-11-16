"""Local lexical retrieval with document-level provenance."""
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from backend.language import dictionary_lookup, load_wordnet, tokens

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "knowledge" / "english_corpus.txt"

class LocalRetriever:
    def __init__(self):
        self.records = self._load_document_records()
        self.documents = [record["text"] for record in self.records]
        self.sources = [record["source"] for record in self.records]
        self.index = defaultdict(list)
        self.lengths = []
        for doc_id, value in enumerate(self.documents):
            counts = Counter(tokens(value))
            self.lengths.append(sum(counts.values()))
            for term, count in counts.items():
                self.index[term].append((doc_id, count))
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))

    @staticmethod
    def _load_document_records():
        records = []
        sources = [CORPUS]
        reference_dir = ROOT / "knowledge" / "references"
        if reference_dir.exists():
            sources.extend(sorted(reference_dir.rglob("*.txt")))
        for source in sources:
            try:
                text = source.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            chunks = [part.strip() for part in re.split(r"(?<=[.!?])\\s+|\\n+", text)
                      if 35 < len(part.strip()) <= 1800]
            label = str(source.relative_to(ROOT))
            records.extend({"text": chunk, "source": label} for chunk in chunks)
        from backend.language import load_dictionary
        for term, entry in load_dictionary().items():
            if isinstance(entry, str) and entry.strip():
                records.append({"text": f"{term}: {entry.strip()}", "source": "local_dictionary"})
            elif isinstance(entry, dict):
                definitions = entry.get("definitions", [])
                if isinstance(definitions, str):
                    definitions = [definitions]
                if isinstance(definitions, list):
                    records.extend({"text": f"{term}: {value.strip()}", "source": "local_dictionary"}
                                   for value in definitions if isinstance(value, str) and value.strip())
        return records

    def retrieve(self, query, limit=3):
        terms = list(dict.fromkeys(tokens(query)))
        if not terms or not self.documents:
            return []
        weighted_terms = {term: 1.0 for term in terms}
        wordnet = load_wordnet()
        for term in terms:
            entry = wordnet.get(term, {})
            if isinstance(entry, dict):
                for synonym in entry.get("synonyms", [])[:6]:
                    synonym = str(synonym).lower()
                    if synonym and synonym not in weighted_terms:
                        weighted_terms[synonym] = 0.35
        scores = defaultdict(float)
        total = len(self.documents)
        for term, weight in weighted_terms.items():
            postings = self.index.get(term, [])
            if not postings:
                continue
            df = len(postings)
            idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
            for doc_id, frequency in postings:
                length = self.lengths[doc_id]
                denominator = frequency + 1.2 * (1 - 0.75 + 0.75 * length / max(1, self.average_length))
                scores[doc_id] += weight * idf * frequency * 2.2 / denominator
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        return [{"text": self.documents[i], "source": self.sources[i], "score": round(score, 4)}
                for i, score in ranked if score > 0]

    @staticmethod
    def define(term):
        entry = dictionary_lookup(term)
        if isinstance(entry, dict):
            definitions = entry.get("definitions", [])
            return {"term": term, "definition": definitions[0] if definitions else "",
                    "part_of_speech": entry.get("part_of_speech", "word"),
                    "synonyms": entry.get("synonyms", [])[:8], "source": "local_dictionary"}
        if isinstance(entry, str) and entry:
            return {"term": term, "definition": entry, "part_of_speech": "word",
                    "synonyms": [], "source": "local_dictionary"}
        return None
