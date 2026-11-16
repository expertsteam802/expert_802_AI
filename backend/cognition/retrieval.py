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
        documents = []
        sources = [CORPUS]
        reference_dir = ROOT / "knowledge" / "references"
        if reference_dir.exists():
            sources.extend(sorted(reference_dir.rglob("*.txt")))
        for source in sources:
            try:
                text = source.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            chunks = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text)
                      if 35 < len(part.strip()) <= 1800]
            documents.extend(chunks)

        # Index dictionary glosses as evidence too. This makes the local
        # lexical resource searchable by meaning, rather than only by exact
        # "define X" requests.
        from backend.language import load_dictionary, load_wordnet
        for term, entry in load_dictionary().items():
            if isinstance(entry, str) and entry.strip():
                documents.append(f"{term}: {entry.strip()}")
            elif isinstance(entry, dict):
                definitions = entry.get("definitions", [])
                if isinstance(definitions, str):
                    definitions = [definitions]
                if isinstance(definitions, list):
                    documents.extend(f"{term}: {value.strip()}" for value in definitions
                                     if isinstance(value, str) and value.strip())
        return documents

    def retrieve(self, query, limit=3):
        terms = list(dict.fromkeys(tokens(query)))
        if not terms or not self.documents:
            return []
        # Expand with WordNet synonym evidence at a lower weight than exact terms.
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
        for term, query_weight in weighted_terms.items():
            postings = self.index.get(term, [])
            if not postings:
                continue
            df = len(postings)
            idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
            for doc_id, frequency in postings:
                length = self.lengths[doc_id]
                denominator = frequency + 1.2 * (1 - 0.75 + 0.75 * length / max(1, self.average_length))
                scores[doc_id] += query_weight * idf * frequency * 2.2 / denominator
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
