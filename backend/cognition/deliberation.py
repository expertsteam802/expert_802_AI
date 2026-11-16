"""General, bounded deliberation over a question and locally available evidence."""
import re
from backend.language import tokens
from .intent import STOP

NEGATIONS = {"not", "never", "no", "neither", "cannot", "can't", "isn't", "aren't"}

def _content_terms(text):
    return [term for term in tokens(text) if term not in STOP and len(term) > 1]

class Deliberator:
    """Decompose, retrieve, compare and verify without fabricating missing facts."""
    def __init__(self, retriever, max_passes=4, evidence_limit=8):
        self.retriever = retriever
        self.max_passes = max(1, min(int(max_passes), 6))
        self.evidence_limit = max(1, min(int(evidence_limit), 16))

    def analyze(self, question, history=None):
        original = (question or "").strip()
        clauses = [part.strip(" ,;") for part in re.split(
            r"(?<=[.!?])\s+|\s+(?:and also|and|but|because|although|while)\s+|\n+",
            original, flags=re.I) if len(_content_terms(part)) >= 2]
        subquestions = list(dict.fromkeys(clauses or [original]))
        queries = [original] + [part for part in subquestions if part != original]
        terms = _content_terms(original)
        if len(terms) >= 3 and len(queries) < self.max_passes:
            focused = " ".join(terms[:8])
            if focused not in queries:
                queries.append(focused)
        evidence, seen, pass_log = [], set(), []
        for number, query in enumerate(queries[:self.max_passes], 1):
            hits = self.retriever.retrieve(query, limit=4)
            added = 0
            for hit in hits:
                key = (hit.get("source", ""), hit.get("text", ""))
                if key in seen:
                    continue
                seen.add(key)
                evidence.append({**hit, "retrieval_query": query, "pass": number})
                added += 1
            pass_log.append({"pass": number, "query": query, "new_evidence": added})
        evidence.sort(key=lambda item: item.get("score", 0), reverse=True)
        conflicts = self._find_conflicts(evidence)
        evidence = evidence[:self.evidence_limit]
        sources = {item.get("source") for item in evidence}
        confidence = 0.0
        if evidence:
            confidence = min(0.82, 0.22 + min(evidence[0].get("score", 0) / 4, 0.35)
                             + min(len(sources) * 0.08, 0.24))
            if conflicts:
                confidence *= 0.55
        return {"question": original, "subquestions": subquestions, "passes": pass_log,
                "evidence": evidence, "conflicts": conflicts, "confidence": round(confidence, 3),
                "supported": bool(evidence and confidence >= 0.35 and not conflicts),
                "method": "decompose_retrieve_compare_verify"}

    @staticmethod
    def _find_conflicts(evidence):
        conflicts = []
        for index, left in enumerate(evidence):
            left_terms = set(_content_terms(left.get("text", "")))
            for right in evidence[index + 1:]:
                if left.get("source") == right.get("source"):
                    continue
                right_terms = set(_content_terms(right.get("text", "")))
                shared = left_terms & right_terms
                union = left_terms | right_terms
                if len(shared) >= 4 and len(shared) / max(1, len(union)) >= 0.35:
                    if bool(left_terms & NEGATIONS) != bool(right_terms & NEGATIONS):
                        conflicts.append({"shared_terms": sorted(shared)[:8],
                            "sources": [left.get("source", "unknown"), right.get("source", "unknown")],
                            "passages": [left.get("text", ""), right.get("text", "")]})
        return conflicts[:3]
