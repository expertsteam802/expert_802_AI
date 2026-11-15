"""Vercel Python endpoint for the provider-independent Expert 802 cognitive core."""
import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from backend.engine import ExpertLanguageEngine
from backend.language import dictionary_lookup, load_dictionary, load_wordnet

_ENGINE = ExpertLanguageEngine()
_MODEL_PATH = Path(__file__).resolve().parents[1] / "backend" / "language_model.json"

def _training_stats():
    try:
        data = json.loads(_MODEL_PATH.read_text(encoding="utf-8"))
        if data.get("format") != "expert802-word-bigram-v1":
            return {}
        return {"trained_artifact": data.get("format"),
                "training_sentences": data.get("sentences_seen", 0),
                "training_vocabulary": data.get("vocabulary_size", 0)}
    except (OSError, ValueError):
        return {}

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self._send(200, {
            "engine": "expert802-cognitive-core",
            "architecture": ["normalization", "intent_analysis", "dialogue_state", "knowledge_graph",
                             "local_retrieval", "symbolic_reasoning"],
            "external_model_provider": False,
            "model_configured": False,
            "neural_generator_ready": False,
            "dictionary_entries": len(load_dictionary()),
            "wordnet_entries": len(load_wordnet()),
            "corpus_documents": len(_ENGINE.retriever.documents),
            **_training_stats(),
        })

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.end_headers()

    def do_POST(self):
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size < 1 or size > 20000:
                return self._send(400, {"error": "Send a message under 20 KB."})
            payload = json.loads(self.rfile.read(size).decode("utf-8"))
            if not isinstance(payload, dict):
                return self._send(400, {"error": "The request body must be a JSON object."})
            if payload.get("action", "chat") == "lookup":
                term = payload.get("term", "")
                if not isinstance(term, str) or not term.strip():
                    return self._send(400, {"error": "Enter a word to look up."})
                term = term.strip().lower()[:120]
                entry = dictionary_lookup(term)
                if isinstance(entry, dict):
                    definitions = entry.get("definitions", [])
                    result = {"term": term, "part_of_speech": entry.get("part_of_speech", "word"),
                              "definition": definitions[0] if definitions else "No definition is available in this entry.",
                              "synonyms": entry.get("synonyms", [])[:8], "found": bool(definitions)}
                elif isinstance(entry, str) and entry:
                    result = {"term": term, "part_of_speech": "word", "definition": entry,
                              "synonyms": [], "found": True}
                else:
                    result = {"term": term, "part_of_speech": None,
                              "definition": "This word is not present in the local dictionary yet.",
                              "synonyms": [], "found": False}
                return self._send(200, result)
            message = payload.get("message", "")
            if not isinstance(message, str) or not message.strip():
                return self._send(400, {"error": "A non-empty message is required."})
            history = payload.get("history", [])
            if not isinstance(history, list):
                history = []
            return self._send(200, _ENGINE.reply(message[:4000], history[-30:]))
        except Exception as error:
            print("Expert 802 cognitive core error:", repr(error))
            return self._send(500, {"error": "The cognitive engine encountered an internal error."})

    def _send(self, status, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)
