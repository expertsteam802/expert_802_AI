"""Vercel Python serverless endpoint for Expert 802 AI."""
import json
from http.server import BaseHTTPRequestHandler
from backend.engine import ExpertLanguageEngine
from backend.language import dictionary_lookup, load_wordnet
from backend.model_client import is_configured, configured_model_name

_ENGINE = ExpertLanguageEngine()

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        model = _ENGINE.model or {}
        self._send(200, {
            "engine": "qwen3-self-hosted" if is_configured() else "python-english-v2",
            "model_configured": is_configured(),
            "model_name": configured_model_name() if is_configured() else None,
            "dictionary_entries": len(_ENGINE.dictionary),
            "wordnet_entries": len(load_wordnet()),
            "trained_artifact": model.get("format"),
            "training_sentences": model.get("sentences_seen", 0),
            "training_vocabulary": model.get("vocabulary_size", 0),
            "neural_model_ready": is_configured(),
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
            action = payload.get("action", "chat")
            if action == "lookup":
                term = payload.get("term", "")
                if not isinstance(term, str) or not term.strip():
                    return self._send(400, {"error": "Enter a word to look up."})
                term = term.strip().lower()[:120]
                entry = dictionary_lookup(term)
                if isinstance(entry, dict):
                    definitions = entry.get("definitions", [])
                    result = {
                        "term": term,
                        "part_of_speech": entry.get("part_of_speech", "word"),
                        "definition": definitions[0] if definitions else "No definition is available in this entry.",
                        "synonyms": entry.get("synonyms", [])[:8],
                        "found": bool(definitions),
                    }
                elif isinstance(entry, str) and entry:
                    result = {"term": term, "part_of_speech": "word", "definition": entry, "synonyms": [], "found": True}
                else:
                    result = {"term": term, "part_of_speech": None, "definition": "This word is not present in the local dictionary yet.", "synonyms": [], "found": False}
                return self._send(200, result)
            message = payload.get("message", "")
            if not isinstance(message, str) or not message.strip():
                return self._send(400, {"error": "A non-empty message is required."})
            history = payload.get("history", [])
            if not isinstance(history, list):
                history = []
            result = _ENGINE.reply(message[:4000], history[-12:])
            return self._send(200, result)
        except Exception:
            return self._send(500, {"error": "The English language engine encountered an internal error."})

    def _send(self, status, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
