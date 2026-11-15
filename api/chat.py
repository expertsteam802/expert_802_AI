"""Vercel Python endpoint for the provider-independent Expert 802 cognitive core."""
import json
import os
import time
import hmac
import hashlib
import secrets
from collections import defaultdict, deque
from urllib.parse import urlparse
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from backend.engine import ExpertLanguageEngine
from backend.language import dictionary_lookup, load_dictionary, load_wordnet
from backend.cognition.agent import AgentPlanner, InspectionError
from backend.cognition.programming import build_work_plan
from backend.cognition.sandbox_runtime import SandboxRuntime

_ENGINE = ExpertLanguageEngine()
_AGENT = AgentPlanner()
_SANDBOX = SandboxRuntime()
_MODEL_PATH = Path(__file__).resolve().parents[1] / "backend" / "language_model.json"
_CSRF_SECRET = os.environ.get("EXPERT802_CSRF_SECRET", "")
_CODE_RUN_ENABLED = os.environ.get("EXPERT802_CODE_RUN_ENABLED", "").lower() == "true"
_RATE_BUCKETS = defaultdict(deque)


def _make_csrf_token():
    if len(_CSRF_SECRET) < 32:
        return ""
    timestamp = str(int(time.time()))
    nonce = secrets.token_urlsafe(16)
    message = timestamp + "." + nonce
    signature = hmac.new(_CSRF_SECRET.encode(), message.encode(), hashlib.sha256).hexdigest()
    return message + "." + signature

def _valid_csrf(handler):
    if len(_CSRF_SECRET) < 32:
        return False
    origin = handler.headers.get("Origin", "")
    host = handler.headers.get("Host", "")
    if not origin or not host or urlparse(origin).netloc.lower() != host.lower():
        return False
    parts = handler.headers.get("X-Expert802-CSRF", "").split(".")
    if len(parts) != 3:
        return False
    timestamp, nonce, signature = parts
    if not timestamp.isdigit():
        return False
    age = time.time() - int(timestamp)
    if age < -60 or age > 900:
        return False
    message = timestamp + "." + nonce
    expected = hmac.new(_CSRF_SECRET.encode(), message.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, expected)

def _rate_allowed(action, client_ip, limit=4, window=300):
    now = time.time()
    bucket = _RATE_BUCKETS[(action, client_ip)]
    while bucket and now - bucket[0] > window:
        bucket.popleft()
    if len(bucket) >= limit:
        return False
    bucket.append(now)
    return True

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
        neural_health = _ENGINE.neural.health()
        sandbox_health = _SANDBOX.health()
        csrf_token = _make_csrf_token()
        sandbox_health["ready"] = bool(sandbox_health["ready"] and _CODE_RUN_ENABLED and csrf_token)
        self._send(200, {
            "engine": "expert802-cognitive-core",
            "architecture": ["normalization", "intent_analysis", "dialogue_state", "knowledge_graph",
                             "local_retrieval", "symbolic_reasoning", "programming_planner", "safe_code_inspection", "tool_registry", "private_neural_runtime", "isolated_test_runner"],
            "external_model_provider": False,
            "model_configured": _ENGINE.neural.configured,
            "neural_generator_ready": neural_health["ready"],
            "code_generation_enabled": bool(neural_health["ready"] and csrf_token),
            "csrf_token": csrf_token,
            "neural_runtime": neural_health,
            "sandbox_runtime": sandbox_health,
            "coding_agent_ready": True,
            "code_execution_enabled": False,
            "isolated_test_runner_ready": sandbox_health["ready"],
            "supported_code_inspection": ["Python", "JavaScript", "TypeScript", "HTML", "CSS", "JSON"],
            "dictionary_entries": len(load_dictionary()),
            "wordnet_entries": len(load_wordnet()),
            "corpus_documents": len(_ENGINE.retriever.documents),
            **_training_stats(),
        })

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_POST(self):
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size < 1 or size > 1100000:
                return self._send(400, {"error": "Request payload must be under 1.1 MB."})
            payload = json.loads(self.rfile.read(size).decode("utf-8"))
            if not isinstance(payload, dict):
                return self._send(400, {"error": "The request body must be a JSON object."})
            action = payload.get("action", "chat")
            if action == "code_run":
                if not _CODE_RUN_ENABLED or not _SANDBOX.health()["ready"]:
                    return self._send(503, {"error": "The isolated runner is disabled or unavailable. An administrator must explicitly enable and configure it."})
                if not _valid_csrf(self):
                    return self._send(403, {"error": "The request did not pass same-origin approval. Reload the app and try again."})
                if not _rate_allowed("code_run", self.client_address[0], limit=3):
                    return self._send(429, {"error": "Test-run limit reached. Wait five minutes before trying again."})
                command = payload.get("command", "python_unittest")
                target = payload.get("target", "")
                result = _SANDBOX.run(payload.get("files", []), command, target)
                return self._send(200 if result.get("ok") else 422, {"engine": "expert802-isolated-docker-runner", **result})
            if action == "code_generate":
                if not _ENGINE.neural.health()["ready"]:
                    return self._send(503, {"error": "The private Expert 802 neural generator is not ready. Train a checkpoint and start the private inference service first."})
                if not _valid_csrf(self):
                    return self._send(403, {"error": "The request did not pass same-origin approval. Reload the app and try again."})
                if not _rate_allowed("code_generate", self.client_address[0], limit=3):
                    return self._send(429, {"error": "Patch-generation limit reached. Wait five minutes before trying again."})
                task = payload.get("task", "")
                if not isinstance(task, str) or not task.strip():
                    return self._send(400, {"error": "Describe the code change you want."})
                files = payload.get("files", [])
                project_context = ""
                if files:
                    inspected = _AGENT.inspect(files)
                    if not inspected["ok"]:
                        return self._send(400, {"error": inspected["error"]})
                    snippets = []
                    for item in files[:5]:
                        if isinstance(item, dict) and isinstance(item.get("path"), str) and isinstance(item.get("content"), str):
                            snippets.append("FILE: " + item["path"] + "\n" + item["content"][:1800])
                    project_context = "\n\n".join(snippets)
                generated = _ENGINE.neural.generate("Implement this programming task and return a focused patch or complete file as appropriate: " + task, [], project_context)
                if not generated["ok"]:
                    return self._send(502, {"error": generated["error"]})
                return self._send(200, {"engine": generated["engine"], "patch": generated["reply"],
                                        "warning": "Generated output has not been applied or executed. Review and test it before use."})
            if action == "code_analyze":
                files = payload.get("files", [])
                result = _AGENT.inspect(files)
                if not result["ok"]:
                    return self._send(400, {"error": result["error"]})
                task = payload.get("task", "Review this project for correctness and maintainability.")
                if not isinstance(task, str):
                    return self._send(400, {"error": "The engineering task must be text."})
                plan = _AGENT.plan(task, files)
                return self._send(200, {"engine": "expert802-coding-agent", "analysis": result["output"], "plan": plan})
            if action == "plan":
                task = payload.get("task", "")
                if not isinstance(task, str) or not task.strip():
                    return self._send(400, {"error": "A non-empty engineering task is required."})
                return self._send(200, {"engine": "expert802-coding-agent", "plan": _AGENT.plan(task)})
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
            if _ENGINE.neural.configured and not _rate_allowed("neural_chat", self.client_address[0], limit=30):
                return self._send(429, {"error": "Conversation generation is temporarily rate limited. Please wait and try again."})
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
        self.end_headers()
        self.wfile.write(body)
