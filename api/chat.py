"""Vercel Python endpoint for the provider-independent Expert 802 cognitive core."""
import json
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
        self._send(200, {
            "engine": "expert802-cognitive-core",
            "architecture": ["normalization", "intent_analysis", "dialogue_state", "knowledge_graph",
                             "local_retrieval", "symbolic_reasoning", "programming_planner", "safe_code_inspection", "tool_registry"],
            "external_model_provider": False,
            "model_configured": _ENGINE.neural.configured,
            "neural_generator_ready": neural_health["ready"],
            "neural_runtime": neural_health,
            "sandbox_runtime": sandbox_health,
            "coding_agent_ready": True,
            "code_execution_enabled": False,\n            "isolated_test_runner_ready": sandbox_health["ready"],
            "supported_code_inspection": ["Python", "JavaScript", "TypeScript", "HTML", "CSS", "JSON"],
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
            if size < 1 or size > 1100000:
                return self._send(400, {"error": "Request payload must be under 1.1 MB."})
            payload = json.loads(self.rfile.read(size).decode("utf-8"))
            if not isinstance(payload, dict):
                return self._send(400, {"error": "The request body must be a JSON object."})
            action = payload.get("action", "chat")
            if action == "code_run":
                if not _SANDBOX.health()["ready"]:
                    return self._send(503, {"error": "The private Docker sandbox is not ready. Configure the authenticated sandbox URL and start its runner."})
                command = payload.get("command", "python_unittest")
                target = payload.get("target", "")
                result = _SANDBOX.run(payload.get("files", []), command, target)
                return self._send(200 if result.get("ok") else 422, {"engine": "expert802-isolated-docker-runner", **result})
            if action == "code_generate":
                if not _ENGINE.neural.health()["ready"]:
                    return self._send(503, {"error": "The private Expert 802 neural generator is not ready. Train a checkpoint and start the private inference service first."})
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
