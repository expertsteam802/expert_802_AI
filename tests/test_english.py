import json
import threading
import unittest
from http.server import HTTPServer
from urllib.request import Request, urlopen

from backend.engine import ExpertLanguageEngine
from backend.cognition.intent import normalize, parse_intent
from backend.cognition.reasoning import calculate
from backend.cognition.code_intelligence import inspect_project, InspectionError
from backend.cognition.programming import build_work_plan, is_programming_request
from backend.language import correct_common_grammar, dictionary_lookup
from api.chat import handler as ChatHandler

class CognitiveArchitectureTests(unittest.TestCase):
    def setUp(self):
        self.engine = ExpertLanguageEngine()

    def test_programming_request_is_classified_and_planned(self):
        self.assertTrue(is_programming_request("Please build a Python API"))
        plan = build_work_plan("Fix the Python API error")
        self.assertEqual(plan["kind"], "debug")
        self.assertTrue(plan["requires_project_files"])

    def test_python_project_inspection_maps_symbols_and_security_warnings(self):
        result = inspect_project([{"path": "app.py", "content": "import os\\n\\ndef run():\\n    eval('1 + 1')\\n"}])
        self.assertEqual(result["file_count"], 1)
        self.assertIn("run", [item["name"] for item in result["symbols"]])
        self.assertIn("dynamic_eval", [item["code"] for item in result["diagnostics"]])

    def test_project_inspection_detects_python_syntax_error(self):
        result = inspect_project([{"path": "broken.py", "content": "def broken(:\\n pass"}])
        self.assertEqual(result["summary"]["errors"], 1)
        self.assertEqual(result["diagnostics"][0]["code"], "python_syntax")

    def test_project_inspection_detects_invalid_json_and_js_delimiters(self):
        result = inspect_project([
            {"path": "config.json", "content": "{bad json}"},
            {"path": "app.js", "content": "function run() { return [1, 2; "}
        ])
        codes = {item["code"] for item in result["diagnostics"]}
        self.assertIn("invalid_json", codes)
        self.assertIn("unclosed_delimiter", codes)

    def test_project_inspection_rejects_path_traversal(self):
        with self.assertRaises(InspectionError):
            inspect_project([{"path": "../secret.py", "content": "print('no')"}])

    def test_spelling_and_joined_words_are_recovered(self):
        fixed = normalize("i nned my gamerightnoww")
        self.assertIn("need", fixed.lower())
        self.assertIn("game", fixed.lower())
        self.assertIn("now", fixed.lower())

    def test_common_grammar_helper(self):
        corrected, notes = correct_common_grammar("She are ready")
        self.assertEqual(corrected, "She is ready")
        self.assertTrue(notes)

    def test_definition_uses_local_knowledge(self):
        result = self.engine.reply("what does curious mean")
        self.assertIn("eager to know", result["reply"].lower())
        self.assertEqual(result["engine"], "expert802-cognitive-core")

    def test_definition_target_parsing(self):
        frame = parse_intent("what is a curious")
        self.assertEqual(frame.kind, "definition")
        self.assertEqual(frame.target, "curious")

    def test_arithmetic_is_parsed_before_definition_intent(self):
        frame = parse_intent("what is 12 plus 8?")
        self.assertEqual(frame.kind, "calculation")
        self.assertEqual(calculate(frame.target), 20)
        self.assertIn("20", self.engine.reply("what is 12 plus 8?")["reply"])

    def test_arithmetic_supports_percentages(self):
        self.assertEqual(calculate("20 percent of 50"), 10)

    def test_arithmetic_supports_precedence_and_parentheses(self):
        self.assertEqual(calculate("18 * (4 + 2)"), 108)

    def test_arithmetic_rejects_code_execution(self):
        with self.assertRaises((ValueError, SyntaxError)):
            calculate("__import__('os').system('echo unsafe')")

    def test_request_extracts_the_requested_object(self):
        result = self.engine.reply("i nned my gamerightnoww")
        self.assertEqual(result["intent"]["kind"], "request")
        self.assertIn("game", result["reply"].lower())
        self.assertNotIn("pretrained conversational model", result["reply"].lower())

    def test_multiple_turns_are_processed_in_order(self):
        result = self.engine.reply("what does curious mean?    what does grammar mean?")
        self.assertTrue(result.get("batch_processed"))
        self.assertEqual(result.get("batch_items"), 2)
        self.assertIn("eager to know", result["reply"].lower())
        self.assertIn("system and rules", result["reply"].lower())

    def test_followup_resolves_previous_topic(self):
        history = [
            {"role": "user", "text": "I need help with Minecraft"},
            {"role": "assistant", "text": "What do you want to do with it?"},
        ]
        result = self.engine.reply("what about it?", history)
        self.assertEqual(result["resolved_topic"], "minecraft")
        self.assertIn("minecraft", result["reply"].lower())

    def test_knowledge_graph_performs_transitive_location_reasoning(self):
        history = [
            {"role": "user", "text": "Paris is in France."},
            {"role": "assistant", "text": "Noted."},
            {"role": "user", "text": "France is in Europe."},
            {"role": "assistant", "text": "Noted."},
        ]
        result = self.engine.reply("Is Paris in Europe?", history)
        self.assertTrue(result.get("memory_used"))
        self.assertEqual(result["reasoning"]["method"], "explicit_fact_graph")
        self.assertIn("Paris -> France -> Europe", result["reply"])

    def test_knowledge_graph_surfaces_conflicting_facts(self):
        history = [
            {"role": "user", "text": "Paris is in France."},
            {"role": "user", "text": "Paris is in Texas."},
        ]
        result = self.engine.reply("Where is Paris?", history)
        self.assertIn("conflicting locations", result["reply"].lower())
        self.assertIn("france", result["reply"].lower())
        self.assertIn("texas", result["reply"].lower())

    def test_proper_names_are_not_autocorrected(self):
        self.assertIn("Paris", normalize("Where is Paris?"))

    def test_explicit_user_facts_can_be_recalled(self):
        history = [
            {"role": "user", "text": "Paris is in France"},
            {"role": "assistant", "text": "Got it."},
        ]
        result = self.engine.reply("Where is Paris?", history)
        self.assertTrue(result.get("memory_used"))
        self.assertIn("Paris is in France", result["reply"])

    def test_identity_describes_real_implemented_components(self):
        result = self.engine.reply("who are you")
        self.assertIn("intent analysis", result["reply"].lower())
        self.assertIn("still being developed", result["reply"].lower())

    def test_dictionary_lookup_still_works(self):
        self.assertIn("eager to know", dictionary_lookup("curious"))

    def test_http_status_has_no_external_model_provider(self):
        server = HTTPServer(("127.0.0.1", 0), ChatHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urlopen(f"http://127.0.0.1:{server.server_port}/api/chat", timeout=5) as response:
                data = json.loads(response.read().decode())
                self.assertEqual(response.status, 200)
                self.assertEqual(data["engine"], "expert802-cognitive-core")
                self.assertFalse(data["external_model_provider"])
                self.assertIn("symbolic_reasoning", data["architecture"])
                self.assertIn("knowledge_graph", data["architecture"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_http_chat_uses_native_engine(self):
        server = HTTPServer(("127.0.0.1", 0), ChatHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            payload = json.dumps({"message": "what is 2 plus 3", "history": []}).encode()
            request = Request(f"http://127.0.0.1:{server.server_port}/api/chat", data=payload,
                              headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(request, timeout=5) as response:
                data = json.loads(response.read().decode())
                self.assertEqual(response.status, 200)
                self.assertEqual(data["engine"], "expert802-cognitive-core")
                self.assertIn("5", data["reply"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

if __name__ == "__main__":
    unittest.main()
