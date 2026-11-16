import json
import threading
import unittest
from http.server import HTTPServer
from urllib.request import Request, urlopen

from backend.engine import ExpertLanguageEngine
from backend.cognition.intent import normalize, parse_intent
from backend.cognition.reasoning import calculate
from backend.cognition.deliberation import Deliberator
from backend.cognition.code_intelligence import inspect_project, InspectionError
from backend.cognition.programming import build_work_plan, is_programming_request
from backend.cognition.agent import AgentPlanner
from sandbox.runner import validate_files, execute
from backend.language import correct_common_grammar, dictionary_lookup
from api.chat import handler as ChatHandler

class CognitiveArchitectureTests(unittest.TestCase):
    def setUp(self):
        self.engine = ExpertLanguageEngine()

    def test_agent_plan_exposes_registered_tools_and_safety_policy(self):
        plan = AgentPlanner().plan("Review this Python project")
        names = {tool["name"] for tool in plan["available_tools"]}
        self.assertIn("inspect_project", names)
        self.assertIn("run_python_tests", names)
        self.assertFalse(plan["code_execution_enabled"])

    def test_sandbox_rejects_unregistered_commands(self):
        with self.assertRaises(ValueError):
            execute([{"path":"app.py","content":"print('safe')"}],"shell")

    def test_sandbox_validates_paths_before_container_start(self):
        with self.assertRaises(InspectionError):
            validate_files([{"path":"../outside.py","content":"print('unsafe')"}])

    def test_programming_request_is_classified_and_planned(self):
        self.assertTrue(is_programming_request("Please build a Python API"))
        self.assertTrue(is_programming_request("Help me with this Python error"))
        plan = build_work_plan("Fix the Python API error")
        self.assertEqual(plan["kind"], "debug")
        self.assertTrue(plan["requires_project_files"])

    def test_python_project_inspection_maps_symbols_and_security_warnings(self):
        result = inspect_project([{"path": "app.py", "content": "import os\n\ndef run():\n    eval('1 + 1')\n"}])
        self.assertEqual(result["file_count"], 1)
        self.assertIn("run", [item["name"] for item in result["symbols"]])
        self.assertIn("dynamic_eval", [item["code"] for item in result["diagnostics"]])

    def test_project_inspection_flags_python_and_javascript_security_risks(self):
        python_source = chr(10).join(["import subprocess", "def run(command):", "    subprocess.run(command, shell=True)", ""])
        js_source = "eval(userInput); element.innerHTML = userInput;"
        result = inspect_project([
            {"path":"unsafe.py","content":python_source},
            {"path":"unsafe.js","content":js_source},
        ])
        codes = {item["code"] for item in result["diagnostics"]}
        self.assertIn("subprocess_shell_true", codes)
        self.assertIn("javascript_eval", codes)
        self.assertIn("unsafe_inner_html", codes)

    def test_project_inspection_detects_python_syntax_error(self):
        result = inspect_project([{"path": "broken.py", "content": "def broken(:\n pass"}])
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

    def test_user_taught_acronym_meaning_is_recalled_in_followup(self):
        history = [
            {"role": "user", "text": "do you know PS5"},
            {"role": "assistant", "text": "Do you mean the gaming device?"},
            {"role": "user", "text": "its play station 5, a gaming device"},
        ]
        result = self.engine.reply("what is PS5?", history)
        self.assertTrue(result.get("memory_used"))
        self.assertIn("play station 5", result["reply"].lower())
        self.assertIn("gaming device", result["reply"].lower())

    def test_user_taught_expansion_is_recalled_in_followup(self):
        history = [
            {"role": "user", "text": "html"},
            {"role": "assistant", "text": "What about HTML?"},
            {"role": "user", "text": "it means hyper text markup language"},
        ]
        result = self.engine.reply("what is html in full?", history)
        self.assertTrue(result.get("memory_used"))
        self.assertIn("hyper text markup language", result["reply"].lower())

    def test_bare_topic_does_not_return_random_dictionary_excerpt(self):
        result = self.engine.reply("html")
        self.assertNotIn("local references", result["reply"].lower())
        self.assertIn("what would you like to know", result["reply"].lower())

    def test_knowledge_question_is_not_misread_as_a_command(self):
        frame = parse_intent("do you know PS5")
        self.assertEqual(frame.kind, "knowledge_check")
        self.assertEqual(frame.target.lower(), "ps5")

    def test_user_correction_is_attached_to_previous_topic(self):
        from backend.cognition.conversation_memory import ConversationMemory
        history = [
            {"role": "user", "text": "do you know PS5"},
            {"role": "assistant", "text": "Which part?"},
        ]
        memory = ConversationMemory(history + [{"role": "user", "text": "its play station 5, a gaming device"}])
        fact = memory.meaning_for("PS5")
        self.assertIsNotNone(fact)
        self.assertIn("play station 5", fact["value"].lower())
        self.assertIn("gaming device", fact["value"].lower())

    def test_pronoun_meaning_correction_is_attached_to_topic(self):
        from backend.cognition.conversation_memory import ConversationMemory
        history = [
            {"role": "user", "text": "html"},
            {"role": "assistant", "text": "What do you mean?"},
            {"role": "user", "text": "it means hyper text markup language"},
        ]
        memory = ConversationMemory(history)
        self.assertIn("hyper text markup language", memory.meaning_for("HTML")["value"].lower())

    def test_additional_meaning_correction_is_attached_to_topic(self):
        from backend.cognition.conversation_memory import ConversationMemory
        history = [
            {"role": "user", "text": "food"},
            {"role": "assistant", "text": "What about food?"},
            {"role": "user", "text": "also it means what we eat to thrive"},
        ]
        memory = ConversationMemory(history)
        self.assertIn("what we eat to thrive", memory.meaning_for("food")["value"].lower())

    def test_user_defined_food_meaning_is_used_in_later_question(self):
        history = [
            {"role": "user", "text": "food"},
            {"role": "assistant", "text": "What about food?"},
            {"role": "user", "text": "also it means what we eat to thrive"},
            {"role": "assistant", "text": "Got it."},
        ]
        result = self.engine.reply("what is food?", history)
        self.assertTrue(result.get("memory_used"))
        self.assertIn("what we eat to thrive", result["reply"].lower())

    def test_harm_request_is_not_treated_as_a_task_to_assist(self):
        self.assertEqual(parse_intent("i want to kill someone").kind, "harm_risk")
        self.assertEqual(parse_intent("materials used to kill someone").kind, "harm_risk")
        result = self.engine.reply("i want materials used to kill someone")
        self.assertIn("can't help find weapons", result["reply"].lower())

    def test_weapon_followup_keeps_recent_harm_context(self):
        history = [
            {"role": "user", "text": "I want to kill someone"},
            {"role": "assistant", "text": "I can't help hurt anyone."},
            {"role": "user", "text": "I need materials used to kill"},
        ]
        result = self.engine.reply("a gun", history)
        self.assertIn("can't help find weapons", result["reply"].lower())

    def test_acknowledgement_without_pending_question_is_natural(self):
        result = self.engine.reply("yes", [{"role": "assistant", "text": "I'll use that meaning."}])
        self.assertEqual(result["intent"]["kind"], "acknowledgement")
        self.assertIn("got it", result["reply"].lower())

    def test_greetings_bypass_topic_retrieval(self):
        frame = parse_intent("hi")
        self.assertEqual(frame.kind, "greeting")
        result = self.engine.reply("hi")
        self.assertEqual(result["intent"]["kind"], "greeting")
        self.assertIn("Hello", result["reply"])

    def test_single_word_topics_are_not_spell_corrected_or_misclassified(self):
        for word in ("html", "code", "hui", "fooli", "gta", "FBI"):
            self.assertEqual(normalize(word), word)
        self.assertEqual(parse_intent("html").kind, "topic")
        self.assertEqual(parse_intent("code").kind, "topic")
        self.assertEqual(parse_intent("gta").target, "gta")
        self.assertEqual(parse_intent("FBI").target, "FBI")

    def test_intent_extracts_unlisted_action_and_object_compositionally(self):
        frame = parse_intent("play GTA")
        self.assertEqual(frame.kind, "action_request")
        self.assertEqual(frame.action, "play")
        self.assertEqual(frame.target.lower(), "gta")

    def test_arbitrary_single_word_is_a_topic_not_a_special_case(self):
        frame = parse_intent("games")
        self.assertEqual(frame.kind, "topic")
        self.assertEqual(frame.target, "games")

    def test_statement_extracts_user_supplied_meaning(self):
        result = self.engine.reply("it refers to letter b typed 5 times")
        self.assertIn("letter b typed 5 times", result["reply"].lower())
        self.assertEqual(result["asserted_fact"]["relation"], "refers_to")

    def test_action_response_does_not_claim_to_launch_external_software(self):
        result = self.engine.reply("play GTA")
        self.assertIn("play", result["action_frame"]["verb"])
        self.assertIn("cannot directly control your computer", result["reply"])

    def test_yes_resolves_as_confirmation_only_when_assistant_asked(self):
        history = [{"role": "assistant", "text": "Would you like me to explain HTML?"}]
        self.assertEqual(parse_intent("yes", history).kind, "confirmation")
        self.assertEqual(parse_intent("yes", []).kind, "acknowledgement")

    def test_short_unknown_word_is_preserved_and_clarified(self):
        result = self.engine.reply("hui")
        self.assertIn("hui", result["reply"])
        self.assertNotIn("Gui", result["reply"])
        self.assertNotIn("statement", result["reply"].lower())

    def test_yes_is_not_treated_as_an_isolated_statement(self):
        history = [{"role": "assistant", "text": "Would you like help with HTML?"}]
        result = self.engine.reply("yes", history)
        self.assertIn("agreement", result["reply"].lower())
        self.assertEqual(result["intent"]["kind"], "confirmation")

    def test_yes_executes_a_single_action_offered_in_previous_turn(self):
        history = [{"role": "assistant", "text": "Would you like me to explain HTML?"}]
        result = self.engine.reply("yes", history)
        self.assertEqual(result["intent"]["kind"], "action_request")
        self.assertEqual(result["action_frame"]["verb"], "explain")

    def test_yes_does_not_guess_between_multiple_offered_options(self):
        history = [{"role": "assistant", "text": "Would you like me to explain it, check it, or compare it?"}]
        result = self.engine.reply("yes", history)
        self.assertIn("more than one option", result["reply"])

    def test_deliberator_splits_compound_questions_and_keeps_source_provenance(self):
        class FakeRetriever:
            def retrieve(self, query, limit=3):
                if "water" in query.lower():
                    return [{"text":"Water boils at 100 degrees Celsius at standard atmospheric pressure.",
                             "source":"science.txt","score":1.8}]
                if "pressure" in query.lower():
                    return [{"text":"The boiling point changes when atmospheric pressure changes.",
                             "source":"physics.txt","score":1.5}]
                return []
        result = Deliberator(FakeRetriever()).analyze(
            "What is water's boiling point, and how does pressure affect it?"
        )
        self.assertGreaterEqual(len(result["subquestions"]), 2)
        self.assertTrue(result["evidence"])
        self.assertIn(result["evidence"][0]["source"], {"science.txt", "physics.txt"})
        self.assertEqual(result["method"], "decompose_retrieve_compare_verify")

    def test_question_route_runs_deliberation_even_without_neural_weights(self):
        result = self.engine.reply("How does water affect climate?")
        self.assertIn("deliberation", result)
        self.assertEqual(result["deliberation"]["method"], "decompose_retrieve_compare_verify")

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

    def test_symbolic_reasoner_uses_universal_rules_transitively(self):
        history = [
            {"role": "user", "text": "Fido is a dog."},
            {"role": "user", "text": "All dogs are mammals."},
            {"role": "user", "text": "All mammals are animals."},
        ]
        result = self.engine.reply("Is Fido an animal?", history)
        self.assertTrue(result.get("memory_used"))
        self.assertIn("yes", result["reply"].lower())
        self.assertEqual(result["reasoning"]["method"], "type_inheritance")
        self.assertEqual(len(result["reasoning"]["inference_chain"]), 3)

    def test_symbolic_reasoner_does_not_assume_unknown_relations(self):
        result = self.engine.reply("Is a dragon an animal?")
        self.assertFalse(result.get("memory_used"))
        self.assertNotIn("Yes.", result["reply"])

    def test_knowledge_graph_performs_transitive_location_reasoning(self):
        history = [
            {"role": "user", "text": "Paris is in France."},
            {"role": "assistant", "text": "Noted."},
            {"role": "user", "text": "France is in Europe."},
            {"role": "assistant", "text": "Noted."},
        ]
        result = self.engine.reply("Is Paris in Europe?", history)
        self.assertTrue(result.get("memory_used"))
        self.assertEqual(result["reasoning"]["method"], "transitive_location_inference")
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
        self.assertIn("programming assistant", result["reply"].lower())
        self.assertIn("static code inspection", result["reply"].lower())

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

    def test_http_code_run_requires_private_sandbox_configuration(self):
        server = HTTPServer(("127.0.0.1", 0), ChatHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            payload = json.dumps({"action":"code_run","command":"python_unittest","files":[]}).encode()
            request = Request(f"http://127.0.0.1:{server.server_port}/api/chat",data=payload,
                              headers={"Content-Type":"application/json"},method="POST")
            with self.assertRaises(Exception) as context:
                urlopen(request,timeout=5)
            self.assertIn("503",str(context.exception))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_http_code_analysis_endpoint(self):
        server = HTTPServer(("127.0.0.1", 0), ChatHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            payload = json.dumps({
                "action": "code_analyze",
                "task": "Review this Python project",
                "files": [{"path": "app.py", "content": "def main():\n    return 42\n"}],
            }).encode()
            request = Request(f"http://127.0.0.1:{server.server_port}/api/chat", data=payload,
                              headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(request, timeout=5) as response:
                data = json.loads(response.read().decode())
                self.assertEqual(response.status, 200)
                self.assertEqual(data["engine"], "expert802-coding-agent")
                self.assertEqual(data["analysis"]["file_count"], 1)
                self.assertIn("main", [item["name"] for item in data["analysis"]["symbols"]])
                self.assertFalse(data["plan"]["code_execution_enabled"])
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
