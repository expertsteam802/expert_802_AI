import json
import threading
import unittest
from http.server import HTTPServer
from urllib.request import Request, urlopen
from unittest.mock import patch

from backend.engine import ExpertLanguageEngine
from backend.language import correct_common_grammar, correct_spelling, dictionary_lookup
from api.chat import handler as ChatHandler


class EnglishFoundationTests(unittest.TestCase):
    def setUp(self):
        self.engine = ExpertLanguageEngine()

    def test_subject_verb_agreement_helper(self):
        corrected, notes = correct_common_grammar("She are ready")
        self.assertEqual(corrected, "She is ready")
        self.assertTrue(notes)

    def test_spelling_and_missing_spaces_are_recovered(self):
        corrected, changes = correct_spelling("i nned my gamerightnoww")
        self.assertIn("need", corrected.lower())
        self.assertTrue(changes)

    def test_dictionary_retrieves_word_meaning(self):
        answer = self.engine.reply("curious")["reply"].lower()
        self.assertIn("eager to know", answer)

    def test_natural_definition_request_uses_dictionary(self):
        answer = self.engine.reply("what does curious mean")["reply"].lower()
        self.assertIn("eager to know", answer)

    def test_multiple_dictionary_lookups_are_processed_in_order(self):
        message = "what does curious mean?    what does grammar mean?"
        result = self.engine.reply(message)
        answer = result["reply"].lower()
        self.assertTrue(result.get("batch_processed"))
        self.assertIn("eager to know", answer)
        self.assertIn("system and rules", answer)

    def test_open_ended_chat_does_not_fall_back_to_scripted_answers(self):
        result = self.engine.reply("I need my game right now")
        self.assertEqual(result["engine"], "dictionary-only")
        self.assertIn("pretrained conversational model is not connected", result["reply"])
        self.assertNotIn("Which do you mean", result["reply"])

    def test_pretrained_model_is_used_when_configured(self):
        with patch("backend.engine.is_configured", return_value=True), patch(
            "backend.engine.generate_reply", return_value="A generated answer based on context."
        ):
            result = self.engine.reply("Explain this sentence in context.")
        self.assertEqual(result["engine"], "qwen3-self-hosted")
        self.assertEqual(result["reply"], "A generated answer based on context.")

    def test_dictionary_contains_english_words(self):
        self.assertIn("eager to know", dictionary_lookup("curious"))

    def test_chat_http_endpoint_returns_dictionary_only_status(self):
        server = HTTPServer(("127.0.0.1", 0), ChatHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            payload = json.dumps({"message": "hello", "history": []}).encode()
            request = Request(
                f"http://127.0.0.1:{server.server_port}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=3) as response:
                data = json.loads(response.read().decode())
                self.assertEqual(response.status, 200)
                self.assertEqual(data["engine"], "dictionary-only")
                self.assertIn("pretrained conversational model is not connected", data["reply"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_dictionary_lookup_http_action(self):
        server = HTTPServer(("127.0.0.1", 0), ChatHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            payload = json.dumps({"action": "lookup", "term": "curious"}).encode()
            request = Request(
                f"http://127.0.0.1:{server.server_port}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=3) as response:
                data = json.loads(response.read().decode())
                self.assertEqual(response.status, 200)
                self.assertTrue(data["found"])
                self.assertIn("eager to know", data["definition"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
