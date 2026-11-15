import json
import threading
import unittest
from http.server import HTTPServer
from urllib.request import Request, urlopen

from backend.engine import ExpertLanguageEngine
from backend.language import correct_common_grammar, correct_spelling, dictionary_lookup
from api.chat import handler as ChatHandler

class EnglishFoundationTests(unittest.TestCase):
    def setUp(self):
        self.engine = ExpertLanguageEngine()

    def test_subject_verb_agreement(self):
        corrected, notes = correct_common_grammar("She are ready")
        self.assertEqual(corrected, "She is ready")
        self.assertTrue(notes)

    def test_spelling_and_missing_spaces_are_recovered(self):
        corrected, changes = correct_spelling("i nned my gamerightnoww")
        self.assertEqual(corrected.lower(), "i need my game right now")
        self.assertTrue(changes)

    def test_engine_responds_to_recovered_sentence(self):
        answer = self.engine.reply("i nned my gamerightnoww")["reply"].lower()
        self.assertIn("i think you meant", answer)
        self.assertIn("you need a game right now", answer)

    def test_short_yes_uses_game_context(self):
        history = [{"role": "user", "text": "I need my game right now"}]
        answer = self.engine.reply("yes", history)["reply"].lower()
        self.assertIn("which do you mean", answer)

    def test_gta_is_not_spell_corrected_in_game_context(self):
        history = [{"role": "user", "text": "I need my game right now"}, {"role": "user", "text": "yes"}]
        answer = self.engine.reply("GTA", history)["reply"]
        self.assertIn("Grand Theft Auto", answer)
        self.assertNotIn("Eta", answer)

    def test_ai_full_form(self):
        answer = self.engine.reply("What is AI in full words?")["reply"]
        self.assertIn("artificial intelligence", answer.lower())

    def test_food_definition(self):
        answer = self.engine.reply("What is food?")["reply"].lower()
        self.assertIn("people or animals eat", answer)

    def test_car_definition_is_not_wrong_wordnet_sense(self):
        answer = self.engine.reply("what is the definition of car")["reply"].lower()
        self.assertIn("road vehicle", answer)

    def test_one_word_input_is_helpful(self):
        answer = self.engine.reply("food")["reply"].lower()
        self.assertIn("food", answer)
        self.assertNotIn("add a little more detail", answer)

    def test_dictionary_contains_english_words(self):
        self.assertIn("eager to know", dictionary_lookup("curious"))

    def test_engine_is_explicit_about_current_facts(self):
        answer = self.engine.reply("What is the capital of France?")["reply"]
        self.assertIn("should not invent", answer)

    def test_chat_http_endpoint_returns_json_reply(self):
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
                self.assertIn("Hello", data["reply"])
                self.assertEqual(data["engine"], "python-english-v2")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

if __name__ == "__main__":
    unittest.main()
