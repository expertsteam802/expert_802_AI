import json
import threading
import unittest
from http.server import HTTPServer
from urllib.request import Request, urlopen

from backend.engine import ExpertLanguageEngine
from backend.language import correct_common_grammar, dictionary_lookup
from api.chat import handler as ChatHandler

class EnglishFoundationTests(unittest.TestCase):
    def test_subject_verb_agreement(self):
        corrected, notes = correct_common_grammar("She are ready")
        self.assertEqual(corrected, "She is ready")
        self.assertTrue(notes)

    def test_dictionary_contains_english_words(self):
        self.assertIn("eager to know", dictionary_lookup("curious"))

    def test_engine_is_explicit_about_its_limits(self):
        answer = ExpertLanguageEngine().reply("What is the capital of France?")["reply"]
        self.assertIn("should not guess", answer)

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
                self.assertEqual(data["engine"], "python-english-v1")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

if __name__ == "__main__":
    unittest.main()
