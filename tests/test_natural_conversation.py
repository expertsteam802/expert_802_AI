import unittest

from backend.cognition.intent import normalize, parse_intent
from backend.cognition.orchestrator import CognitiveEngine


class NaturalConversationTests(unittest.TestCase):
    def test_casual_yooo_is_a_greeting(self):
        self.assertEqual(parse_intent("yooo").kind, "greeting")

    def test_common_html_typo_is_corrected(self):
        self.assertEqual(normalize("hitml"), "html")
        frame = parse_intent("hitml")
        self.assertEqual(frame.kind, "topic")
        self.assertEqual(frame.target, "html")

    def test_i_want_to_code_is_not_parsed_as_to_code(self):
        frame = parse_intent("i want to code")
        self.assertEqual(frame.kind, "request")
        reply = CognitiveEngine._respond_to_request(frame.target)
        self.assertIn("let us code", reply.lower())
        self.assertNotIn("to code", reply.lower())

    def test_html_topic_gets_a_useful_response(self):
        reply = CognitiveEngine._respond_to_topic("html")
        self.assertIn("structures the content of web pages", reply)
        self.assertIn("build", reply.lower())

    def test_code_topic_invites_a_concrete_build(self):
        reply = CognitiveEngine._respond_to_topic("code")
        self.assertIn("let us build something", reply.lower())
        self.assertIn("website", reply.lower())


if __name__ == "__main__":
    unittest.main()
