import unittest
from backend.engine import ExpertLanguageEngine
from backend.language import correct_common_grammar, dictionary_lookup

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

if __name__ == "__main__":
    unittest.main()
