import os
import unittest
from unittest.mock import patch

from backend.cognition import online


class OnlineAssistantTests(unittest.TestCase):
    def test_greetings_do_not_trigger_web_search(self):
        self.assertFalse(online.should_search_web("Hello!"))
        self.assertFalse(online.should_search_web("What is your name?"))

    def test_factual_questions_trigger_web_search(self):
        self.assertTrue(online.should_search_web("Who invented the telephone?"))
        self.assertTrue(online.should_search_web("Explain quantum computing"))
        self.assertTrue(online.should_search_web("Can you explain how transformers work?"))

    def test_freshness_questions_trigger_web_search(self):
        self.assertTrue(online.should_search_web("What is the latest Python version today?"))

    def test_explicit_search_requests_trigger_web_search(self):
        self.assertTrue(online.should_search_web("Please search the web for current Vercel limits."))
        self.assertTrue(online.should_search_web("Search the internet for the official API docs."))

    def test_result_normalization_rejects_unsafe_urls_and_cleans_html(self):
        results = online._normalize_results([
            {"title": "<b>Useful</b> result", "description": "A <b>clear</b> snippet",
             "url": "https://example.com/page"},
            {"title": "Bad link", "description": "Ignore this", "url": "javascript:alert(1)"},
        ], "test", 5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Useful result")
        self.assertEqual(results[0]["snippet"], "A clear snippet")

    def test_external_language_model_adapter_is_not_part_of_runtime(self):
        self.assertFalse(hasattr(online, "online_model_config"))
        self.assertFalse(hasattr(online, "generate_online_reply"))

    def test_google_search_uses_programmable_search_and_normalizes_results(self):
        payload = {"items": [{"title": "Python documentation", "snippet": "Official Python docs.", "link": "https://docs.python.org/3/"}]}
        with patch.dict(os.environ, {"GOOGLE_CSE_API_KEY": "google-test-key", "GOOGLE_CSE_ID": "engine-test-id"}, clear=True), patch.object(online, "_json_request", return_value=payload) as request:
            results = online.search_web("Python documentation", 5)
        self.assertEqual(results[0]["source"], "Google Search")
        self.assertEqual(results[0]["url"], "https://docs.python.org/3/")
        self.assertIn("key=google-test-key", request.call_args.args[0])
        self.assertIn("cx=engine-test-id", request.call_args.args[0])

    def test_google_search_falls_through_when_not_configured(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(online, "_google_search", return_value=None), patch.object(online, "_wikipedia_search", return_value=[]) as fallback:
            online.search_web("example", 3)
        fallback.assert_called_once_with("example", 3)

    def test_search_uses_wikipedia_when_no_paid_provider_is_configured(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            online, "_wikipedia_search", return_value=[{
                "title": "Example", "snippet": "Evidence", "url": "https://en.wikipedia.org/?curid=1",
                "source": "Wikipedia",
            }]
        ) as fallback:
            results = online.search_web("example", 3)
        fallback.assert_called_once_with("example", 3)
        self.assertEqual(results[0]["source"], "Wikipedia")


if __name__ == "__main__":
    unittest.main()
