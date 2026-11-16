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

    def test_result_normalization_rejects_unsafe_urls_and_cleans_html(self):
        results = online._normalize_results([
            {"title": "<b>Useful</b> result", "description": "A <b>clear</b> snippet",
             "url": "https://example.com/page"},
            {"title": "Bad link", "description": "Ignore this", "url": "javascript:alert(1)"},
        ], "test", 5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Useful result")
        self.assertEqual(results[0]["snippet"], "A clear snippet")

    def test_hosted_model_is_disabled_without_server_key(self):
        with patch.dict(os.environ, {"EXPERT802_LLM_PROVIDER": "gemini"}, clear=True):
            self.assertFalse(online.online_model_config()["configured"])

    def test_existing_gemini_environment_key_is_detected(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "existing-secret"}, clear=True):
            config = online.online_model_config()
        self.assertTrue(config["configured"])
        self.assertEqual(config["provider"], "gemini")

    def test_gemini_uses_secure_openai_compatible_endpoint(self):
        with patch.dict(os.environ, {
            "EXPERT802_LLM_PROVIDER": "gemini",
            "EXPERT802_LLM_API_KEY": "test-secret",
        }, clear=True):
            config = online.online_model_config()
        self.assertTrue(config["configured"])
        self.assertEqual(config["model"], "gemini-2.5-flash")
        self.assertEqual(
            config["base_url"],
            "https://generativelanguage.googleapis.com/v1beta/openai",
        )

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
