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

    def test_informal_current_us_president_question_is_normalized(self):
        self.assertEqual(
            online.normalize_search_query("whos is the USA current President"),
            "current president of the United States",
        )
        self.assertTrue(online.should_search_web("whos is the USA current President"))

    def test_grounded_answer_selects_relevant_web_evidence(self):
        results = [
            {"title": "List of presidents of the United States",
             "snippet": "Donald Trump is the 47th and current president of the United States.",
             "url": "https://example.org/presidents", "source": "test"},
            {"title": "Thomas Jefferson",
             "snippet": "Thomas Jefferson was the third president.",
             "url": "https://example.org/jefferson", "source": "test"},
        ]
        answer = online.answer_from_web_results("whos is the USA current President", results)
        self.assertIn("Donald Trump", answer)

    def test_current_us_president_uses_official_usagov_source(self):
        from io import BytesIO

        page = (
            "<h2>Current president</h2>"
            "<p>The 47th and current president of the United States is "
            "Donald John Trump.</p>"
        ).encode("utf-8")
        with patch.object(online.urllib.request, "urlopen", return_value=BytesIO(page)):
            results = online._current_us_president_search()
        self.assertEqual(len(results), 1)
        self.assertIn("Donald John Trump", results[0]["snippet"])
        self.assertEqual(results[0]["url"], "https://www.usa.gov/presidents")
        answer = online.answer_from_web_results(
            "whos is the USA current President", results
        )
        self.assertIn("Donald John Trump", answer)

    def test_current_president_query_prefers_official_source(self):
        official = [{
            "title": "USAGov: Current president of the United States",
            "snippet": "The current president of the United States is Donald John Trump.",
            "url": "https://www.usa.gov/presidents",
            "source": "USAGov (official U.S. government)",
        }]
        with patch.object(online, "_current_us_president_search", return_value=official), patch.object(
            online, "_duckduckgo_search"
        ) as search:
            results = online.search_web("whos is the USA current President", 5)
        search.assert_not_called()
        self.assertEqual(results, official)

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

    def test_duckduckgo_html_parser_extracts_search_result(self):
        parser = online._DuckDuckGoParser()
        parser.feed('<a class="result__a" href="https://example.org">Example title</a><a class="result__snippet">Useful answer details.</a>')
        self.assertEqual(parser.items[0]["title"], "Example title")
        self.assertEqual(parser.items[0]["snippet"], "Useful answer details.")
        self.assertEqual(parser.items[0]["url"], "https://example.org")

    def test_search_uses_wikipedia_when_no_paid_provider_is_configured(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            online, "_duckduckgo_search", return_value=None
        ), patch.object(
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
