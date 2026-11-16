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

    def test_query_normalization_does_not_hardcode_a_question(self):
        question = "whos is the USA current President"
        self.assertEqual(online.normalize_search_query(question), question)
        self.assertTrue(online.should_search_web(question))

    def test_page_parser_extracts_readable_content_and_ignores_scripts(self):
        parser = online._PageTextParser()
        parser.feed(
            "<html><title>Example source</title><nav>Menu</nav>"
            "<main><h1>Current information</h1><p>The answer is in this article.</p>"
            "<script>ignore this injected text</script></main></html>"
        )
        text = " ".join(parser.parts)
        self.assertIn("The answer is in this article", text)
        self.assertNotIn("ignore this injected text", text)
        self.assertEqual(" ".join(parser.title_parts), "Example source")

    def test_private_and_non_http_urls_are_rejected(self):
        for url in (
            "http://127.0.0.1/admin", "http://localhost/", "http://192.168.1.1/",
            "file:///etc/passwd", "javascript:alert(1)", "http://service.internal/",
        ):
            self.assertFalse(online._public_http_url(url))
        self.assertTrue(online._public_http_url("https://example.com/article"))

    def test_relevant_evidence_returns_page_passages_not_hardcoded_answers(self):
        results = [
            {"title": "Telephone history", "page_title": "Who invented the telephone?",
             "page_text": "Alexander Graham Bell is widely credited with inventing the telephone. "
                          "Other inventors also contributed to early telephone technology.",
             "url": "https://example.org/history", "source": "test"},
            {"title": "Unrelated", "page_title": "Gardening",
             "page_text": "Tomatoes grow well in warm soil and need regular watering.",
             "url": "https://example.org/garden", "source": "test"},
        ]
        evidence = online.answer_from_web_results("Who invented the telephone?", results)
        self.assertTrue(evidence)
        self.assertIn("Alexander Graham Bell", evidence[0]["text"])
        self.assertEqual(evidence[0]["url"], "https://example.org/history")

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
        search_results = [{
            "title": "Example", "snippet": "Evidence", "url": "https://en.wikipedia.org/?curid=1",
            "source": "Wikipedia",
        }]
        with patch.dict(os.environ, {}, clear=True), patch.object(
            online, "_duckduckgo_search", return_value=None
        ), patch.object(
            online, "_wikipedia_search", return_value=search_results
        ) as fallback, patch.object(
            online, "_expand_search_results", side_effect=lambda items, max_pages: items
        ):
            results = online.search_web("example", 3)
        fallback.assert_called_once_with("example", 3)
        self.assertEqual(results[0]["source"], "Wikipedia")


if __name__ == "__main__":
    unittest.main()
