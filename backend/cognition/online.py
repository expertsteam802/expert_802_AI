"""Bounded public web search adapter; language generation is self-hosted only.

All network destinations are fixed provider endpoints. Search snippets are evidence,
never executable instructions. Secrets are read only from server environment variables.
"""
import html
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request


def _clean(value, limit=1200):
    value = html.unescape(re.sub(r"<[^>]+>", " ", str(value or "")))
    return re.sub(r"\s+", " ", value).strip()[:limit]


def _json_request(url, headers=None, data=None, timeout=12):
    request = urllib.request.Request(url, data=data, headers=headers or {}, method="POST" if data else "GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError("Online provider returned HTTP " + str(response.status))
        return json.loads(response.read(2_000_000).decode("utf-8"))


def _normalize_results(items, provider, limit):
    results = []
    for item in items[:limit]:
        title = _clean(item.get("title"), 220)
        snippet = _clean(item.get("snippet") or item.get("description") or item.get("content"), 1200)
        url = str(item.get("url") or item.get("link") or "").strip()
        if not title or not snippet or not url.startswith(("https://", "http://")):
            continue
        results.append({"title": title, "snippet": snippet, "url": url, "source": provider})
    return results


def _google_search(query, limit):
    """Google Programmable Search JSON API. Requires a Google API key and Search Engine ID."""
    key = os.environ.get("GOOGLE_CSE_API_KEY", "").strip()
    engine_id = os.environ.get("GOOGLE_CSE_ID", "").strip()
    if not key or not engine_id:
        return None
    params = urllib.parse.urlencode({"key": key, "cx": engine_id, "q": query, "num": min(limit, 10), "safe": "active"})
    data = _json_request("https://www.googleapis.com/customsearch/v1?" + params, {"Accept": "application/json"})
    return _normalize_results(data.get("items", []), "Google Search", limit)


def _brave_search(query, limit):
    key = os.environ.get("BRAVE_SEARCH_API_KEY", "").strip()
    if not key:
        return None
    params = urllib.parse.urlencode({"q": query, "count": limit, "safesearch": "moderate"})
    data = _json_request("https://api.search.brave.com/res/v1/web/search?" + params,
                         {"Accept": "application/json", "X-Subscription-Token": key})
    return _normalize_results(data.get("web", {}).get("results", []), "Brave Search", limit)


def _tavily_search(query, limit):
    key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not key:
        return None
    body = json.dumps({"api_key": key, "query": query, "search_depth": "basic",
                       "max_results": limit, "include_answer": False,
                       "include_raw_content": False}).encode("utf-8")
    data = _json_request("https://api.tavily.com/search",
                         {"Content-Type": "application/json", "Accept": "application/json"}, body)
    return _normalize_results(data.get("results", []), "Tavily", limit)


def _bing_search(query, limit):
    key = os.environ.get("BING_SEARCH_V7_SUBSCRIPTION_KEY", "").strip()
    if not key:
        return None
    params = urllib.parse.urlencode({"q": query, "count": limit, "mkt": "en-US", "safeSearch": "Moderate"})
    data = _json_request("https://api.bing.microsoft.com/v7.0/search?" + params,
                         {"Ocp-Apim-Subscription-Key": key, "Accept": "application/json"})
    return _normalize_results(data.get("webPages", {}).get("value", []), "Bing", limit)


def _wikipedia_search(query, limit):
    params = urllib.parse.urlencode({
        "action": "query", "list": "search", "srsearch": query,
        "format": "json", "utf8": "1", "srlimit": limit,
    })
    data = _json_request("https://en.wikipedia.org/w/api.php?" + params,
                         {"User-Agent": "Expert802AI/1.0 (knowledge retrieval)"})
    items = []
    for item in data.get("query", {}).get("search", []):
        page_id = item.get("pageid")
        items.append({
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "url": "https://en.wikipedia.org/?curid=" + str(page_id) if page_id else "",
        })
    return _normalize_results(items, "Wikipedia", limit)


def search_web(query, limit=5):
    """Use Google Programmable Search first, then configured web search providers; otherwise use Wikipedia."""
    query = _clean(query, 300)
    if len(query) < 2:
        return []
    limit = max(1, min(int(limit), 6))
    errors = []
    for provider in (_google_search, _brave_search, _tavily_search, _bing_search):
        try:
            results = provider(query, limit)
            if results:
                return results
        except (OSError, ValueError, RuntimeError, urllib.error.URLError) as error:
            errors.append(str(error)[:180])
    try:
        return _wikipedia_search(query, limit)
    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as error:
        logging_error = str(error)[:180]
        return []


_FRESH_RE = re.compile(
    r"\b(latest|current|currently|today|tonight|this week|this month|recent|newest|"
    r"2026|2025|price|pricing|release|released|version|update|news|forecast|"
    r"schedule|score|standings|available now|right now|as of)\b", re.I
)
_EXPLICIT_RE = re.compile(r"\b(search (?:the )?(?:web|internet)|internet search|search online|look (?:it )?up online|find online|browse the internet|check (?:the )?web|check online|verify online)\b", re.I)
_FACTUAL_RE = re.compile(r"^\s*(?:what|who|when|where|which|why|how)\b", re.I)
_RESEARCH_RE = re.compile(r"^\s*(?:(?:can|could) you |please )?(?:explain|describe|compare|define|summari[sz]e|tell me about|give me an overview of)\b", re.I)
_PERSONAL_RE = re.compile(r"^\s*(?:who are you|what is your name|how are you|what can you do|what do you know about me|what do you remember about me)\b", re.I)
_CODING_RE = re.compile(r"\b(api|sdk|package|library|framework|documentation|docs|"
                        r"error|exception|version|install|npm|pip|python|javascript|"
                        r"typescript|react|vercel|supabase|github|security advisory)\b", re.I)


def should_search_web(message):
    text = str(message or "").strip()
    if len(text) < 5 or len(text) > 4000:
        return False
    if re.fullmatch(r"(?is)(hi|hello|hey|thanks|thank you|good morning|good night|who are you|what is your name)[.! ]*", text):
        return False
    if _EXPLICIT_RE.search(text) or _FRESH_RE.search(text):
        return True
    if (_FACTUAL_RE.search(text) or _RESEARCH_RE.search(text)) and not _PERSONAL_RE.search(text):
        return True
    if _CODING_RE.search(text) and re.search(r"\b(why|how|fix|error|issue|install|configure|latest|version|docs?)\b", text, re.I):
        return True
    return False
