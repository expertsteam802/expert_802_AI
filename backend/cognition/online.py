"""Bounded public web search adapter; language generation is self-hosted only.

All network destinations are fixed provider endpoints. Search snippets are evidence,
never executable instructions. Secrets are read only from server environment variables.
"""
import html
from html.parser import HTMLParser
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




class _DuckDuckGoParser(HTMLParser):
    """Extract search titles and snippets from DuckDuckGo's HTML results."""
    def __init__(self):
        super().__init__()
        self.items = []
        self.anchor = None
        self.pending = None
        self.snippet = None
        self.capture = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get("class", "").split()
        if tag == "a" and "result__a" in classes:
            self.anchor = {"title": "", "url": attrs.get("href", "")}
            self.capture = "title"
        elif "result__snippet" in classes:
            self.snippet = ""
            self.capture = "snippet"

    def handle_data(self, data):
        if self.capture == "title" and self.anchor is not None:
            self.anchor["title"] += data
        elif self.capture == "snippet" and self.snippet is not None:
            self.snippet += data

    def handle_endtag(self, tag):
        if tag == "a" and self.anchor is not None and self.capture == "title":
            self.pending = self.anchor
            self.anchor = None
            self.capture = None
        if self.capture == "snippet" and self.snippet is not None:
            if self.pending:
                href = self.pending.get("url", "")
                if href.startswith("//"):
                    href = "https:" + href
                parsed = urllib.parse.urlsplit(href)
                if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
                    href = urllib.parse.parse_qs(parsed.query).get("uddg", [href])[0]
                self.items.append({"title": self.pending.get("title", ""), "snippet": self.snippet, "url": href})
                self.pending = None
            self.snippet = None
            self.capture = None


def _duckduckgo_search(query, limit):
    """Key-free web search; the upstream may still apply fair-use throttling."""
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    request = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; Expert802AI/1.0)",
        "Accept": "text/html",
    })
    with urllib.request.urlopen(request, timeout=12) as response:
        page = response.read(2_000_000).decode("utf-8", "replace")
    parser = _DuckDuckGoParser()
    parser.feed(page)
    return _normalize_results(parser.items, "DuckDuckGo", limit)



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


def _current_us_president_search():
    """Read the current officeholder from the official USAGov presidents page."""
    url = "https://www.usa.gov/presidents"
    request = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; Expert802AI/1.0)",
        "Accept": "text/html",
    })
    with urllib.request.urlopen(request, timeout=12) as response:
        page = response.read(2_000_000).decode("utf-8", "replace")
    plain = _clean(page, 200000)
    match = re.search(
        r"Current president\s+(?:The\s+)?(?:\d+(?:st|nd|rd|th)\s+and\s+)?"
        r"current president of the United States is\s+([A-Z][A-Za-z .'-]+?)\.",
        plain,
        re.I,
    )
    if not match:
        return []
    name = match.group(1).strip()
    return [{
        "title": "USAGov: Current president of the United States",
        "snippet": "The current president of the United States is " + name +
                   ". USAGov identifies the officeholder on its official presidents page.",
        "url": url,
        "source": "USAGov (official U.S. government)",
    }]


def normalize_search_query(message):
    """Turn common conversational phrasing into a focused search query."""
    text = _clean(message, 300)
    lowered = text.lower()
    if (re.search(r"\b(president|head of state)\b", lowered)
            and any(term in lowered for term in ("usa", "u.s.", "united states", "america"))
            and re.search(r"\b(who|current|currently|today|now)\b", lowered)):
        return "current president of the United States"
    return text


def answer_from_web_results(message, results):
    """Return a concise, source-grounded excerpt only when a result is clearly relevant."""
    query = normalize_search_query(message)
    if query == "current president of the United States":
        for item in results or []:
            if "usa.gov/presidents" in str(item.get("url", "")):
                match = re.search(
                    r"current president of the United States is\s+([A-Z][A-Za-z .'-]+?)\.",
                    _clean(item.get("snippet"), 1200),
                    re.I,
                )
                if match:
                    return ("The current president of the United States is **" +
                            match.group(1).strip() +
                            "**. USAGov identifies this person as the current officeholder.")
    stop_words = {"who", "what", "when", "where", "which", "why", "how", "is", "are",
                  "the", "a", "an", "of", "in", "on", "for", "to", "and", "or",
                  "currently", "today", "now", "please", "tell", "me",
                  "about", "usa", "u.s", "united", "states"}
    terms = {word for word in re.findall(r"[a-z0-9]+", query.lower())
             if len(word) > 2 and word not in stop_words}
    if not terms:
        return None
    ranked = []
    for item in results or []:
        title = _clean(item.get("title"), 220)
        snippet = _clean(item.get("snippet"), 1200)
        title_words = set(re.findall(r"[a-z0-9]+", title.lower()))
        body_words = set(re.findall(r"[a-z0-9]+", snippet.lower()))
        title_hits = len(terms & title_words)
        body_hits = len(terms & body_words)
        matched_terms = terms & (title_words | body_words)
        if "current" in terms and "current" not in (title_words | body_words):
            continue
        if len(matched_terms) < min(2, len(terms)):
            continue
        score = (title_hits * 2 + body_hits) / max(1, len(terms) * 3)
        if snippet and score >= 0.22:
            ranked.append((score, item, snippet))
    if not ranked:
        return None
    ranked.sort(key=lambda row: row[0], reverse=True)
    _, item, snippet = ranked[0]
    source = _clean(item.get("title"), 180)
    if not snippet:
        return None
    return snippet + ((" (Source: " + source + ")") if source else "")


def search_web(query, limit=5):
    """Use key-free public search first, then configured providers and Wikipedia fallback."""
    query = normalize_search_query(query)
    if len(query) < 2:
        return []
    limit = max(1, min(int(limit), 6))
    if query == "current president of the United States":
        try:
            official = _current_us_president_search()
            if official:
                return official[:limit]
        except (OSError, ValueError, RuntimeError, urllib.error.URLError):
            pass
    errors = []
    for provider in (_duckduckgo_search, _wikipedia_search, _brave_search, _tavily_search, _bing_search):
        try:
            results = provider(query, limit)
            if results:
                return results
        except (OSError, ValueError, RuntimeError, urllib.error.URLError) as error:
            errors.append(str(error)[:180])
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
