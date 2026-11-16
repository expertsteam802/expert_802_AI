"""Bounded public web search adapter; language generation is self-hosted only.

All network destinations are fixed provider endpoints. Search snippets are evidence,
never executable instructions. Secrets are read only from server environment variables.
"""
import html
from html.parser import HTMLParser
import json
import os
import re
import socket
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


class _PageTextParser(HTMLParser):
    """Extract readable page text while ignoring navigation, scripts and styling."""
    BLOCKED = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form"}
    BLOCKS = {"article", "main", "section", "p", "div", "li", "h1", "h2", "h3", "h4", "br"}

    def __init__(self):
        super().__init__()
        self.parts = []
        self.blocked_depth = 0
        self.title_parts = []
        self.in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.BLOCKED:
            self.blocked_depth += 1
        if tag == "title":
            self.in_title = True
        if not self.blocked_depth and tag in self.BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.BLOCKED and self.blocked_depth:
            self.blocked_depth -= 1
        if tag == "title":
            self.in_title = False
        if not self.blocked_depth and tag in self.BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.in_title:
            self.title_parts.append(data)
        if not self.blocked_depth:
            cleaned = re.sub(r"\s+", " ", data).strip()
            if cleaned:
                self.parts.append(cleaned + " ")


def _public_http_url(value):
    """Reject non-web schemes and obvious local/private destinations before fetching."""
    from ipaddress import ip_address
    parsed = urllib.parse.urlsplit(str(value or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    host = parsed.hostname.lower().rstrip(".")
    if host in {"localhost", "localhost.localdomain"} or host.endswith((".localhost", ".local", ".internal")):
        return False
    try:
        address = ip_address(host)
        if not address.is_global:
            return False
    except ValueError:
        if not re.fullmatch(r"[a-z0-9.-]+", host) or "." not in host:
            return False
    return True


def _resolved_public_url(value):
    """Resolve a destination and reject private, loopback, link-local and reserved IPs."""
    from ipaddress import ip_address
    parsed = urllib.parse.urlsplit(str(value or "").strip())
    if not _public_http_url(value):
        return False
    try:
        addresses = socket.getaddrinfo(
            parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except (OSError, ValueError):
        return False
    if not addresses:
        return False
    for address in addresses:
        try:
            if not ip_address(address[4][0]).is_global:
                return False
        except ValueError:
            return False
    return True


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Validate every redirect destination, not just the original search URL."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _resolved_public_url(newurl):
            raise urllib.error.URLError("Blocked redirect to a non-public destination")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _read_webpage(url, timeout=10, max_bytes=900000):
    """Fetch and extract a bounded public HTML/text page; page content is untrusted data."""
    if not _resolved_public_url(url):
        return None
    request = urllib.request.Request(str(url), headers={
        "User-Agent": "Mozilla/5.0 (compatible; Expert802AI/1.0; +https://example.org/bot)",
        "Accept": "text/html,text/plain;q=0.9,application/xhtml+xml;q=0.8",
    })
    opener = urllib.request.build_opener(_SafeRedirectHandler())
    with opener.open(request, timeout=timeout) as response:
        final_url = response.geturl()
        if not _public_http_url(final_url):
            return None
        content_type = response.headers.get("Content-Type", "").lower()
        if content_type and not any(kind in content_type for kind in ("text/html", "text/plain", "application/xhtml+xml")):
            return None
        raw = response.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raw = raw[:max_bytes]
    page = raw.decode("utf-8", "replace")
    parser = _PageTextParser()
    parser.feed(page)
    title = _clean(" ".join(parser.title_parts), 220)
    text = _clean(" ".join(parser.parts), 9000)
    if len(text) < 120:
        return None
    return {"title": title, "text": text, "url": final_url}


def _expand_search_results(results, max_pages=3):
    """Read top search-result pages and attach bounded page text to their search metadata."""
    expanded = []
    for item in (results or [])[:max_pages]:
        record = dict(item)
        try:
            page = _read_webpage(record.get("url", ""))
        except (OSError, ValueError, RuntimeError, urllib.error.URLError):
            page = None
        if page:
            record["page_title"] = page["title"] or record.get("title", "")
            record["page_text"] = page["text"]
            record["url"] = page["url"]
            record["content_type"] = "fetched_page"
        else:
            record["content_type"] = "search_excerpt"
        expanded.append(record)
    return expanded


def normalize_search_query(message):
    """Prepare user language for retrieval without mapping questions to canned queries."""
    return _clean(message, 300)


def answer_from_web_results(message, results):
    """Select relevant source passages; generation/reasoning belongs to the private model."""
    query = normalize_search_query(message)
    stop_words = {
        "who", "what", "when", "where", "which", "why", "how", "is", "are", "was", "were",
        "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "please", "tell",
        "me", "about", "can", "could", "would", "should", "do", "does", "did", "with",
        "from", "that", "this", "it", "they", "them", "current", "currently", "today", "now",
        "whos", "usa", "united", "states", "u", "s",
    }
    terms = {word for word in re.findall(r"[a-z0-9]+", query.lower())
             if len(word) > 2 and word not in stop_words}
    if not terms:
        return None
    ranked = []
    for item in results or []:
        title = _clean(item.get("page_title") or item.get("title"), 220)
        body = _clean(item.get("page_text") or item.get("snippet"), 9000)
        if not body:
            continue
        title_words = set(re.findall(r"[a-z0-9]+", title.lower()))
        body_words = set(re.findall(r"[a-z0-9]+", body.lower()))
        matched = terms & (title_words | body_words)
        if not matched:
            continue
        title_hits = len(terms & title_words)
        body_hits = len(terms & body_words)
        coverage = len(matched) / max(1, len(terms))
        score = (title_hits * 2 + body_hits) / max(1, len(terms) * 3)
        if coverage >= 0.35 and score >= 0.12:
            ranked.append((score, item, body))
    if not ranked:
        return None
    ranked.sort(key=lambda row: row[0], reverse=True)
    passages = []
    for score, item, body in ranked[:3]:
        title = _clean(item.get("page_title") or item.get("title"), 220)
        title_words = set(re.findall(r"[a-z0-9]+", title.lower()))
        sentences = re.split(r"(?<=[.!?])\s+|[\r\n]+", body)
        sentence_scores = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 35:
                continue
            sentence_words = set(re.findall(r"[a-z0-9]+", sentence.lower()))
            hits = len(terms & sentence_words)
            title_hits = len(terms & title_words)
            sentence_score = (hits * 2 + title_hits) / max(1, len(terms) * 3)
            if hits or title_hits:
                sentence_scores.append((sentence_score, sentence))
        sentence_scores.sort(key=lambda row: row[0], reverse=True)
        selected = []
        seen_sentences = set()
        for _, sentence in sentence_scores:
            normalized = sentence.lower()
            if normalized in seen_sentences:
                continue
            seen_sentences.add(normalized)
            selected.append(sentence)
            if len(selected) == 5:
                break
        excerpt = " ".join(selected) if selected else body[:1400]
        passages.append({
            "source": item.get("source", "web"),
            "title": title,
            "url": item.get("url", ""),
            "score": round(score, 3),
            "text": excerpt[:5000],
        })
    return passages


def search_web(query, limit=5):
    """Search broadly, then read result pages; no question-specific routes or answer templates."""
    query = normalize_search_query(query)
    if len(query) < 2:
        return []
    limit = max(1, min(int(limit), 6))
    for provider in (_duckduckgo_search, _wikipedia_search, _brave_search, _tavily_search, _bing_search):
        try:
            results = provider(query, limit)
            if results:
                return _expand_search_results(results, max_pages=min(3, limit))
        except (OSError, ValueError, RuntimeError, urllib.error.URLError):
            continue
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
