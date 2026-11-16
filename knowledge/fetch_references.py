"""Download the project's cited public-domain English reference books.

Run manually: python knowledge/fetch_references.py
Books are kept as local retrieval sources; runtime inference never needs network access.
"""
from pathlib import Path
from urllib.request import Request, urlopen
import re

ROOT = Path(__file__).resolve().parent
DESTINATION = ROOT / "references"
SOURCES = {
    "elements_of_style_37134.txt": "https://www.gutenberg.org/cache/epub/37134/pg37134.txt",
    "kings_english_75439.txt": "https://www.gutenberg.org/cache/epub/75439/pg75439.txt",
}
MAX_BYTES = 8 * 1024 * 1024


def clean_gutenberg(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    start = re.search(r"\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*{3}", text, re.I)
    end = re.search(r"\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*{3}", text, re.I)
    if start:
        text = text[start.end():]
    if end:
        text = text[:end.start()]
    return text.strip() + "\n"


def download():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    results = []
    for filename, url in SOURCES.items():
        request = Request(url, headers={"User-Agent": "Expert802ReferenceFetcher/1.0"})
        with urlopen(request, timeout=30) as response:
            data = response.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise ValueError(f"{filename} exceeds the {MAX_BYTES}-byte safety limit.")
        text = clean_gutenberg(data.decode("utf-8-sig"))
        if len(text) < 10000:
            raise ValueError(f"{filename} did not contain a complete readable book.")
        destination = DESTINATION / filename
        destination.write_text(text, encoding="utf-8")
        results.append({"file": str(destination), "characters": len(text)})
    return results


if __name__ == "__main__":
    for result in download():
        print(f"Saved {result['file']} ({result['characters']:,} characters)")
