"""Train a tiny word-level n-gram language model from local English text.

Run: python -m learning.train [path/to/corpus.txt]
The output is backend/language_model.json. This is a statistical language model
baseline, not a neural network or a general-purpose chatbot.
"""
import json, math, re, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "knowledge" / "english_corpus.txt"
OUTPUT = ROOT / "backend" / "language_model.json"

def sentences_from(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [re.findall(r"[a-z]+(?:'[a-z]+)?|[.!?]", s.lower())
            for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]

def train(paths):
    unigram = Counter()
    bigram = defaultdict(Counter)
    sentence_count = 0
    for path in paths:
        for words in sentences_from(path):
            words = [w for w in words if re.fullmatch(r"[a-z]+(?:'[a-z]+)?", w)]
            if not words:
                continue
            sentence_count += 1
            sequence = ["<s>"] + words + ["</s>"]
            unigram.update(words)
            for left, right in zip(sequence, sequence[1:]):
                bigram[left][right] += 1
    model = {
        "format": "expert802-word-bigram-v1",
        "language": "English",
        "sentences_seen": sentence_count,
        "vocabulary_size": len(unigram),
        "unigram": dict(unigram),
        "bigram": {left: dict(next_words) for left, next_words in bigram.items()},
    }
    OUTPUT.write_text(json.dumps(model, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return model

if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]] or [DEFAULT_CORPUS]
    paths = [p for p in paths if p.exists() and p.is_file()]
    if not paths:
        raise SystemExit("No readable .txt training corpus was supplied.")
    result = train(paths)
    print(f"Trained English word bigram model: {result['sentences_seen']} sentences, "
          f"{result['vocabulary_size']} words. Saved to {OUTPUT}")
