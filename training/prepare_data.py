"""Prepare byte-tokenized text and chat data for training; inputs are explicitly supplied."""
import argparse
import json
from pathlib import Path
import numpy as np

TEXT_SUFFIXES = {".txt", ".md", ".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".sql", ".json"}

def read_source(path):
    if path.suffix.lower() == ".jsonl":
        conversations = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            messages = item.get("messages", []) if isinstance(item, dict) else []
            if not isinstance(messages, list):
                continue
            turns = []
            for message in messages:
                if isinstance(message, dict) and message.get("role") in {"system", "user", "assistant"}:
                    body = message.get("content", "")
                    if isinstance(body, str) and body.strip():
                        turns.append(message["role"].upper() + ": " + body.strip())
            if turns:
                conversations.append("\n".join(turns))
        return "\n\n".join(conversations)
    if path.suffix.lower() in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="ignore")
    return ""

def prepare(paths, output_dir, validation_fraction=0.02):
    chunks = []
    for source in paths:
        path = Path(source)
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = [item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in TEXT_SUFFIXES | {".jsonl"}]
        else:
            continue
        for candidate in candidates:
            if candidate.stat().st_size > 20_000_000:
                continue
            text = read_source(candidate).strip()
            if text:
                chunks.append(text.encode("utf-8"))
    if not chunks:
        raise ValueError("No supported text or conversation data was found.")
    raw = b"\n\n".join(chunks)
    if len(raw) < 2048:
        raise ValueError("Training data is too small. Supply a larger licensed corpus.")
    validation_size = max(513, int(len(raw) * validation_fraction))
    split = len(raw) - validation_size
    if split <= 513:
        raise ValueError("Supply more data so both splits exceed the model context length.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.frombuffer(raw[:split], dtype=np.uint8).tofile(output / "train.bin")
    np.frombuffer(raw[split:], dtype=np.uint8).tofile(output / "val.bin")
    metadata = {"tokenizer": "utf8-byte-v1", "vocab_size": 256, "bytes": len(raw),
                "train_bytes": split, "validation_bytes": len(raw)-split,
                "source_count": len(chunks), "sources": [str(path) for path in paths]}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="+", help="Explicit text/code folders or files, including JSONL conversations.")
    parser.add_argument("--output", default="training/data")
    args = parser.parse_args()
    result = prepare(args.sources, args.output)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
