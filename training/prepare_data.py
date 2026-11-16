"""Prepare a locally tokenized corpus with source-level train/validation isolation."""
import argparse
import json
import random
import tempfile
from pathlib import Path
import numpy as np
from tokenizer import train_tokenizer, encode_text

TEXT_SUFFIXES = {".txt", ".md", ".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".sql", ".json", ".jsonl"}

def read_source(path):
    if path.suffix.lower() == ".jsonl":
        conversations = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            messages = item.get("messages", []) if isinstance(item, dict) else []
            turns = []
            for message in messages if isinstance(messages, list) else []:
                if isinstance(message, dict) and message.get("role") in {"system", "user", "assistant"}:
                    body = message.get("content", "")
                    if isinstance(body, str) and body.strip():
                        turns.append("<|" + message["role"] + "|> " + body.strip())
            if turns:
                conversations.append("\n".join(turns))
        return "\n\n".join(conversations)
    if path.suffix.lower() in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="ignore")
    return ""

def split_documents(text):
    pieces = [part.strip() for part in text.split("\n\n") if len(part.strip()) > 80]
    if len(pieces) > 1:
        return pieces
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 40]
    return lines if len(lines) > 1 else ([text.strip()] if text.strip() else [])

def prepare(sources, output_dir, validation_fraction=0.02, vocab_size=16000, seed=802):
    source_groups = []
    for source in sources:
        path = Path(source)
        candidates = [path] if path.is_file() else (
            [item for item in path.rglob("*") if item.is_file() and not item.is_symlink()
             and item.suffix.lower() in TEXT_SUFFIXES] if path.is_dir() else [])
        for candidate in candidates:
            if candidate.stat().st_size > 20_000_000:
                continue
            docs = split_documents(read_source(candidate))
            if docs:
                source_groups.append((candidate.as_posix(), docs))
    if len(source_groups) < 2:
        raise ValueError("At least two independent source files are required to prevent train/validation leakage.")
    random.Random(seed).shuffle(source_groups)
    validation_count = max(1, int(len(source_groups) * validation_fraction))
    validation_groups = source_groups[:validation_count]
    training_groups = source_groups[validation_count:]
    training_docs = [doc for _, docs in training_groups for doc in docs]
    validation_docs = [doc for _, docs in validation_groups for doc in docs]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        corpus_path = Path(handle.name)
        handle.write("\n\n".join(training_docs))
    try:
        tokenizer = train_tokenizer([corpus_path], output / "tokenizer.json", vocab_size=vocab_size)
    finally:
        corpus_path.unlink(missing_ok=True)
    separator = tokenizer.token_to_id("<|endoftext|>")
    def tokenize(documents):
        ids = []
        for document in documents:
            ids.extend(encode_text(tokenizer, document))
            ids.append(separator)
        return ids
    train_ids, validation_ids = tokenize(training_docs), tokenize(validation_docs)
    if len(train_ids) <= 1025 or len(validation_ids) <= 1025:
        raise ValueError("Both source-isolated splits must exceed 1025 tokens for the default 1,024-token context.")
    dtype = np.uint16 if tokenizer.get_vocab_size() <= 65535 else np.uint32
    np.asarray(train_ids, dtype=dtype).tofile(output / "train.bin")
    np.asarray(validation_ids, dtype=dtype).tofile(output / "val.bin")
    metadata = {"tokenizer": "byte-level-bpe-v1", "vocab_size": tokenizer.get_vocab_size(),
                "token_dtype": str(np.dtype(dtype)), "train_tokens": len(train_ids),
                "validation_tokens": len(validation_ids), "training_documents": len(training_docs),
                "validation_documents": len(validation_docs), "training_source_files": len(training_groups),
                "validation_source_files": len(validation_groups), "split_unit": "source_file"}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="+", help="Explicit reviewed text/code corpus folders or files.")
    parser.add_argument("--output", default="training/data")
    parser.add_argument("--vocab-size", type=int, default=16000)
    args = parser.parse_args()
    print(json.dumps(prepare(args.sources, args.output, vocab_size=args.vocab_size), indent=2))

if __name__ == "__main__":
    main()
