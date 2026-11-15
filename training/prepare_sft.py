"""Build a held-out-by-conversation byte dataset for supervised chat fine-tuning."""
import argparse
import json
import random
from pathlib import Path
import numpy as np

def prepare(input_path, output_dir, validation_fraction=0.05, seed=802):
    randomizer = random.Random(seed)
    training, validation = [], []
    source = Path(input_path)
    for line in source.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        messages = record.get("messages", []) if isinstance(record, dict) else []
        turns = []
        for message in messages if isinstance(messages, list) else []:
            if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}:
                continue
            content = message.get("content", "")
            if isinstance(content, str) and content.strip():
                turns.append(message["role"].upper() + ": " + content.strip())
        if not turns:
            continue
        conversation = ("\n".join(turns) + "\n\n").encode("utf-8")
        split = record.get("split") if isinstance(record, dict) else None
        if split == "validation" or (split not in {"train", "validation"} and randomizer.random() < validation_fraction):
            validation.append(conversation)
        else:
            training.append(conversation)
    if not training or not validation:
        raise ValueError("Provide enough conversations for both train and validation splits.")
    if sum(map(len, training)) <= 513 or sum(map(len, validation)) <= 513:
        raise ValueError("Both conversation splits must exceed 513 bytes for the default context.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_bytes, val_bytes = b"".join(training), b"".join(validation)
    np.frombuffer(train_bytes, dtype=np.uint8).tofile(output / "train.bin")
    np.frombuffer(val_bytes, dtype=np.uint8).tofile(output / "val.bin")
    metadata = {"tokenizer":"utf8-byte-v1","vocab_size":256,"train_conversations":len(training),
                "validation_conversations":len(validation),"train_bytes":len(train_bytes),"validation_bytes":len(val_bytes)}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--output", default="training/sft_data")
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset, args.output), indent=2))
