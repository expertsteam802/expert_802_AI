"""Prepare assistant-only supervised targets with conversation-level data isolation."""
import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
from tokenizer import load_tokenizer


def encode_conversation(messages, tokenizer):
    """Return token IDs and a 1/0 mask; only assistant response tokens incur loss."""
    ids, loss_mask = [], []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}:
            continue
        body = message.get("content", "")
        if not isinstance(body, str) or not body.strip():
            continue
        role = message["role"]
        prefix_ids = tokenizer.encode("<|" + role + "|> ").ids
        body_ids = tokenizer.encode(body.strip()).ids
        ids.extend(prefix_ids)
        loss_mask.extend([0] * len(prefix_ids))
        ids.extend(body_ids)
        loss_mask.extend(([1] if role == "assistant" else [0]) * len(body_ids))
        ids.extend(tokenizer.encode("\n").ids)
        loss_mask.extend([0] * len(tokenizer.encode("\n").ids))
    if ids and messages and isinstance(messages[-1], dict) and messages[-1].get("role") == "assistant":
        end_id = tokenizer.token_to_id("<|endoftext|>")
        ids.append(end_id)
        loss_mask.append(1)
    return ids, loss_mask


def prepare(input_path, output_dir, tokenizer_path, validation_fraction=0.05, seed=802):
    randomizer = random.Random(seed)
    training_ids, training_mask, validation_ids, validation_mask = [], [], [], []
    tokenizer = load_tokenizer(tokenizer_path)
    source = Path(input_path)
    for line in source.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        messages = record.get("messages", []) if isinstance(record, dict) else []
        ids, mask = encode_conversation(messages, tokenizer)
        if not ids or sum(mask) == 0:
            continue
        split = record.get("split") if isinstance(record, dict) else None
        is_validation = split == "validation" or (
            split not in {"train", "validation"} and randomizer.random() < validation_fraction
        )
        if is_validation:
            validation_ids.extend(ids)
            validation_mask.extend(mask)
        else:
            training_ids.extend(ids)
            training_mask.extend(mask)
    if len(training_ids) <= 1025 or len(validation_ids) <= 1025:
        raise ValueError("Supply enough reviewed conversations for both splits (more than 1025 tokens each).")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tokenizer_path, output / "tokenizer.json")
    dtype = np.uint16 if tokenizer.get_vocab_size() <= 65535 else np.uint32
    np.asarray(training_ids, dtype=dtype).tofile(output / "train.bin")
    np.asarray(validation_ids, dtype=dtype).tofile(output / "val.bin")
    np.asarray(training_mask, dtype=np.uint8).tofile(output / "train_mask.bin")
    np.asarray(validation_mask, dtype=np.uint8).tofile(output / "val_mask.bin")
    metadata = {
        "tokenizer": "byte-level-bpe-v1",
        "vocab_size": tokenizer.get_vocab_size(),
        "token_dtype": str(np.dtype(dtype)),
        "train_tokens": len(training_ids),
        "validation_tokens": len(validation_ids),
        "train_supervised_tokens": int(sum(training_mask)),
        "validation_supervised_tokens": int(sum(validation_mask)),
        "objective": "assistant_only_next_token_prediction",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--tokenizer", default="training/data/tokenizer.json")
    parser.add_argument("--output", default="training/sft_data")
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset, args.output, args.tokenizer), indent=2))
