"""Train Expert802GPT with optional assistant-only supervised loss masks."""
import argparse
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from model import Expert802GPT, ModelConfig


def load_tokens(path, dtype):
    return np.memmap(Path(path), dtype=np.dtype(dtype), mode="r")


def get_batch(data, batch_size, block_size, device, mask_data=None):
    if len(data) <= block_size + 1:
        raise ValueError("Token file is shorter than the configured context.")
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(np.array(data[i:i + block_size], dtype=np.int64)) for i in starts])
    y = torch.stack([torch.from_numpy(np.array(data[i + 1:i + block_size + 1], dtype=np.int64)) for i in starts])
    x, y = x.to(device), y.to(device)
    if mask_data is None:
        return x, y, None
    weights = torch.stack([
        torch.from_numpy(np.array(mask_data[i + 1:i + block_size + 1], dtype=np.float32))
        for i in starts
    ]).to(device)
    if not torch.any(weights):
        for _ in range(31):
            starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
            x = torch.stack([torch.from_numpy(np.array(data[i:i + block_size], dtype=np.int64)) for i in starts])
            y = torch.stack([torch.from_numpy(np.array(data[i + 1:i + block_size + 1], dtype=np.int64)) for i in starts])
            weights = torch.stack([
                torch.from_numpy(np.array(mask_data[i + 1:i + block_size + 1], dtype=np.float32))
                for i in starts
            ])
            if torch.any(weights):
                return x.to(device), y.to(device), weights.to(device)
        raise ValueError("Could not sample assistant targets. Increase the number and density of reviewed assistant responses.")
    return x, y, weights


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, device,
                  train_mask=None, val_mask=None, batches=20):
    model.eval()
    results = {}
    for name, data, mask in (("train", train_data, train_mask), ("validation", val_data, val_mask)):
        losses = []
        for _ in range(batches):
            x, y, weights = get_batch(data, batch_size, block_size, device, mask)
            _, loss = model(x, y, weights)
            losses.append(float(loss.item()))
        results[name] = sum(losses) / len(losses)
    model.train()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="training/data")
    parser.add_argument("--output", default="training/checkpoints")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()
    torch.manual_seed(802)
    random.seed(802)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    data_dir = Path(args.data)
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(args.resume, map_location=device, weights_only=False) if args.resume else None
    if checkpoint:
        config = ModelConfig(**checkpoint["config"])
        if config.vocab_size != int(metadata["vocab_size"]):
            raise ValueError("The fine-tuning tokenizer vocabulary must match the base checkpoint.")
    else:
        config = ModelConfig(vocab_size=int(metadata["vocab_size"]), block_size=args.block_size,
                            n_layer=args.layers, n_head=args.heads, n_embd=args.width)
    model = Expert802GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    start_step = 0
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint.get("step", 0))
    train_data = load_tokens(data_dir / "train.bin", metadata["token_dtype"])
    val_data = load_tokens(data_dir / "val.bin", metadata["token_dtype"])
    train_mask_path, val_mask_path = data_dir / "train_mask.bin", data_dir / "val_mask.bin"
    train_mask = np.memmap(train_mask_path, dtype=np.uint8, mode="r") if train_mask_path.exists() else None
    val_mask = np.memmap(val_mask_path, dtype=np.uint8, mode="r") if val_mask_path.exists() else None
    if (train_mask is None) != (val_mask is None):
        raise ValueError("Training and validation loss masks must either both exist or both be absent.")
    if train_mask is not None and (len(train_mask) != len(train_data) or len(val_mask) != len(val_data)):
        raise ValueError("Loss masks must have exactly one entry per token.")
    if len(train_data) <= args.block_size + 1 or len(val_data) <= args.block_size + 1:
        raise ValueError("Both train and validation splits must exceed block_size + 1 tokens.")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(data_dir / "tokenizer.json", output / "tokenizer.json")
    print(json.dumps({"device": device, "parameters": model.parameter_count(), "config": model.config_dict(),
                      "train_tokens": len(train_data), "validation_tokens": len(val_data),
                      "objective": metadata.get("objective", "all_token_next_token_prediction")}))
    best_validation = float("inf")
    best_checkpoint_path = output / "best.pt"
    if best_checkpoint_path.exists():
        previous_best = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
        best_validation = float(previous_best.get("metrics", {}).get("validation", float("inf")))
    for step in range(start_step, args.steps):
        progress = step / max(1, args.steps)
        learning_rate = args.learning_rate * 0.1 + 0.9 * args.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        x, y, weights = get_batch(train_data, args.batch_size, args.block_size, device, train_mask)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y, weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.eval_interval == 0 or step == args.steps - 1:
            metrics = estimate_loss(model, train_data, val_data, args.batch_size, args.block_size, device,
                                     train_mask, val_mask)
            print(json.dumps({"step": step, "loss": float(loss.item()), **metrics}))
            payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step + 1,
                       "config": model.config_dict(), "metrics": metrics,
                       "objective": metadata.get("objective", "all_token_next_token_prediction"),
                       "tokenizer": "byte-level-bpe-v1"}
            torch.save(payload, output / "latest.pt")
            if metrics["validation"] < best_validation:
                best_validation = metrics["validation"]
                torch.save(payload, output / "best.pt")
    print("Training complete. Checkpoints are stored outside the web application.")


if __name__ == "__main__":
    main()
