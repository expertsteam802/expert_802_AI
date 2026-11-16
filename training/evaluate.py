"""Evaluate a trained Expert 802 checkpoint on held-out tokens and fixed prompts."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from generate import load_model
from tokenizer import load_tokenizer


@torch.no_grad()
def evaluate(checkpoint_path, data_dir, tokenizer_path, batch_size=4, batches=50, prompts=None):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model, payload = load_model(checkpoint_path, device)
    metadata = json.loads((Path(data_dir) / "metadata.json").read_text(encoding="utf-8"))
    validation = np.memmap(
        Path(data_dir) / "val.bin",
        dtype=np.dtype(metadata["token_dtype"]),
        mode="r",
    )
    if len(validation) <= model.config.block_size + 1:
        raise ValueError("Validation split is shorter than the model context.")
    losses = []
    model.eval()
    for _ in range(max(1, batches)):
        starts = torch.randint(0, len(validation) - model.config.block_size - 1, (batch_size,))
        x = torch.stack([
            torch.from_numpy(np.array(validation[i:i + model.config.block_size], dtype=np.int64))
            for i in starts
        ]).to(device)
        y = torch.stack([
            torch.from_numpy(np.array(validation[i + 1:i + model.config.block_size + 1], dtype=np.int64))
            for i in starts
        ]).to(device)
        _, loss = model(x, y)
        losses.append(float(loss.item()))

    tokenizer = load_tokenizer(tokenizer_path)
    sample_prompts = prompts or [
        "<|user|> Hello. Please introduce yourself.\n<|assistant|>",
        "<|user|> Explain what a Python function is.\n<|assistant|>",
        "<|user|> Write a JavaScript function that adds two numbers.\n<|assistant|>",
        "<|user|> I made a mistake in my code. How should I debug it?\n<|assistant|>",
    ]
    samples = []
    for prompt in sample_prompts:
        ids = tokenizer.encode(prompt).ids[-model.config.block_size:]
        if not ids:
            continue
        context = torch.tensor([ids], dtype=torch.long, device=device)
        generated = model.generate(
            context,
            max_new_tokens=96,
            temperature=0.7,
            top_k=40,
            stop_token_id=tokenizer.token_to_id("<|endoftext|>"),
        )
        completion = tokenizer.decode(generated[0].tolist()[len(ids):], skip_special_tokens=True)
        samples.append({"prompt": prompt, "completion": completion})

    mean_loss = sum(losses) / len(losses)
    return {
        "checkpoint": str(checkpoint_path),
        "step": int(payload.get("step", 0)),
        "device": device,
        "parameters": model.parameter_count(),
        "validation_loss": mean_loss,
        "validation_perplexity": math.exp(min(mean_loss, 20)),
        "validation_batches": len(losses),
        "samples": samples,
        "warning": "Loss and perplexity measure next-token prediction, not conversational helpfulness or coding correctness. Review generated samples and run task-specific tests before deployment.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="training/checkpoints/best.pt")
    parser.add_argument("--data", default="training/data")
    parser.add_argument("--tokenizer", default="training/checkpoints/tokenizer.json")
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.data, args.tokenizer, args.batch_size, args.batches)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
