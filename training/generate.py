"""Generate text from a trained Expert802GPT checkpoint."""
import argparse
import torch
from model import Expert802GPT, ModelConfig\nfrom tokenizer import load_tokenizer

def load_model(checkpoint_path, device):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ModelConfig(**payload["config"])
    model = Expert802GPT(config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="training/checkpoints/best.pt")\n    parser.add_argument("--tokenizer", default="training/checkpoints/tokenizer.json")
    parser.add_argument("--prompt", default="USER: Hello.\nASSISTANT:")
    parser.add_argument("--tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model, payload = load_model(args.checkpoint, device)
    tokenizer = load_tokenizer(args.tokenizer)\n    context = torch.tensor([tokenizer.encode(args.prompt).ids], dtype=torch.long, device=device)
    result = model.generate(context, max_new_tokens=args.tokens, temperature=args.temperature)
    print(result[0].tolist()[len(context[0]):])
    print(tokenizer.decode(result[0].tolist(), skip_special_tokens=False))

if __name__ == "__main__":
    main()
