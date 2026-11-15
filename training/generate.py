"""Generate text from a trained Expert802GPT checkpoint."""
import argparse
import torch
from model import Expert802GPT, ModelConfig

def load_model(checkpoint_path, device):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ModelConfig(**payload["config"])
    model = Expert802GPT(config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="training/checkpoints/best.pt")
    parser.add_argument("--prompt", default="USER: Hello.\nASSISTANT:")
    parser.add_argument("--tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model, payload = load_model(args.checkpoint, device)
    context = torch.tensor([list(args.prompt.encode("utf-8"))], dtype=torch.long, device=device)
    result = model.generate(context, max_new_tokens=args.tokens, temperature=args.temperature)
    print(result[0].tolist()[len(context[0]):])
    print(bytes(result[0].tolist()).decode("utf-8", errors="replace"))

if __name__ == "__main__":
    main()
