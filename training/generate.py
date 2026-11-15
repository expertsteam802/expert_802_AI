"""Generate text from a trained Expert802GPT checkpoint."""
import argparse
import torch
from model import Expert802GPT,ModelConfig
from tokenizer import load_tokenizer

def load_model(checkpoint_path,device):
    payload=torch.load(checkpoint_path,map_location=device,weights_only=False)
    config=ModelConfig(**payload["config"])
    model=Expert802GPT(config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model,payload

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",default="training/checkpoints/best.pt")
    parser.add_argument("--tokenizer",default="training/checkpoints/tokenizer.json")
    parser.add_argument("--prompt",default="<|user|> Hello.\n<|assistant|>")
    parser.add_argument("--tokens",type=int,default=200)
    parser.add_argument("--temperature",type=float,default=0.8)
    args=parser.parse_args()
    device="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model,payload=load_model(args.checkpoint,device)
    tokenizer=load_tokenizer(args.tokenizer)
    prompt_ids=tokenizer.encode(args.prompt).ids[-model.config.block_size:]
    context=torch.tensor([prompt_ids],dtype=torch.long,device=device)
    result=model.generate(context,max_new_tokens=max(1,min(args.tokens,8192)),temperature=args.temperature,stop_token_id=tokenizer.token_to_id("<|endoftext|>"))
    print(tokenizer.decode(result[0].tolist()[len(prompt_ids):],skip_special_tokens=True))

if __name__=="__main__":
    main()
