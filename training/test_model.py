"""Tiny forward/backward/generation smoke test for the native Transformer."""
import torch
from model import Expert802GPT, ModelConfig

def main():
    torch.manual_seed(802)
    config=ModelConfig(vocab_size=512,block_size=64,n_layer=2,n_head=4,n_embd=64,dropout=0.0)
    model=Expert802GPT(config)
    tokens=torch.randint(0,config.vocab_size,(2,32))
    logits,loss=model(tokens,tokens)
    assert logits.shape==(2,32,config.vocab_size)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    generated=model.generate(tokens[:1,:8],max_new_tokens=3)
    assert generated.shape==(1,11)
    print({"parameters":model.parameter_count(),"loss":float(loss.item()),"generation_shape":list(generated.shape)})

if __name__=="__main__":
    main()
