"""Small decoder-only Transformer trained from random initialization; no pretrained weights."""
from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class ModelConfig:
    vocab_size: int = 16000
    block_size: int = 1024
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.1

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.n_embd % config.n_head:
            raise ValueError("Embedding size must be divisible by head count.")
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dropout = config.dropout

    def forward(self, x):
        batch, length, channels = x.shape
        q, k, v = self.qkv(x).split(channels, dim=2)
        q = q.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
        value = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        value = value.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.proj(value)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.n_embd)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x

class Expert802GPT(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.n_embd)
        self.position_embedding = nn.Embedding(self.config.block_size, self.config.n_embd)
        self.dropout = nn.Dropout(self.config.dropout)
        self.blocks = nn.Sequential(*[TransformerBlock(self.config) for _ in range(self.config.n_layer)])
        self.final_norm = nn.LayerNorm(self.config.n_embd)
        self.language_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        self.language_head.weight = self.token_embedding.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, targets=None):
        batch, length = tokens.shape
        if length > self.config.block_size:
            raise ValueError("Input exceeds configured context length.")
        positions = torch.arange(length, device=tokens.device)
        hidden = self.dropout(self.token_embedding(tokens) + self.position_embedding(positions))
        hidden = self.final_norm(self.blocks(hidden))
        logits = self.language_head(hidden)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, tokens, max_new_tokens=300, temperature=0.8, top_k=40, stop_token_id=None):
        self.eval()
        for _ in range(max_new_tokens):
            context = tokens[:, -self.config.block_size:]
            logits, _ = self(context)
            logits = logits[:, -1, :] / max(temperature, 1e-4)
            if top_k:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            tokens = torch.cat((tokens, next_token), dim=1)
            if stop_token_id is not None and int(next_token[0, 0]) == int(stop_token_id):
                break
        return tokens

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def config_dict(self):
        return asdict(self.config)
