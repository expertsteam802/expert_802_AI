"""Locally trained byte-level BPE tokenizer. No downloaded vocabulary or hosted service."""
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

SPECIAL_TOKENS = ["<|endoftext|>", "<|system|>", "<|user|>", "<|assistant|>", "<|tool|>"]

def train_tokenizer(files, output_path, vocab_size=16000, min_frequency=2):
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<|unk|>"] + SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train(files=[str(path) for path in files], trainer=trainer)
    tokenizer.save(str(output_path))
    return tokenizer

def load_tokenizer(path):
    return Tokenizer.from_file(str(path))

def encode_text(tokenizer, text):
    return tokenizer.encode(text).ids

def decode_tokens(tokenizer, ids):
    return tokenizer.decode([int(value) for value in ids], skip_special_tokens=False)
