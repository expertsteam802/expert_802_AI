# Expert 802 native neural training

Expert 802's neural layer is project-owned and trained from random initialization. It does not call Qwen, OpenAI, or any hosted model API. The Vercel application does not install PyTorch or download model weights.

## Architecture in this repository

- `training/tokenizer.py`: train a byte-level BPE vocabulary on the supplied corpus. No downloaded tokenizer vocabulary is required.
- `training/model.py`: decoder-only causal Transformer with multi-head self-attention, residual blocks, feed-forward layers, learned positional embeddings and tied input/output embeddings.
- `training/prepare_data.py`: tokenize explicitly selected text, documentation and source-code corpora; hold out validation documents.
- `training/train.py`: pretrain from random weights, evaluate held-out loss, clip gradients, schedule learning rate and save resumable checkpoints.
- `training/prepare_sft.py`: prepare supervised chat conversations while keeping whole conversations in one split.
- `training/generate.py`: local checkpoint inference.
- `training/serve.py`: private HTTP inference process for a trained checkpoint.
- `training/checkpoints/`: model weights and tokenizer, excluded from Git.

The starter configuration is intentionally modest: 6 Transformer blocks, width 256, 8 attention heads, a 1,024-token context and a corpus-trained vocabulary up to 16,000 tokens. Change the training arguments only when the available GPU memory and dataset justify it.

## Install training dependencies

Use a machine or rented GPU runner with a compatible PyTorch build. Training dependencies are isolated from the Vercel application:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r training/requirements.txt
```

For CUDA, install the PyTorch wheel matching the GPU host's CUDA runtime as directed by the official PyTorch installation selector, then install the remaining requirements.

## Stage 1 — prepare and pretrain

Choose legally usable general-language and programming corpora. You may supply documentation, your own code, public-domain books, and properly licensed code. Keep source/license records. Do not silently train on private customer data or personal conversations.

```bash
python training/prepare_data.py ./licensed-text ./licensed-code --output training/data --vocab-size 16000
python training/train.py --data training/data --output training/checkpoints --steps 10000
```

Training starts with random weights. It does not load Qwen or any other pretrained checkpoint. The validation split is held out by document. The output contains `latest.pt`, `best.pt` and the matching `tokenizer.json`.

## Stage 2 — conversation and coding fine-tuning

Create a JSONL file with one complete conversation per line:

```json
{"messages":[{"role":"system","content":"You are Expert 802 AI. Be accurate, explain uncertainty, and never invent tool results."},{"role":"user","content":"Write a Python function that validates an email address."},{"role":"assistant","content":"Here is a function with a clear validation boundary..."}],"split":"train","skills":["python","coding","instruction_following"]}
```

Use real, reviewed examples with varied instructions, explanations, code, debugging traces, test design, tool-use records, clarification and refusals. Do not train the model to claim it executed code when it did not. The `split` field may be `train` or `validation`; otherwise the script assigns whole conversations to splits.

```bash
python training/prepare_sft.py conversations.jsonl --tokenizer training/checkpoints/tokenizer.json --output training/sft_data
python training/train.py --data training/sft_data --output training/sft-checkpoints --resume training/checkpoints/best.pt --steps 20000
```

The fine-tuning tokenizer must match the base checkpoint. `--steps` is the total target step number, so choose a number greater than the checkpoint's saved step.

## Stage 3 — test generation and serve privately

```bash
python training/generate.py --checkpoint training/sft-checkpoints/best.pt --tokenizer training/sft-checkpoints/tokenizer.json --prompt "<|user|> Explain a Python decorator.\n<|assistant|>"
python training/serve.py --checkpoint training/sft-checkpoints/best.pt --tokenizer training/sft-checkpoints/tokenizer.json --host 127.0.0.1 --port 8787
```

The inference service exposes `GET /health` and `POST /generate`. It binds to localhost by default. Remote binding requires a 32-character `EXPERT802_INFERENCE_TOKEN` and a TLS reverse proxy. Do not expose the process directly to the public internet.

## What this does and does not mean

This is a real trainable Transformer architecture, not a dictionary lookup or bigram trick. But the starter model is small and begins with random weights. It will initially produce poor language. It needs a large, high-quality, legally usable corpus, substantial GPU training, reviewed coding examples, held-out evaluation and repeated improvements. It is **not GPT-equivalent** simply because it uses attention or because its checkpoint becomes large.

The symbolic cognition and Code Lab layers can provide deterministic calculations, conversation-state tracking, evidence retrieval, task plans and static source inspection. The neural model is intended to improve open-ended language and code generation after it has actually been trained.

The neural inference service is not yet connected to the production Vercel chat route. It is kept separate until a real trained checkpoint exists and can be evaluated. No model weights or corpora are committed to Git.
