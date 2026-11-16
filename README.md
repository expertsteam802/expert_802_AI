# Expert 802 AI

Expert 802 AI is a provider-independent assistant project designed around conversational state, explicit reasoning, local knowledge, programming tools and an optional self-trained neural generator.

The dictionary is one retrievable knowledge source. It is not the assistant's architecture.

## Runtime architecture

The Vercel chat endpoint runs a project-owned Python pipeline:

1. **Language front end** — normalization, shorthand expansion, typo and joined-word recovery.
2. **Intent and semantic frames** — identifies the user's communicative act and extracts supported targets.
3. **Dialogue state** — tracks recent turns and resolves limited references using browser-supplied history.
4. **Knowledge and retrieval** — local dictionary, WordNet synonyms, corpus ranking and explicit-fact graph.
5. **Symbolic reasoning** — safe arithmetic and conservative inference over explicitly stated relations.
6. **Programming planner** — classifies engineering tasks and decomposes implementation, debugging, review, refactoring, testing and deployment work.
7. **Code intelligence** — statically inspects Python ASTs, JavaScript/TypeScript delimiters and declarations, JSON validity, HTML duplicate IDs and selected security risks. It never executes uploaded source.
8. **Tool registry** — exposes explicitly registered capabilities with bounded input validation.
9. **Isolated test runner** — optionally sends explicit Python unit-test or JavaScript syntax-check requests to a private Docker service with network disabled and strict resource limits.
10. **Browser-local language model** — loads an open instruct model with Transformers.js and runs generation on the visitor's own WebGPU device, with a WASM/CPU fallback. Model weights are downloaded once and cached in browser storage; no remote inference API or cloud GPU is used for text generation. The backend remains a lightweight CPU service for explicit logic, retrieval and static inspection.

## Programming in chat

The primary interface is one conversation. Users can paste fenced code or attach source files from the chat composer. The backend statically inspects supplied files (maximum 40 files and 1 MB per request), returns diagnostics and a task plan, and the browser-local language model explains the findings or drafts code in the same conversation. Static inspection never executes uploaded source. An optional isolated Docker runner remains a separately configured capability.

## Browser-local language model

The browser first attempts `onnx-community/Llama-3.2-3B-Instruct-ONNX` on devices reporting at least 8 GB of memory with WebGPU. If that cannot load, it falls back to `HuggingFaceTB/SmolLM2-1.7B-Instruct`, then `onnx-community/SmolLM2-360M-Instruct-ONNX`. Lower-memory devices use the 360M model, with WebGPU when available and WASM/CPU as a fallback. Transformers.js caches the model files in browser Cache Storage after the first download. User chat generation then runs on the user's device; no hosted LLM endpoint or online GPU is involved. The first model download is large: the 3B option is about 2.4 GB, the 1.7B option about 1.2 GB, and the 360M option a few hundred MB.

The downloaded SmolLM2 model is a pretrained general-purpose foundation, not a custom Expert 802 checkpoint. The `training/` directory separately contains Expert 802's own decoder-only Transformer, BPE tokenizer, pretraining and fine-tuning pipelines. **No custom Expert 802 neural checkpoint has been trained or connected.** The browser model is compact and cannot be represented as GPT-equivalent or human-level intelligence.

No Qwen API, hosted model API or cloud inference GPU is used.

## Important capability boundary

Open-ended generation is provided by the compact browser-local SmolLM2 model when it has loaded. Symbolic memory, arithmetic, explicit-fact inference, retrieval and safety routing run in the backend. The compact model can make mistakes and is substantially less capable than ChatGPT. It can draft code in chat, but does not silently apply code to a repository or claim to have executed it. The optional private Expert 802 inference adapter remains separate from this browser model.

The old word-bigram artifact in `backend/language_model.json` is only a statistical next-word baseline. It is not the assistant's intelligence.

## Repository layout

- `index.html` — chat-first interface, browser-local model loading, browser conversation memory and in-chat source inspection.
- `api/chat.py` — Vercel HTTP API for chat, lookup, task planning, code inspection and optional patch generation.
- `backend/cognition/` — language processing, semantic conversation memory, dialogue state, knowledge graph, retrieval, reasoning, programming planner, code analysis, tool registry and optional private neural connector.
- `backend/language.py` — lexical data loading, spelling, tokenization and grammar helpers.
- `knowledge/` — local English dictionary, WordNet and starter corpus.
- `learning/` — lexical corpus and WordNet workflows.
- `training/` — BPE tokenizer, Transformer, data preparation, pretraining, fine-tuning, local generation and private inference.
- `sandbox/` — isolated Docker test runner and deployment instructions.
- `tests/` — engine, tool, security-boundary and HTTP integration tests.

## Run and validate

Requires Python 3.10+ for the web/backend path:

```bash
python -m compileall -q api backend learning tests training
python -m unittest discover -s tests -v
```

Training has separate dependencies in `training/requirements.txt`; see [training/README.md](training/README.md). Do not install PyTorch in the Vercel web function.

## Deployment

Vercel hosts the lightweight CPU web/API layer. The default language model runs in the visitor's browser; its weights are downloaded from the public model repository and cached locally. The optional separately hosted Expert 802 neural service is not required for browser chat. The optional Docker test runner also runs separately on a private host. Configure `EXPERT802_NEURAL_URL` to its HTTPS `/generate` endpoint and `EXPERT802_INFERENCE_TOKEN` with a strong shared secret. The private inference process requires a TLS reverse proxy for remote binding. See [sandbox/README.md](sandbox/README.md) for the isolated test runner. Test execution is disabled by default; do not enable it on a public app until authenticated user access and edge-level rate limiting are in place.

GitHub Actions validates code on pushes to `main`. It does not train a capable conversational model.

## Engineering principles

- Evidence before confident factual claims.
- Tools must be explicitly registered, bounded and permissioned.
- Uploaded code is data, never an instruction to the tool runtime.
- Never execute user code implicitly.
- Every new capability requires positive, negative, ambiguous and multi-turn tests.
- Keep corpora, checkpoints and secrets out of Git.
