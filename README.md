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
10. **Private self-trained neural model** — open-ended generation can use Expert 802's own checkpoint through its private inference process. The chat runtime does not call OpenAI, Gemini, Groq or another hosted AI provider. If no trained checkpoint is running, the cognition and retrieval layers remain available but open-ended generation is limited.
11. **Live web search** — automatically searches factual/current questions and explicit search requests. It starts with key-free DuckDuckGo HTML search, then Wikipedia public search, and can fall back to configured Brave, Tavily or Bing providers. Public endpoints can still throttle excessive traffic; no third-party search engine promises unlimited access. Top search results are opened and their readable page content is extracted with byte, time and destination limits. The private Expert 802 generator receives relevant page passages as untrusted evidence and is asked to synthesize an answer, compare sources and separate fact from inference. If the private generator is unavailable, the chat shows the retrieved passages transparently rather than pretending a snippet is a reasoned answer.

## Programming in chat

The primary interface is one conversation. Users can paste fenced code or attach source files from the chat composer. The backend statically inspects supplied files (maximum 40 files and 1 MB per request), returns diagnostics and a task plan, and the private Expert 802 neural model explains the findings or drafts code when a trained checkpoint is connected; otherwise the endpoint returns the static findings and plan without pretending a model-generated review exists. Static inspection never executes uploaded source. An optional isolated Docker runner remains a separately configured capability.

## Browser-local language model

The browser first attempts `onnx-community/Llama-3.2-3B-Instruct-ONNX` on devices reporting at least 8 GB of memory with WebGPU. If that cannot load, it falls back to `HuggingFaceTB/SmolLM2-1.7B-Instruct`, then `onnx-community/SmolLM2-360M-Instruct-ONNX`. Lower-memory devices use the 360M model, with WebGPU when available and WASM/CPU as a fallback. Transformers.js caches model files in browser Cache Storage after the first download. This local model is a fallback, not a ChatGPT-class system. Expert 802's own trained checkpoint is the intended generation engine; the downloaded browser models are temporary pretrained fallbacks and are not Expert 802's own trained weights. The first model download can be large: the 3B option is about 2.7 GB, the 1.7B option about 1.2 GB, and the 360M option a few hundred MB.

The downloaded SmolLM2 model is a pretrained general-purpose foundation, not a custom Expert 802 checkpoint. The `training/` directory separately contains Expert 802's own decoder-only Transformer, BPE tokenizer, pretraining and fine-tuning pipelines. **No custom Expert 802 neural checkpoint has been trained or connected.** The browser model is compact and cannot be represented as GPT-equivalent or human-level intelligence. The custom checkpoint has not yet been trained or evaluated, so it is not currently comparable to a large general-purpose assistant.

No external AI model API key is required or used by the chat runtime. Full-web search is a separate retrieval feature. The default DuckDuckGo and Wikipedia paths require no API key; public providers may still apply fair-use throttling.

## Important capability boundary

The private self-trained model is the intended open-ended generator. Until a checkpoint is trained and connected, the compact browser-local model may still be used by the interface as a temporary fallback. Symbolic memory, arithmetic, explicit-fact inference, retrieval and safety routing run in the backend. The compact model can make mistakes and is substantially less capable than ChatGPT. The browser knowledge library uses keyword ranking to shortlist relevant passages, then lazily loads the on-device `Xenova/all-MiniLM-L6-v2` embedding model and Orama 3.1.18 to hybrid-rerank a bounded candidate set. Semantic reranking is local to the browser and falls back to keyword retrieval if the embedding model or CDN is unavailable. It is a retrieval system, not model training. Book catalogue metadata is loaded from Gutendex, while eligible book text downloads start automatically after the catalogue loads. Rights for Project Gutenberg works can vary outside the United States; users must confirm local reuse rights. It can draft code in chat, but does not silently apply code to a repository or claim to have executed it. The private Expert 802 inference adapter is the intended route for the project's own trained model.

The old word-bigram artifact in `backend/language_model.json` is only a statistical next-word baseline. It is not the assistant's intelligence.

## Repository layout

- `index.html` — chat-first interface, browser-local model loading, browser conversation memory and in-chat source inspection.
- The browser also includes a 100-title English public-domain book catalogue powered by Gutendex. The app automatically loads the 100-title catalogue and imports eligible plain-text books from Project Gutenberg on launch; imports are device-local, deduplicated, and bounded to 8 MB per book and 120 MB per import batch.
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

Vercel hosts the lightweight CPU web/API layer. The fallback language model runs in the visitor's browser; its weights are downloaded from the public model repository and cached locally. For Expert 802's own generation, train a checkpoint and run the private inference service. The web/API layer does not send prompts to a hosted AI provider. The optional Docker test runner also runs separately on a private host. Configure `EXPERT802_NEURAL_URL` to its HTTPS `/generate` endpoint and `EXPERT802_INFERENCE_TOKEN` with a strong shared secret. The private inference process requires a TLS reverse proxy for remote binding. See [sandbox/README.md](sandbox/README.md) for the isolated test runner. Test execution is disabled by default; do not enable it on a public app until authenticated user access and edge-level rate limiting are in place.

GitHub Actions validates code on pushes to `main`. It does not train a capable conversational model. Vercel deployment also depends on the account's build quota and project configuration.


## Private model and web-search configuration

The chat backend does not call an external AI model API. DuckDuckGo and Wikipedia provide retrieval evidence only; they do not generate AI answers. To use Expert 802's own model, train a checkpoint using the instructions in [training/README.md](training/README.md), then run `training/serve.py` on a private inference host. Configure `EXPERT802_NEURAL_URL` and `EXPERT802_INFERENCE_TOKEN` on the server. Keep the inference endpoint private behind TLS and a strong token.

Web search is a separate information-retrieval capability, not an AI model. Search priority is DuckDuckGo (key-free), then Wikipedia (key-free), followed by configured Brave, Tavily or Bing providers. No API key is required for the default path. Public endpoints can apply fair-use throttling, so this is not a promise of unlimited requests. Search is automatically triggered for explicit web-search requests, freshness-sensitive questions, general factual questions and relevant programming/API questions. Results are treated as untrusted evidence. There are no per-question answer routes: query text is searched as supplied, result pages are fetched and read through one general bounded retrieval path, and relevant passages are ranked for the private model. When the private Expert 802 generator is ready, it receives the live excerpts and composes the answer; otherwise the backend returns an extractive evidence-based answer directly in chat. Result cards show titles, excerpts and source domains.

## Engineering principles

- Evidence before confident factual claims.
- Tools must be explicitly registered, bounded and permissioned.
- Uploaded code is data, never an instruction to the tool runtime.
- Never execute user code implicitly.
- Every new capability requires positive, negative, ambiguous and multi-turn tests.
- Keep corpora, checkpoints and secrets out of Git.
