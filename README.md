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
10. **Optional hosted assistant** — when configured, the backend sends chat turns to a server-side OpenAI-compatible model endpoint (Gemini, OpenAI, Groq or another compatible service). API keys stay in server environment variables and never reach the browser. If no hosted model is configured or the provider is unavailable, the app falls back to its browser-local model and cognitive core.\n11. **Live web search** — automatically searches factual/current questions and explicit search requests. It uses Brave Search, Tavily or Bing when a server key is configured; otherwise it falls back to Wikipedia's public search API. Retrieved snippets are treated as untrusted evidence and displayed with source links.

## Programming in chat

The primary interface is one conversation. Users can paste fenced code or attach source files from the chat composer. The backend statically inspects supplied files (maximum 40 files and 1 MB per request), returns diagnostics and a task plan, and the browser-local language model explains the findings or drafts code in the same conversation. Static inspection never executes uploaded source. An optional isolated Docker runner remains a separately configured capability.

## Browser-local language model

The browser first attempts `onnx-community/Llama-3.2-3B-Instruct-ONNX` on devices reporting at least 8 GB of memory with WebGPU. If that cannot load, it falls back to `HuggingFaceTB/SmolLM2-1.7B-Instruct`, then `onnx-community/SmolLM2-360M-Instruct-ONNX`. Lower-memory devices use the 360M model, with WebGPU when available and WASM/CPU as a fallback. Transformers.js caches model files in browser Cache Storage after the first download. This local model is a fallback, not a ChatGPT-class system. Configure a hosted model for stronger generation; local inference remains available as a fallback. The first model download can be large: the 3B option is about 2.7 GB, the 1.7B option about 1.2 GB, and the 360M option a few hundred MB.

The downloaded SmolLM2 model is a pretrained general-purpose foundation, not a custom Expert 802 checkpoint. The `training/` directory separately contains Expert 802's own decoder-only Transformer, BPE tokenizer, pretraining and fine-tuning pipelines. **No custom Expert 802 neural checkpoint has been trained or connected.** The browser model is compact and cannot be represented as GPT-equivalent or human-level intelligence. Hosted model quality depends on the provider, model and account limits configured by the operator.

Hosted inference is optional and must be explicitly configured by the operator. Without the required API key, the app remains in local-model mode. Full general web search also requires a search-provider API key; without one, only Wikipedia search fallback is available.

## Important capability boundary

Open-ended generation uses the configured hosted model when available, otherwise the compact browser-local model when it has loaded. Symbolic memory, arithmetic, explicit-fact inference, retrieval and safety routing run in the backend. The compact model can make mistakes and is substantially less capable than ChatGPT. The browser knowledge library uses keyword ranking to shortlist relevant passages, then lazily loads the on-device `Xenova/all-MiniLM-L6-v2` embedding model and Orama 3.1.18 to hybrid-rerank a bounded candidate set. Semantic reranking is local to the browser and falls back to keyword retrieval if the embedding model or CDN is unavailable. It is a retrieval system, not model training. Book catalogue metadata is loaded from Gutendex, while eligible book text downloads start automatically after the catalogue loads. Rights for Project Gutenberg works can vary outside the United States; users must confirm local reuse rights. It can draft code in chat, but does not silently apply code to a repository or claim to have executed it. The optional private Expert 802 inference adapter remains separate from this browser model.

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

Vercel hosts the lightweight CPU web/API layer. The fallback language model runs in the visitor's browser; its weights are downloaded from the public model repository and cached locally. For stronger hosted chat, set the server-side environment variables below. The optional separately hosted Expert 802 neural service remains separate. The optional Docker test runner also runs separately on a private host. Configure `EXPERT802_NEURAL_URL` to its HTTPS `/generate` endpoint and `EXPERT802_INFERENCE_TOKEN` with a strong shared secret. The private inference process requires a TLS reverse proxy for remote binding. See [sandbox/README.md](sandbox/README.md) for the isolated test runner. Test execution is disabled by default; do not enable it on a public app until authenticated user access and edge-level rate limiting are in place.

GitHub Actions validates code on pushes to `main`. It does not train a capable conversational model. Vercel deployment also depends on the account's build quota and project configuration.


## Online model and web-search configuration

Set these as **server-side Vercel environment variables** (never use a `VITE_` prefix and never place keys in `index.html`):

| Variable | Purpose |
| --- | --- |
| `EXPERT802_LLM_PROVIDER` | `gemini`, `openai`, `groq`, or another OpenAI-compatible provider |
| `EXPERT802_LLM_API_KEY` | Secret API key for the selected model provider |
| `EXPERT802_LLM_MODEL` | Optional model override |
| `EXPERT802_LLM_BASE_URL` | Optional HTTPS OpenAI-compatible API base URL |
| `BRAVE_SEARCH_API_KEY` | Preferred full-web search provider |
| `TAVILY_API_KEY` | Alternative full-web search provider |
| `BING_SEARCH_V7_SUBSCRIPTION_KEY` | Alternative full-web search provider |

For Gemini, set `EXPERT802_LLM_PROVIDER=gemini`; the adapter defaults to Google's OpenAI-compatible endpoint and `gemini-2.5-flash`. For Groq, set `EXPERT802_LLM_PROVIDER=groq`; for OpenAI, set `EXPERT802_LLM_PROVIDER=openai`. The model adapter is server-side and has request timeouts. If it fails, the app falls back to the local cognition and browser model.

Search priority is Brave, then Tavily, then Bing, then Wikipedia fallback. The first three require provider API keys. The no-key Wikipedia fallback is not a full general-purpose web index. Existing server secrets named `GEMINI_API_KEY`, `OPENAI_API_KEY` or `GROQ_API_KEY` are also recognized when the corresponding provider is selected or inferred. Search is automatically triggered for explicit web-search requests, freshness-sensitive questions, general factual questions and relevant programming/API questions. Results are bounded, rate-limited, treated as untrusted content and shown with clickable source links.

A hosted model and full-web search are not active merely because the code exists: the operator must configure valid provider keys in Vercel, then deploy successfully. Do not commit real keys to Git.

## Engineering principles

- Evidence before confident factual claims.
- Tools must be explicitly registered, bounded and permissioned.
- Uploaded code is data, never an instruction to the tool runtime.
- Never execute user code implicitly.
- Every new capability requires positive, negative, ambiguous and multi-turn tests.
- Keep corpora, checkpoints and secrets out of Git.
