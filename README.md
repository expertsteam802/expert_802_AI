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
9. **Neural runtime adapter** — optionally calls a private Expert 802 inference process when a trained checkpoint is actually available. No hosted model provider is used.

## Code Lab

The Code Lab UI lets a user select a project folder or source files and submit a bounded project snapshot to the backend. It returns a file map, language counts, discovered symbols/imports, selected diagnostics and an engineering plan. It accepts at most 40 files and 1 MB per inspection request.

A trained private model can optionally produce an unapplied patch draft through the self-hosted inference adapter. The app does not claim that a draft has been applied, tested or deployed. Arbitrary uploaded code is not executed.

## Native neural model

The `training/` directory now contains a project-owned decoder-only Transformer, corpus-trained byte-level BPE tokenizer, pretraining pipeline, conversation-level fine-tuning data preparation, checkpointed training loop, local generation and a private inference server.

It starts from random weights. **No trained neural checkpoint is currently included in the repository or connected to production.** The initial model configuration is a small research baseline, not GPT-level intelligence. It needs substantial licensed language and code data, GPU compute, careful evaluation and iterative training.

No Qwen model, Qwen endpoint or third-party model API is part of the architecture.

## Important capability boundary

The current production chat path is the symbolic cognition system plus optional private neural inference if configured. Without trained weights, open-ended language generation and autonomous code generation are not available. Code Lab can statically inspect source and build work plans, but it does not yet apply changes to arbitrary uploaded projects or execute them.

The old word-bigram artifact in `backend/language_model.json` is only a statistical next-word baseline. It is not the assistant's intelligence.

## Repository layout

- `index.html` — chat, dictionary, Code Lab and architecture-status views.
- `api/chat.py` — Vercel HTTP API for chat, lookup, task planning, code inspection and optional patch generation.
- `backend/cognition/` — language processing, dialogue state, knowledge graph, retrieval, reasoning, programming planner, code analysis, tool registry and private neural connector.
- `backend/language.py` — lexical data loading, spelling, tokenization and grammar helpers.
- `knowledge/` — local English dictionary, WordNet and starter corpus.
- `learning/` — lexical corpus and WordNet workflows.
- `training/` — BPE tokenizer, Transformer, data preparation, pretraining, fine-tuning, local generation and private inference.
- `tests/` — engine, tool, security-boundary and HTTP integration tests.

## Run and validate

Requires Python 3.10+ for the web/backend path:

```bash
python -m compileall -q api backend learning tests training
python -m unittest discover -s tests -v
```

Training has separate dependencies in `training/requirements.txt`; see [training/README.md](training/README.md). Do not install PyTorch in the Vercel web function.

## Deployment

Vercel hosts the lightweight web/API layer. A trained neural model must run separately on a private machine or GPU service. Configure `EXPERT802_NEURAL_URL` to its HTTPS `/generate` endpoint and `EXPERT802_INFERENCE_TOKEN` with a strong shared secret. The private inference process requires a TLS reverse proxy for remote binding.

GitHub Actions validates code on pushes to `main`. It does not train a capable conversational model.

## Engineering principles

- Evidence before confident factual claims.
- Tools must be explicitly registered, bounded and permissioned.
- Uploaded code is data, never an instruction to the tool runtime.
- Never execute user code implicitly.
- Every new capability requires positive, negative, ambiguous and multi-turn tests.
- Keep corpora, checkpoints and secrets out of Git.
