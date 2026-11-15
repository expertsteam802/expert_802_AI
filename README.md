# Expert 802 AI

Expert 802 AI is a provider-independent English cognitive system with a lightweight web interface and a Python reasoning backend.

## Current architecture

The live chat endpoint runs the project-owned pipeline in `backend/cognition/`:

1. **Normalization** — common chat shorthand, spelling recovery and joined-word recovery.
2. **Intent and slot analysis** — turns a message into a structured frame (definition, calculation, request, follow-up, explanation, question or statement).
3. **Dialogue state** — extracts the current topic and resolves simple references such as “it” and “that” against supplied conversation history.
4. **Local retrieval** — ranks passages from the project corpus and expands terms with WordNet synonyms.
5. **Fact graph** — stores explicit relations from the supplied conversation history, answers supported questions and detects conflicts; it can infer transitive location relations when each link was explicitly stated.
6. **Symbolic reasoning** — evaluates safe arithmetic expressions with a restricted AST; it never evaluates arbitrary Python.
7. **Response orchestration** — combines the frame, context, retrieved evidence and computed results, and exposes confidence/evidence metadata to callers.

There is no Qwen integration, external model endpoint, API key, or provider token in the active chat path.

## Important capability boundary

This is a real logic-based cognitive pipeline, but it is **not yet a general neural language generator** and it is not GPT-equivalent. It can normalize language, recognize supported intents, resolve limited dialogue references, retrieve local evidence and perform bounded symbolic calculations. For an open-ended question outside its local knowledge, it should say what evidence is missing rather than fabricate an answer.

The word-bigram artifact in `backend/language_model.json` is only a next-word statistical baseline. It is not the cognitive engine and it does not provide general language understanding.

## Repository layout

- `index.html` — fixed-layout chat, dictionary and architecture-status views.
- `api/chat.py` — Vercel HTTP endpoint and live architecture status.
- `backend/engine.py` — compatibility facade for the native engine.
- `backend/cognition/` — normalization, intent analysis, dialogue memory, retrieval, reasoning and orchestration.
- `backend/language.py` — lexical data loading and spelling/grammar helpers.
- `knowledge/` — English dictionary, imported WordNet and the small starter corpus.
- `learning/` — reproducible corpus and WordNet import workflows.
- `training/README.md` — pinned, from-scratch neural-training path using MIT-licensed source code; not yet run or connected.
- `tests/` — behavior and HTTP integration tests.

## Run locally

Requires Python 3.10+.

```bash
python -m compileall -q api backend learning tests
python -m unittest discover -s tests -v
python -m learning.train
```

For the complete Vercel function environment, install the Vercel CLI and run `vercel dev`.

## Deploy

Import this repository into Vercel. Use the Other framework preset and leave the build command empty. Vercel detects Python files under `api/` as serverless functions.

GitHub Actions runs validation on pushes to `main`. The separate training workflow runs on GitHub-hosted CPU compute and updates the small word-bigram baseline and WordNet import; it does not train a neural conversational model.

## Roadmap

- Improve robust parsing with a tested grammar and labeled semantic-frame data.
- Add durable, user-controlled memory storage rather than relying only on browser-supplied recent history.
- Expand the knowledge base with licensed, source-attributed documents and retrieval evaluations.
- Add explicit multi-step planning, tool permissions, execution traces and recovery.
- Train a native neural generator only when a suitable GPU training runtime, large legally usable corpus, checkpoint storage and held-out evaluation set are available. Source code alone is not trained intelligence.
