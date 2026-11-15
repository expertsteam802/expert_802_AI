# Expert 802 AI — native cognitive architecture

## Design goal

Build a system that can process language into explicit representations, use evidence, preserve conversation state, reason over supported structures and execute only permitted actions. The system must not depend on a hosted LLM API or hide unsupported behavior behind canned responses.

## Request path

```text
HTTP request
  -> normalization and spelling recovery
  -> intent frame + slots
  -> dialogue-state extraction and reference resolution
  -> local lexical / corpus retrieval
  -> deterministic reasoning modules
  -> response realization + confidence + evidence
```

## Modules

- `intent.py`: normalizes chat language and classifies the user's communicative act.
- `memory.py`: builds a bounded dialogue state from the history supplied by the client. It is not yet durable cross-session memory.
- `retrieval.py`: indexes local corpus passages with BM25-style term ranking and retrieves dictionary/WordNet entries.
- `reasoning.py`: evaluates a restricted arithmetic AST and refuses arbitrary code.
- `orchestrator.py`: routes frames through capabilities, combines evidence and reports uncertainty.
- `backend/engine.py`: stable import facade for the API.

## Why this is not a GPT replacement yet

A symbolic pipeline can be inspectable and useful, but it cannot generate arbitrary fluent answers from language patterns it has never learned. A transformer implementation without trained weights is random initialization, not an intelligent model. Training a capable generator requires a substantial text corpus, accelerator compute, checkpointing, evaluation and a separately hosted inference runtime. GitHub Actions CPU runners can validate code and train small statistical baselines; they are not a practical substitute for large-scale neural training.

## Production constraints

Vercel remains the web/API host. The request path is intentionally dependency-light and does not download model weights or contact a model provider. Long-lived memory, large indexes and future neural inference should be moved to a dedicated service with explicit persistence, resource limits, authentication, observability and backups.

## Evaluation policy

Every new capability must have:
- positive examples with paraphrases and spelling noise;
- negative / ambiguous examples;
- multi-turn cases where the referent is in earlier context;
- tests proving unsupported inputs are not falsely answered;
- measured precision/recall or task success on held-out examples.

Do not mark the system as generally intelligent based on a passing unit-test suite.
