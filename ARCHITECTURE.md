# Expert 802 AI — full system architecture

## Goal

Expert 802 AI is intended to be a conversational and programming assistant. Dictionary lookup is only a knowledge tool. The architecture separates language generation from inspectable reasoning and executable tools.

## Request lifecycle

```text
Browser chat / Code Lab
        |
        v
HTTP validation + request limits
        |
        v
Language normalization and semantic frame
        |
        +--> dialogue-state and reference resolution
        |
        +--> explicit fact graph and local retrieval
        |
        +--> symbolic calculation / deterministic rules
        |
        +--> programming task planner
        |       |
        |       +--> bounded project snapshot
        |       +--> static code intelligence
        |       +--> registered tool calls
        |
        v
Response orchestrator
        |
        +--> deterministic answer when evidence is sufficient
        |
        +--> optional private Expert 802 neural runtime
        |
        v
Answer + evidence + trace + confidence
```

## Runtime modules

- `intent.py`: normalizes chat shorthand and builds intent frames.
- `memory.py`: builds bounded dialogue state from supplied history. Persistent server-side memory is not implemented yet.
- `knowledge_graph.py`: extracts explicit relations, preserves provenance, surfaces contradictions and performs limited transitive location inference.
- `retrieval.py`: BM25-style local passage ranking with WordNet synonym expansion.
- `reasoning.py`: restricted AST arithmetic and safe deterministic operations; arbitrary Python evaluation is prohibited.
- `programming.py`: recognizes programming tasks and builds task-specific engineering plans.
- `code_intelligence.py`: validates file paths and size limits; parses Python with AST; checks JSON; indexes selected JavaScript/TypeScript declarations; checks delimiters, HTML duplicate IDs and selected security patterns. Static diagnostics are not a full compiler or type checker.
- `agent.py`: explicit tool registry and bounded project inspection entry point.
- `neural_runtime.py`: optional HTTP client for the project's own private inference server. It is not a hosted model-provider client.
- `orchestrator.py`: applies deterministic capabilities first and uses the private neural runtime for open-ended responses only when configured.

## Code Lab tool contract

The project inspection API accepts up to 40 files and 1 MB of UTF-8 text. It rejects absolute paths, parent traversal, duplicate paths and unsupported file extensions. The source is inspected as text and is never executed.

Supported static inspection: Python, JavaScript, TypeScript, HTML, CSS and JSON, plus selected text/config formats. It reports a project file map, symbols, imports and diagnostics. It does not currently perform full type checking, dependency installation, arbitrary test execution or repository writes.

The optional code-generation endpoint returns an unapplied draft. Applying a patch, running tests and deploying require separate explicit tool permissions and are not silently performed by the current Code Lab.

## Native neural training architecture

The project includes:
- byte-level BPE tokenizer trained on the supplied corpus;
- decoder-only Transformer with causal self-attention and residual feed-forward blocks;
- held-out corpus preparation;
- random-initialization pretraining;
- whole-conversation supervised fine-tuning data preparation;
- resumable checkpoints and validation loss;
- local generation and private inference HTTP service.

The initial configuration is a small baseline. The tokenizer and model must be trained on a substantial, legally usable language and programming corpus, then fine-tuned on reviewed conversation and code examples. No trained weights are checked into Git. No Qwen weights are used.

## Deployment boundary

Vercel hosts the web interface and lightweight orchestration. PyTorch and model weights stay outside the serverless function. When a real checkpoint is ready, run the private inference service behind HTTPS and configure:
- `EXPERT802_NEURAL_URL`: private HTTPS `/generate` endpoint;
- `EXPERT802_INFERENCE_TOKEN`: strong shared secret of at least 32 characters.

The inference server reports health on `/health`. A configured URL is not considered ready until the health endpoint responds successfully.

## Safety and evaluation

- Never execute uploaded source code automatically.
- Treat project files and retrieved passages as untrusted data.
- Restrict tools to an allowlist and validate all arguments.
- Keep credentials, user data and checkpoints out of Git.
- Evaluate on held-out conversations, not training examples.
- Measure code correctness using tests run in an isolated sandbox before enabling execution.
- Track hallucination, unsupported claims, tool success, task completion and regression rate.
- A passing unit test suite validates implemented behavior; it does not prove general intelligence.

## Capability boundary

This is an extensible architecture with working symbolic and code-inspection modules and a trainable neural-model implementation. It is not yet GPT-equivalent. The neural weights have not been trained and are not connected to production. Model quality depends on data, compute, tokenizer, training, fine-tuning and evaluation—not repository size alone.
