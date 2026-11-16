# Expert 802 AI — full system architecture

## Goal

Expert 802 AI is intended to be a conversational and programming assistant. Dictionary lookup is only a knowledge tool. The architecture separates language generation from inspectable reasoning and executable tools.

## Request lifecycle

```text
Browser chat / in-chat source attachments
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
        +--> browser-local SmolLM2 text generation (WebGPU/WASM)
        |       +--> model weights cached in browser Cache Storage
        |
        +--> optional private Expert 802 neural runtime (separate)
        v
Answer + evidence + trace + confidence
```

## Runtime modules

- `intent.py`: normalizes chat shorthand and builds intent frames.
- `memory.py`: builds bounded dialogue state and resolves topic phrases from supplied history.
- `conversation_memory.py`: extracts explicit user-taught meanings with provenance and attaches pronoun corrections to the active topic. The browser persists chat and learned corrections locally; the server does not maintain a cross-user memory store.
- `knowledge_graph.py`: extracts explicit relations, preserves provenance, surfaces contradictions and performs limited transitive location inference.
- `retrieval.py`: BM25-style local passage ranking with WordNet synonym expansion.
- `reasoning.py`: restricted AST arithmetic and safe deterministic operations; arbitrary Python evaluation is prohibited.
- `programming.py`: recognizes programming tasks and builds task-specific engineering plans.
- `code_intelligence.py`: validates file paths and size limits; parses Python with AST; checks JSON; indexes selected JavaScript/TypeScript declarations; checks delimiters, HTML duplicate IDs and selected security patterns. Static diagnostics are not a full compiler or type checker.
- `agent.py`: explicit tool registry and bounded project inspection entry point.
- `neural_runtime.py`: optional HTTP client for the project's own private inference server. It is not a hosted model-provider client.
- `sandbox_runtime.py`: authenticated HTTP client for a separately hosted, allowlisted Docker test runner.
- `orchestrator.py`: applies deterministic capabilities first, keeps explicit facts and safety decisions authoritative, and supplies structured evidence and engineering plans to the browser-local language model.

## In-chat programming tool contract

The chat composer accepts pasted fenced code or source-file attachments. The project inspection API accepts up to 40 files and 1 MB of UTF-8 text. It rejects absolute paths, parent traversal, duplicate paths and unsupported file extensions. The source is inspected as text and is never executed. Static findings are returned in the same chat thread, where the browser-local model can explain them and draft changes.

Supported static inspection: Python, JavaScript, TypeScript, HTML, CSS and JSON, plus selected text/config formats. It reports a project file map, symbols, imports and diagnostics. It does not currently perform full type checking, dependency installation, arbitrary test execution or repository writes.

The browser-local model can draft code in chat. Drafts are not silently applied to repositories. The optional Docker runner supports only Python unittest discovery and JavaScript syntax checks; it uses network-disabled, read-only containers with resource limits. Running tests and deploying require separate explicit tool permissions.

## Browser-local language generation

The web interface loads an instruction-tuned ONNX model through Transformers.js. Devices reporting WebGPU and at least 8 GB of memory attempt Llama 3.2 3B in q4f16, then fall back to SmolLM2 1.7B and finally SmolLM2 360M. Other devices use the 360M instruct model. If WebGPU initialization fails, the client falls back to 360M on WASM/CPU. Transformers.js caches model weights in the browser Cache API, so later visits reuse the local copy. Inference happens on the visitor's device, not on a hosted GPU. The first download ranges from a few hundred MB to about 2.4 GB.

Separately, the project includes:
- byte-level BPE tokenizer trained on the supplied corpus;
- decoder-only Transformer with causal self-attention and residual feed-forward blocks;
- held-out corpus preparation;
- random-initialization pretraining;
- whole-conversation supervised fine-tuning data preparation;
- resumable checkpoints and validation loss;
- local generation and private inference HTTP service.

The custom Expert 802 Transformer still starts from random weights; no trained Expert 802 checkpoint is checked into Git. The browser uses the separate pretrained SmolLM2 foundation model for current language generation. Neither component should be described as GPT-equivalent. No Qwen weights or API are used.

## Deployment boundary

Vercel hosts the web interface and lightweight CPU orchestration. Browser model weights are fetched directly by the browser and cached in browser storage; chat generation does not call an online GPU service. The optional custom Expert 802 private inference service can still be configured separately with:
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

This is an extensible architecture with working symbolic and code-inspection modules and a trainable neural-model implementation. It is not yet GPT-equivalent. The custom Expert 802 neural weights have not been trained and are not connected to production. The currently used browser-local SmolLM2 model is a compact pretrained model and remains less capable than a large frontier assistant.
