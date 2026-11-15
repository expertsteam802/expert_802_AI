# Train Expert 802's own neural generator (no provider API)

This is the planned neural-generation layer for the native cognition stack. It is deliberately separate from Vercel: model weights and training dependencies do not belong in a serverless web deployment.

## Selected source-code foundation

Use [karpathy/nanochat](https://github.com/karpathy/nanochat), pinned to commit `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`. The repository is MIT-licensed; keep its MIT notice with any copied or modified source. It contains the tokenizer, GPT architecture, pretraining, supervised fine-tuning, evaluation, checkpointing and inference engine. This is source code, not a hosted inference service and not a pretrained Qwen model.

Pinned upstream commit: `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`

## What training actually requires

The upstream reference run trains a GPT-2-capability model from scratch using an 8xH100 GPU node and reports roughly 1.5–2 hours for its speedrun. Its own README describes that model as a very small, weak assistant. That is a useful first self-owned neural generator, **not** GPT-4/5/6 capability and not a promise of human-level reasoning. GPU rental, storage, bandwidth and later inference hosting are separate costs.

The current Expert 802 GitHub Actions runners are CPU runners. They can test code and train the small lexical baseline, but they cannot reproduce the 8xH100 run. Do not start the expensive run until a GPU host and budget are explicitly chosen.

## Expert 802 training stages

1. **Base pretraining:** train random-initialized weights on a large, legally usable general-language corpus. The upstream speedrun uses its own ClimbMix dataset; this is not Expert 802 user data.
2. **Conversation fine-tuning:** adapt the SFT task mixture to include the Expert 802 conversation dataset described below. Keep general conversation, instruction following, clarification, tool-use and refusal examples balanced.
3. **Evaluation:** hold out complete conversations and evaluate typo recovery, multi-turn references, factual grounding, arithmetic, tool selection and hallucination rate. Never train on the held-out split.
4. **Checkpointing:** store tokenizer and model checkpoints in external object storage or a persistent GPU volume, not Git history.
5. **Inference:** run the trained model on a private GPU service. Vercel should remain the frontend and lightweight orchestration layer. The symbolic graph and deterministic tools remain authoritative for facts and calculations.

## Conversation dataset format

Use JSON Lines. One line is one complete conversation:

```json
{"messages":[{"role":"system","content":"You are Expert 802 AI. Use evidence, ask when context is missing, and never claim an action you did not perform."},{"role":"user","content":"i nned my gamerightnoww"},{"role":"assistant","content":"Do you want help finding a game, launching one, or fixing a problem with a game?"}],"split":"train","skills":["typo_recovery","intent","clarification"]}
```

Rules:
- Include varied paraphrases, misspellings, abbreviations, joined words and natural multi-turn references.
- Include tool-call traces and tool results as structured examples; do not teach the model to invent tool outcomes.
- Include uncertainty and clarification examples.
- Keep personal conversations out of the dataset unless the user explicitly chooses to contribute them.
- Split by whole conversation, not by individual message.
- Store source/license metadata for every imported dataset.

## Current status

The live app currently uses the native symbolic cognition engine. No neural weights are installed, no Qwen endpoint is configured, and no provider token is required. The training harness above is a documented next stage; it has **not** been run or connected to the live app yet.
