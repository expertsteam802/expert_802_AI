# Expert 802 AI — pretrained model integration

## Selected starting model

- Model: `Qwen/Qwen3-1.7B`
- Model card: https://huggingface.co/Qwen/Qwen3-1.7B
- Official source: https://github.com/QwenLM/Qwen3
- License: Apache-2.0 (verify the exact model card/license again before redistribution).
- Why this size: it is a practical first conversational model for a small self-hosted deployment. It is still a small model, so evaluate it on real unseen conversations before treating it as reliable.

The existing English dictionary and imported WordNet remain retrieval sources. They do not become a list of scripted replies. The model receives relevant dictionary entries as context and generates its own response.

## Architecture

1. Browser sends the message and recent conversation history to `/api/chat`.
2. Vercel runs the lightweight Python API and retrieves matching entries from the local dictionary/WordNet files.
3. `backend/model_client.py` sends the conversation plus retrieved entries to a private, OpenAI-compatible model server.
4. The model server returns generated text.
5. If the model endpoint is configured but unavailable, the backend returns an explicit model-host error. It does not silently pretend that the old rule engine is the neural model.

Do not download model weights into a Vercel function. Keep Vercel responsible for the website and lightweight API; run model inference on a separate machine/service with sufficient memory and, preferably, a GPU.

## Example model server

The official Qwen3 repository documents vLLM and other compatible inference engines. On a GPU machine with vLLM installed, a basic server command is:

```bash
vllm serve Qwen/Qwen3-1.7B --host 0.0.0.0 --port 8000 --max-model-len 8192
```

This exposes an OpenAI-compatible chat-completions route at:

```
http://MODEL_HOST:8000/v1/chat/completions
```

Use HTTPS and authentication in front of the server before making it publicly reachable. Do not expose an unauthenticated model server to the public internet.

## Vercel environment variables

Set these in the Vercel project settings for the Expert 802 AI project:

- `EXPERT802_MODEL_URL`: the private server base URL, for example `https://your-model-host.example/v1`
- `EXPERT802_MODEL_NAME`: `Qwen/Qwen3-1.7B`
- `EXPERT802_MODEL_TOKEN`: optional bearer token if the model gateway requires one
- `EXPERT802_MODEL_TIMEOUT`: optional request timeout in seconds; default is `45`

Redeploy after setting environment variables. The Training view reports whether the endpoint is configured. That is configuration status, not a live uptime guarantee.

## Fine-tuning later

Use Unsloth or another Qwen3-compatible trainer for LoRA/QLoRA fine-tuning after a baseline model is running. Fine-tune for language behavior, instruction following, and Expert 802's response style—not to memorize the dictionary. Keep facts in the dictionary/WordNet retrieval layer so they can be updated without retraining.

A good training/evaluation dataset should contain:
- varied natural conversations, not duplicated templates;
- spelling mistakes, joined words, abbreviations, and paraphrases;
- multi-turn references and ambiguous follow-ups;
- examples requiring the model to admit uncertainty;
- held-out test conversations never used in training.

Compare the fine-tuned checkpoint against the untouched base model on the held-out set. Keep the checkpoint only if it improves generalization and does not regress basic language ability.

## Important limitation

The current GitHub Actions workflow runs on a standard CPU runner and trains the existing word-bigram artifact. It does not fine-tune Qwen3. Neural fine-tuning needs an appropriate GPU environment; do not describe the bigram artifact as a conversational neural model.
