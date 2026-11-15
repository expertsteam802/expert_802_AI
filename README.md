# Expert 802 AI

An API-free, English-only language learning project with its own Python backend.

## What is in this version

- Static chat interface in `index.html`.
- Vercel Python serverless endpoints: `/api/chat` and `/api/health`.
- Project-owned English processing code in `backend/`.
- A starter English dictionary in `knowledge/english_dictionary.json`.
- A small English teaching corpus in `knowledge/english_corpus.txt`.
- A reproducible word-level bigram training script in `learning/train.py`.
- No external AI inference APIs, no database integration, and no third-party model dependency.

## Run locally

Requires Python 3.10+.

```bash
python -m learning.train
python -m http.server 8000
```

The static page can be opened at `http://localhost:8000`. The Python chat API itself is deployed as a Vercel Function; to test the complete application locally, install the Vercel CLI and run `vercel dev`.

## Deploy on Vercel

Import this repository. Use the Other framework preset and leave the build command empty. Vercel detects Python files under `api/` as serverless functions. The site is served from the repository root.

## Add English knowledge and train

1. Add trusted, legally usable English text to `knowledge/english_corpus.txt` (or pass another UTF-8 text file to the trainer).
2. Run `python -m learning.train`.
3. The trainer writes `backend/language_model.json`, containing word frequencies and next-word transition counts.

For example:

```bash
python -m learning.train knowledge/english_corpus.txt
```

The starter trainer is a statistical word-bigram model. It learns word co-occurrence and next-word probabilities from the supplied text. It is **not** a neural network, does not automatically acquire human-like understanding, and is not yet used as a free-form chat generator. The current conversation endpoint combines explicit English vocabulary, grammar patterns, and intent handling. This is an honest first foundation, not a finished general intelligence.

## Training compute limitation

Vercel Functions are for short request/response work, not long model training. The training script is designed to run in a separate Python environment (for example, a future GitHub Actions training workflow or another hosted CPU/GPU runtime) and save its resulting model artifact to the repository or object storage. No local-PC training is required once a hosted training runner is configured. This commit does not claim that such a runner is already configured.

## Next milestones

- Expand the dictionary with licensed lexical data and word forms.
- Add sentence parsing, morphology, grammar tests, and user correction datasets.
- Add a hosted, resumable training workflow and versioned checkpoints.
- Upgrade from n-gram statistics to a small neural language model when a suitable training runtime and corpus are available.
