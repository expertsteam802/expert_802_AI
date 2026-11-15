# Expert 802 AI

A temporary, API-free natural-language chat prototype.

## Current implementation

- Static browser application; no server or database required.
- No Supabase integration.
- No external AI provider or inference API.
- Local English vocabulary definitions and grammar-pattern checks.
- Basic intent detection and in-session conversation context.
- Explicit uncertainty handling: unsupported factual questions are not answered with invented facts.

## Important limitation

This first version is a rule-based language program, not a trained neural language model. Its vocabulary is a small starter lexicon, not the complete English dictionary. It can demonstrate grammar and language processing, but it cannot provide broad human-level understanding or reliable open-domain factual answers. A complete dictionary would also provide word forms and definitions, not by itself teach reasoning or conversation.

## Run and deploy

The application is a static `index.html` file. Open it in a browser for local testing or import this repository into Vercel and deploy it as a static site. No build command or environment variables are required. Set the output directory to the repository root if Vercel asks.

## Next development step

Expand the lexical and grammar datasets, add compositional parsing and tests, then introduce a genuinely trainable small language model with versioned weights when a suitable training runtime is available.
