# English reference sources

The training workflow downloads these English-language references from Project Gutenberg at training time. They are used as text examples for word-pattern statistics; the project does not claim that reading them creates human-like understanding.

- **The Elements of Style**, William Strunk (Project Gutenberg eBook 37134): https://www.gutenberg.org/ebooks/37134
- **The King's English**, H. W. Fowler and F. G. Fowler (Project Gutenberg eBook 75439): https://www.gutenberg.org/ebooks/75439
- **Open English WordNet 2025**: https://en-word.net/downloads

Project Gutenberg identifies the two books as public domain in the United States. Users should check applicable copyright rules in their own country before redistributing the source texts. The WordNet-derived dictionary requires attribution under its CC BY 4.0 terms; see https://en-word.net/ and https://github.com/globalwordnet/english-wordnet.


## Downloading runtime reference books

The books are separate from model training data. To make them available to Expert 802's retrieval engine, run:

\`\`\`bash
python knowledge/fetch_references.py
\`\`\`

The script stores cleaned local text copies under \`knowledge/references/\`. The cognitive retriever indexes those files on startup alongside the bundled English corpus and dictionary glosses. Runtime chat does not make network requests. Keep the source URLs and verify local redistribution rules before committing downloaded texts or deploying them.
