"""Convert the official Open English WordNet WNDB files to a compact lookup JSON."""
import json
import sys
from pathlib import Path

def parse_file(path, pos):
    entries = {}
    if not path.exists():
        return entries
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line or line.startswith("  ") or line.startswith(" "):
            continue
        parts = line.split("|", 1)
        if len(parts) != 2:
            continue
        fields = parts[0].split()
        try:
            word_count = int(fields[3], 16)
            lemmas = [fields[4 + i * 2].replace("_", " ") for i in range(word_count)]
            gloss = parts[1].strip()
            definition = gloss.split('"', 2)[1] if '"' in gloss else gloss.split(";", 1)[0]
            definition = definition.strip()
            if not definition:
                continue
            for lemma in lemmas:
                key = lemma.lower().replace("_", " ")
                record = entries.setdefault(key, {"part_of_speech": pos, "definitions": [], "synonyms": []})
                if definition not in record["definitions"] and len(record["definitions"]) < 3:
                    record["definitions"].append(definition)
                record["synonyms"] = sorted(set(record["synonyms"] + [w for w in lemmas if w.lower() != key]))[:12]
        except (ValueError, IndexError):
            continue
    return entries

def main(source, output):
    source = Path(source)
    merged = {}
    for filename, pos in (("data.noun", "noun"), ("data.verb", "verb"),
                          ("data.adj", "adjective"), ("data.adv", "adverb")):
        merged.update(parse_file(source / filename, pos))
    Path(output).write_text(json.dumps(merged, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Imported {len(merged)} WordNet entries into {output}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python -m learning.import_wordnet WNDB_DIRECTORY OUTPUT_JSON")
    main(sys.argv[1], sys.argv[2])
