"""Audit a text/code corpus before training; require explicit provenance and license records."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

TEXT_SUFFIXES = {".txt", ".md", ".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".sql", ".json", ".jsonl"}
SECRET_PATTERNS = [
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"(?i)(?:api[_-]?key|secret|password|access[_-]?token)\s*[:=]\s*['\"][^'\"]{12,}"),
]

def audit(corpus_dir, manifest_path, output_path, max_file_mb=20):
    root = Path(corpus_dir).resolve()
    manifest = {}
    with Path(manifest_path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rel = (row.get("path") or "").strip()
            if rel:
                manifest[rel.replace("\\", "/")] = row
    report = {"corpus": str(root), "files_scanned": 0, "accepted_files": 0,
              "rejected": [], "warnings": [], "accepted": [], "duplicate_groups": []}
    hashes = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.is_symlink():
            report["rejected"].append({"path": str(path.relative_to(root)), "reason": "symbolic links are not accepted"})
            continue
        rel = path.relative_to(root).as_posix()
        if path.suffix.lower() not in TEXT_SUFFIXES:
            report["rejected"].append({"path": rel, "reason": "unsupported file type"})
            continue
        if path.stat().st_size == 0 or path.stat().st_size > max_file_mb * 1024 * 1024:
            report["rejected"].append({"path": rel, "reason": "empty or exceeds size limit"})
            continue
        try:
            raw = path.read_bytes()
            content = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            report["rejected"].append({"path": rel, "reason": "not valid UTF-8 text"})
            continue
        report["files_scanned"] += 1
        row = manifest.get(rel)
        if not row or not (row.get("license") or "").strip() or not (row.get("permission_basis") or "").strip():
            report["rejected"].append({"path": rel, "reason": "missing license and permission_basis in corpus manifest"})
            continue
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            report["rejected"].append({"path": rel, "reason": "possible credential or private key detected"})
            continue
        digest = hashlib.sha256(raw).hexdigest()
        if digest in hashes:
            report["duplicate_groups"].append({"sha256": digest, "paths": [hashes[digest], rel]})
            report["rejected"].append({"path": rel, "reason": "duplicate content"})
            continue
        hashes[digest] = rel
        item = {"path": rel, "sha256": digest, "bytes": len(raw),
                "license": row["license"].strip(),
                "source_url": (row.get("source_url") or "").strip(),
                "permission_basis": row["permission_basis"].strip()}
        report["accepted"].append(item)
        report["accepted_files"] += 1
        if not item["source_url"]:
            report["warnings"].append({"path": rel, "reason": "source_url is empty; retain independent provenance records"})
    report["rejected_files"] = len(report["rejected"])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("files_scanned", "accepted_files", "rejected_files", "warnings")}, indent=2))
    if report["accepted_files"] == 0 or report["rejected_files"]:
        raise SystemExit("Corpus audit failed: fix rejected files or provide a fully reviewed corpus before training.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus")
    parser.add_argument("--manifest", required=True, help="CSV with path,license,source_url,permission_basis,notes columns")
    parser.add_argument("--output", default="training/corpus_audit.json")
    parser.add_argument("--max-file-mb", type=int, default=20)
    args = parser.parse_args()
    audit(args.corpus, args.manifest, args.output, args.max_file_mb)

if __name__ == "__main__":
    main()
