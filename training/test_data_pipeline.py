"""Smoke tests for corpus auditing, source-isolated BPE preparation and SFT preparation."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from audit_corpus import audit
from prepare_data import prepare
from prepare_sft import prepare as prepare_sft
from tokenizer import load_tokenizer

class TrainingDataPipelineTests(unittest.TestCase):
    def _make_corpus(self, root):
        corpus = root / "corpus"
        corpus.mkdir()
        paragraph = ("Expert 802 learns programming language patterns, software design, debugging, "
                     "conversation, mathematics and clear explanations. " * 120)
        files = []
        for index in range(20):
            path = corpus / f"source_{index}.txt"
            path.write_text((paragraph + f" Independent source {index}.\n\n") * 2, encoding="utf-8")
            files.append(path)
        manifest = root / "manifest.csv"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "license", "source_url", "permission_basis", "notes"])
            writer.writeheader()
            for path in files:
                writer.writerow({"path": path.name, "license": "CC0-1.0", "source_url": "https://example.invalid/source",
                                 "permission_basis": "public-domain test fixture", "notes": "synthetic test content"})
        return corpus, manifest, paragraph

    def test_corpus_audit_requires_provenance_and_rejects_secrets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus, manifest, _ = self._make_corpus(root)
            (corpus / "leak.txt").write_text("api_key = 'abcdefghijklmnopqrstu'", encoding="utf-8")
            with manifest.open("a", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerow(["leak.txt", "CC0-1.0", "", "synthetic test fixture", "credential scanner test"])
            report = root / "audit.json"
            with self.assertRaises(SystemExit):
                audit(corpus, manifest, report)
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertGreaterEqual(payload["accepted_files"], 1)
            self.assertTrue(any("credential" in item["reason"] for item in payload["rejected"]))

    def test_base_corpus_uses_source_isolated_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus, _, _ = self._make_corpus(root)
            output = root / "prepared"
            metadata = prepare([str(corpus)], str(output), validation_fraction=0.1, vocab_size=1024)
            self.assertGreater(metadata["train_tokens"], 1025)
            self.assertGreater(metadata["validation_tokens"], 1025)
            self.assertEqual(metadata["split_unit"], "source_file")
            self.assertGreater(metadata["training_source_files"], 0)
            self.assertGreater(metadata["validation_source_files"], 0)
            tokenizer = load_tokenizer(output / "tokenizer.json")
            self.assertTrue(tokenizer.encode("Expert 802 can write code.").ids)
            self.assertEqual(metadata["vocab_size"], tokenizer.get_vocab_size())

    def test_sft_keeps_explicit_train_and_validation_conversations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus, _, paragraph = self._make_corpus(root)
            base = root / "base"
            prepare([str(corpus)], str(base), validation_fraction=0.1, vocab_size=1024)
            dataset = root / "conversations.jsonl"
            records = []
            for split in ("train", "validation"):
                for index in range(20):
                    records.append({"split": split, "messages": [
                        {"role": "user", "content": f"Write and explain a safe software function {index}. " + paragraph},
                        {"role": "assistant", "content": "Use clear inputs, validation, tests and documentation. " + paragraph},
                    ]})
            dataset.write_text("\n".join(json.dumps(item) for item in records), encoding="utf-8")
            output = root / "sft"
            metadata = prepare_sft(str(dataset), str(output), str(base / "tokenizer.json"))
            self.assertGreater(metadata["train_tokens"], 1025)
            self.assertGreater(metadata["validation_tokens"], 1025)
            self.assertTrue((output / "tokenizer.json").exists())

if __name__ == "__main__":
    unittest.main()
