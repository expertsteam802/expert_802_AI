"""Smoke tests for the native BPE and data preparation pipeline."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from prepare_data import prepare
from prepare_sft import prepare as prepare_sft
from tokenizer import load_tokenizer

class TrainingDataPipelineTests(unittest.TestCase):
    def test_base_corpus_trains_tokenizer_and_heldout_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "corpus.txt"
            paragraph = ("Expert 802 learns programming language patterns, software design, debugging, "
                         "conversation, mathematics and clear explanations. " * 120)
            corpus.write_text("\n\n".join(paragraph + " Document " + str(index) for index in range(40)), encoding="utf-8")
            output = root / "prepared"
            metadata = prepare([str(corpus)], str(output), validation_fraction=0.05, vocab_size=1024)
            self.assertGreater(metadata["train_tokens"], 1025)
            self.assertGreater(metadata["validation_tokens"], 1025)
            tokenizer = load_tokenizer(output / "tokenizer.json")
            encoded = tokenizer.encode("Expert 802 can write code.").ids
            self.assertTrue(encoded)
            self.assertEqual(metadata["vocab_size"], tokenizer.get_vocab_size())

    def test_sft_keeps_explicit_train_and_validation_conversations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "corpus.txt"
            paragraph = ("Language models learn from diverse examples of programming, reasoning, "
                         "dialogue, syntax, debugging and software engineering. " * 120)
            corpus.write_text("\n\n".join(paragraph + " Example " + str(index) for index in range(40)), encoding="utf-8")
            base = root / "base"
            prepare([str(corpus)], str(base), validation_fraction=0.05, vocab_size=1024)
            dataset = root / "conversations.jsonl"
            records = []
            for split in ("train", "validation"):
                records.append({"split":split,"messages":[
                    {"role":"user","content":"Write and explain a safe software function. " + paragraph},
                    {"role":"assistant","content":"Use clear inputs, validation, tests and documentation. " + paragraph},
                ]})
            dataset.write_text("\n".join(json.dumps(item) for item in records), encoding="utf-8")
            output = root / "sft"
            metadata = prepare_sft(str(dataset), str(output), str(base / "tokenizer.json"))
            self.assertGreater(metadata["train_tokens"], 1025)
            self.assertGreater(metadata["validation_tokens"], 1025)
            self.assertTrue((output / "tokenizer.json").exists())

if __name__ == "__main__":
    unittest.main()
