"""Validate the frozen 30-question Section relevance gold set."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GOLD = ROOT / "evaluation" / "section_relevance_gold_30.jsonl"
FREEZE = ROOT / "evaluation" / "section_relevance_gold_30.freeze.json"


def sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Check gold records, source evidence, and frozen fingerprints."""
    records = [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines()]
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    questions = {
        int(row["id"]): row["question"].strip('"')
        for row in csv.DictReader(
            (ROOT / "question_public.csv").open(encoding="utf-8-sig", newline="")
        )
    }
    sections = {
        f"{section['product']}|{section['section_id']}": section
        for section in json.loads(
            (ROOT / "data" / "section_chunks.json").read_text(encoding="utf-8")
        )
    }

    assert len(records) == 30
    assert len({record["id"] for record in records}) == 30
    for record in records:
        assert record["judgeable"] is True
        assert questions[record["id"]] == record["question"]
        assert record["relevant_sections"]
        assert {item["section_key"] for item in record["evidence"]} == set(
            record["relevant_sections"]
        )
        for item in record["evidence"]:
            section = sections[item["section_key"]]
            assert section["product"] == record["expected_product"]
            assert section["heading"] == item["heading"]
            assert item["quote"] in section["text"]

    assert freeze["record_count"] == len(records)
    assert freeze["gold_sha256"] == sha256(GOLD)
    for relative_path, expected in freeze["source_sha256"].items():
        assert sha256(ROOT / relative_path) == expected
    print(f"section_gold=OK records={len(records)} sha256={freeze['gold_sha256']}")


if __name__ == "__main__":
    main()
