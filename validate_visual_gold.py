"""Validate derived and external-style visual relevance records."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from visual_retrieval import ROOT


DERIVED = ROOT / "evaluation" / "visual_relevance_gold_30_derived.jsonl"
EXTERNAL = ROOT / "evaluation" / "visual_relevance_gold_10_external.jsonl"


def sha256(path: Path) -> str:
    """Return a file fingerprint."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> list[dict]:
    """Load JSONL records when the optional file exists."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate(
    records: list[dict],
    sections: set[str],
    image_sections: dict[tuple[str, str], set[str]],
    case_ids: set[str],
    image_hashes: set[str],
) -> None:
    """Validate paths, labels, uniqueness, and derived-image independence."""
    for record in records:
        case_id = str(record["case_id"])
        assert case_id not in case_ids
        case_ids.add(case_id)
        assert record["judgeable"] is True
        assert record["expected_product"]
        assert record["relevant_sections"]
        assert record["relevant_images"]
        assert set(record["relevant_sections"]) <= sections
        image = ROOT / record["query_image"]
        assert image.is_file()
        assert image.stat().st_size <= 5 * 1024 * 1024
        digest = sha256(image)
        assert digest not in image_hashes
        image_hashes.add(digest)
        for image_id in record["relevant_images"]:
            mapped = image_sections[(record["expected_product"], image_id)]
            assert mapped.intersection(record["relevant_sections"])
        if record["source_type"] == "derived":
            source = ROOT / record["source_image"]
            assert source.is_file()
            assert digest != sha256(source)


def main() -> None:
    """Validate the fixed 30 derived plus 10 external-style records."""
    sections = {
        f"{item['product']}|{item['section_id']}"
        for item in json.loads((ROOT / "data" / "section_chunks.json").read_text(encoding="utf-8"))
    }
    metadata = json.loads(
        (ROOT / "data" / "visual_index" / "visual_metadata.json").read_text(encoding="utf-8")
    )["items"]
    image_sections = {
        (item["product"], item["image_id"]): set(item["section_keys"])
        for item in metadata
    }
    derived = load(DERIVED)
    external = load(EXTERNAL)
    assert len(derived) == 30
    assert Counter(record["category"] for record in derived) == {
        "part_view": 10,
        "schematic": 10,
        "info_table": 10,
    }
    assert len(external) == 10
    assert Counter(record["source_type"] for record in external) == {
        "synthetic": 6,
        "user_photo": 1,
        "public_photo": 3,
    }
    assert Counter(record["category"] for record in external) == {
        "part_view": 3,
        "procedure": 4,
        "ocr": 3,
    }
    assert all(record["source_group"] == "external_style" for record in external)
    assert len({record["expected_product"] for record in external}) >= 5
    assert all(record.get("distributable") is False for record in external if record["source_type"] == "public_photo")
    validate(derived + external, sections, image_sections, set(), set())
    print(
        "visual_gold=OK "
        f"derived={len(derived)} external={len(external)} "
        "synthetic=6 user_photo=1 public_photo=3 total=40"
    )


if __name__ == "__main__":
    main()
