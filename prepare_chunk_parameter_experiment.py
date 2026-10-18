"""Prepare and validate P120/P160/P200 chunk datasets without remote calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import parse_manuals


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
GROUPS = {
    "P120": {"target": 120, "overlap": 18, "minimum": 53, "maximum": 180},
    "P160": {"target": 160, "overlap": 24, "minimum": 70, "maximum": 240},
    "P200": {"target": 200, "overlap": 30, "minimum": 88, "maximum": 300},
}


def sha256(path: Path) -> str:
    """Return a lowercase SHA-256 digest for a saved experiment input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[int], ratio: float) -> int:
    """Return the nearest-rank percentile for a non-empty integer sample."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(ratio * len(ordered)) - 1)]


def approx_units(text: str) -> int:
    """Use the production parser's whitespace/PIC-free length measure."""
    return parse_manuals.approx_units(text)


def validate_chunks(sections: list[dict], chunks: list[dict]) -> dict:
    """Validate parent/image/range invariants and return structural metrics."""
    section_lookup = {
        (section["product"], section["section_id"]): section for section in sections
    }
    chunks_by_parent: dict[tuple[str, int], list[dict]] = defaultdict(list)

    assert [chunk["chunk_id"] for chunk in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        key = (chunk["product"], chunk["parent_section_id"])
        assert key in section_lookup, f"missing parent: {key}"
        assert chunk["text"].strip(), f"empty chunk: {chunk['chunk_id']}"
        assert chunk["char_len"] == len(chunk["text"])
        assert chunk["pic_count"] == len(chunk["pics"])
        section = section_lookup[key]
        start, end = chunk.get("char_start"), chunk.get("char_end")
        assert isinstance(start, int) and isinstance(end, int)
        assert 0 <= start < end <= len(section["text"])
        assert section["text"][start:end].strip() == chunk["text"]
        assert chunk["pics"] == parse_manuals.select_pics_for_span(
            section["text"], section["pics"], start, end
        )
        chunks_by_parent[key].append(chunk)

    missing = sorted(set(section_lookup) - set(chunks_by_parent))
    assert not missing, f"sections without chunks: {missing[:10]}"

    units = [approx_units(chunk["text"]) for chunk in chunks]
    zero_unit_chunks = [chunk for chunk, unit in zip(chunks, units) if unit == 0]
    assert all(chunk["pics"] for chunk in zero_unit_chunks), (
        "zero-unit chunk without image evidence"
    )
    per_section = [len(items) for items in chunks_by_parent.values()]
    overlaps: list[int] = []
    for key, items in chunks_by_parent.items():
        section_text = section_lookup[key]["text"]
        ordered = sorted(items, key=lambda item: item["char_start"])
        for previous, current in zip(ordered, ordered[1:]):
            overlap_start = current["char_start"]
            overlap_end = min(previous["char_end"], current["char_end"])
            overlaps.append(
                approx_units(section_text[overlap_start:overlap_end])
                if overlap_end > overlap_start else 0
            )

    split_kinds = Counter(chunk["split_kind"] for chunk in chunks)
    return {
        "section_count": len(sections),
        "chunk_count": len(chunks),
        "split_kinds": dict(sorted(split_kinds.items())),
        "picture_only_chunk_count": len(zero_unit_chunks),
        "units": {
            "minimum": min(units),
            "mean": round(statistics.mean(units), 3),
            "median": statistics.median(units),
            "p95": percentile(units, 0.95),
            "maximum": max(units),
        },
        "chunks_per_section": {
            "mean": round(statistics.mean(per_section), 3),
            "median": statistics.median(per_section),
            "p95": percentile(per_section, 0.95),
            "maximum": max(per_section),
        },
        "actual_overlap_units": {
            "samples": len(overlaps),
            "minimum": min(overlaps) if overlaps else 0,
            "mean": round(statistics.mean(overlaps), 3) if overlaps else 0,
            "median": statistics.median(overlaps) if overlaps else 0,
            "p95": percentile(overlaps, 0.95) if overlaps else 0,
            "maximum": max(overlaps) if overlaps else 0,
        },
    }


def write_json(path: Path, value: object) -> None:
    """Write one deterministic UTF-8 JSON artifact."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Generate three isolated chunk datasets and refuse an invalid baseline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        raise SystemExit(f"output already exists: {out_dir}")
    out_dir.mkdir(parents=True)

    sections_path = DATA / "section_chunks.json"
    current_chunks_path = DATA / "retrieval_chunks.json"
    catalog_path = DATA / "catalog.json"
    summaries_path = DATA / "section_summaries.json"
    gold_path = ROOT / "evaluation" / "section_relevance_gold_30.jsonl"
    sections = json.loads(sections_path.read_text(encoding="utf-8"))
    current_chunks = json.loads(current_chunks_path.read_text(encoding="utf-8"))
    base_catalog = json.loads(catalog_path.read_text(encoding="utf-8"))

    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "groups": GROUPS,
        "source_sha256": {
            "prepare_chunk_parameter_experiment.py": sha256(
                ROOT / "prepare_chunk_parameter_experiment.py"
            ),
            "parse_manuals.py": sha256(ROOT / "parse_manuals.py"),
            "data/section_chunks.json": sha256(sections_path),
            "data/retrieval_chunks.json": sha256(current_chunks_path),
            "data/catalog.json": sha256(catalog_path),
            "evaluation/section_relevance_gold_30.jsonl": sha256(gold_path),
        },
        "remote_calls": False,
    }
    write_json(out_dir / "config.json", config)

    original = (
        parse_manuals.CHUNK_TARGET_UNITS,
        parse_manuals.CHUNK_OVERLAP_UNITS,
        parse_manuals.CHUNK_MIN_UNITS,
        parse_manuals.CHUNK_MAX_UNITS,
    )
    summaries: dict[str, dict] = {}
    try:
        for name, values in GROUPS.items():
            parse_manuals.CHUNK_TARGET_UNITS = values["target"]
            parse_manuals.CHUNK_OVERLAP_UNITS = values["overlap"]
            parse_manuals.CHUNK_MIN_UNITS = values["minimum"]
            parse_manuals.CHUNK_MAX_UNITS = values["maximum"]

            chunks = parse_manuals.build_retrieval_chunks(sections)
            summary = validate_chunks(sections, chunks)
            if name == "P160":
                summary["matches_current_retrieval_chunks"] = chunks == current_chunks
                assert summary["matches_current_retrieval_chunks"], (
                    "P160 output differs from data/retrieval_chunks.json"
                )

            group_dir = out_dir / name.lower()
            data_dir = group_dir / "data"
            (data_dir / "index").mkdir(parents=True)
            write_json(data_dir / "retrieval_chunks.json", chunks)
            write_json(data_dir / "section_chunks.json", sections)

            counts = Counter(chunk["product"] for chunk in chunks)
            catalog = json.loads(json.dumps(base_catalog, ensure_ascii=False))
            for product, entry in catalog.items():
                entry["retrieval_chunk_count"] = counts[product]
            write_json(data_dir / "catalog.json", catalog)
            if summaries_path.exists():
                (data_dir / "section_summaries.json").write_bytes(summaries_path.read_bytes())

            summary["retrieval_chunks_sha256"] = sha256(data_dir / "retrieval_chunks.json")
            summaries[name] = summary
            print(
                f"{name}: chunks={summary['chunk_count']} "
                f"units_median={summary['units']['median']} "
                f"overlap_median={summary['actual_overlap_units']['median']}"
            )
    finally:
        (
            parse_manuals.CHUNK_TARGET_UNITS,
            parse_manuals.CHUNK_OVERLAP_UNITS,
            parse_manuals.CHUNK_MIN_UNITS,
            parse_manuals.CHUNK_MAX_UNITS,
        ) = original

    write_json(out_dir / "prepare_summary.json", summaries)
    print(f"chunk_prepare=OK groups={len(summaries)} out_dir={out_dir}")


if __name__ == "__main__":
    main()
