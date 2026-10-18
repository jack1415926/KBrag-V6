"""Compare the frozen 30 text queries with visual retrieval off versus on."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from retrieval_engine import RetrievalEngine
from visual_retrieval import ROOT


GOLD = ROOT / "evaluation" / "section_relevance_gold_30.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    """Load required JSONL records."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    """Run interleaved baseline and visual-enabled text retrieval."""
    parser = argparse.ArgumentParser(description="视觉功能开启前后的30题文本回归")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = args.out_dir / "runs.jsonl"
    summary_path = args.out_dir / "summary.json"
    if (runs_path.exists() or summary_path.exists()) and not args.reset:
        raise FileExistsError("output already exists; use a new directory or --reset")
    for path in (runs_path, summary_path):
        if args.reset and path.exists():
            path.unlink()

    gold = load_jsonl(GOLD)
    engine = RetrievalEngine()
    engine.ensure_index()
    os.environ["VISUAL_RETRIEVAL_STRICT"] = "1"
    modes = ("baseline", "visual_enabled")
    for repeat in range(args.repeats):
        ordered_modes = modes[repeat % 2:] + modes[:repeat % 2]
        cases = list(reversed(gold)) if repeat % 2 else gold
        for mode in ordered_modes:
            os.environ["VISUAL_RETRIEVAL_ENABLED"] = "1" if mode == "visual_enabled" else "0"
            for case in cases:
                started = time.time()
                error = None
                rank = None
                try:
                    results, _ = engine.search_manual(
                        [], semantic_query=case["question"], original_query=case["question"],
                        top_k=5, products=[case["expected_product"]],
                        visual_products=[case["expected_product"]],
                    )
                    relevant = set(case["relevant_sections"])
                    keys = [f"{result.product}|{result.source['parent_section_id']}" for result in results]
                    rank = next((index for index, key in enumerate(keys, 1) if key in relevant), None)
                except Exception as exc:
                    error = str(exc)
                record = {
                    "repeat": repeat,
                    "mode": mode,
                    "id": case["id"],
                    "rank": rank,
                    "recall5": int(rank is not None and rank <= 5),
                    "rr": 0.0 if rank is None else 1.0 / rank,
                    "elapsed": round(time.time() - started, 3),
                    "error": error,
                }
                with runs_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"visual_text repeat={repeat + 1} mode={mode} id={case['id']} error={bool(error)}")

    rows = load_jsonl(runs_path)
    summary = {}
    for mode in modes:
        selected = [row for row in rows if row["mode"] == mode]
        valid = [row for row in selected if not row["error"]]
        summary[mode] = {
            "records": len(selected),
            "errors": len(selected) - len(valid),
            "recall5": sum(row["recall5"] for row in valid) / max(1, len(valid)),
            "mrr": sum(row["rr"] for row in valid) / max(1, len(valid)),
            "latency_median": statistics.median(row["elapsed"] for row in valid) if valid else 0.0,
        }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"visual_text_regression=OK records={len(rows)} summary={summary_path}")


if __name__ == "__main__":
    main()
