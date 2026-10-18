"""Compare a few rerank candidate-pool sizes on the frozen Section gold set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from retrieval_engine import RetrievalEngine


ROOT = Path(__file__).resolve().parent
GOLD = ROOT / "evaluation" / "section_relevance_gold_30.jsonl"
FREEZE = ROOT / "evaluation" / "section_relevance_gold_30.freeze.json"


def sha256(path: Path) -> str:
    """Return a file's lowercase SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], ratio: float) -> float:
    """Return a nearest-rank percentile for a non-empty sample."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(ratio * len(ordered)) - 1)]


def summarize(records: list[dict], pools: list[int], repeats: int) -> dict:
    """Aggregate quality, latency, candidate counts, and failures by pool."""
    summary = {}
    for pool in pools:
        rows = [row for row in records if row["pool"] == pool]
        valid = [row for row in rows if not row.get("error")]
        elapsed = [row["elapsed_s"] for row in valid]
        rerank_elapsed = [row["rerank_elapsed_s"] for row in valid]
        by_repeat = []
        for repeat in range(repeats):
            batch = [row for row in valid if row["repeat"] == repeat]
            by_repeat.append({
                "repeat": repeat + 1,
                "questions": len(batch),
                "hits_at_5": sum(row["hit_at_5"] for row in batch),
                "recall_at_5": sum(row["hit_at_5"] for row in batch) / len(batch) if batch else None,
                "mrr": statistics.mean(row["reciprocal_rank"] for row in batch) if batch else None,
            })
        summary[f"R{pool}"] = {
            "samples": len(valid),
            "failures": len(rows) - len(valid),
            "rerank_fallbacks": sum(bool(row.get("rerank_error")) for row in rows),
            "quality_by_repeat": by_repeat,
            "mean_recall_at_5": statistics.mean(item["recall_at_5"] for item in by_repeat if item["recall_at_5"] is not None),
            "mean_mrr": statistics.mean(item["mrr"] for item in by_repeat if item["mrr"] is not None),
            "retrieval_median_s": statistics.median(elapsed),
            "retrieval_p95_s": percentile(elapsed, 0.95),
            "rerank_median_s": statistics.median(rerank_elapsed),
            "rerank_p95_s": percentile(rerank_elapsed, 0.95),
            "candidate_median": statistics.median(row["candidate_count"] for row in valid),
            "candidate_max": max(row["candidate_count"] for row in valid),
        }
    return summary


def main() -> None:
    """Run interleaved live retrievals and save raw and aggregate artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pools", default="80,20,10")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    pools = [int(value) for value in args.pools.split(",")]
    if len(pools) not in {2, 3} or any(value < 1 for value in pools):
        raise SystemExit("--pools must contain two or three positive integers")
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")

    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        raise SystemExit(f"output already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    raw_path = out_dir / "runs.jsonl"

    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze["gold_sha256"] != sha256(GOLD):
        raise SystemExit("frozen gold hash mismatch")
    gold = [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines()]
    if len(gold) != 30:
        raise SystemExit(f"expected 30 gold records, got {len(gold)}")

    engine = RetrievalEngine()
    engine.ensure_index()
    rerank_calls: list[dict] = []
    original_rerank = engine.rerank_client.rerank

    def timed_rerank(*, query: str, documents: list[str], top_n: int | None = None):
        started = time.perf_counter()
        call = {"candidate_count": len(documents), "error": ""}
        try:
            return original_rerank(query=query, documents=documents, top_n=top_n)
        except Exception as exc:
            call["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            call["elapsed_s"] = round(time.perf_counter() - started, 6)
            rerank_calls.append(call)

    engine.rerank_client.rerank = timed_rerank
    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "pools": pools,
        "repeats": args.repeats,
        "question_count": len(gold),
        "query_method": "keywords=[], semantic_query=question, original_query=question, top_k=5, products=[expected_product]",
        "quality_metrics": ["Section Recall@5", "MRR"],
        "latency_metrics": ["full retrieval median/P95", "rerank median/P95"],
        "gold_sha256": sha256(GOLD),
        "source_sha256": {
            "retrieval_engine.py": sha256(ROOT / "retrieval_engine.py"),
            "data/index/dense.faiss": sha256(ROOT / "data" / "index" / "dense.faiss"),
            "data/index/retrieval_index.pkl": sha256(ROOT / "data" / "index" / "retrieval_index.pkl"),
        },
        "embedding_model": engine.embedding_model,
        "rerank_model": engine.rerank_client.model,
    }
    (out_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    records = []
    for repeat in range(args.repeats):
        pool_order = pools[repeat % len(pools):] + pools[:repeat % len(pools)]
        questions = gold if repeat % 2 == 0 else list(reversed(gold))
        for pool in pool_order:
            engine.rerank_candidate_pool = pool
            for item in questions:
                rerank_calls.clear()
                started = time.perf_counter()
                record = {"repeat": repeat, "pool": pool, "id": item["id"]}
                try:
                    results, _ = engine.search_manual(
                        [],
                        semantic_query=item["question"],
                        original_query=item["question"],
                        top_k=5,
                        products=[item["expected_product"]],
                    )
                    section_keys = [
                        f"{result.product}|{result.source['parent_section_id']}"
                        for result in results
                    ]
                    rank = next(
                        (index for index, key in enumerate(section_keys, start=1) if key in item["relevant_sections"]),
                        None,
                    )
                    rerank_call = rerank_calls[-1] if rerank_calls else {
                        "candidate_count": 0, "elapsed_s": 0.0, "error": "missing rerank call"
                    }
                    record.update({
                        "top_sections": section_keys,
                        "first_relevant_rank": rank,
                        "hit_at_5": int(rank is not None),
                        "reciprocal_rank": 1.0 / rank if rank else 0.0,
                        "candidate_count": rerank_call["candidate_count"],
                        "rerank_elapsed_s": rerank_call["elapsed_s"],
                        "rerank_error": rerank_call["error"],
                        "error": "",
                    })
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                record["elapsed_s"] = round(time.perf_counter() - started, 6)
                records.append(record)
                with raw_path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"repeat={repeat + 1} pool={pool} id={item['id']} elapsed={record['elapsed_s']:.3f}s")

    summary = summarize(records, pools, args.repeats)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
