"""Evaluate P120/P160/P200 indexes on the frozen 30-question Section gold set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from retrieval_engine import RetrievalEngine


ROOT = Path(__file__).resolve().parent
GROUPS = ("P120", "P160", "P200")
GOLD = ROOT / "evaluation" / "section_relevance_gold_30.jsonl"
FREEZE = ROOT / "evaluation" / "section_relevance_gold_30.freeze.json"


def sha256(path: Path) -> str:
    """Return a lowercase SHA-256 digest for an experiment input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], ratio: float) -> float:
    """Return the nearest-rank percentile for a non-empty sample."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(ratio * len(ordered)) - 1)]


def quality(rows: list[dict]) -> dict:
    """Compute Section recall and reciprocal-rank metrics for one batch."""
    return {
        "questions": len(rows),
        "hits_at_1": sum(row["hit_at_1"] for row in rows),
        "hits_at_3": sum(row["hit_at_3"] for row in rows),
        "hits_at_5": sum(row["hit_at_5"] for row in rows),
        "recall_at_1": statistics.mean(row["hit_at_1"] for row in rows),
        "recall_at_3": statistics.mean(row["hit_at_3"] for row in rows),
        "recall_at_5": statistics.mean(row["hit_at_5"] for row in rows),
        "mrr": statistics.mean(row["reciprocal_rank"] for row in rows),
    }


def summarize(records: list[dict], repeats: int) -> dict:
    """Aggregate quality, latency, parent diversity, failures, and stability."""
    summary: dict[str, dict] = {}
    for group in GROUPS:
        rows = [row for row in records if row["group"] == group]
        valid = [row for row in rows if not row.get("error")]
        by_repeat = [
            quality([row for row in valid if row["repeat"] == repeat])
            for repeat in range(repeats)
        ]
        retrieval_elapsed = [row["elapsed_s"] for row in valid]
        rerank_elapsed = [row["rerank_elapsed_s"] for row in valid]
        ratios = [row["candidate_unique_parent_ratio"] for row in valid]
        unstable = 0
        for qid in {row["id"] for row in valid}:
            qrows = [row for row in valid if row["id"] == qid]
            signatures = {
                (row["first_relevant_rank"], tuple(row["top_sections"])) for row in qrows
            }
            unstable += len(signatures) > 1
        summary[group] = {
            "samples": len(valid),
            "failures": len(rows) - len(valid),
            "rerank_fallbacks": sum(bool(row.get("rerank_error")) for row in rows),
            "quality_by_repeat": by_repeat,
            "mean_recall_at_1": statistics.mean(item["recall_at_1"] for item in by_repeat),
            "mean_recall_at_3": statistics.mean(item["recall_at_3"] for item in by_repeat),
            "mean_recall_at_5": statistics.mean(item["recall_at_5"] for item in by_repeat),
            "mean_mrr": statistics.mean(item["mrr"] for item in by_repeat),
            "retrieval_median_s": statistics.median(retrieval_elapsed),
            "retrieval_p95_s": percentile(retrieval_elapsed, 0.95),
            "rerank_median_s": statistics.median(rerank_elapsed),
            "rerank_p95_s": percentile(rerank_elapsed, 0.95),
            "candidate_median": statistics.median(row["candidate_count"] for row in valid),
            "candidate_max": max(row["candidate_count"] for row in valid),
            "unique_parent_ratio_mean": statistics.mean(ratios),
            "unique_parent_ratio_median": statistics.median(ratios),
            "unstable_question_count": unstable,
        }
    return summary


def load_engine(experiment_dir: Path, group: str) -> RetrievalEngine:
    """Load one group's isolated BM25, FAISS, and metadata artifacts."""
    data_dir = experiment_dir / group.lower() / "data"
    engine = RetrievalEngine(
        retrieval_chunks_path=data_dir / "retrieval_chunks.json",
        section_chunks_path=data_dir / "section_chunks.json",
        catalog_path=data_dir / "catalog.json",
        index_dir=data_dir / "index",
    )
    engine.ensure_index()
    return engine


def instrument_rerank(engine: RetrievalEngine, events: list[dict]) -> None:
    """Record the real RRF candidate IDs without duplicating search logic."""
    original_candidates = engine._rerank_candidates
    original_remote = engine.rerank_client.rerank
    remote_errors: list[str] = []

    def observed_remote(*, query: str, documents: list[str], top_n: int | None = None):
        try:
            return original_remote(query=query, documents=documents, top_n=top_n)
        except Exception as exc:
            remote_errors.append(f"{type(exc).__name__}: {exc}")
            raise

    def observed_candidates(query: str, candidate_ids: list[int], top_n: int):
        started = time.perf_counter()
        remote_errors.clear()
        result = original_candidates(query, candidate_ids, top_n)
        parent_keys = [engine._result_dedup_key(doc_id) for doc_id in candidate_ids]
        events.append({
            "candidate_count": len(candidate_ids),
            "candidate_parent_keys": parent_keys,
            "unique_parent_count": len(set(parent_keys)),
            "elapsed_s": round(time.perf_counter() - started, 6),
            "error": remote_errors[-1] if remote_errors else "",
        })
        return result

    engine.rerank_client.rerank = observed_remote
    engine._rerank_candidates = observed_candidates


def main() -> None:
    """Run interleaved retrievals and save raw records plus aggregate metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")

    experiment_dir = Path(args.experiment_dir).resolve()
    build_summary_path = experiment_dir / "build_summary.json"
    if not build_summary_path.exists():
        raise SystemExit("missing build_summary.json; build all indexes first")
    out_dir = experiment_dir / "retrieval_eval"
    if out_dir.exists():
        raise SystemExit(f"output already exists: {out_dir}")
    out_dir.mkdir()
    raw_path = out_dir / "runs.jsonl"

    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze["gold_sha256"] != sha256(GOLD):
        raise SystemExit("frozen gold hash mismatch")
    gold = [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines()]
    if len(gold) != 30:
        raise SystemExit(f"expected 30 gold records, got {len(gold)}")

    engines = {group: load_engine(experiment_dir, group) for group in GROUPS}
    events = {group: [] for group in GROUPS}
    for group, engine in engines.items():
        instrument_rerank(engine, events[group])

    build_summary = json.loads(build_summary_path.read_text(encoding="utf-8"))
    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "groups": list(GROUPS),
        "repeats": args.repeats,
        "question_count": len(gold),
        "query_method": "keywords=[], semantic_query=question, original_query=question, top_k=5, products=[expected_product]",
        "rerank_candidate_pool": {group: engines[group].rerank_candidate_pool for group in GROUPS},
        "embedding_model": {group: engines[group].embedding_model for group in GROUPS},
        "rerank_model": {group: engines[group].rerank_client.model for group in GROUPS},
        "gold_sha256": sha256(GOLD),
        "build_summary_sha256": sha256(build_summary_path),
        "source_sha256": {"run_chunk_parameter_experiment.py": sha256(Path(__file__))},
        "index_sha256": {
            group: {
                "dense.faiss": build_summary["groups"][group]["dense_faiss_sha256"],
                "retrieval_index.pkl": build_summary["groups"][group]["retrieval_index_sha256"],
            }
            for group in GROUPS
        },
    }
    (out_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    records: list[dict] = []
    for repeat in range(args.repeats):
        group_order = GROUPS[repeat % len(GROUPS):] + GROUPS[:repeat % len(GROUPS)]
        questions = gold if repeat % 2 == 0 else list(reversed(gold))
        for group in group_order:
            engine = engines[group]
            for item in questions:
                events[group].clear()
                started = time.perf_counter()
                record = {"repeat": repeat, "group": group, "id": item["id"]}
                try:
                    results, _ = engine.search_manual(
                        [],
                        semantic_query=item["question"],
                        original_query=item["question"],
                        top_k=5,
                        products=[item["expected_product"]],
                    )
                    if len(events[group]) != 1:
                        raise RuntimeError(f"expected one rerank call, got {len(events[group])}")
                    event = events[group][0]
                    section_keys = [
                        f"{result.product}|{result.source['parent_section_id']}"
                        for result in results
                    ]
                    rank = next(
                        (index for index, key in enumerate(section_keys, start=1)
                         if key in item["relevant_sections"]),
                        None,
                    )
                    record.update({
                        "top_sections": section_keys,
                        "matched_chunk_ids": [result.source["matched_chunk_id"] for result in results],
                        "candidate_parent_keys": event["candidate_parent_keys"],
                        "first_relevant_rank": rank,
                        "hit_at_1": int(rank == 1),
                        "hit_at_3": int(rank is not None and rank <= 3),
                        "hit_at_5": int(rank is not None and rank <= 5),
                        "reciprocal_rank": 1.0 / rank if rank else 0.0,
                        "candidate_count": event["candidate_count"],
                        "candidate_unique_parent_count": event["unique_parent_count"],
                        "candidate_unique_parent_ratio": (
                            event["unique_parent_count"] / event["candidate_count"]
                        ),
                        "rerank_elapsed_s": event["elapsed_s"],
                        "rerank_error": event["error"],
                        "error": "",
                    })
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                record["elapsed_s"] = round(time.perf_counter() - started, 6)
                records.append(record)
                with raw_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(
                    f"repeat={repeat + 1} group={group} id={item['id']} "
                    f"elapsed={record['elapsed_s']:.3f}s"
                )

    summary = summarize(records, args.repeats)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
