"""Compare three text-query strategies with batch embedding on the frozen Section gold set."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from retrieval_engine import RetrievalEngine
from run_search_trace_baseline import append_jsonl, percentile, write_json
from run_text_query_embed_mode_gold import FREEZE, GOLD, PROGRESS_PATH, query_keywords, sha256


ROOT = Path(__file__).resolve().parent
STRATEGIES = ("full_only", "keywords_only", "hybrid")


def strategy_inputs(strategy: str, question: str, keywords: list[str]) -> tuple[list[str], str, str]:
    """Map one strategy to search_manual keywords, semantic query, and original query."""
    if strategy == "full_only":
        return [], question, question
    if strategy == "keywords_only":
        return keywords, "", question
    if strategy == "hybrid":
        return keywords, question, question
    raise ValueError(f"unknown strategy: {strategy}")


def summarize(rows: list[dict], repeats: int) -> tuple[dict, dict]:
    """Aggregate quality, latency, request shape, and paired stability by strategy."""
    summary: dict[str, dict] = {}
    for strategy in STRATEGIES:
        strategy_rows = [row for row in rows if row["strategy"] == strategy]
        valid = [row for row in strategy_rows if not row.get("error")]
        elapsed = [float(row["elapsed"]) for row in valid]
        embedding = [float(row["embedding_elapsed"]) for row in valid]
        quality_by_repeat = []
        for repeat in range(1, repeats + 1):
            batch = [row for row in valid if row["repeat"] == repeat]
            quality_by_repeat.append({
                "repeat": repeat,
                "questions": len(batch),
                "hits_at_5": sum(row["hit_at_5"] for row in batch),
                "recall_at_5": None if not batch else sum(row["hit_at_5"] for row in batch) / len(batch),
                "mrr": None if not batch else statistics.mean(row["reciprocal_rank"] for row in batch),
            })
        summary[strategy] = {
            "samples": len(valid),
            "errors": len(strategy_rows) - len(valid),
            "quality_by_repeat": quality_by_repeat,
            "mean_recall_at_5": statistics.mean(item["recall_at_5"] for item in quality_by_repeat if item["recall_at_5"] is not None),
            "mean_mrr": statistics.mean(item["mrr"] for item in quality_by_repeat if item["mrr"] is not None),
            "retrieval_median": statistics.median(elapsed),
            "retrieval_p95": percentile(elapsed, 0.95),
            "embedding_median": statistics.median(embedding),
            "embedding_p95": percentile(embedding, 0.95),
            "embedding_inputs_median": statistics.median(row["embedding_inputs"] for row in valid),
            "rerank_calls_median": statistics.median(row["rerank_calls"] for row in valid),
        }

    comparisons: dict[str, dict] = {}
    by_key = {(row["repeat"], row["id"], row["strategy"]): row for row in rows if not row.get("error")}
    ids = sorted({row["id"] for row in rows})
    for strategy in ("keywords_only", "hybrid"):
        pairs = []
        for repeat in range(1, repeats + 1):
            for question_id in ids:
                baseline = by_key.get((repeat, question_id, "full_only"))
                candidate = by_key.get((repeat, question_id, strategy))
                if baseline is None or candidate is None:
                    continue
                pairs.append({
                    "top5_equal": baseline["top_sections"] == candidate["top_sections"],
                    "rank_equal": baseline["first_relevant_rank"] == candidate["first_relevant_rank"],
                })
        comparisons[strategy] = {
            "paired_samples": len(pairs),
            "top5_equal": sum(pair["top5_equal"] for pair in pairs),
            "rank_equal": sum(pair["rank_equal"] for pair in pairs),
            "recall_delta": summary[strategy]["mean_recall_at_5"] - summary["full_only"]["mean_recall_at_5"],
            "mrr_delta": summary[strategy]["mean_mrr"] - summary["full_only"]["mean_mrr"],
            "retrieval_median_change": summary[strategy]["retrieval_median"] / summary["full_only"]["retrieval_median"] - 1.0,
        }

    hybrid_recall_pass = all(
        hybrid["recall_at_5"] >= baseline["recall_at_5"]
        for baseline, hybrid in zip(summary["full_only"]["quality_by_repeat"], summary["hybrid"]["quality_by_repeat"])
    )
    gate = {
        "hybrid_recall_not_lower_each_repeat": hybrid_recall_pass,
        "hybrid_mrr_delta_at_least_minus_0_01": comparisons["hybrid"]["mrr_delta"] >= -0.01,
        "zero_errors": all(summary[strategy]["errors"] == 0 for strategy in STRATEGIES),
        "hybrid_uses_one_rerank": summary["hybrid"]["rerank_calls_median"] == 1,
    }
    gate["passed"] = all(gate.values())
    return {"strategies": summary, "comparisons_vs_full_only": comparisons}, gate


def parse_args() -> argparse.Namespace:
    """Parse the bounded three-strategy experiment settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--pause", type=float, default=0.5)
    parser.add_argument("--out-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    """Run interleaved batch-embedding strategy comparisons with visible progress."""
    args = parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze["gold_sha256"] != sha256(GOLD):
        raise SystemExit("frozen gold hash mismatch")
    gold = [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(gold) != 30:
        raise SystemExit(f"expected 30 gold records, got {len(gold)}")

    out_dir = args.out_dir or ROOT / "experiments" / f"text_query_strategies_{datetime.now():%Y%m%d_%H%M%S}"
    runs_path = out_dir / "runs.jsonl"
    if runs_path.exists():
        raise SystemExit(f"refusing to overwrite existing runs: {runs_path}")
    keywords_by_id = {str(item["id"]): query_keywords(item["question"]) for item in gold}
    write_json(out_dir / "config.json", {
        "gold_sha256": sha256(GOLD),
        "strategies": STRATEGIES,
        "embedding_mode": "batch",
        "repeats": args.repeats,
        "question_count": len(gold),
        "keywords_by_id": keywords_by_id,
    })

    total = len(gold) * len(STRATEGIES) * args.repeats
    progress = {
        "status": "loading_index",
        "current": 0,
        "total": total,
        "mode": None,
        "repeat": 0,
        "question_id": None,
        "errors": 0,
        "last_elapsed": None,
        "out_dir": str(out_dir.relative_to(ROOT)).replace("\\", "/"),
        "updated_at": time.time(),
    }
    write_json(PROGRESS_PATH, progress)
    engine = RetrievalEngine(text_query_embed_mode="batch")
    engine.ensure_index()

    rows: list[dict] = []
    current = 0
    for repeat_index in range(args.repeats):
        questions = gold if repeat_index % 2 == 0 else list(reversed(gold))
        for question_index, item in enumerate(questions):
            offset = (repeat_index + question_index) % len(STRATEGIES)
            strategies = STRATEGIES[offset:] + STRATEGIES[:offset]
            keywords = keywords_by_id[str(item["id"])]
            for strategy in strategies:
                selected_keywords, semantic_query, original_query = strategy_inputs(
                    strategy, item["question"], keywords
                )
                progress.update({
                    "status": "running",
                    "current": current,
                    "mode": strategy,
                    "repeat": repeat_index + 1,
                    "question_id": item["id"],
                    "updated_at": time.time(),
                })
                write_json(PROGRESS_PATH, progress)
                trace: dict = {}
                started = time.perf_counter()
                row = {
                    "repeat": repeat_index + 1,
                    "id": item["id"],
                    "strategy": strategy,
                    "keywords": selected_keywords,
                    "error": None,
                }
                try:
                    results, _ = engine.search_manual(
                        selected_keywords,
                        semantic_query=semantic_query,
                        original_query=original_query,
                        top_k=5,
                        products=[item["expected_product"]],
                        timing_trace=trace,
                    )
                    top_sections = [
                        f"{result.product}|{result.source['parent_section_id']}"
                        for result in results
                    ]
                    rank = next(
                        (index for index, key in enumerate(top_sections, start=1) if key in item["relevant_sections"]),
                        None,
                    )
                    batch_trace = trace.get("dense_batch") or {}
                    rerank_calls = int(bool((trace.get("keyword_rerank") or {}).get("executed"))) + int(
                        bool((trace.get("original_rerank") or {}).get("executed"))
                    )
                    row.update({
                        "top_sections": top_sections,
                        "first_relevant_rank": rank,
                        "hit_at_5": int(rank is not None),
                        "reciprocal_rank": 1.0 / rank if rank else 0.0,
                        "embedding_elapsed": float(batch_trace.get("elapsed") or 0.0),
                        "embedding_inputs": int(batch_trace.get("input_count") or 0),
                        "embedding_http_status": batch_trace.get("http_status"),
                        "rerank_calls": rerank_calls,
                    })
                except Exception as exc:
                    row.update({
                        "error": f"{type(exc).__name__}: {exc}",
                        "top_sections": [],
                        "first_relevant_rank": None,
                        "hit_at_5": 0,
                        "reciprocal_rank": 0.0,
                        "embedding_elapsed": 0.0,
                        "embedding_inputs": 0,
                        "rerank_calls": 0,
                    })
                    progress["errors"] = int(progress["errors"]) + 1
                row["elapsed"] = round(time.perf_counter() - started, 6)
                rows.append(row)
                append_jsonl(runs_path, row)
                current += 1
                progress.update({"current": current, "last_elapsed": row["elapsed"], "updated_at": time.time()})
                write_json(PROGRESS_PATH, progress)
                print(
                    f"run={current}/{total} repeat={repeat_index + 1} strategy={strategy} "
                    f"id={item['id']} elapsed={row['elapsed']:.3f}s error={row['error'] or 'none'}",
                    flush=True,
                )
                if current < total and args.pause > 0:
                    time.sleep(args.pause)

    summary, gate = summarize(rows, args.repeats)
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "gate_result.json", gate)
    progress.update({
        "status": "complete" if gate["zero_errors"] else "complete_with_errors",
        "mode": None,
        "summary": summary,
        "gate": gate,
        "updated_at": time.time(),
    })
    write_json(PROGRESS_PATH, progress)
    print(json.dumps({"summary": summary, "gate": gate}, ensure_ascii=False), flush=True)
    return 0 if gate["zero_errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
