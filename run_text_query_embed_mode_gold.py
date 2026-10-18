"""Compare serial and batch query embedding on the frozen 30-question Section gold set."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from retrieval_engine import RetrievalEngine, tokenize_mixed
from run_search_trace_baseline import append_jsonl, percentile, write_json


ROOT = Path(__file__).resolve().parent
GOLD = ROOT / "evaluation" / "section_relevance_gold_30.jsonl"
FREEZE = ROOT / "evaluation" / "section_relevance_gold_30.freeze.json"
PROGRESS_PATH = ROOT / "validation_outputs" / "search_trace_baseline_progress.json"
MODES = ("serial", "batch")
_GENERIC_QUERY_TOKENS = {"如何", "什么", "哪些", "需要", "是否", "可以", "怎么", "怎样", "使用"}


def sha256(path: Path) -> str:
    """Return one lowercase file digest for experiment provenance."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def query_keywords(question: str, limit: int = 4) -> list[str]:
    """Choose deterministic question-only keywords without using gold evidence."""
    keywords: list[str] = []
    for token in tokenize_mixed(question):
        cleaned = token.strip()
        if len(cleaned) < 2 or cleaned in _GENERIC_QUERY_TOKENS or cleaned in keywords:
            continue
        keywords.append(cleaned)
        if len(keywords) >= limit:
            break
    return keywords


def embedding_elapsed(trace: dict) -> float:
    """Read total remote query-embedding time for either configured mode."""
    if trace.get("text_query_embed_mode") == "batch":
        return float((trace.get("dense_batch") or {}).get("elapsed") or 0.0)
    return float((trace.get("main_dense") or {}).get("elapsed") or 0.0) + float(
        (trace.get("keyword_dense") or {}).get("elapsed") or 0.0
    )


def summarize(rows: list[dict], repeats: int) -> tuple[dict, dict]:
    """Aggregate quality, latency, request counts, and paired Top-5 stability."""
    summary: dict[str, dict] = {}
    for mode in MODES:
        mode_rows = [row for row in rows if row["mode"] == mode]
        valid = [row for row in mode_rows if not row.get("error")]
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
        summary[mode] = {
            "samples": len(valid),
            "errors": len(mode_rows) - len(valid),
            "quality_by_repeat": quality_by_repeat,
            "mean_recall_at_5": statistics.mean(item["recall_at_5"] for item in quality_by_repeat if item["recall_at_5"] is not None),
            "mean_mrr": statistics.mean(item["mrr"] for item in quality_by_repeat if item["mrr"] is not None),
            "retrieval_median": statistics.median(elapsed),
            "retrieval_p95": percentile(elapsed, 0.95),
            "embedding_median": statistics.median(embedding),
            "embedding_p95": percentile(embedding, 0.95),
            "embedding_requests": sum(row["embedding_requests"] for row in valid),
        }

    paired = []
    by_key = {(row["repeat"], row["id"], row["mode"]): row for row in rows if not row.get("error")}
    for repeat in range(1, repeats + 1):
        for question_id in sorted({row["id"] for row in rows}):
            serial = by_key.get((repeat, question_id, "serial"))
            batch = by_key.get((repeat, question_id, "batch"))
            if serial is None or batch is None:
                continue
            paired.append({
                "repeat": repeat,
                "id": question_id,
                "top5_equal": serial["top_sections"] == batch["top_sections"],
                "rank_equal": serial["first_relevant_rank"] == batch["first_relevant_rank"],
            })
    comparison = {
        "paired_samples": len(paired),
        "top5_equal": sum(item["top5_equal"] for item in paired),
        "rank_equal": sum(item["rank_equal"] for item in paired),
        "batch_minus_serial_mrr": summary["batch"]["mean_mrr"] - summary["serial"]["mean_mrr"],
        "batch_retrieval_median_change": summary["batch"]["retrieval_median"] / summary["serial"]["retrieval_median"] - 1.0,
        "batch_embedding_median_change": summary["batch"]["embedding_median"] / summary["serial"]["embedding_median"] - 1.0,
    }
    recall_pass = all(
        batch["recall_at_5"] >= serial["recall_at_5"]
        for serial, batch in zip(summary["serial"]["quality_by_repeat"], summary["batch"]["quality_by_repeat"])
    )
    gate = {
        "recall_not_lower_each_repeat": recall_pass,
        "mrr_delta_at_least_minus_0_01": comparison["batch_minus_serial_mrr"] >= -0.01,
        "zero_errors": summary["serial"]["errors"] == 0 and summary["batch"]["errors"] == 0,
        "batch_latency_lower": summary["batch"]["retrieval_median"] < summary["serial"]["retrieval_median"],
    }
    gate["passed"] = all(gate.values())
    return {"modes": summary, "comparison": comparison}, gate


def parse_args() -> argparse.Namespace:
    """Parse the bounded fixed-gold experiment settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--pause", type=float, default=0.5)
    parser.add_argument("--out-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    """Run interleaved serial/batch retrievals and persist frontend progress."""
    args = parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze["gold_sha256"] != sha256(GOLD):
        raise SystemExit("frozen gold hash mismatch")
    gold = [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(gold) != 30:
        raise SystemExit(f"expected 30 gold records, got {len(gold)}")

    out_dir = args.out_dir or ROOT / "experiments" / f"text_query_embed_modes_{datetime.now():%Y%m%d_%H%M%S}"
    runs_path = out_dir / "runs.jsonl"
    if runs_path.exists():
        raise SystemExit(f"refusing to overwrite existing runs: {runs_path}")
    keywords_by_id = {str(item["id"]): query_keywords(item["question"]) for item in gold}
    write_json(out_dir / "config.json", {
        "gold_sha256": sha256(GOLD),
        "modes": MODES,
        "repeats": args.repeats,
        "question_count": len(gold),
        "keyword_policy": "first four unique non-generic tokenize_mixed terms from question only",
        "keywords_by_id": keywords_by_id,
        "current_default": "serial",
    })

    total = len(gold) * len(MODES) * args.repeats
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
    engine = RetrievalEngine(text_query_embed_mode="serial")
    engine.ensure_index()

    rows: list[dict] = []
    current = 0
    for repeat_index in range(args.repeats):
        questions = gold if repeat_index % 2 == 0 else list(reversed(gold))
        for question_index, item in enumerate(questions):
            modes = MODES if (repeat_index + question_index) % 2 == 0 else tuple(reversed(MODES))
            for mode in modes:
                engine.text_query_embed_mode = mode
                progress.update({
                    "status": "running",
                    "current": current,
                    "mode": mode,
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
                    "mode": mode,
                    "keywords": keywords_by_id[str(item["id"])],
                    "error": None,
                }
                try:
                    results, _ = engine.search_manual(
                        row["keywords"],
                        original_query=item["question"],
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
                    serial_calls = 1 + len(row["keywords"])
                    row.update({
                        "top_sections": top_sections,
                        "first_relevant_rank": rank,
                        "hit_at_5": int(rank is not None),
                        "reciprocal_rank": 1.0 / rank if rank else 0.0,
                        "embedding_elapsed": embedding_elapsed(trace),
                        "embedding_requests": 1 if mode == "batch" else serial_calls,
                        "embedding_http_status": batch_trace.get("http_status") if mode == "batch" else [
                            (trace.get("main_dense") or {}).get("http_status"),
                            *[(call or {}).get("http_status") for call in (trace.get("keyword_dense") or {}).get("calls", [])],
                        ],
                    })
                except Exception as exc:
                    row.update({
                        "error": f"{type(exc).__name__}: {exc}",
                        "top_sections": [],
                        "first_relevant_rank": None,
                        "hit_at_5": 0,
                        "reciprocal_rank": 0.0,
                        "embedding_elapsed": embedding_elapsed(trace),
                        "embedding_requests": 0,
                    })
                    progress["errors"] = int(progress["errors"]) + 1
                row["elapsed"] = round(time.perf_counter() - started, 6)
                rows.append(row)
                append_jsonl(runs_path, row)
                current += 1
                progress.update({"current": current, "last_elapsed": row["elapsed"], "updated_at": time.time()})
                write_json(PROGRESS_PATH, progress)
                print(
                    f"run={current}/{total} repeat={repeat_index + 1} mode={mode} "
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
