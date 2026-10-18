"""Compare serial, batch, and bounded-parallel online query embedding modes."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from retrieval_engine import RetrievalEngine
from run_search_trace_baseline import append_jsonl, percentile, write_json


ROOT = Path(__file__).resolve().parent
PROGRESS_PATH = ROOT / "validation_outputs" / "search_trace_baseline_progress.json"
QUESTION = "椅子的扶手使用一段时间后为什么会松动？"
KEYWORDS = ["扶手", "松动", "螺丝", "拧紧"]
PRODUCTS = ["人体工学椅手册"]
MODES = ("serial", "batch", "parallel")


def balanced_order(repeat: int) -> list[str]:
    """Rotate mode order between repeats to reduce simple time-order bias."""
    offset = repeat % len(MODES)
    return list(MODES[offset:] + MODES[:offset])


def embed_queries(engine: RetrievalEngine, queries: list[str], mode: str) -> tuple[list[list[float]], dict]:
    """Embed the same queries using one of the three request schedules."""
    trace: dict = {"mode": mode, "query_count": len(queries), "calls": []}
    started = time.perf_counter()
    if mode == "batch":
        call: dict = {}
        vectors = engine.client.embed_texts(queries, engine.embedding_model, timing_trace=call)
        trace.update({"request_count": 1, "calls": [call]})
    elif mode == "serial":
        vectors = []
        for query in queries:
            call = {}
            vectors.append(engine.client.embed_texts([query], engine.embedding_model, timing_trace=call)[0])
            trace["calls"].append(call)
        trace["request_count"] = len(queries)
    elif mode == "parallel":
        calls = [{} for _query in queries]
        with ThreadPoolExecutor(max_workers=len(queries)) as pool:
            futures = [
                pool.submit(engine.client.embed_texts, [query], engine.embedding_model, calls[index])
                for index, query in enumerate(queries)
            ]
            vectors = [future.result()[0] for future in futures]
        trace.update({"request_count": len(queries), "calls": calls})
    else:
        raise ValueError(f"unknown mode: {mode}")
    trace["elapsed"] = round(time.perf_counter() - started, 6)
    trace["http_statuses"] = [call.get("http_status") for call in trace["calls"]]
    return vectors, trace


def search_with_mode(engine: RetrievalEngine, mode: str) -> tuple[list, dict]:
    """Run the unchanged sparse/RRF/rerank/parent pipeline with one embedding mode."""
    allowed = sorted({doc_id for product in PRODUCTS for doc_id in engine.product_chunk_ids.get(product, [])})
    sparse_query = " ".join(KEYWORDS)
    main_sparse = engine._sparse_recall(sparse_query, top_n=20, allowed_doc_ids=allowed)
    extra_sparse: list[int] = []
    for keyword in KEYWORDS:
        extra_sparse.extend(engine._sparse_recall(keyword, top_n=5, allowed_doc_ids=allowed))

    queries = [sparse_query, *KEYWORDS]
    vectors, embedding_trace = embed_queries(engine, queries, mode)
    main_dense = engine._dense_recall_from_vector(vectors[0], top_n=20, allowed_doc_ids=allowed)
    extra_dense: list[int] = []
    for vector in vectors[1:]:
        extra_dense.extend(engine._dense_recall_from_vector(vector, top_n=5, allowed_doc_ids=allowed))

    ranked_lists = [
        list(dict.fromkeys(main_sparse)),
        list(dict.fromkeys(main_dense)),
        list(dict.fromkeys(extra_sparse)),
        list(dict.fromkeys(extra_dense)),
    ]
    rrf_started = time.perf_counter()
    candidates = engine._rrf_merge(ranked_lists, top_n=engine.rerank_candidate_pool)
    rrf_elapsed = time.perf_counter() - rrf_started

    keyword_rerank: dict = {}
    keyword_top = engine._rerank_candidates(sparse_query, candidates, top_n=6, timing_trace=keyword_rerank)[:6]
    original_rerank: dict = {}
    original_top = engine._rerank_candidates(QUESTION, candidates, top_n=4, timing_trace=original_rerank)[:4]
    seen: set[str] = set()
    final_ids: list[int] = []
    for doc_id in keyword_top + original_top:
        key = engine._result_dedup_key(doc_id)
        if key in seen:
            continue
        seen.add(key)
        final_ids.append(doc_id)
    results = engine._build_results(final_ids[:8])
    trace = {
        "embedding": embedding_trace,
        "rrf": {"elapsed": round(rrf_elapsed, 6), "candidate_count": len(candidates)},
        "keyword_rerank": keyword_rerank,
        "original_rerank": original_rerank,
    }
    return results, trace


def summarize(rows: list[dict]) -> dict:
    """Aggregate latency, HTTP status, and result-signature stability by mode."""
    summary: dict[str, dict] = {}
    serial_by_repeat = {
        row["repeat"]: row["signature"]
        for row in rows
        if row["mode"] == "serial" and not row.get("error")
    }
    for mode in MODES:
        mode_rows = [row for row in rows if row["mode"] == mode]
        valid = [row for row in mode_rows if not row.get("error")]
        totals = [float(row["elapsed"]) for row in valid]
        embeddings = [float(row["timings"]["embedding"]["elapsed"]) for row in valid]
        matching = [row for row in valid if row["signature"] == serial_by_repeat.get(row["repeat"])]
        summary[mode] = {
            "runs": len(mode_rows),
            "successful": len(valid),
            "errors": len(mode_rows) - len(valid),
            "request_count_per_run": None if not valid else valid[0]["timings"]["embedding"]["request_count"],
            "total_median": None if not totals else round(statistics.median(totals), 6),
            "total_p95": None if not totals else round(percentile(totals, 0.95) or 0.0, 6),
            "total_max": None if not totals else round(max(totals), 6),
            "embedding_median": None if not embeddings else round(statistics.median(embeddings), 6),
            "embedding_p95": None if not embeddings else round(percentile(embeddings, 0.95) or 0.0, 6),
            "embedding_max": None if not embeddings else round(max(embeddings), 6),
            "serial_signature_matches": len(matching),
            "http_statuses": sorted({
                status
                for row in valid
                for status in row["timings"]["embedding"]["http_statuses"]
                if status is not None
            }),
        }
    return summary


def parse_args() -> argparse.Namespace:
    """Parse the bounded three-mode experiment settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--pause", type=float, default=1.0)
    parser.add_argument("--out-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    """Run a balanced real-service comparison while persisting frontend progress."""
    args = parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    out_dir = args.out_dir or ROOT / "experiments" / f"embedding_request_modes_{datetime.now():%Y%m%d_%H%M%S}"
    runs_path = out_dir / "runs.jsonl"
    if runs_path.exists():
        raise SystemExit(f"refusing to overwrite existing runs: {runs_path}")
    total = args.repeats * len(MODES)
    write_json(out_dir / "config.json", {
        "question": QUESTION,
        "keywords": KEYWORDS,
        "products": PRODUCTS,
        "modes": MODES,
        "repeats": args.repeats,
        "pause_seconds": args.pause,
    })
    progress = {
        "status": "loading_index",
        "current": 0,
        "total": total,
        "mode": None,
        "repeat": 0,
        "errors": 0,
        "last_elapsed": None,
        "out_dir": str(out_dir.relative_to(ROOT)).replace("\\", "/"),
        "updated_at": time.time(),
    }
    write_json(PROGRESS_PATH, progress)
    engine = RetrievalEngine()
    engine.ensure_index()

    rows: list[dict] = []
    current = 0
    for repeat in range(args.repeats):
        order = balanced_order(repeat)
        for order_index, mode in enumerate(order):
            progress.update({
                "status": "running",
                "current": current,
                "mode": mode,
                "repeat": repeat + 1,
                "updated_at": time.time(),
            })
            write_json(PROGRESS_PATH, progress)
            started = time.perf_counter()
            row = {"repeat": repeat + 1, "order_index": order_index, "mode": mode, "error": None}
            try:
                results, timings = search_with_mode(engine, mode)
                row.update({
                    "timings": timings,
                    "signature": [f"{result.product}|{result.source.get('parent_section_id')}" for result in results],
                    "headings": [result.heading for result in results],
                })
            except Exception as exc:
                row.update({"error": f"{type(exc).__name__}: {exc}", "timings": {}, "signature": [], "headings": []})
                progress["errors"] = int(progress["errors"]) + 1
            row["elapsed"] = round(time.perf_counter() - started, 6)
            rows.append(row)
            append_jsonl(runs_path, row)
            current += 1
            progress.update({"current": current, "last_elapsed": row["elapsed"], "updated_at": time.time()})
            write_json(PROGRESS_PATH, progress)
            print(
                f"run={current}/{total} repeat={repeat + 1} mode={mode} "
                f"elapsed={row['elapsed']:.3f}s error={row['error'] or 'none'}",
                flush=True,
            )
            if current < total and args.pause > 0:
                time.sleep(args.pause)

    summary = summarize(rows)
    write_json(out_dir / "summary.json", summary)
    errors = sum(item["errors"] for item in summary.values())
    progress.update({
        "status": "complete" if errors == 0 else "complete_with_errors",
        "mode": None,
        "summary": summary,
        "updated_at": time.time(),
    })
    write_json(PROGRESS_PATH, progress)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if any(item["successful"] for item in summary.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
