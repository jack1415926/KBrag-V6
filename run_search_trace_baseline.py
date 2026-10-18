"""Run a small real search_manual latency baseline and persist progress/results."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from retrieval_engine import RetrievalEngine


ROOT = Path(__file__).resolve().parent
PROGRESS_PATH = ROOT / "validation_outputs" / "search_trace_baseline_progress.json"
QUESTION = "椅子的扶手使用一段时间后为什么会松动？"
KEYWORDS = ["扶手", "松动", "螺丝", "拧紧"]
PRODUCTS = ["人体工学椅手册"]


def write_json(path: Path, payload: dict) -> None:
    """Atomically replace one UTF-8 JSON result file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict) -> None:
    """Append one durable UTF-8 JSONL record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def percentile(values: list[float], ratio: float) -> float | None:
    """Return a nearest-rank percentile for a small latency sample."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * ratio + 0.999999) - 1))
    return ordered[index]


def stage_elapsed(trace: dict, name: str) -> float | None:
    """Read one optional stage elapsed value from retrieval timings."""
    stage = trace.get(name)
    if not isinstance(stage, dict) or stage.get("elapsed") is None:
        return None
    return float(stage["elapsed"])


def summarize(rows: list[dict]) -> dict:
    """Aggregate successful total and per-stage latency samples."""
    successful = [row for row in rows if not row.get("error")]
    totals = [float(row["elapsed"]) for row in successful]
    stages: dict[str, dict] = {}
    for name in (
        "main_sparse",
        "main_dense",
        "keyword_sparse",
        "keyword_dense",
        "rrf",
        "keyword_rerank",
        "original_rerank",
        "parent_build",
    ):
        values = [value for row in successful if (value := stage_elapsed(row["retrieval_timings"], name)) is not None]
        stages[name] = {
            "samples": len(values),
            "median": None if not values else round(statistics.median(values), 6),
            "p95": None if not values else round(percentile(values, 0.95) or 0.0, 6),
            "max": None if not values else round(max(values), 6),
        }
    return {
        "runs": len(rows),
        "successful": len(successful),
        "errors": len(rows) - len(successful),
        "total_median": None if not totals else round(statistics.median(totals), 6),
        "total_p95": None if not totals else round(percentile(totals, 0.95) or 0.0, 6),
        "total_max": None if not totals else round(max(totals), 6),
        "stages": stages,
    }


def parse_args() -> argparse.Namespace:
    """Parse the bounded baseline run settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--pause", type=float, default=1.0)
    parser.add_argument("--out-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    """Execute sequential real searches while persisting visible progress."""
    args = parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be positive")
    out_dir = args.out_dir or ROOT / "experiments" / f"search_trace_baseline_{datetime.now():%Y%m%d_%H%M%S}"
    runs_path = out_dir / "runs.jsonl"
    summary_path = out_dir / "summary.json"
    if runs_path.exists():
        raise SystemExit(f"refusing to overwrite existing runs: {runs_path}")

    write_json(out_dir / "config.json", {
        "question": QUESTION,
        "keywords": KEYWORDS,
        "products": PRODUCTS,
        "runs": args.runs,
        "pause_seconds": args.pause,
    })
    progress = {
        "status": "loading_index",
        "current": 0,
        "total": args.runs,
        "errors": 0,
        "last_elapsed": None,
        "out_dir": str(out_dir.relative_to(ROOT)).replace("\\", "/"),
        "updated_at": time.time(),
    }
    write_json(PROGRESS_PATH, progress)

    engine = RetrievalEngine()
    engine.ensure_index()
    rows: list[dict] = []
    for run_index in range(1, args.runs + 1):
        progress.update({"status": "running", "current": run_index - 1, "updated_at": time.time()})
        write_json(PROGRESS_PATH, progress)
        trace: dict = {}
        started = time.perf_counter()
        row = {"run": run_index, "error": None, "retrieval_timings": trace}
        try:
            results, filtered = engine.search_manual(
                KEYWORDS,
                original_query=QUESTION,
                products=PRODUCTS,
                timing_trace=trace,
            )
            row.update({
                "filtered": filtered,
                "result_count": len(results),
                "headings": [result.heading for result in results],
            })
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            progress["errors"] = int(progress["errors"]) + 1
        row["elapsed"] = round(time.perf_counter() - started, 6)
        rows.append(row)
        append_jsonl(runs_path, row)
        progress.update({
            "current": run_index,
            "last_elapsed": row["elapsed"],
            "updated_at": time.time(),
        })
        write_json(PROGRESS_PATH, progress)
        print(
            f"run={run_index}/{args.runs} elapsed={row['elapsed']:.3f}s error={row['error'] or 'none'}",
            flush=True,
        )
        if run_index < args.runs and args.pause > 0:
            time.sleep(args.pause)

    summary = summarize(rows)
    write_json(summary_path, summary)
    progress.update({
        "status": "complete" if summary["errors"] == 0 else "complete_with_errors",
        "summary": summary,
        "updated_at": time.time(),
    })
    write_json(PROGRESS_PATH, progress)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
