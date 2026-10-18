"""Build isolated P120/P160/P200 indexes for the chunk parameter experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import faiss

from retrieval_engine import RetrievalEngine


ROOT = Path(__file__).resolve().parent
GROUPS = ("p120", "p160", "p200")


def sha256(path: Path) -> str:
    """Return a lowercase SHA-256 digest for one index artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    """Persist build progress after each completed group."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def inspect_index(index_dir: Path, expected_chunks: list[dict]) -> dict:
    """Verify model, metadata corpus, FAISS count, and vector dimension."""
    faiss_path = index_dir / "dense.faiss"
    metadata_path = index_dir / "retrieval_index.pkl"
    with metadata_path.open("rb") as stream:
        metadata = pickle.load(stream)
    assert metadata["retrieval_chunks"] == expected_chunks
    index = faiss.read_index(str(faiss_path))
    assert index.ntotal == len(expected_chunks)
    assert metadata["dense_vectors"].shape == (len(expected_chunks), index.d)
    return {
        "embedding_model": metadata["embedding_model"],
        "chunk_count": len(expected_chunks),
        "dimension": index.d,
        "faiss_bytes": faiss_path.stat().st_size,
        "metadata_bytes": metadata_path.stat().st_size,
        "dense_faiss_sha256": sha256(faiss_path),
        "retrieval_index_sha256": sha256(metadata_path),
    }


def main() -> None:
    """Build two new indexes, reuse the identical P160 baseline, and verify all."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")

    experiment_dir = Path(args.experiment_dir).resolve()
    if not (experiment_dir / "prepare_summary.json").exists():
        raise SystemExit("missing prepare_summary.json; run local preparation first")

    summary_path = experiment_dir / "build_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if args.resume and summary_path.exists() else {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "batch_size": args.batch_size,
        "groups": {},
    }

    for group in GROUPS:
        data_dir = experiment_dir / group / "data"
        index_dir = data_dir / "index"
        chunks_path = data_dir / "retrieval_chunks.json"
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        outputs = (index_dir / "dense.faiss", index_dir / "retrieval_index.pkl")
        existing = [path.exists() for path in outputs]
        if any(existing) and not all(existing):
            raise SystemExit(f"partial index exists for {group}: {index_dir}")
        if all(existing):
            if not args.resume:
                raise SystemExit(f"index already exists for {group}: {index_dir}")
            result = inspect_index(index_dir, chunks)
            result["status"] = "verified_existing"
            summary["groups"][group.upper()] = result
            write_json(summary_path, summary)
            print(f"{group.upper()}: verified existing index")
            continue

        started = time.perf_counter()
        if group == "p160":
            current_chunks = json.loads((ROOT / "data" / "retrieval_chunks.json").read_text(encoding="utf-8"))
            assert chunks == current_chunks, "P160 chunks no longer match the current corpus"
            shutil.copy2(ROOT / "data" / "index" / "dense.faiss", outputs[0])
            shutil.copy2(ROOT / "data" / "index" / "retrieval_index.pkl", outputs[1])
            reused_current = True
        else:
            engine = RetrievalEngine(
                retrieval_chunks_path=chunks_path,
                section_chunks_path=data_dir / "section_chunks.json",
                catalog_path=data_dir / "catalog.json",
                index_dir=index_dir,
            )
            engine.build_index(batch_size=args.batch_size)
            reused_current = False

        result = inspect_index(index_dir, chunks)
        result.update({
            "status": "built",
            "reused_current_index": reused_current,
            "elapsed_s": round(time.perf_counter() - started, 3),
        })
        summary["groups"][group.upper()] = result
        write_json(summary_path, summary)
        print(
            f"{group.upper()}: chunks={result['chunk_count']} dim={result['dimension']} "
            f"elapsed={result['elapsed_s']}s reused_current={reused_current}"
        )

    summary["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(summary_path, summary)
    print(f"chunk_index_build=OK groups={len(summary['groups'])}")


if __name__ == "__main__":
    main()
