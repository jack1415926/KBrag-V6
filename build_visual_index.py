"""Build the independent Qwen3-VL manual-image FAISS index.

This command sends eligible manual images to the configured remote embedding
service. Run ``--dry-run`` first; do not run a real build without data-egress
authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import faiss
import numpy as np
from numpy.lib.format import open_memmap

from visual_retrieval import (
    DEFAULT_VISUAL_EMBEDDING_MODEL,
    IMAGE_DIR,
    ROOT,
    VISUAL_DIMENSION,
    VISUAL_FAISS_PATH,
    VISUAL_INDEX_DIR,
    VISUAL_METADATA_PATH,
    VisualEmbeddingClient,
    file_to_data_url,
)


SECTION_PATH = ROOT / "data" / "section_chunks.json"
CAPTION_PATH = ROOT / "data" / "image_captions_v4_final.json"
SUMMARY_PATH = VISUAL_INDEX_DIR / "visual_build_summary.json"
PROGRESS_PATH = VISUAL_INDEX_DIR / "visual_build_progress.jsonl"
PARTIAL_VECTORS_PATH = VISUAL_INDEX_DIR / ".visual_vectors.partial.npy"


def sha256(path: Path) -> str:
    """Return a file fingerprint for build provenance."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_items() -> tuple[list[dict], dict[str, int]]:
    """Collect section-linked, non-noise images in stable order."""
    sections = json.loads(SECTION_PATH.read_text(encoding="utf-8"))
    captions = json.loads(CAPTION_PATH.read_text(encoding="utf-8")).get("items", {})
    files = {path.stem: path for path in IMAGE_DIR.iterdir() if path.is_file()}
    mappings: dict[str, dict] = {}
    for section in sections:
        for image_id in section.get("pics") or []:
            key = f"{section['product']}|{image_id}"
            item = mappings.setdefault(
                key,
                {"image_id": image_id, "product": section["product"], "section_keys": []},
            )
            section_key = f"{section['product']}|{section['section_id']}"
            if section_key not in item["section_keys"]:
                item["section_keys"].append(section_key)

    eligible: list[dict] = []
    noise = 0
    missing_caption = 0
    for key in sorted(mappings):
        item = mappings[key]
        caption = captions.get(key) or {}
        if caption.get("category") == "noise":
            noise += 1
            continue
        path = files.get(item["image_id"])
        if path is None:
            raise FileNotFoundError(f"section image is missing: {item['image_id']}")
        if not caption:
            missing_caption += 1
        eligible.append(
            {
                **item,
                "relative_path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
                "short_caption": (caption.get("short_caption") or "").strip(),
                "caption": (caption.get("content") or caption.get("short_caption") or "").strip(),
                "category": caption.get("category") or "unknown",
            }
        )
    stats = {
        "physical_files": len(files),
        "referenced_images": len(mappings),
        "eligible_images": len(eligible),
        "excluded_noise": noise,
        "missing_caption": missing_caption,
        "unreferenced_files": len(set(files) - {item["image_id"] for item in mappings.values()}),
    }
    eligible.sort(key=lambda item: item["image_id"])
    return eligible, stats


def load_completed() -> set[str]:
    """Read successfully persisted image IDs for resumable builds."""
    if not PROGRESS_PATH.exists():
        return set()
    completed: set[str] = set()
    for line in PROGRESS_PATH.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("status") == "ok":
            completed.add(str(record.get("image_id")))
    return completed


def append_progress(record: dict) -> None:
    """Append one non-image progress record."""
    with PROGRESS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def progress_stats() -> tuple[float, int]:
    """Aggregate successful remote batch time and historical error rows."""
    if not PROGRESS_PATH.exists():
        return 0.0, 0
    success_elapsed = 0.0
    errors = 0
    for line in PROGRESS_PATH.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("status") == "ok":
            success_elapsed += float(record.get("elapsed") or 0.0) / max(1, int(record.get("batch_size") or 4))
        elif record.get("status") == "error":
            errors += 1
    return round(success_elapsed, 3), errors


def main() -> None:
    """Validate inputs or build the remote visual embedding index."""
    parser = argparse.ArgumentParser(description="构建独立视觉FAISS索引")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="只检查图片和Section映射，不发送图片")
    parser.add_argument("--reset", action="store_true", help="清除本脚本的断点文件并重新构建")
    args = parser.parse_args()

    items, stats = collect_items()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if len(items) != 2504:
        raise RuntimeError(f"eligible image count drifted: expected 2504, got {len(items)}")
    if args.dry_run:
        print("visual_index_dry_run=OK")
        return

    VISUAL_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if args.reset:
        for path in (VISUAL_FAISS_PATH, VISUAL_METADATA_PATH, SUMMARY_PATH, PROGRESS_PATH, PARTIAL_VECTORS_PATH):
            if path.exists():
                path.unlink()
    elif VISUAL_FAISS_PATH.exists() or VISUAL_METADATA_PATH.exists():
        raise FileExistsError("visual index already exists; pass --reset to rebuild")

    completed = load_completed()
    if completed and not PARTIAL_VECTORS_PATH.exists():
        raise RuntimeError("progress exists without partial vectors; pass --reset to rebuild")
    mode = "r+" if PARTIAL_VECTORS_PATH.exists() else "w+"
    vectors = open_memmap(
        PARTIAL_VECTORS_PATH,
        mode=mode,
        dtype=np.float32,
        shape=(len(items), VISUAL_DIMENSION),
    )
    base_url = __import__("os").getenv("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
    api_key = __import__("os").getenv("EMBEDDING_API_KEY", "")
    client = VisualEmbeddingClient(base_url, api_key)
    started = time.time()
    failures: list[dict] = []
    pending = [(idx, item) for idx, item in enumerate(items) if item["image_id"] not in completed]

    for start in range(0, len(pending), max(1, args.batch_size)):
        batch = pending[start:start + max(1, args.batch_size)]
        batch_started = time.time()
        try:
            inputs = [{"image": file_to_data_url(ROOT / item["relative_path"])} for _, item in batch]
            embedded = client.embed_inputs(inputs, DEFAULT_VISUAL_EMBEDDING_MODEL, VISUAL_DIMENSION)
            array = np.asarray(embedded, dtype=np.float32)
            if array.shape != (len(batch), VISUAL_DIMENSION) or not np.isfinite(array).all():
                raise RuntimeError(f"unexpected embedding shape: {array.shape}")
            norms = np.linalg.norm(array, axis=1, keepdims=True)
            if np.any(norms == 0):
                raise RuntimeError("zero-norm visual embedding")
            array /= norms
            for row, (idx, item) in enumerate(batch):
                vectors[idx] = array[row]
                append_progress(
                    {
                        "image_id": item["image_id"],
                        "vector_id": idx,
                        "status": "ok",
                        "dimension": VISUAL_DIMENSION,
                        "batch_size": len(batch),
                        "elapsed": round(time.time() - batch_started, 3),
                    }
                )
                completed.add(item["image_id"])
            vectors.flush()
            print(f"visual_embedding={len(completed)}/{len(items)}")
        except Exception as exc:
            error = {"batch_start": start, "image_ids": [item["image_id"] for _, item in batch], "error": str(exc)}
            failures.append(error)
            for _, item in batch:
                append_progress({"image_id": item["image_id"], "status": "error", "error": str(exc)[:500]})
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code in {400, 401, 403, 404}:
                break

    successful_batch_elapsed, historical_error_records = progress_stats()
    summary = {
        **stats,
        "embedding_model": DEFAULT_VISUAL_EMBEDDING_MODEL,
        "embedding_dimension": VISUAL_DIMENSION,
        "completed": len(completed),
        "failures": failures,
        "last_run_elapsed": round(time.time() - started, 3),
        "successful_batch_elapsed": successful_batch_elapsed,
        "historical_error_records": historical_error_records,
        "source_sha256": {"section_chunks.json": sha256(SECTION_PATH), "image_captions_v4_final.json": sha256(CAPTION_PATH)},
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures or len(completed) != len(items):
        raise RuntimeError("visual index build incomplete; rerun without --reset to resume")

    final_vectors = np.asarray(vectors, dtype=np.float32)
    index = faiss.IndexFlatIP(VISUAL_DIMENSION)
    index.add(final_vectors)
    faiss.write_index(index, str(VISUAL_FAISS_PATH))
    metadata_items = [
        {
            "vector_id": idx,
            **item,
            "embedding_model": DEFAULT_VISUAL_EMBEDDING_MODEL,
            "embedding_dimension": VISUAL_DIMENSION,
        }
        for idx, item in enumerate(items)
    ]
    VISUAL_METADATA_PATH.write_text(
        json.dumps(
            {
                "version": 1,
                "embedding_model": DEFAULT_VISUAL_EMBEDDING_MODEL,
                "embedding_dimension": VISUAL_DIMENSION,
                "items": metadata_items,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    del final_vectors
    vectors.flush()
    mmap = getattr(vectors, "_mmap", None)
    if mmap is not None:
        mmap.close()
    del vectors
    PARTIAL_VECTORS_PATH.unlink()
    print(f"visual_index=OK images={len(items)} dimension={VISUAL_DIMENSION}")


if __name__ == "__main__":
    main()
