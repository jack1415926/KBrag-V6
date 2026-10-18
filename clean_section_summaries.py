"""Safely rebuild misaligned Section summaries and prune stale keys."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import gen_section_summaries as generator


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "data" / "section_summaries.json"
STAGING = ROOT / "data" / "section_summaries.cleanup_staging.json"
AUDIT = ROOT / "validation_outputs" / "section_summary_cleanup.json"


def sha256(path: Path) -> str:
    """Return a file's lowercase SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write(path: Path, payload: dict) -> None:
    """Write JSON through a same-directory temporary file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def generate_with_retries(section: dict) -> tuple[str, str]:
    """Generate one normalized summary with bounded retries."""
    key = f"{section['product']}|{section['section_id']}"
    last_error = ""
    for attempt in range(4):
        try:
            summary = " ".join(generator.build_one(section).split())
            if not summary:
                raise ValueError("empty summary")
            return key, summary
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"[:200]
            if attempt < 3:
                time.sleep(8 * (attempt + 1))
    return key, f"__ERROR__ {last_error}"


def main() -> int:
    """Build a complete staging map, validate it, back up, then commit atomically."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--products", required=True, help="Comma-separated products to rebuild")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    products = {item.strip() for item in args.products.split(",") if item.strip()}
    sections = generator.SECTIONS
    section_lookup = {
        f"{section['product']}|{section['section_id']}": section for section in sections
    }
    unknown = products - {section["product"] for section in sections}
    if unknown:
        raise SystemExit(f"unknown products: {sorted(unknown)}")

    original = json.loads(TARGET.read_text(encoding="utf-8"))
    stale_keys = sorted(original.keys() - section_lookup.keys())
    rebuild_keys = {
        key for key, section in section_lookup.items() if section["product"] in products
    }
    base = {
        key: value
        for key, value in original.items()
        if key in section_lookup and key not in rebuild_keys
    }
    missing_outside_rebuild = sorted(
        key for key in section_lookup.keys() - original.keys() if key not in rebuild_keys
    )
    if missing_outside_rebuild:
        raise SystemExit(f"missing keys outside rebuild scope: {missing_outside_rebuild}")

    print(
        f"sections={len(sections)} original={len(original)} stale={len(stale_keys)} "
        f"rebuild={len(rebuild_keys)} keep={len(base)}",
        flush=True,
    )
    if args.plan:
        return 0

    if STAGING.exists() and not args.resume:
        raise SystemExit(f"staging exists; inspect it, then use --resume: {STAGING}")
    staged = json.loads(STAGING.read_text(encoding="utf-8")) if STAGING.exists() else base
    staged = {
        key: value
        for key, value in staged.items()
        if key in section_lookup and (key in base or key in rebuild_keys)
    }
    staged.update(base)
    todo = [section_lookup[key] for key in sorted(rebuild_keys - staged.keys())]
    print(f"completed={len(rebuild_keys) - len(todo)} todo={len(todo)}", flush=True)

    started = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(generate_with_retries, section) for section in todo]
        for future in as_completed(futures):
            key, summary = future.result()
            staged[key] = summary
            completed += 1
            if completed % 10 == 0:
                atomic_write(STAGING, staged)
                print(f"generated={completed}/{len(todo)} last={key}", flush=True)
    atomic_write(STAGING, staged)

    error_keys = sorted(
        key for key, value in staged.items() if not value or value.startswith("__ERROR__")
    )
    missing_keys = sorted(section_lookup.keys() - staged.keys())
    extra_keys = sorted(staged.keys() - section_lookup.keys())
    if error_keys or missing_keys or extra_keys:
        print(
            f"not committed: errors={len(error_keys)} missing={len(missing_keys)} "
            f"extra={len(extra_keys)} staging={STAGING}",
            flush=True,
        )
        return 1

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = TARGET.with_name(f"section_summaries.pre_cleanup_{timestamp}.json")
    shutil.copy2(TARGET, backup)
    original_hash = sha256(TARGET)
    os.replace(STAGING, TARGET)
    audit = {
        "completed_at": datetime.now().astimezone().isoformat(),
        "model": os.getenv("SILICONFLOW_MODEL", ""),
        "rebuild_products": sorted(products),
        "rebuild_count": len(rebuild_keys),
        "stale_removed_count": len(stale_keys),
        "stale_removed_keys": stale_keys,
        "original_count": len(original),
        "final_count": len(staged),
        "original_sha256": original_hash,
        "final_sha256": sha256(TARGET),
        "backup": str(backup.relative_to(ROOT)),
        "elapsed_s": round(time.time() - started, 3),
        "errors": [],
    }
    atomic_write(AUDIT, audit)
    print(
        f"committed summaries={len(staged)} stale_removed={len(stale_keys)} "
        f"backup={backup} audit={AUDIT}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
