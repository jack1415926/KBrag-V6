"""Create the deterministic 30-case derived-image visual relevance set."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from build_visual_index import collect_items
from visual_retrieval import ROOT


OUTPUT_DIR = ROOT / "evaluation" / "visual_queries" / "derived"
GOLD_PATH = ROOT / "evaluation" / "visual_relevance_gold_30_derived.jsonl"
EXTERNAL_TEMPLATE_PATH = ROOT / "evaluation" / "visual_external_style_template.json"
CATEGORIES = ("part_view", "schematic", "info_table")


def sha256(path: Path) -> str:
    """Return a file fingerprint."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose_items(items: list[dict], category: str, count: int) -> list[dict]:
    """Round-robin products so one manual cannot dominate a category."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        if item["category"] == category:
            grouped[item["product"]].append(item)
    for values in grouped.values():
        values.sort(key=lambda item: item["image_id"])
    selected: list[dict] = []
    products = sorted(grouped)
    offset = 0
    while len(selected) < count:
        progressed = False
        for product in products:
            if offset < len(grouped[product]):
                selected.append(grouped[product][offset])
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise RuntimeError(f"not enough {category} images")
        offset += 1
    return selected


def transform_image(source: Path, destination: Path, variant: int, seed: int) -> str:
    """Create one deterministic, non-identical query image."""
    rng = random.Random(seed)
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    width, height = image.size
    mode = variant % 5
    if mode == 0:
        margin_x, margin_y = max(1, width // 12), max(1, height // 12)
        image = image.crop((margin_x, margin_y, width - margin_x, height - margin_y))
        transform = "center_crop"
    elif mode == 1:
        image = image.rotate(rng.choice((-4, -3, 3, 4)), expand=True, fillcolor=(240, 240, 240))
        transform = "rotate"
    elif mode == 2:
        image = image.filter(ImageFilter.GaussianBlur(radius=1.4))
        transform = "blur"
    elif mode == 3:
        draw = ImageDraw.Draw(image)
        box_width, box_height = max(8, width // 5), max(8, height // 7)
        left = rng.randint(0, max(0, width - box_width))
        top = rng.randint(0, max(0, height - box_height))
        draw.rectangle((left, top, left + box_width, top + box_height), fill=(128, 128, 128))
        transform = "partial_occlusion"
    else:
        scale = 0.65
        image = image.resize((max(32, int(width * scale)), max(32, int(height * scale))))
        transform = "resize_compress"
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="JPEG", quality=65, optimize=True)
    return transform


def main() -> None:
    """Generate the fixed derived-image set and external-style template."""
    parser = argparse.ArgumentParser(description="生成30条视觉派生金标")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    items, _stats = collect_items()
    selected = [item for category in CATEGORIES for item in choose_items(items, category, 10)]
    if GOLD_PATH.exists() and not args.reset:
        raise FileExistsError(f"{GOLD_PATH} already exists; pass --reset to replace generated data")
    if args.reset and OUTPUT_DIR.exists():
        for path in OUTPUT_DIR.iterdir():
            if path.is_file():
                path.unlink()

    records = []
    for index, item in enumerate(selected, start=1):
        source = ROOT / item["relative_path"]
        destination = OUTPUT_DIR / f"V{index:03d}_{item['image_id']}.jpg"
        transform = transform_image(source, destination, index - 1, seed=20260915 + index)
        if sha256(source) == sha256(destination):
            raise RuntimeError(f"derived query is byte-identical to source: {item['image_id']}")
        label = item.get("short_caption") or item.get("caption") or item["image_id"]
        label = " ".join(str(label).split())[:100]
        records.append(
            {
                "case_id": f"V{index:03d}",
                "source_type": "derived",
                "category": item["category"],
                "query_image": destination.relative_to(ROOT).as_posix(),
                "question": f"请查找展示“{label}”的手册图片，并定位对应章节。",
                "expected_product": item["product"],
                "relevant_sections": item["section_keys"],
                "relevant_images": [item["image_id"]],
                "source_image": item["relative_path"],
                "transform": transform,
                "judgeable": True,
            }
        )
    GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLD_PATH.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    if not EXTERNAL_TEMPLATE_PATH.exists():
        EXTERNAL_TEMPLATE_PATH.write_text(
            json.dumps(
                {
                    "required_records": 10,
                    "output_file": "evaluation/visual_relevance_gold_10_external.jsonl",
                    "record_schema": {
                        "case_id": "R001",
                        "source_group": "external_style",
                        "source_type": "synthetic | user_photo | public_photo",
                        "category": "part_view",
                        "query_image": "evaluation/visual_queries/external/R001.jpg",
                        "question": "用户真实问题",
                        "expected_product": "产品名",
                        "relevant_sections": ["产品名|SectionID"],
                        "relevant_images": ["ManualXX_Y"],
                        "visual_style": "photorealistic",
                        "judgeable": True,
                    },
                    "distribution": {"part_view": 3, "procedure": 4, "ocr": 3, "minimum_products": 5},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print(f"visual_derived_gold=OK records={len(records)} path={GOLD_PATH}")


if __name__ == "__main__":
    main()
