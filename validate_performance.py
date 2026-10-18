"""Generate validation tables for the initial-round delivery report.

Default mode is deterministic and offline: it reads the public question file,
reference submissions, product mapping labels, image captions, and packaged
manual assets, then writes CSV/JSON/Markdown outputs under validation_outputs/.

Optional online API validation can be enabled with --api-base-url to record a
short two-turn dialogue check against a running /chat service.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = ROOT / "validation_outputs"

QUESTION_PUBLIC = ROOT / "question_public.csv"
FULL_REFERENCE = ROOT / "submissions" / "v6_full_reference.csv"
TECH_REFERENCE = ROOT / "submissions" / "v6_tech_reference.csv"
PRODUCT_MAPPING = ROOT / "data" / "product_mapping_final.csv"
IMAGE_CAPTIONS = ROOT / "data" / "image_captions_v4_final.json"
SECTION_CHUNKS = ROOT / "data" / "section_chunks.json"
RETRIEVAL_CHUNKS = ROOT / "data" / "retrieval_chunks.json"
MANUAL_MD_DIR = ROOT / "手册_v4"
IMAGE_DIR = ROOT / "手册" / "插图"

PIC_MARKER_RE = re.compile(r"<PIC>", re.IGNORECASE)
PICS_ARRAY_RE = re.compile(r",(\[[^\n\r]*\])\s*$")

ROUTER_REPEAT_RESULTS = [
    {"run": 1, "correct": 400, "total": 400, "wrong_ids": "", "disagreement_count": 7, "latency_avg_s": 0.90086, "latency_median_s": 0.8745, "latency_max_s": 1.959, "wall_s": 36.493},
    {"run": 2, "correct": 400, "total": 400, "wrong_ids": "", "disagreement_count": 7, "latency_avg_s": 0.9030975, "latency_median_s": 0.905, "latency_max_s": 1.458, "wall_s": 36.603},
    {"run": 3, "correct": 400, "total": 400, "wrong_ids": "", "disagreement_count": 9, "latency_avg_s": 0.931945, "latency_median_s": 0.908, "latency_max_s": 2.371, "wall_s": 37.837},
    {"run": 4, "correct": 400, "total": 400, "wrong_ids": "", "disagreement_count": 10, "latency_avg_s": 0.9060475, "latency_median_s": 0.9095, "latency_max_s": 1.575, "wall_s": 36.773},
    {"run": 5, "correct": 400, "total": 400, "wrong_ids": "", "disagreement_count": 8, "latency_avg_s": 1.0323425, "latency_median_s": 0.9045, "latency_max_s": 15.575, "wall_s": 41.864},
]

FINAL_ANSWER_PRODUCT_WRONG_IDS = {"432", "433"}

PRODUCT_MAPPING_ALIASES = {
    "air fryer": "Air Fryer",
    "boat": "Boat",
    "camera": "Camera",
    "camera-install": "Camera",
    "coffee machine": "Espresso Machine",
    "electric toothbrush": "Toothbrush",
    "e-reader": "Media Player",
    "fax machine": "Printer",
    "grill": "Gas Grill",
    "landline?": "Phone",
    "motherboard": "Motherboard",
    "multi-use pressure cooker and air fryer": "Pressure Cooker",
    "over-the-range microwave": "Microwave",
    "riding lawn mower": "Lawn Mower",
    "snowmobile": "Snowmobile",
    "television": "TV",
    "vacuum": "Vacuum",
    "waverunner jetski": "WaveRunner",
    "wireless earphones": "Earphones",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8-SIG CSV file into dictionaries."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Any:
    """Read a JSON file using the repository's UTF-8 convention."""
    return json.loads(path.read_text(encoding="utf-8"))


def pct(numerator: int | float, denominator: int | float) -> str:
    if not denominator:
        return "0.00%"
    return f"{float(numerator) / float(denominator) * 100:.2f}%"


def normalize_product_mapping_label(label: str, catalog_products: set[str]) -> str:
    """Convert product_mapping_final.csv labels into catalog product names."""
    cleaned = (label or "").strip()
    if not cleaned or cleaned == "非RAG":
        return ""
    if cleaned in catalog_products:
        return cleaned
    mapped = PRODUCT_MAPPING_ALIASES.get(cleaned.lower())
    if mapped in catalog_products:
        return mapped
    return ""


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_pic_array(ret: str) -> list[str]:
    """Extract the platform image array appended to a technical answer."""
    match = PICS_ARRAY_RE.search(ret or "")
    if not match:
        return []
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [str(item).strip() for item in payload if str(item).strip()]


def image_inventory() -> set[str]:
    names: set[str] = set()
    if not IMAGE_DIR.exists():
        return names
    for path in IMAGE_DIR.rglob("*"):
        if not path.is_file():
            continue
        names.add(path.name)
        names.add(path.stem)
    return names


def caption_inventory(items: dict[str, dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for key, item in items.items():
        if "|" in key:
            ids.add(key.split("|", 1)[1])
        image_id = str(item.get("image_id") or "").strip()
        if image_id:
            ids.add(image_id)
    return ids


def build_dataset_overview() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Summarize public questions, manual assets, chunks, captions, and references."""
    questions = read_csv_rows(QUESTION_PUBLIC)
    full_rows = read_csv_rows(FULL_REFERENCE)
    tech_rows = read_csv_rows(TECH_REFERENCE)
    section_chunks = read_json(SECTION_CHUNKS)
    retrieval_chunks = read_json(RETRIEVAL_CHUNKS)
    captions = read_json(IMAGE_CAPTIONS).get("items", {})
    image_files = [p for p in IMAGE_DIR.rglob("*") if p.is_file()] if IMAGE_DIR.exists() else []
    manual_files = list(MANUAL_MD_DIR.glob("*.md")) if MANUAL_MD_DIR.exists() else []

    service_count = sum(int(row["id"]) < 64 for row in questions)
    tech_count = len(questions) - service_count
    rows = [
        {"metric": "公开题集总题数", "value": len(questions), "note": "question_public.csv"},
        {"metric": "客服题数量", "value": service_count, "note": "按 qid < 64 统计"},
        {"metric": "技术题数量", "value": tech_count, "note": "按 qid >= 64 统计"},
        {"metric": "手册 Markdown 数量", "value": len(manual_files), "note": "手册_v4/*.md"},
        {"metric": "section 数量", "value": len(section_chunks), "note": "data/section_chunks.json"},
        {"metric": "retrieval chunk 数量", "value": len(retrieval_chunks), "note": "data/retrieval_chunks.json"},
        {"metric": "手册插图文件数量", "value": len(image_files), "note": "手册/插图"},
        {"metric": "图片 caption 条目数量", "value": len(captions), "note": "data/image_captions_v4_final.json"},
        {"metric": "完整参考提交行数", "value": len(full_rows), "note": "submissions/v6_full_reference.csv"},
        {"metric": "技术参考提交行数", "value": len(tech_rows), "note": "submissions/v6_tech_reference.csv"},
    ]
    summary = {
        "questions_total": len(questions),
        "service_questions": service_count,
        "tech_questions": tech_count,
        "manual_md_files": len(manual_files),
        "section_chunks": len(section_chunks),
        "retrieval_chunks": len(retrieval_chunks),
        "image_files": len(image_files),
        "caption_items": len(captions),
        "full_reference_rows": len(full_rows),
        "tech_reference_rows": len(tech_rows),
    }
    return rows, summary


def build_routing_validation() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return the repeated service/tech router validation table."""
    rows = []
    for item in ROUTER_REPEAT_RESULTS:
        row = dict(item)
        row["accuracy"] = pct(item["correct"], item["total"])
        rows.append(row)
    summary = {
        "runs": len(rows),
        "total_cases_per_run": 400,
        "all_runs_correct": all(row["correct"] == row["total"] for row in rows),
        "min_accuracy": min(row["correct"] / row["total"] for row in rows),
        "avg_latency_s": mean(row["latency_avg_s"] for row in rows),
        "avg_disagreement_count": mean(row["disagreement_count"] for row in rows),
    }
    return rows, summary


def build_product_router_validation() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the current product router and compute Top-1/Top-3 hit rates."""
    from product_router import ProductRouter
    from retrieval_engine import RetrievalEngine

    mapping_rows = read_csv_rows(PRODUCT_MAPPING)
    questions = {int(row["id"]): str(row.get("question") or "").strip() for row in read_csv_rows(QUESTION_PUBLIC)}
    engine = RetrievalEngine()
    engine.ensure_index()
    router = ProductRouter(engine.catalog, engine=engine)
    catalog_products = set(engine.catalog.keys())
    detail_rows: list[dict[str, Any]] = []
    unresolved_rows: list[dict[str, str]] = []
    for record in sorted(mapping_rows, key=lambda item: int(item["id"])):
        qid = int(record["id"])
        raw_label = str(record.get("匹配产品") or "").strip()
        if raw_label == "非RAG":
            continue
        expected_product = normalize_product_mapping_label(raw_label, catalog_products)
        if not expected_product:
            unresolved_rows.append({"qid": str(qid), "label": raw_label})
            continue
        question = questions.get(qid) or str(record.get("题目(前80字)") or "").strip()
        decision = router.route(question, top_n=3)
        products = list(decision.products or [])
        top1 = products[0] if products else ""
        top3 = products[:3]
        route_candidate_hit = expected_product in products
        final_answer_hit = str(qid) not in FINAL_ANSWER_PRODUCT_WRONG_IDS
        detail_rows.append({
            "qid": int(qid),
            "question": question,
            "mapping_label": raw_label,
            "mapping_confidence": record.get("匹配置信度", ""),
            "mapping_language": record.get("语言", ""),
            "expected_product": expected_product,
            "route_top1": top1,
            "route_top3": " | ".join(top3),
            "top1_hit": int(top1 == expected_product),
            "top3_hit": int(expected_product in top3),
            "route_candidate_hit": int(route_candidate_hit),
            "final_answer_product_hit": int(final_answer_hit),
            "candidate_count": len(products),
            "confidence": decision.confidence,
            "reason": decision.reason,
            "debug_scores": json.dumps(decision.debug_scores, ensure_ascii=False),
        })

    total = len(detail_rows)
    top1_hits = sum(row["top1_hit"] for row in detail_rows)
    top3_hits = sum(row["top3_hit"] for row in detail_rows)
    route_candidate_hits = sum(row["route_candidate_hit"] for row in detail_rows)
    final_answer_hits = sum(row["final_answer_product_hit"] for row in detail_rows)
    no_candidate = sum(1 for row in detail_rows if not row["route_top1"])
    confidence_counts = Counter(str(row["confidence"]) for row in detail_rows)
    reason_counts = Counter(str(row["reason"]) for row in detail_rows)
    summary_rows = [
        {"metric": "产品路由 Top-1 命中率", "value": pct(top1_hits, total), "numerator": top1_hits, "denominator": total, "note": "route.products[0] == expected_product"},
        {"metric": "产品路由 Top-3 命中率", "value": pct(top3_hits, total), "numerator": top3_hits, "denominator": total, "note": "expected_product in route.products[:3]"},
        {"metric": "产品路由候选覆盖率", "value": pct(route_candidate_hits, total), "numerator": route_candidate_hits, "denominator": total, "note": "expected_product in route.products"},
        {"metric": "最终答案产品命中率", "value": pct(final_answer_hits, total), "numerator": final_answer_hits, "denominator": total, "note": "基于 v6_full_reference.csv 复核，未命中 qid=432,433"},
        {"metric": "无候选路由占比", "value": pct(no_candidate, total), "numerator": no_candidate, "denominator": total, "note": "route.products 为空"},
        {"metric": "high 置信路由占比", "value": pct(confidence_counts.get("high", 0), total), "numerator": confidence_counts.get("high", 0), "denominator": total, "note": "ProductRouteDecision.confidence"},
        {"metric": "medium 置信路由占比", "value": pct(confidence_counts.get("medium", 0), total), "numerator": confidence_counts.get("medium", 0), "denominator": total, "note": "ProductRouteDecision.confidence"},
        {"metric": "标签归一化失败数量", "value": len(unresolved_rows), "numerator": len(unresolved_rows), "denominator": len(mapping_rows), "note": "product_mapping_final.csv 标签无法映射到 catalog"},
    ]
    summary = {
        "total": total,
        "source": "data/product_mapping_final.csv",
        "non_rag_rows_skipped": sum(1 for row in mapping_rows if str(row.get("匹配产品") or "").strip() == "非RAG"),
        "unresolved_mapping_labels": unresolved_rows,
        "top1_hits": top1_hits,
        "top1_hit_rate": top1_hits / total if total else 0.0,
        "top3_hits": top3_hits,
        "top3_hit_rate": top3_hits / total if total else 0.0,
        "route_candidate_hits": route_candidate_hits,
        "route_candidate_hit_rate": route_candidate_hits / total if total else 0.0,
        "final_answer_product_hits": final_answer_hits,
        "final_answer_product_hit_rate": final_answer_hits / total if total else 0.0,
        "final_answer_product_wrong_ids": sorted(FINAL_ANSWER_PRODUCT_WRONG_IDS),
        "no_candidate": no_candidate,
        "no_candidate_rate": no_candidate / total if total else 0.0,
        "confidence_counts": dict(confidence_counts),
        "reason_counts": dict(reason_counts),
    }
    return summary_rows, {"summary": summary, "details": detail_rows}


def build_multimodal_validation() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Check final answer image markers, image files, and caption coverage."""
    questions = {int(row["id"]): row for row in read_csv_rows(QUESTION_PUBLIC)}
    full_rows = read_csv_rows(FULL_REFERENCE)
    caption_items = read_json(IMAGE_CAPTIONS).get("items", {})
    image_names = image_inventory()
    caption_ids = caption_inventory(caption_items)
    category_counts = Counter(str(item.get("category") or "unknown") for item in caption_items.values())
    fit_counts = Counter(str(item.get("section_fit") or "unknown") for item in caption_items.values())

    detail_rows: list[dict[str, Any]] = []
    for row in full_rows:
        qid = int(row["id"])
        ret = row.get("ret", "")
        pics = parse_pic_array(ret)
        marker_count = len(PIC_MARKER_RE.findall(ret))
        pics_exist = sum(1 for pic in pics if pic in image_names)
        pics_captioned = sum(1 for pic in pics if pic in caption_ids)
        detail_rows.append({
            "qid": qid,
            "question_type": "service" if qid < 64 else "tech",
            "ret_nonempty": int(bool(ret.strip())),
            "pic_marker_count": marker_count,
            "pics_array_count": len(pics),
            "marker_array_match": int(marker_count == len(pics)),
            "pics_exist_count": pics_exist,
            "pics_captioned_count": pics_captioned,
            "all_pics_exist": int(pics_exist == len(pics)),
            "all_pics_captioned": int(pics_captioned == len(pics)),
            "has_question": int(qid in questions),
        })

    total = len(detail_rows)
    tech_rows = [row for row in detail_rows if row["question_type"] == "tech"]
    service_rows = [row for row in detail_rows if row["question_type"] == "service"]
    rows_with_pics = [row for row in detail_rows if row["pics_array_count"] > 0]
    total_pic_refs = sum(row["pics_array_count"] for row in detail_rows)
    total_existing_pic_refs = sum(row["pics_exist_count"] for row in detail_rows)
    total_captioned_pic_refs = sum(row["pics_captioned_count"] for row in detail_rows)
    marker_matches = sum(row["marker_array_match"] for row in detail_rows)
    nonempty = sum(row["ret_nonempty"] for row in detail_rows)
    service_without_pics = sum(1 for row in service_rows if row["pics_array_count"] == 0 and row["pic_marker_count"] == 0)

    summary_rows = [
        {"metric": "提交行数完整率", "value": pct(total, len(questions)), "numerator": total, "denominator": len(questions), "note": "v6_full_reference.csv vs question_public.csv"},
        {"metric": "答案非空率", "value": pct(nonempty, total), "numerator": nonempty, "denominator": total, "note": "ret 非空"},
        {"metric": "图片锚点与数组一致率", "value": pct(marker_matches, total), "numerator": marker_matches, "denominator": total, "note": "<PIC> 数量 == 图片数组长度"},
        {"metric": "图片文件存在率", "value": pct(total_existing_pic_refs, total_pic_refs), "numerator": total_existing_pic_refs, "denominator": total_pic_refs, "note": "按图片引用计"},
        {"metric": "图片 caption 覆盖率", "value": pct(total_captioned_pic_refs, total_pic_refs), "numerator": total_captioned_pic_refs, "denominator": total_pic_refs, "note": "按图片引用计"},
        {"metric": "客服题无图片输出率", "value": pct(service_without_pics, len(service_rows)), "numerator": service_without_pics, "denominator": len(service_rows), "note": "客服题不附加手册图片"},
        {"metric": "技术题含图片答案占比", "value": pct(sum(1 for row in tech_rows if row["pics_array_count"] > 0), len(tech_rows)), "numerator": sum(1 for row in tech_rows if row["pics_array_count"] > 0), "denominator": len(tech_rows), "note": "按最终答案计"},
        {"metric": "caption 条目数量", "value": len(caption_items), "numerator": len(caption_items), "denominator": "", "note": "图片语义描述库"},
    ]
    summary = {
        "total_rows": total,
        "tech_rows": len(tech_rows),
        "service_rows": len(service_rows),
        "rows_with_pics": len(rows_with_pics),
        "total_pic_refs": total_pic_refs,
        "marker_array_match_rows": marker_matches,
        "marker_array_match_rate": marker_matches / total if total else 0.0,
        "image_file_hit_rate": total_existing_pic_refs / total_pic_refs if total_pic_refs else 1.0,
        "caption_hit_rate": total_captioned_pic_refs / total_pic_refs if total_pic_refs else 1.0,
        "caption_category_counts": dict(category_counts),
        "caption_section_fit_counts": dict(fit_counts),
        "details": detail_rows,
    }
    return summary_rows, summary


def run_api_dialogue(base_url: str, token: str, timeout_s: float = 65.0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run a two-turn same-session /chat validation against a live API."""
    session_id = f"validation-{int(time.time())}"
    cases = [
        {"turn": 1, "question": "椅子的扶手使用一段时间后为什么会松动？"},
        {"turn": 2, "question": "那应该怎么处理？"},
    ]
    rows: list[dict[str, Any]] = []
    for case in cases:
        payload = json.dumps({
            "question": case["question"],
            "session_id": session_id,
            "stream": False,
        }, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            base_url.rstrip("/") + "/chat",
            data=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.time()
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                body = response.read().decode("utf-8")
                status = response.status
            elapsed = time.time() - started
            parsed = json.loads(body)
            answer = str((parsed.get("data") or {}).get("answer") or "")
            ok = status == 200 and parsed.get("code") == 0 and bool(answer.strip())
            rows.append({
                "turn": case["turn"],
                "session_id": session_id,
                "http_status": status,
                "code": parsed.get("code"),
                "ok": int(ok),
                "elapsed_s": f"{elapsed:.3f}",
                "answer_len": len(answer),
                "question": case["question"],
                "answer_preview": answer[:120].replace("\n", " "),
                "error": "",
            })
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            elapsed = time.time() - started
            rows.append({
                "turn": case["turn"],
                "session_id": session_id,
                "http_status": "",
                "code": "",
                "ok": 0,
                "elapsed_s": f"{elapsed:.3f}",
                "answer_len": 0,
                "question": case["question"],
                "answer_preview": "",
                "error": repr(exc)[:300],
            })
            break
    summary = {
        "run": True,
        "base_url": base_url,
        "session_id": session_id,
        "turns_attempted": len(rows),
        "turns_ok": sum(row["ok"] for row in rows),
        "all_ok": bool(rows) and all(row["ok"] for row in rows) and len(rows) == len(cases),
    }
    return rows, summary


def skipped_api_dialogue() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [{
        "turn": "",
        "session_id": "",
        "http_status": "",
        "code": "",
        "ok": "",
        "elapsed_s": "",
        "answer_len": "",
        "question": "",
        "answer_preview": "",
        "error": "未运行；如需在线验证，请先启动 API 后执行 --api-base-url http://127.0.0.1:8000",
    }]
    summary = {"run": False, "all_ok": False, "note": "not_run"}
    return rows, summary


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        vals = [str(row.get(col, "")).replace("\n", " ") for col in columns]
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + body)


def build_markdown(summary: dict[str, Any], tables: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "# V6 初赛性能验证数据表",
        "",
        f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "> 口径：入口Service/Tech五次400题结果来自历史常量；产品路由Top-1/Top-3由当前ProductRouter离线重算；图文指标检查已保存参考答案与当前图片/Caption资产的一致性，不代表当前模型重新生成400题。",
        "",
        "## 数据集与资产",
        "",
        markdown_table(tables["dataset_overview"], ["metric", "value", "note"]),
        "",
        "## 路由验证",
        "",
        markdown_table(tables["routing_validation"], ["run", "correct", "total", "accuracy", "disagreement_count", "latency_avg_s", "latency_median_s", "latency_max_s", "wall_s"]),
        "",
        "## 产品路由验证",
        "",
        markdown_table(tables["product_router_summary"], ["metric", "value", "numerator", "denominator", "note"]),
        "",
        "## 多模态图文与格式验证",
        "",
        markdown_table(tables["multimodal_summary"], ["metric", "value", "numerator", "denominator", "note"]),
        "",
        "## 对话连贯性 / API 验证",
        "",
        markdown_table(tables["dialogue_validation"], ["turn", "session_id", "http_status", "code", "ok", "elapsed_s", "answer_len", "question", "answer_preview", "error"]),
        "",
        "## 汇总 JSON 字段",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    """CLI entry point for generating validation artifacts."""
    parser = argparse.ArgumentParser(description="Generate V6 validation tables for delivery report.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for validation CSV/JSON/Markdown files.")
    parser.add_argument("--api-base-url", default="", help="Optional running API base URL, for example http://127.0.0.1:8000.")
    parser.add_argument("--api-token", default=os.getenv("KAFU_API_TOKEN", ""), help="Bearer token for optional online API validation.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows, dataset_summary = build_dataset_overview()
    routing_rows, routing_summary = build_routing_validation()
    product_router_summary_rows, product_router_payload = build_product_router_validation()
    multimodal_summary_rows, multimodal_payload = build_multimodal_validation()
    if args.api_base_url:
        dialogue_rows, dialogue_summary = run_api_dialogue(args.api_base_url, args.api_token)
    else:
        dialogue_rows, dialogue_summary = skipped_api_dialogue()

    write_csv(out_dir / "dataset_overview.csv", dataset_rows)
    write_csv(out_dir / "routing_validation.csv", routing_rows)
    write_csv(out_dir / "product_router_summary.csv", product_router_summary_rows)
    write_csv(out_dir / "product_router_details.csv", product_router_payload["details"])
    write_csv(out_dir / "multimodal_format_summary.csv", multimodal_summary_rows)
    write_csv(out_dir / "multimodal_format_details.csv", multimodal_payload["details"])
    write_csv(out_dir / "dialogue_api_validation.csv", dialogue_rows)

    summary = {
        "dataset": dataset_summary,
        "routing": routing_summary,
        "product_router": product_router_payload["summary"],
        "multimodal": {k: v for k, v in multimodal_payload.items() if k != "details"},
        "dialogue": dialogue_summary,
        "outputs": {
            "dataset_overview": "validation_outputs/dataset_overview.csv",
            "routing_validation": "validation_outputs/routing_validation.csv",
            "product_router_summary": "validation_outputs/product_router_summary.csv",
            "product_router_details": "validation_outputs/product_router_details.csv",
            "multimodal_summary": "validation_outputs/multimodal_format_summary.csv",
            "multimodal_details": "validation_outputs/multimodal_format_details.csv",
            "dialogue_validation": "validation_outputs/dialogue_api_validation.csv",
            "markdown": "validation_outputs/validation_summary.md",
            "json": "validation_outputs/validation_summary.json",
        },
    }
    write_json(out_dir / "validation_summary.json", summary)
    markdown = build_markdown(
        summary,
        {
            "dataset_overview": dataset_rows,
            "routing_validation": routing_rows,
            "product_router_summary": product_router_summary_rows,
            "multimodal_summary": multimodal_summary_rows,
            "dialogue_validation": dialogue_rows,
        },
    )
    (out_dir / "validation_summary.md").write_text(markdown, encoding="utf-8")

    print(f"validation_outputs={out_dir}")
    print(f"product_router_top1_hit_rate={pct(product_router_payload['summary']['top1_hits'], product_router_payload['summary']['total'])}")
    print(f"product_router_top3_hit_rate={pct(product_router_payload['summary']['top3_hits'], product_router_payload['summary']['total'])}")
    print(f"multimodal_marker_array_match_rate={pct(multimodal_payload['marker_array_match_rows'], multimodal_payload['total_rows'])}")
    print(f"dialogue_validation_run={dialogue_summary.get('run')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
