"""Apply the predeclared quality and latency gates to visual experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    """Fail unless every predeclared rollout gate passes."""
    parser = argparse.ArgumentParser(description="校验视觉检索上线门槛")
    parser.add_argument("--visual-summary", type=Path, required=True)
    parser.add_argument("--text-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    visual = json.loads(args.visual_summary.read_text(encoding="utf-8"))
    text = json.loads(args.text_summary.read_text(encoding="utf-8"))
    baseline = visual["vision_to_text"]
    fusion = visual["fusion"]
    checks = {
        "complete_visual_records": all(visual[mode]["records"] == 120 for mode in visual),
        "no_visual_errors": all(visual[mode]["errors"] == 0 for mode in visual),
        "no_visual_fallbacks": all(visual[mode].get("fallbacks", 0) == 0 for mode in visual),
        "section_recall_gain": fusion["section_r5"] >= baseline["section_r5"] + 0.05,
        "image_recall_gain": fusion["image_r5"] >= baseline["image_r5"] + 0.05,
        "product_r1_no_regression": fusion["product_r1"] >= baseline["product_r1"],
        "mrr_tolerance": fusion["section_rr"] >= baseline["section_rr"] - 0.01,
        "derived_no_regression": (
            fusion["subsets"]["derived"]["section_r5"] >= baseline["subsets"]["derived"]["section_r5"]
            and fusion["subsets"]["derived"]["image_r5"] >= baseline["subsets"]["derived"]["image_r5"]
        ),
        "external_style_no_regression": (
            fusion["subsets"]["external_style"]["section_r5"]
            >= baseline["subsets"]["external_style"]["section_r5"]
            and fusion["subsets"]["external_style"]["image_r5"]
            >= baseline["subsets"]["external_style"]["image_r5"]
        ),
        "latency_absolute": fusion["latency_p95"] < 50.0,
        "latency_relative": fusion["latency_p95"] <= baseline["latency_p95"] * 1.35,
        "text_no_errors": text["visual_enabled"]["errors"] == 0,
        "text_recall_30_of_30": text["visual_enabled"]["recall5"] == 1.0,
        "text_mrr_tolerance": text["visual_enabled"]["mrr"] >= text["baseline"]["mrr"] - 0.01,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {"status": "FAILED" if failed else "OK", "checks": checks, "failed_checks": failed},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(f"visual_rollout_gate=FAILED checks={','.join(failed)}")
    print("visual_rollout_gate=OK")


if __name__ == "__main__":
    main()
