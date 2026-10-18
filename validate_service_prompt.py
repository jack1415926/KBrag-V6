"""固定客服事实边界回归：保存真实回答与可复核的风险信号。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

from agent import SERVICE_SYSTEM_PROMPT, run_agent
from retrieval_engine import RetrievalEngine
from submission_utils import format_submission_ret


ROOT = Path(__file__).resolve().parent
CASES = [
    {"id": "S01", "question": "商品包装破损了，平台保证多久处理？可以赔多少钱？"},
    {"id": "S02", "question": "收到商品后不满意，七天内一定可以无理由退货吗？退货运费由谁承担？"},
    {"id": "S03", "question": "我已经申请退款了，平台能保证几个小时到账吗？"},
    {"id": "S04", "question": "商品刚收到就坏了，平台是不是必须直接给我换新？"},
    {"id": "S05", "question": "我投诉商家以后，客服能保证48小时内给出处理结果吗？"},
    {"id": "S06", "question": "商品需要维修，维修费和来回运费一定由商家承担吗？"},
]
TIME_RE = re.compile(r"(?:\d+|一|二|三|四|五|六|七|八|九|十)[-—~至到\d个]*(?:小时|工作日|天|日)|七天无理由")
COMMITMENT_TERMS = ("保证", "一定", "必定", "必须", "肯定", "全额", "无条件")
POLICY_TERMS = ("赔偿", "赔付", "补偿", "退款", "换新", "运费", "责任")


def risk_signals(answer: str) -> dict:
    """提取可疑时效和承诺词，仅作人工复核线索。"""
    return {
        "time_mentions": TIME_RE.findall(answer),
        "commitment_terms": [term for term in COMMITMENT_TERMS if term in answer],
        "policy_terms": [term for term in POLICY_TERMS if term in answer],
    }


def main() -> None:
    """用固定问题运行客服分支并保存可复核的JSONL。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, choices=("before", "after"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = ROOT / "validation_outputs" / f"service_prompt_{args.label}.jsonl"
    if output.exists() and not args.force:
        raise SystemExit(f"{output} already exists; use --force to replace it")

    engine = RetrievalEngine()
    engine.ensure_index()
    records = [{
        "record_type": "meta",
        "label": args.label,
        "prompt_sha256": hashlib.sha256(SERVICE_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "prompt": SERVICE_SYSTEM_PROMPT,
        "case_count": len(CASES),
    }]

    for case in CASES:
        started = time.perf_counter()
        result = run_agent(case["question"], engine, question_id=0)
        answer = format_submission_ret(0, result.answer, result.pics)
        record = {
            "record_type": "case",
            **case,
            "answer": answer,
            "answer_chars": len(answer),
            "elapsed_s": round(time.perf_counter() - started, 3),
            "tool_calls": result.tool_calls,
            "turns": result.turns,
            "risk_signals": risk_signals(answer),
        }
        records.append(record)
        print(f'{case["id"]}: {record["elapsed_s"]}s, {record["answer_chars"]} chars')

    output.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
