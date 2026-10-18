"""用真实会话历史比较V0/V1/V2客服Prompt的连续追问表现。"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import agent
from api_server import _build_question_with_history
from retrieval_engine import RetrievalEngine
from submission_utils import format_submission_ret
from validate_service_prompt import risk_signals


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "validation_outputs" / "service_prompt_followups.jsonl"
SCENARIOS = [
    {
        "id": "S01",
        "initial": "商品包装破损了，平台保证多久处理？可以赔多少钱？",
        "followups": [
            "那我现在应该先做什么？需要准备哪些材料？",
            "如果售后页面一直没有更新，我还能怎么处理？",
        ],
    },
    {
        "id": "S03",
        "initial": "我已经申请退款了，平台能保证几个小时到账吗？",
        "followups": [
            "页面现在显示退款成功，但银行卡里还没有到账，我应该怎么排查？",
            "如果还是没有到账，我联系人工客服时应该准备什么信息？",
        ],
    },
]


def load_run(path: Path) -> tuple[str, dict[str, dict]]:
    """读取一次单轮测试保存的Prompt与逐题回答。"""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    meta = next(row for row in rows if row["record_type"] == "meta")
    cases = {row["id"]: row for row in rows if row["record_type"] == "case"}
    return meta["prompt"], cases


def call_service(engine: RetrievalEngine, question: str, history: list[dict[str, str]]) -> dict:
    """按API相同的历史拼接方式执行一次客服回答。"""
    started = time.perf_counter()
    result = agent.run_agent(
        _build_question_with_history(question, history),
        engine,
        question_id=0,
    )
    answer = format_submission_ret(0, result.answer, result.pics)
    return {
        "question": question,
        "answer": answer,
        "answer_chars": len(answer),
        "elapsed_s": round(time.perf_counter() - started, 3),
        "tool_calls": result.tool_calls,
        "turns": result.turns,
        "risk_signals": risk_signals(answer),
    }


def main() -> None:
    """复用V0/V1首答并运行两组固定追问链。"""
    if OUTPUT.exists():
        raise SystemExit(f"{OUTPUT} already exists")

    v0_prompt, v0_cases = load_run(ROOT / "validation_outputs" / "service_prompt_before.jsonl")
    v1_prompt, v1_cases = load_run(ROOT / "validation_outputs" / "service_prompt_after_v1.jsonl")
    v2_prompt = agent.SERVICE_SYSTEM_PROMPT
    versions = [
        ("v0", v0_prompt, v0_cases),
        ("v1", v1_prompt, v1_cases),
        ("v2", v2_prompt, None),
    ]

    engine = RetrievalEngine()
    engine.ensure_index()
    original_prompt = agent.SERVICE_SYSTEM_PROMPT
    records = []
    try:
        for version, prompt, saved_cases in versions:
            agent.SERVICE_SYSTEM_PROMPT = prompt
            records.append({
                "record_type": "meta",
                "version": version,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            })
            for scenario in SCENARIOS:
                history: list[dict[str, str]] = []
                if saved_cases is None:
                    initial = call_service(engine, scenario["initial"], history)
                else:
                    saved = saved_cases[scenario["id"]]
                    initial = {
                        "question": saved["question"],
                        "answer": saved["answer"],
                        "answer_chars": saved["answer_chars"],
                        "elapsed_s": saved["elapsed_s"],
                        "tool_calls": saved["tool_calls"],
                        "turns": saved["turns"],
                        "risk_signals": saved["risk_signals"],
                    }
                initial.update({
                    "record_type": "turn",
                    "version": version,
                    "scenario_id": scenario["id"],
                    "turn": 1,
                    "reused_saved_answer": saved_cases is not None,
                })
                records.append(initial)
                history.extend([
                    {"role": "user", "content": initial["question"]},
                    {"role": "assistant", "content": initial["answer"]},
                ])

                for turn, question in enumerate(scenario["followups"], start=2):
                    record = call_service(engine, question, history)
                    record.update({
                        "record_type": "turn",
                        "version": version,
                        "scenario_id": scenario["id"],
                        "turn": turn,
                        "reused_saved_answer": False,
                    })
                    records.append(record)
                    history.extend([
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": record["answer"]},
                    ])
                    print(
                        f'{version}/{scenario["id"]}/T{turn}: '
                        f'{record["elapsed_s"]}s, {record["answer_chars"]} chars'
                    )
    finally:
        agent.SERVICE_SYSTEM_PROMPT = original_prompt

    OUTPUT.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
