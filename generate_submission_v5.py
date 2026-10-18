"""
V5 批量生成提交文件。

V5 相比 V4 仅做 prompt 层改动（agent.py）：
- 枚举/部件题保护改为窄触发：只命中"有哪些/列出/列举性"问句，不再覆盖"如何/怎样/操作步骤"
- 图编号禁令：禁止在正文写"图N / Figure N / 第N张图"，需要回指写"上图/下图"
- 锚点筛选规则：只保留正文实际描述到的图，不再为"完整"堆砌整章图

V4 相比 V3.5 的核心变化不是单点 prompt 微调，而是：
- 软路由升级：增强别名、泛词降级、query 扩展、内容投票加权
- 召回重构：保留主 query 的 `BM25 20 + dense 20`，并给每个 keyword phrase 追加 `top5 + top5` 补召回
- rerank 改为按 rank 截断，不再使用固定分数阈值
- 指定 `products` 时先在产品子集里召回，避免目标产品被全库热门片段提前挤出

输出：
- submissions/v4_submit.csv         最终提交
- submissions/v4_submit.raw.jsonl   每题原始结果（answer + pics + 工具调用数 + 耗时）
- submissions/v4_submit.trace.jsonl 每题精简调用链（产品路由 + LLM/工具决策摘要）
"""

from __future__ import annotations

import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# 默认启用 highspeed，但允许命令行/外部环境覆盖
os.environ.setdefault("DISABLE_HIGHSPEED", "0")
os.environ.setdefault("HIGHSPEED_MAX_CONCURRENCY", "20")

from dotenv import load_dotenv
load_dotenv(override=False)

# 若外部已显式指定路由并发/禁用策略，则保留外部设置

from agent import run_agent
from agent import SERVICE_SYSTEM_PROMPT, _extract_text_from_response
from llm_router import describe_routes, create_message_with_fallback
from retrieval_engine import RetrievalEngine
from submission_utils import format_submission_ret, is_customer_service_question

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "submissions"
OUT_DIR.mkdir(exist_ok=True)

# V4 仍支持 --out-prefix 切换输出文件名（便于同版本代码跑不同模型/实验）
DEFAULT_OUT_PREFIX = "v4_submit"
OUT_CSV = OUT_DIR / f"{DEFAULT_OUT_PREFIX}.csv"
OUT_RAW = OUT_DIR / f"{DEFAULT_OUT_PREFIX}.raw.jsonl"
OUT_TRACE = OUT_DIR / f"{DEFAULT_OUT_PREFIX}.trace.jsonl"

MAX_WORKERS = 8
PER_Q_TIMEOUT = 180
MAX_RETRIES = 6
SERVICE_DEFAULT_ANSWER: str | None = None

# ─── 客服多轮对话支持 ───
QUOTES = '""'  # 直引号 + 中文弯引号


def split_turns(question: str) -> list[str]:
    """通用多轮拆分：question 由 N 行独立的引号包裹字符串组成时返回 N 段，否则原样返回单段。"""
    parts: list[str] = []
    for line in question.splitlines():
        s = line.strip().rstrip(",，").strip()
        if not s:
            continue
        if s[0] in QUOTES and s[-1] in QUOTES:
            inner = s[1:-1].replace('\\"', '"').strip()
            if inner:
                parts.append(inner)
        else:
            return [question]
    return parts if len(parts) >= 2 else [question]


def is_multi_turn_service(qid: int, question: str) -> bool:
    """客服多轮题判定：客服硬路由命中 + question 是结构化多行引号格式。"""
    if not is_customer_service_question(qid):
        return False
    return len(split_turns(question)) >= 2


def call_llm_service(messages: list[dict], max_tokens: int = 8000) -> str:
    """调用 LLM 生成客服回答。"""
    response, _route = create_message_with_fallback(
        max_tokens=max_tokens,
        system=SERVICE_SYSTEM_PROMPT,
        messages=messages,
    )
    return _extract_text_from_response(response).strip()


def is_retryable_error(exc: Exception) -> bool:
    text = str(exc).lower()
    markers = [
        "429",
        "rate_limit_error",
        "rate limit",
        "当前请求量较高",
        "too many requests",
        "temporarily overloaded",
        "connection error",
        "api connection error",
        "apiconnectionerror",
        "read timeout",
        "readtimeout",
        "timed out",
        "timeout",
        "remoteprotocolerror",
        "protocol error",
        "max retries exceeded",
        "failed to establish a new connection",
        "connection reset",
        "connection aborted",
        "server disconnected",
        "temporary failure",
        "exceeded max_turns",
        "工具调用次数达到上限",
    ]
    return any(marker in text for marker in markers)


def retry_sleep_seconds(attempt: int) -> int:
    schedule = [10, 20, 40, 60, 90, 120]
    idx = min(attempt, len(schedule) - 1)
    return schedule[idx]


def load_questions() -> list[dict]:
    questions = []
    with open(ROOT / "question_public.csv", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row or not row[0].strip().isdigit():
                continue
            questions.append({"id": int(row[0]), "question": row[1]})
    return questions


def process_one(engine: RetrievalEngine, q: dict) -> dict:
    t0 = time.time()
    qid = q["id"]
    question = q["question"]

    if SERVICE_DEFAULT_ANSWER is not None and is_customer_service_question(qid):
        return {
            "id": qid,
            "question": question,
            "answer": SERVICE_DEFAULT_ANSWER,
            "pics": [],
            "tool_calls": 0,
            "turns": 0,
            "elapsed": round(time.time() - t0, 2),
            "error": None,
            "trace": {
                "id": qid,
                "question": question,
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "events": [{"kind": "service_default_answer"}],
                "result": {"answer": SERVICE_DEFAULT_ANSWER, "pics": []},
                "error": None,
                "elapsed": round(time.time() - t0, 2),
            },
        }

    # 客服多轮题：使用两轮对话生成
    if is_multi_turn_service(qid, question):
        return _process_multi_turn_service(qid, question, t0)

    # 其他题（客服单轮 + 技术题）：使用 agent 跑
    for attempt in range(MAX_RETRIES):
        try:
            result = run_agent(question, engine, question_id=qid, collect_trace=True)
            return {
                "id": qid,
                "question": question,
                "answer": result.answer,
                "pics": result.pics,
                "tool_calls": result.tool_calls,
                "turns": result.turns,
                "elapsed": round(time.time() - t0, 2),
                "error": None,
                "trace": result.trace,
            }
        except Exception as e:
            if attempt < MAX_RETRIES - 1 and is_retryable_error(e):
                sleep_s = retry_sleep_seconds(attempt)
                print(f"  id={qid} 命中限流，{sleep_s}s 后重试（{attempt + 1}/{MAX_RETRIES}）", flush=True)
                time.sleep(sleep_s)
                continue
            return {
                "id": qid,
                "question": question,
                "answer": "拒绝回答",
                "pics": [],
                "tool_calls": 0,
                "turns": 0,
                "elapsed": round(time.time() - t0, 2),
                "error": str(e)[:300],
                "trace": {
                    "id": qid,
                    "question": question,
                    "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "events": [],
                    "result": None,
                    "error": {"message": str(e)[:300], "type": type(e).__name__},
                    "elapsed": round(time.time() - t0, 2),
                },
            }


def _process_multi_turn_service(qid: int, question: str, t0: float) -> dict:
    """处理客服多轮题：先回答第一个子问题，再把上下文传给第二轮。"""
    turns = split_turns(question)
    assert len(turns) >= 2, f"qid={qid} 被判定为多轮题但只有 {len(turns)} 段"

    # 第一轮：只回答第一个子问题
    msgs1 = [{"role": "user", "content": turns[0]}]
    a1 = call_llm_service(msgs1)

    # 第二轮：带着第一轮回答，回答第二个子问题
    msgs2 = [
        {"role": "user", "content": turns[0]},
        {"role": "assistant", "content": a1},
        {"role": "user", "content": turns[1]},
    ]
    a2 = call_llm_service(msgs2)

    combined = f"{a1}\n\n{a2}"
    return {
        "id": qid,
        "question": question,
        "turns_split": turns,
        "answer_turn1": a1,
        "answer_turn2": a2,
        "answer": combined,
        "pics": [],
        "tool_calls": 0,
        "turns": 2,
        "elapsed": round(time.time() - t0, 2),
        "error": None,
        "trace": {
            "id": qid,
            "question": question,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "events": [
                {"kind": "multi_turn_split", "turns": turns},
                {"kind": "llm_call", "turn": 1, "answer_len": len(a1)},
                {"kind": "llm_call", "turn": 2, "answer_len": len(a2)},
            ],
            "result": {"answer_preview": combined[:200]},
        },
    }


def append_raw(rec: dict) -> None:
    with open(OUT_RAW, "a", encoding="utf-8") as f:
        raw_rec = {k: v for k, v in rec.items() if k != "trace"}
        f.write(json.dumps(raw_rec, ensure_ascii=False) + "\n")


def append_trace(rec: dict) -> None:
    trace = rec.get("trace")
    if trace is None:
        return
    with open(OUT_TRACE, "a", encoding="utf-8") as f:
        f.write(json.dumps(trace, ensure_ascii=False) + "\n")


def write_submission(records: dict[int, dict]) -> None:
    qs = load_questions()
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["id", "ret"])
        for q in qs:
            rec = records.get(q["id"])
            if rec is None:
                w.writerow([q["id"], "拒绝回答"])
                continue
            ret = format_submission_ret(q["id"], rec["answer"], rec["pics"])
            w.writerow([q["id"], ret])


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 个（0=全部）")
    parser.add_argument("--ids", type=str, default="", help="逗号分隔的 id 列表，仅跑这些")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--resume", action="store_true", help="跳过 raw.jsonl 中已完成的")
    parser.add_argument("--retry-errors", action="store_true", help="只重跑 raw.jsonl 里 error 非空的")
    parser.add_argument("--reset", action="store_true", help="清空 raw.jsonl 从头跑")
    parser.add_argument("--rewrite-only", action="store_true",
                        help="仅基于现有 raw.jsonl 重写 CSV，不调用 LLM")
    parser.add_argument("--out-prefix", type=str, default=DEFAULT_OUT_PREFIX,
                        help="输出文件前缀，如 'v4_submit' → submissions/v4_submit.{csv,raw.jsonl}")
    parser.add_argument("--cs-only", action="store_true",
                        help="仅跑客服题（qid < 64），适用于验证多轮对话合并后的效果")
    parser.add_argument("--service-default-answer", type=str, default="",
                        help="qid < 64 时不调用 LLM，直接填入该答案；为空则保持原客服逻辑")
    args = parser.parse_args()

    # 根据 --out-prefix 覆盖全局输出路径
    global OUT_CSV, OUT_RAW, OUT_TRACE, SERVICE_DEFAULT_ANSWER
    OUT_CSV = OUT_DIR / f"{args.out_prefix}.csv"
    OUT_RAW = OUT_DIR / f"{args.out_prefix}.raw.jsonl"
    OUT_TRACE = OUT_DIR / f"{args.out_prefix}.trace.jsonl"
    SERVICE_DEFAULT_ANSWER = args.service_default_answer or None

    if args.reset and OUT_RAW.exists():
        OUT_RAW.unlink()
        print(f"已清空 {OUT_RAW}")
    if args.reset and OUT_TRACE.exists():
        OUT_TRACE.unlink()
        print(f"已清空 {OUT_TRACE}")

    records: dict[int, dict] = {}
    done_ids: set[int] = set()

    if args.rewrite_only:
        if not OUT_RAW.exists():
            raise SystemExit(f"{OUT_RAW} 不存在，无法 rewrite")
        with open(OUT_RAW, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                records[rec["id"]] = rec
        write_submission(records)
        print(f"已根据 {len(records)} 条 raw 记录重写 {OUT_CSV}")
        return

    print("LLM 路由配置：")
    for r in describe_routes():
        print(f"  - {r['name']:12s} model={r['model']:30s} concurrency={r['concurrency']}")

    engine = RetrievalEngine()
    engine.ensure_index()
    print(f"索引: {len(engine.retrieval_chunks)} 块, {len(engine.catalog)} 产品")

    questions = load_questions()
    if args.cs_only:
        questions = [q for q in questions if q["id"] < 64]
        print(f"客服题模式：仅跑 {len(questions)} 道客服题")
    if args.ids:
        want = {int(x) for x in args.ids.split(",") if x.strip()}
        questions = [q for q in questions if q["id"] in want]
    if args.limit > 0:
        questions = questions[: args.limit]

    if args.retry_errors and OUT_RAW.exists():
        retry_ids: set[int] = set()
        with open(OUT_RAW, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("error"):
                    retry_ids.add(rec["id"])
                else:
                    records[rec["id"]] = rec
                    done_ids.add(rec["id"])
        retry_ids -= done_ids
        with open(OUT_RAW, "w", encoding="utf-8") as f:
            for rec in records.values():
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"retry-errors: 保留 {len(done_ids)} 成功, 重跑 {len(retry_ids)} 失败")
        questions = [q for q in questions if q["id"] in retry_ids]
    elif args.resume and OUT_RAW.exists():
        with open(OUT_RAW, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                records[rec["id"]] = rec
                done_ids.add(rec["id"])
        print(f"resume: 已完成 {len(done_ids)} 题，跳过")
        questions = [q for q in questions if q["id"] not in done_ids]

    print(f"待处理: {len(questions)} 题，并发 {args.workers}")
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, engine, q): q["id"] for q in questions}
        done = 0
        for fut in as_completed(futures):
            rec = fut.result()
            records[rec["id"]] = rec
            append_raw(rec)
            append_trace(rec)
            done += 1
            tag = "✗" if rec["error"] else "✓"
            if done % 10 == 0 or done <= 5 or rec["error"]:
                elapsed = time.time() - t_start
                eta = (elapsed / done) * (len(questions) - done) if done else 0
                print(f"  [{done}/{len(questions)}] {tag} id={rec['id']} "
                      f"tools={rec['tool_calls']} pics={len(rec['pics'])} "
                      f"t={rec['elapsed']}s  total={elapsed:.0f}s eta={eta:.0f}s",
                      flush=True)

    write_submission(records)
    total = time.time() - t_start
    errors = sum(1 for r in records.values() if r.get("error"))
    print(f"\n完成: {len(records)} 题，{errors} 错，耗时 {total:.0f}s")
    print(f"提交文件: {OUT_CSV}")
    print(f"原始记录: {OUT_RAW}")
    print(f"调用链记录: {OUT_TRACE}")


if __name__ == "__main__":
    main()
