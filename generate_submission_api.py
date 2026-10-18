#!/usr/bin/env python3
"""Run question_public.csv through the /chat API endpoint.

This script intentionally does not use qid for routing. It sends normal /chat
requests and only uses id for output ordering/session ids. Multi-line quoted
customer-service questions are sent as multiple turns with the same session_id,
then the answers are concatenated for the submission ret.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "submissions"
OUT_DIR.mkdir(exist_ok=True)

DEFAULT_OUT_PREFIX = "chat_api_submit"
QUOTES = '"“”'


def split_turns(question: str) -> list[str]:
    """Split CSV cells like '"turn1",\n"turn2"' into separate user turns."""
    parts: list[str] = []
    for line in question.splitlines():
        s = line.strip().rstrip(",，").strip()
        if not s:
            continue
        if len(s) >= 2 and s[0] in QUOTES and s[-1] in QUOTES:
            inner = s[1:-1].replace('\\"', '"').strip()
            if inner:
                parts.append(inner)
        else:
            return [question]
    return parts if len(parts) >= 2 else [question]


def load_questions() -> list[dict[str, Any]]:
    """Load question_public.csv while preserving qid for output ordering only."""
    questions: list[dict[str, Any]] = []
    with (ROOT / "question_public.csv").open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row or not row[0].strip().isdigit():
                continue
            questions.append({"id": int(row[0]), "question": row[1]})
    return questions


def post_chat(url: str, token: str, question: str, session_id: str, timeout_s: float, stream: bool = False) -> dict[str, Any]:
    """POST one turn to /chat and return the decoded JSON response."""
    payload = json.dumps({
        "question": question,
        "session_id": session_id,
        "stream": stream,
    }, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {token}",
            "X-Request-Id": f"kf_batch_{session_id}_{int(time.time() * 1000)}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"URL error: {exc}") from exc


def _response_summary(resp: dict[str, Any]) -> dict[str, Any]:
    """Keep compact response metadata for client-side trace files."""
    data = resp.get("data") or {}
    answer = data.get("answer") or ""
    return {
        "code": resp.get("code"),
        "msg": resp.get("msg"),
        "session_id": data.get("session_id"),
        "timestamp": data.get("timestamp"),
        "answer_len": len(answer),
        "answer_preview": answer[:300],
    }


def process_one(q: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Send one public question through /chat, including split multi-turn cells."""
    qid = q["id"]
    question = q["question"]
    session_id = f"kf_batch_q{qid}"
    t0 = time.time()
    turns = split_turns(question)
    answers: list[str] = []
    responses: list[dict[str, Any]] = []
    turn_records: list[dict[str, Any]] = []
    error = None

    try:
        for idx, turn in enumerate(turns, start=1):
            turn_t0 = time.time()
            resp = post_chat(
                args.chat_url,
                args.api_token,
                turn,
                session_id,
                args.timeout,
                stream=False,
            )
            turn_elapsed = round(time.time() - turn_t0, 3)
            responses.append(resp)
            if resp.get("code") != 0:
                raise RuntimeError(f"API code={resp.get('code')} msg={resp.get('msg')}")
            data = resp.get("data") or {}
            answer = (data.get("answer") or "").strip()
            if not answer:
                raise RuntimeError("API returned empty answer")
            answers.append(answer)
            turn_records.append({
                "turn": idx,
                "question": turn,
                "elapsed": turn_elapsed,
                "answer": answer,
                "answer_len": len(answer),
                "response": resp,
            })
    except Exception as exc:  # noqa: BLE001
        error = str(exc)[:500]

    final_answer = "\n\n".join(answers) if answers else "拒绝回答"
    return {
        "id": qid,
        "question": question,
        "turns_split": turns,
        "turn_count": len(turns),
        "answer": final_answer,
        "ret": final_answer,
        "tool_calls": 0,
        "turns": len(turns),
        "elapsed": round(time.time() - t0, 3),
        "error": error,
        "responses": responses,
        "turn_records": turn_records,
        "trace": {
            "id": qid,
            "question": question,
            "session_id": session_id,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "kind": "chat_api_batch",
            "events": [
                {
                    "kind": "chat_api_turn",
                    "turn": tr["turn"],
                    "question": tr["question"],
                    "elapsed": tr["elapsed"],
                    "response": _response_summary(tr["response"]),
                }
                for tr in turn_records
            ],
            "result": {
                "ret": final_answer,
                "answer_len": len(final_answer),
                "turn_count": len(turns),
            },
            "error": error,
            "elapsed": round(time.time() - t0, 3),
        },
    }


def append_raw(rec: dict[str, Any], out_raw: Path) -> None:
    """Append one raw batch record without the verbose trace payload."""
    raw_rec = {k: v for k, v in rec.items() if k != "trace"}
    with out_raw.open("a", encoding="utf-8") as f:
        f.write(json.dumps(raw_rec, ensure_ascii=False) + "\n")


def append_trace(rec: dict[str, Any], out_trace: Path | None) -> None:
    """Append the optional client-side HTTP trace for one batch record."""
    if out_trace is None:
        return
    trace = rec.get("trace")
    if trace is None:
        return
    with out_trace.open("a", encoding="utf-8") as f:
        f.write(json.dumps(trace, ensure_ascii=False) + "\n")


def write_csv(records: dict[int, dict[str, Any]], out_csv: Path) -> None:
    """Write platform-style id/ret CSV in original question order."""
    questions = load_questions()
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["id", "ret"])
        for q in questions:
            rec = records.get(q["id"])
            writer.writerow([q["id"], rec["ret"] if rec else "拒绝回答"])


def write_outputs(records: dict[int, dict[str, Any]], out_csv: Path, out_raw: Path, out_trace: Path | None) -> None:
    """Rewrite CSV/raw/trace outputs from accumulated records."""
    write_csv(records, out_csv)
    with out_raw.open("w", encoding="utf-8") as f:
        for qid in sorted(records):
            rec = records[qid]
            raw_rec = {k: v for k, v in rec.items() if k != "trace"}
            f.write(json.dumps(raw_rec, ensure_ascii=False) + "\n")
    if out_trace is not None:
        with out_trace.open("w", encoding="utf-8") as f:
            for qid in sorted(records):
                trace = records[qid].get("trace")
                if trace is not None:
                    f.write(json.dumps(trace, ensure_ascii=False) + "\n")


def main() -> None:
    """CLI entrypoint for replaying the public CSV through the live /chat API."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat-url", default=os.getenv("CHAT_API_URL", "http://127.0.0.1:8000/chat"))
    parser.add_argument("--api-token", default=os.getenv("KAFU_API_TOKEN", "sk-test"))
    parser.add_argument("--out-prefix", default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--ids", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=float(os.getenv("CHAT_API_CLIENT_TIMEOUT", "35")))
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true",
                        help="与 --resume 类似读取已有 raw，但会重跑缺失题和 error 非空的题")
    parser.add_argument("--rewrite-every", type=int, default=10,
                        help="每完成 N 题重写一次 CSV；0 表示只在最后写")
    parser.add_argument("--client-trace", action="store_true",
                        help="额外写客户端 HTTP 层 trace；默认关闭，内部链路看 API 服务端 trace")
    args = parser.parse_args()

    if not args.api_token:
        raise SystemExit("KAFU_API_TOKEN/--api-token is required")

    out_csv = OUT_DIR / f"{args.out_prefix}.csv"
    out_raw = OUT_DIR / f"{args.out_prefix}.raw.jsonl"
    out_trace = OUT_DIR / f"{args.out_prefix}.trace.jsonl" if args.client_trace else None
    if args.reset and out_raw.exists():
        out_raw.unlink()
    if args.reset and out_trace is not None and out_trace.exists():
        out_trace.unlink()

    records: dict[int, dict[str, Any]] = {}
    if (args.resume or args.retry_errors) and out_raw.exists():
        with out_raw.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                records[int(rec["id"])] = rec
        print(f"resume: loaded {len(records)} existing records")
        if args.retry_errors:
            error_ids = sorted(qid for qid, rec in records.items() if rec.get("error"))
            if error_ids:
                records = {qid: rec for qid, rec in records.items() if not rec.get("error")}
                print(f"retry-errors: 清理旧错误记录 {len(error_ids)} 条，首批 {error_ids[:20]}")
                write_outputs(records, out_csv, out_raw, out_trace)

    questions = load_questions()
    if args.ids:
        want = {int(x) for x in args.ids.split(",") if x.strip()}
        questions = [q for q in questions if q["id"] in want]
    if args.limit > 0:
        questions = questions[:args.limit]
    if args.resume:
        questions = [q for q in questions if q["id"] not in records]
    if args.retry_errors:
        questions = [
            q for q in questions
            if q["id"] not in records or records[q["id"]].get("error")
        ]

    print(f"POST {args.chat_url}")
    print(f"待处理: {len(questions)} 题，并发 {args.workers}；qid 仅用于输出，不参与路由")
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, q, args): q["id"] for q in questions}
        done = 0
        for fut in as_completed(futures):
            rec = fut.result()
            records[rec["id"]] = rec
            append_raw(rec, out_raw)
            append_trace(rec, out_trace)
            done += 1
            tag = "✗" if rec["error"] else "✓"
            if done <= 5 or done % 10 == 0 or rec["error"]:
                elapsed = time.time() - t_start
                eta = (elapsed / done) * (len(questions) - done) if done else 0
                print(
                    f"  [{done}/{len(questions)}] {tag} id={rec['id']} "
                    f"turns={rec['turn_count']} t={rec['elapsed']}s total={elapsed:.0f}s eta={eta:.0f}s"
                    + (f" err={rec['error'][:120]}" if rec["error"] else ""),
                    flush=True,
                )
            if args.rewrite_every > 0 and done % args.rewrite_every == 0:
                write_csv(records, out_csv)

    write_outputs(records, out_csv, out_raw, out_trace)
    errors = sum(1 for rec in records.values() if rec.get("error"))
    print(f"\n完成: {len(records)} 题，{errors} 错，耗时 {time.time() - t_start:.0f}s")
    print(f"提交文件: {out_csv}")
    print(f"原始记录: {out_raw}")
    if out_trace is not None:
        print(f"客户端调用链记录: {out_trace}")


if __name__ == "__main__":
    main()
