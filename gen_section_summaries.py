"""离线给每个 section 生成「图文合一」的功能总结。

历史主体产物由GPT-5.5生成；2026-09-08对8个编号位移产品的373条摘要使用当前配置的
DeepSeek模型重建，具体范围、哈希和备份见 validation_outputs/section_summary_cleanup.json。

为什么：章节目录 summary 曾只是正文前 120 字截断（summarize_text），
约 11% 以警告/套话开头、信息量为零，导致大模型在目录里选不准该读哪节。
本脚本把 section 正文 + 该节所有配图的 caption 一起喂给配置的主回答模型，产出一句话功能总结，
让"只看这一句就能判断要不要打开本节"。

产物：data/section_summaries.json  {"产品|section_id": "总结"}（可断点续跑）。
落地：加载索引时挂到 section["llm_summary"]；当前已建BM25/Dense和Agent返回的
section_summary仍使用Chunk/Section原summary，因此单独更新本文件不要求重建索引。
"""
from __future__ import annotations
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=False)

from llm_router import create_message_with_fallback
from agent import _extract_text_from_response

ROOT = Path(__file__).resolve().parent
SECTIONS = json.load(open(ROOT / "data" / "section_chunks.json", encoding="utf-8"))
CAPS = json.load(open(ROOT / "data" / "image_captions_v4_final.json", encoding="utf-8")).get("items", {})
OUT = ROOT / "data" / "section_summaries.json"

SYS_ZH = (
    "你是产品手册目录编纂助手。给定一个章节的正文和它的配图说明，用一句话（≤70字）概括本节功能，"
    "让人只看这一句就能判断要不要打开本节。必须同时覆盖：正文讲的关键部件/动作/规格/步骤，"
    "以及配图展示的内容（例如『含3张分步图』『含规格参数表』『含部件标注图』）。"
    "只输出这一句，不要任何前缀、引号或解释。"
)
SYS_EN = (
    "You are a manual table-of-contents editor. Given a section's body text and its image captions, "
    "write ONE concise line (<=40 words) that lets a reader decide whether to open this section. "
    "Cover both the key parts/actions/specs/steps in the text AND what the images show "
    "(e.g. 'includes 3 step diagrams', 'includes a spec table', 'includes a labeled parts view'). "
    "Output only that single line, no prefix, quotes, or explanation."
)


def caption_block(product: str, pics: list[str]) -> str:
    """Build a compact caption context block for one section's images."""
    out = []
    for p in pics or []:
        it = CAPS.get(f"{product}|{p}")
        if not it:
            continue
        cat = it.get("category", "")
        short = (it.get("short_caption") or "").strip()
        content = (it.get("content") or "").strip()
        seg = f"[{cat}] {short}"
        if cat == "info_table" and content:
            seg += f" 表内容:{content[:200]}"
        out.append(seg)
    return "\n".join(out)


def build_one(sec: dict) -> str:
    """Ask the configured LLM to summarize one manual section."""
    zh = sec.get("lang") != "en"
    heading = " / ".join(sec.get("heading_path") or [sec.get("heading", "")])
    body = (sec.get("text") or "").strip()[:2500]
    caps = caption_block(sec["product"], sec.get("pics") or [])
    parts = [f"章节标题: {heading}", f"正文:\n{body}"]
    if caps:
        parts.append(f"本节配图说明（{sec.get('pic_count',0)}张）:\n{caps}")
    else:
        parts.append("本节无配图。")
    user = "\n\n".join(parts)
    resp, _ = create_message_with_fallback(
        system=SYS_ZH if zh else SYS_EN,
        messages=[{"role": "user", "content": user}],
        max_tokens=300,
    )
    return _extract_text_from_response(resp).strip().strip('"').strip()


def main():
    """Resume or create data/section_summaries.json for all manual sections."""
    workers = int(os.getenv("SUMMARY_WORKERS", "8"))
    done = {}
    if OUT.exists():
        done = json.loads(OUT.read_text(encoding="utf-8"))
    todo = [s for s in SECTIONS if f"{s['product']}|{s['section_id']}" not in done]
    print(f"全量 {len(SECTIONS)} 节，已完成 {len(done)}，待跑 {len(todo)}，并发 {workers}", flush=True)

    lock_n = [0]
    t0 = time.time()

    def work(sec):
        """Run one section summary task with a few conservative retries."""
        key = f"{sec['product']}|{sec['section_id']}"
        for attempt in range(4):
            try:
                return key, build_one(sec)
            except Exception as e:
                if attempt < 3:
                    time.sleep(8 * (attempt + 1)); continue
                return key, f"__ERROR__ {str(e)[:120]}"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(work, s): s for s in todo}
        for fut in as_completed(futs):
            key, summ = fut.result()
            done[key] = summ
            lock_n[0] += 1
            if lock_n[0] % 20 == 0:
                OUT.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
                el = time.time() - t0
                eta = el / lock_n[0] * (len(todo) - lock_n[0])
                print(f"  [{lock_n[0]}/{len(todo)}] {key} :: {summ[:50]}  el={el:.0f}s eta={eta:.0f}s", flush=True)
    OUT.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
    errs = sum(1 for v in done.values() if v.startswith("__ERROR__"))
    print(f"完成 {len(done)} 节，{errs} 错，耗时 {time.time()-t0:.0f}s → {OUT}", flush=True)


if __name__ == "__main__":
    main()
