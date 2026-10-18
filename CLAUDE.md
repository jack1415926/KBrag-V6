# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

KBrag V6 is a multimodal customer-service Agent used for source-level reproduction, RAG learning, and retrieval optimization. It answers ordinary service questions directly or retrieves evidence from 39 product manuals (20 Chinese in `手册_v4/`, 19 English) and returns text with image references. The codebase is primarily Chinese. This is a runnable project snapshot rather than a git repository: there is no CI or `pytest` suite, but a focused stdlib `unittest` suite covers request context/concurrency/timeout behavior, and the pre-built indexes/data allow local reproduction without rebuilding first.

## Commands

Python 3.10+; the current `.venv` is verified on Python 3.12.4. Install deps first: `pip install -r requirements.txt`.

```bash
./run_api.sh              # start FastAPI server (uvicorn api_server:app, port 8000)
./smoke_test.sh           # hit /health and /chat, assert OK (server must be running)

python agent.py "你的问题"      # run a single question through the agent (CLI)
python agent.py -i             # interactive REPL mode

python validate_delivery.py    # static self-check (files, syntax, secret sprawl, comment coverage)
python validate_performance.py                 # regenerate validation report CSVs (offline)
python validate_performance.py --api-base-url http://127.0.0.1:8000   # + online dialogue check

python -m unittest discover -s tests -p "test_request_context.py" -v  # request identity + cooperative timeout tests
python -m unittest discover -s tests -p "test_visual_retrieval.py" -v  # visual routing/fusion/fallback tests
python -m unittest discover -s tests -p "test_search_manual_trace.py" -v  # formal search phase timings
python -m unittest discover -s tests -p "test_api_images.py" -v  # image ID/route/response contract
python run_search_trace_baseline.py --runs 5 --pause 1  # real remote search baseline; explicit external-data authorization required
python run_embedding_request_mode_experiment.py --repeats 5 --pause 1  # serial/batch/parallel real-service comparison; explicit authorization required
python run_text_query_embed_mode_gold.py --repeats 3 --pause 0.5  # 30-gold serial/batch acceptance; explicit authorization required
python run_text_query_strategy_gold.py --repeats 3 --pause 0.5  # batch full/keywords/hybrid 30-gold comparison; explicit authorization required
python validate_section_gold.py  # verify the frozen 30-query Section relevance set and hashes
python validate_visual_gold.py   # verify 30 derived + 10 external-style visual cases
python validate_visual_index.py  # verify 2,504×4,096 visual FAISS and metadata

python generate_submission_v5.py --ids 246,303,401 --workers 3 --out-prefix v6_smoke --reset   # offline CSV batch

python build_retrieval_index.py  # rebuild BM25+FAISS index — NOT needed normally; data/index/ is pre-built
```

There is no broad unit-test suite or pytest setup. Current checks are the focused stdlib suites under `tests/`, `smoke_test.sh` (end-to-end), `validate_delivery.py` / `validate_performance.py` (static/offline checks), and `python submission_utils.py` (self-check in its `__main__`).

## Architecture

The system is a **two-track ReAct agent**. Every question resolves to one of two tracks, selected by `question_id`:

- `qid < 64` → **customer-service track** (pure LLM, no retrieval; `SERVICE_SYSTEM_PROMPT`).
- `qid >= 64` → **technical track** (RAG over manuals; `TECH_SYSTEM_PROMPT`).

Online `/chat` has no qid, so `api_server.py` classifies first (see flow below) and maps to a fake qid (0 or 64).

### Request flow (`api_server.py` → `agent.py`)

```
POST /chat
  → Bearer-token auth (KAFU_API_TOKEN)
  → classify service vs tech: DeepSeek "3-way binary vote" (_classify_question)
        fallback = local regex rules (_local_fallback_route)
  → fake_qid = 0 (service) | 64 (tech)
  → run_agent(question, engine, question_id=fake_qid)
  → format_submission_ret()   # normalize answer + pics
  → {code, msg, data:{answer, session_id, timestamp}}
```

`api_server.py` is a **thin wrapper** — it deliberately reuses `run_agent` + `format_submission_ret` so online answers are byte-identical in format to offline CSV submission. It owns auth, validation, timeout (504), in-memory session history, and JSONL trace logging.

Request identity and timeout state live in `request_context.py`. Per-request qid uses `ContextVar`, not a process environment variable. The API propagates one monotonic deadline and a cooperative cancellation event through LLM, embedding, rerank, semaphore wait, and retry backoff. A 504 does not safely kill a Python thread immediately: an in-flight socket call exits on its capped downstream timeout, then the worker stops at the next cancellation check. `/health.background_agents` reports timed-out/disconnected workers still exiting; normal steady state is `0`.

### Agent loop (`agent.py:run_agent`)

1. **Product routing** — `product_router.ProductRouter.route()` picks which manual(s) to search (explicit name → alias → phrase-grep → content vote → BM25 fallback). Only constrains the search scope; the agent can widen with `products=[]`.
2. **Pre-retrieval** — dense recall over the routed product, injected as a hint (not a substitute for tool calls).
3. **ReAct loop** — max `MAX_TURNS=2` tool-turning rounds, single tool `search_manual`. A state machine (`run_agent` locals: `search_attempts`, `empty_search_streak`, `low_gain_streak`, `section_focus_streak`, …) emits Chinese "route notes" steering the LLM toward/away from products and headings. Exceeding turns → `_finalize_without_tools`.
4. **Postprocess** — `_postprocess_final_answer` then `_resolve_answer_pics`.

### Retrieval (`retrieval_engine.py`)

Hybrid: **BM25** (jieba + `rank_bm25`) + **dense** (SiliconFlow `Qwen/Qwen3-Embedding-8B`, 4096 dimensions, over a `faiss.IndexFlatIP`) → **RRF** merge → **rerank** (`Qwen/Qwen3-Reranker-8B` cross-encoder). Chunks locate the answer, but results return the **full parent section** (`RETURN_PARENT_SECTION=1`) so the model sees the complete topic. Embedding/rerank are remote APIs by default; `embedding_server.py` / `rerank_server.py` are optional local fallbacks only. The original 1024-dimensional BGE-M3 index is preserved under `data/index_archives/bge-m3-original-20260607/`.

Online text-query embedding is switchable with `TEXT_QUERY_EMBED_MODE=serial|batch`; the delivery default is now `batch`, with `serial` retained as an environment override. Batch mode only groups the dynamic text inputs the formal path already selected into one `/embeddings` request, then searches each returned vector separately; it does not generate extra keywords or change RRF inputs.

### LLM routing (`llm_router.py`)

`agent.py` speaks one **Anthropic-shaped** message format. `llm_router` translates it to OpenAI-compatible DeepSeek calls. Text requests use `deepseek-v4-flash`; requests with uploaded images use `deepseek-v4-flash-vision-exp`. `_convert_messages_for_openai` handles the Anthropic↔OpenAI tool/multimodal mapping. Technical requests require a tool call until formal retrieval is confirmed; service requests receive no tools at code level.

## The `[[PIC:...]]` / `<PIC>` answer format contract

This is the most important cross-file convention:

- **Internal (LLM-facing)**: answers carry inline image anchors `[[PIC:Manual04_22]]`.
- **Final (submission)**: `submission_utils.format_submission_ret` converts them — for tech questions the `ret` field becomes `正文...,["Manual04_22","Manual04_23"]` (a JSON list appended after a comma); `<PIC>` markers in the text + the pics list. For service questions, all images are stripped and contact info (phone/URL/email/微信/官网) is removed.
- **Online image access**: `/chat` preserves the existing `answer` string and additionally returns ordered `data.images` entries (`id`, `/manual-images/{id}`, MIME, marker index). `GET /manual-images/{image_id}` resolves only the startup-built JPG/PNG allowlist; it never treats the path parameter as a filesystem path.
- **Browser display**: `data.display_answer` keeps `<PIC>` markers without the legacy trailing image array. `GET /` serves `web/index.html`, which inserts ordered `data.images`; `/latency-baseline/status` reads only local progress written by the standalone real-search runner.
- `normalize_technical_answer` / `normalize_customer_service_answer` also strip markdown, leaked filenames, and unsupported Unicode.

## Configuration

`config_runtime.py` holds only non-secret defaults. `apply_default_env()` loads the ignored local `.env` without overriding exported environment variables, then fills non-secret defaults with `os.environ.setdefault(...)`. Keep credentials in `.env` or environment variables only; `.env.example` contains empty credential fields. `validate_delivery.py` scans distributable text, including data and configuration, for secret patterns without credential allowlists; it skips local-only environment files and directories.

## Key data files (`data/`)

`catalog.json` (product catalog), `retrieval_chunks.json` + `section_chunks.json` (chunks), `section_summaries.json` (1,943/1,943 aligned LLM summaries after the 2026-09-08 cleanup, loaded onto `section["llm_summary"]` at index load), `image_captions_v4_final.json` (per-image captions, injected as "图片内容标注" evidence), `index/dense.faiss` + `index/retrieval_index.pkl` (pre-built indexes loaded by `RetrievalEngine.ensure_index()`). The loaded `llm_summary` is currently metadata only: cached BM25/Dense text and Agent `section_summary` still use the original chunk/section `summary`, so summary-only cleanup does not require an index rebuild. Manuals live in `手册_v4/` (Markdown); `手册/插图/` has images.

The visual-FAISS four-path code, 2,504×4,096 independent index, and 40-case gold set now exist; the gold provenance is 30 manual-derived, 6 GPT Image synthetic, 1 user photo, and 3 non-distributable public photos. `VISUAL_RETRIEVAL_ENABLED=0`. The canonical paired result versus the no-visual-FAISS Vision-to-text baseline is Section Recall@5 60.83%→78.33% and Image Recall@5 55.00%→67.50%. Never mix VL vectors into the text index or call the external-style subset "10 real photos".

The manual parser is heading-driven: it recognizes only line-start Markdown headings matching `^(#{1,3})\s+(.*)$`. Flattening headings merges bodies under the previous Section and weakens parent/image boundaries. The current `tokenize_mixed()` also produces duplicate two-character Chinese terms and, when mixed text contains ASCII such as `<PIC>` or `USB`, may add `pic` and whole Chinese clauses through the English fallback. This behavior is verified but its retrieval impact is not; preserve it as the baseline until a fixed Recall/MRR comparison justifies a change.

Retrieval experiments are intentionally limited to a shared frozen 30-query Section relevance set: rerank candidate-pool sizes, three neighboring Chunk parameter bundles, and baseline versus clean BM25 tokenization. The B34/C20/C10 rerank comparison is complete: C20 preserved 30/30 Recall@5 and 0.9611 MRR while reducing rerank median/P95, and the production default is now 20 (environment override remains available). The P120/P160/P200 Chunk comparison is also complete: all groups preserved 30/30 Recall@5 per round; P200 reproducibly lowered three correct Sections from rank 1 to 2, while P120 did not prove a stable gain over P160 and added 32.8% Chunk/FAISS cost, so production remains P160. `tokenize_mixed_clean()` is experimental and must not replace the indexed tokenizer before measured comparison.

Text and VL embedding/rerank clients reuse one `requests.Session` per worker thread. Do not replace this with one globally shared Session: online text/visual branches run concurrently and `requests.Session` is not a cross-thread synchronization primitive.

## Current verified progress

- Completed: Qwen3 model/index migration, RRF/default-product/source fixes, code-level service tool isolation, frozen 30-query Section gold set, C20 production default, P160 retention, Section Summary cleanup, ContextVar/cooperative timeout, formal retrieval stage tracing, dynamic batch query embedding with serial fallback, per-thread HTTP Session reuse, structured image API, and the minimal web UI.
- Current only task: rerun four small end-to-end Agent scenarios on the modified runtime, then freeze one real text and one real image Tool Call for serial/batch replay. Do not start another retrieval/query-strategy experiment before this closes.
- Visual retrieval is implemented and measured but remains a PoC with `VISUAL_RETRIEVAL_ENABLED=0`. The latest two Fusion rounds were 80/80 valid with Section Recall@5 78.75%, Image Recall@5 68.75%, median 5.788s and P95 15.256s; text Embedding and text Rerank produced the tail. That runner used one text input per branch, so it is not evidence for batch speedup. Query-strategy runs based on generated deterministic keywords are exploratory only and must not drive production behavior.
- Human-facing current status and frozen backlog: `项目面试，运行问题说明文档/项目当前进展与下一步.md`.
