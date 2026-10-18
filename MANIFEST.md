# KBrag V6 项目清单

这是 KBrag V6 的源码、知识库、索引与验证材料清单，用于本地复现 `/chat` 服务并开展检索优化实验。

## 包含

### 源码

- `agent.py`
- `request_context.py`
- `api_server.py`
- `llm_router.py`
- `retrieval_engine.py`
- `product_router.py`
- `rerank_client.py`
- `submission_utils.py`
- `generate_submission_v5.py`
- `parse_manuals.py`
- `build_retrieval_index.py`
- `visual_retrieval.py`
- `build_visual_index.py`
- `prepare_visual_gold.py`
- `validate_visual_gold.py`
- `validate_visual_index.py`
- `probe_visual_api.py`
- `smoke_visual_retrieval.py`
- `run_visual_retrieval_experiment.py`
- `run_visual_text_regression.py`
- `validate_visual_experiment.py`
- `gen_section_summaries.py`
- `clean_section_summaries.py`
- `embedding_server.py`
- `rerank_server.py`
- `generate_submission_api.py`
- `validate_delivery.py`
- `validate_performance.py`
- `validate_service_prompt.py`
- `validate_service_prompt_followups.py`
- `validate_section_gold.py`
- `run_rerank_pool_experiment.py`
- `prepare_chunk_parameter_experiment.py`
- `build_chunk_parameter_indexes.py`
- `run_chunk_parameter_experiment.py`
- `tests/test_request_context.py`
- `tests/test_visual_retrieval.py`
- `tests/test_search_manual_trace.py`
- `tests/test_api_images.py`
- `run_search_trace_baseline.py`
- `run_embedding_request_mode_experiment.py`
- `run_text_query_embed_mode_gold.py`
- `run_text_query_strategy_gold.py`
- `web/index.html`

### Prompt / skill

- `skills/search_manual.md`

### 运行数据

- `data/catalog.json`
- `data/section_chunks.json`
- `data/retrieval_chunks.json`
- `data/section_summaries.json`
- `data/section_summaries.pre_cleanup_20260908-083850.json`
- `data/section_aliases.json`
- `data/_product_map.json`
- `data/product_mapping_final.csv`
- `data/image_captions_v4_final.json`
- `data/manual_sections/`
- `data/index/dense.faiss`
- `data/index/retrieval_index.pkl`
- `data/visual_index/visual.faiss`（2,504×4,096，独立于文本索引）
- `data/visual_index/visual_metadata.json`
- `data/visual_index/visual_build_summary.json`
- `data/visual_index/visual_build_progress.jsonl`
- `data/index_archives/bge-m3-original-20260607/dense.faiss`
- `data/index_archives/bge-m3-original-20260607/retrieval_index.pkl`
- `手册_v4/`
- `手册/插图/`

### 启动材料

- `README.md`
- `API.md`
- `requirements.txt`
- `config_runtime.py`
- `.env.example`
- `run_api.sh`
- `smoke_test.sh`
- `项目技术架构说明文档/01_API接口说明.md`
- `项目技术架构说明文档/02_源码运行说明.md`
- `项目技术架构说明文档/03_技术方案说明.md`
- `项目技术架构说明文档/04_验证报告.md`
- `项目技术架构说明文档/05_项目深度解剖与面试答辩白皮书.md`
- `项目技术架构说明文档/06_分词与检索优化实验设计.md`
- `项目技术架构说明文档/07_客服Prompt事实边界优化记录.md`
- `项目技术架构说明文档/08_视觉FAISS四路检索实验与优化复盘.md`
- `项目面试，运行问题说明文档/项目当前进展与下一步.md`
- `validation_outputs/dataset_overview.csv`
- `validation_outputs/routing_validation.csv`
- `validation_outputs/product_router_summary.csv`
- `validation_outputs/product_router_details.csv`
- `validation_outputs/multimodal_format_summary.csv`
- `validation_outputs/multimodal_format_details.csv`
- `validation_outputs/dialogue_api_validation.csv`
- `validation_outputs/validation_summary.md`
- `validation_outputs/validation_summary.json`
- `validation_outputs/service_prompt_before.jsonl`
- `validation_outputs/service_prompt_after_v1.jsonl`
- `validation_outputs/service_prompt_followups.jsonl`
- `validation_outputs/service_prompt_full_transcript.md`
- `validation_outputs/service_prompt_followups_full_transcript.md`
- `validation_outputs/service_prompt_v2_full_transcript.md`
- `validation_outputs/service_prompt_v2_review.json`
- `validation_outputs/section_summary_cleanup.json`
- `evaluation/section_relevance_gold_30.jsonl`
- `evaluation/section_relevance_gold_30.freeze.json`
- `evaluation/visual_relevance_gold_30_derived.jsonl`
- `evaluation/visual_queries/derived/`
- `evaluation/visual_external_style_template.json`
- `evaluation/visual_relevance_gold_10_external.jsonl`
- `evaluation/visual_queries/external/`
- `validation_outputs/visual_api_probe.json`
- `experiments/rerank_pool_20260908_r80_r20_r10/`
- `experiments/chunk_params_20260909_p120_p160_p200/`
- `experiments/visual_faiss_20260915/`
- `experiments/visual_retrieval_20260916/`
- `experiments/visual_text_20260916/`
- `experiments/visual_original_text_anchor_20260916/`
- `experiments/visual_latency_20260917_133735/`
- `experiments/visual_merge_concurrency_20260917_150134/`
- `experiments/visual_pms_confirm_20260917_173923/`
- `submissions/v6_full_reference.csv`
- `submissions/v6_tech_reference.csv`

## 不包含

- `.git/`
- `.claude/`
- `history_versions/`
- `scratch/`
- `reports/`
- `docs_archive/`
- 其他未固定的 `experiments/` 历史实验目录
- `submissions/` 全量历史实验结果
- API 运行 trace/raw 临时日志
- `手册_inlined/`
- Python 缓存和系统临时文件

## 运行方式

```bash
pip install -r requirements.txt
./run_api.sh
```

另开终端：

```bash
./smoke_test.sh
```

生成或刷新验证报告数据表：

```bash
python validate_performance.py
python validate_performance.py --api-base-url http://127.0.0.1:8000
```

## 配置说明

`config_runtime.py` 仅保留非敏感默认配置；真实凭据放在被 Git 忽略的本机 `.env`，或通过环境变量提供，已导出的环境变量优先；`RERANK_CANDIDATE_POOL`现役默认20。B34/C20/C10候选池对照已完成：历史Baseline执行参数为80、固定查询实测候选最大34，C20保持Recall@5与MRR并降低Rerank延迟；结果与复核步骤保存在 `experiments/rerank_pool_20260908_r80_r20_r10/`。该目录不是公开开源脱敏版，如需公开发布请先完成脱敏、授权确认和配置替换。
