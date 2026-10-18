# KBrag V6 历史验证数据表

生成时间：2026-06-15 08:52:07

> 口径说明：入口Service/Tech五次400题结果来自2026-06历史常量；产品路由Top-1/Top-3可由当前ProductRouter离线重算，2026-09-08复算结果仍为343/350和347/350；图文指标检查的是已保存参考答案与当前图片/Caption资产的一致性。2026-09-02现役Dense索引已迁移到 `Qwen/Qwen3-Embedding-8B`（6,570×4,096），但当前模型尚未重新生成全部400题，因此上述历史答案与格式指标不能视为Qwen3版本的新成绩。

## 数据集与资产

| metric | value | note |
| --- | --- | --- |
| 现有离线问题总数 | 400 | question_public.csv |
| 客服题数量 | 50 | 按 qid < 64 统计 |
| 技术题数量 | 350 | 按 qid >= 64 统计 |
| 手册 Markdown 数量 | 39 | 手册_v4/*.md |
| section 数量 | 1943 | data/section_chunks.json |
| retrieval chunk 数量 | 6570 | data/retrieval_chunks.json |
| 手册插图文件数量 | 2605 | 手册/插图 |
| 图片 caption 条目数量 | 2567 | data/image_captions_v4_final.json |
| 完整参考输出行数 | 400 | submissions/v6_full_reference.csv |
| 技术参考输出行数 | 400 | submissions/v6_tech_reference.csv |

## 路由验证

| run | correct | total | accuracy | disagreement_count | latency_avg_s | latency_median_s | latency_max_s | wall_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 400 | 400 | 100.00% | 7 | 0.90086 | 0.8745 | 1.959 | 36.493 |
| 2 | 400 | 400 | 100.00% | 7 | 0.9030975 | 0.905 | 1.458 | 36.603 |
| 3 | 400 | 400 | 100.00% | 9 | 0.931945 | 0.908 | 2.371 | 37.837 |
| 4 | 400 | 400 | 100.00% | 10 | 0.9060475 | 0.9095 | 1.575 | 36.773 |
| 5 | 400 | 400 | 100.00% | 8 | 1.0323425 | 0.9045 | 15.575 | 41.864 |

## 产品路由验证

| metric | value | numerator | denominator | note |
| --- | --- | --- | --- | --- |
| 产品路由 Top-1 命中率 | 98.00% | 343 | 350 | route.products[0] == expected_product |
| 产品路由 Top-3 命中率 | 99.14% | 347 | 350 | expected_product in route.products[:3] |
| 产品路由候选覆盖率 | 99.14% | 347 | 350 | expected_product in route.products |
| 最终答案产品命中率 | 99.43% | 348 | 350 | 基于 v6_full_reference.csv 复核，未命中 qid=432,433 |
| 无候选路由占比 | 0.00% | 0 | 350 | route.products 为空 |
| high 置信路由占比 | 93.14% | 326 | 350 | ProductRouteDecision.confidence |
| medium 置信路由占比 | 6.86% | 24 | 350 | ProductRouteDecision.confidence |
| 标签归一化失败数量 | 0 | 0 | 400 | product_mapping_final.csv 标签无法映射到 catalog |

## 多模态图文与格式验证

| metric | value | numerator | denominator | note |
| --- | --- | --- | --- | --- |
| 输出行数完整率 | 100.00% | 400 | 400 | v6_full_reference.csv vs question_public.csv |
| 答案非空率 | 100.00% | 400 | 400 | ret 非空 |
| 图片锚点与数组一致率 | 100.00% | 400 | 400 | <PIC> 数量 == 图片数组长度 |
| 图片文件存在率 | 100.00% | 780 | 780 | 按图片引用计 |
| 图片 caption 覆盖率 | 99.87% | 779 | 780 | 按图片引用计 |
| 客服题无图片输出率 | 100.00% | 50 | 50 | 客服题不附加手册图片 |
| 技术题含图片答案占比 | 89.14% | 312 | 350 | 按最终答案计 |
| caption 条目数量 | 2567 | 2567 |  | 图片语义描述库 |

## 对话连贯性 / API 验证

| turn | session_id | http_status | code | ok | elapsed_s | answer_len | question | answer_preview | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | validation-1781484694 | 200 | 0 | 1 | 13.562 | 55 | 椅子的扶手使用一段时间后为什么会松动？ | 大多数情况下，扶手使用一段时间后变松，是因为扶手和椅背的螺丝在使用过程中松动了。  重新拧紧螺丝即可解决问题。 |  |
| 2 | validation-1781484694 | 200 | 0 | 1 | 19.482 | 94 | 那应该怎么处理？ | 把扶手和椅背连接处的螺丝重新拧紧即可。  大多数情况下，扶手使用一段时间后变松，是因为扶手和椅背的螺丝松动了。处理时检查扶手与椅背连接位置，把松动的螺丝重新拧紧，扶手松动问题通常就能解决。 |  |

## 汇总 JSON 字段

```json
{
  "dataset": {
    "questions_total": 400,
    "service_questions": 50,
    "tech_questions": 350,
    "manual_md_files": 39,
    "section_chunks": 1943,
    "retrieval_chunks": 6570,
    "image_files": 2605,
    "caption_items": 2567,
    "full_reference_rows": 400,
    "tech_reference_rows": 400
  },
  "routing": {
    "runs": 5,
    "total_cases_per_run": 400,
    "all_runs_correct": true,
    "min_accuracy": 1.0,
    "avg_latency_s": 0.9348585,
    "avg_disagreement_count": 8.2
  },
  "product_router": {
    "total": 350,
    "source": "data/product_mapping_final.csv",
    "non_rag_rows_skipped": 50,
    "unresolved_mapping_labels": [],
    "top1_hits": 343,
    "top1_hit_rate": 0.98,
    "top3_hits": 347,
    "top3_hit_rate": 0.9914285714285714,
    "route_candidate_hits": 347,
    "route_candidate_hit_rate": 0.9914285714285714,
    "final_answer_product_hits": 348,
    "final_answer_product_hit_rate": 0.9942857142857143,
    "final_answer_product_wrong_ids": [
      "432",
      "433"
    ],
    "no_candidate": 0,
    "no_candidate_rate": 0.0,
    "confidence_counts": {
      "high": 326,
      "medium": 24
    },
    "reason_counts": {
      "explicit_product_name": 262,
      "name_and_content_agree": 10,
      "phrase_grep_unique": 5,
      "content_vote_only": 23,
      "functional_alias": 1,
      "explicit_product_nickname": 49
    }
  },
  "multimodal": {
    "total_rows": 400,
    "tech_rows": 350,
    "service_rows": 50,
    "rows_with_pics": 312,
    "total_pic_refs": 780,
    "marker_array_match_rows": 400,
    "marker_array_match_rate": 1.0,
    "image_file_hit_rate": 1.0,
    "caption_hit_rate": 0.9987179487179487,
    "caption_category_counts": {
      "info_table": 628,
      "part_view": 829,
      "schematic": 1047,
      "noise": 60,
      "icon": 1,
      "warning": 2
    },
    "caption_section_fit_counts": {
      "match": 2565,
      "mismatch": 2
    }
  },
  "dialogue": {
    "run": true,
    "base_url": "http://127.0.0.1:8000",
    "session_id": "validation-1781484694",
    "turns_attempted": 2,
    "turns_ok": 2,
    "all_ok": true
  },
  "outputs": {
    "dataset_overview": "validation_outputs/dataset_overview.csv",
    "routing_validation": "validation_outputs/routing_validation.csv",
    "product_router_summary": "validation_outputs/product_router_summary.csv",
    "product_router_details": "validation_outputs/product_router_details.csv",
    "multimodal_summary": "validation_outputs/multimodal_format_summary.csv",
    "multimodal_details": "validation_outputs/multimodal_format_details.csv",
    "dialogue_validation": "validation_outputs/dialogue_api_validation.csv",
    "markdown": "validation_outputs/validation_summary.md",
    "json": "validation_outputs/validation_summary.json"
  }
}
```
