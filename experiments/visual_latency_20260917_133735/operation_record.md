# 四路视觉检索阶段级延迟剖析记录

## 1. 目的与配置

本轮不修改检索结构，只为40条固定视觉金标的 `vision_to_text` 与原始文本锚定 `fusion` 两组增加调用级时间线。每组3轮，共240条查询级记录；记录Vision、原始/扩展文本Embedding与Rerank、VL文本/图片Embedding、VL Rerank、HTTP端点、状态码、负载和线程时间区间。

完整性：240条物理记录、240条成功解析、240个唯一 `repeat|mode|case_id`，错误0、回退0；阶段事件2,640条，错误0。

## 2. 质量与总延迟

| 方案 | Section R@5 | Image R@5 | 中位延迟 | P95 | 最大值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vision转文字文本检索 | 60.00% | 53.33% | 4.494秒 | 10.235秒 | 41.72秒 |
| 原始文本锚定四路融合 | 81.67% | 70.00% | 6.999秒 | 16.468秒 | 75.85秒 |

本轮质量继续高于无视觉FAISS的Vision文本基线；延迟仍存在不稳定长尾。

## 3. 阶段定位

| 阶段 | 中位数 | P95 | 最大值 |
| --- | ---: | ---: | ---: |
| 文本Embedding HTTP | 0.771秒 | 4.423秒 | 72.926秒 |
| 文本Rerank HTTP | 2.008秒 | 7.424秒 | 19.223秒 |
| Vision分析 | 1.622秒 | 2.849秒 | 3.883秒 |
| VL Embedding HTTP | 0.952秒 | 2.516秒 | 4.453秒 |
| VL Rerank HTTP | 1.536秒 | 2.269秒 | 4.554秒 |

最慢的融合V001中，扩展文本Embedding等待72.93秒、原始文本Embedding等待20.11秒，而VL两次Embedding约1秒、文本Rerank约1.75秒、VL Rerank约0.81秒。所有HTTP调用均返回200且没有重试。由此确认极端延迟位于远程文本Embedding HTTP请求，不是本地FAISS、RRF或VL Rerank。

本轮只能确定“远程调用长尾”，不能仅凭客户端总耗时继续拆分公网传输、服务端排队和模型计算。

## 4. 固定产物

| 文件 | SHA-256 |
| --- | --- |
| `runs.jsonl` | `6d64fae3c0b3c066e2ee6ce05828e342a4b8672d2fa7f362913f98dc1a34d50e` |
| `summary.json` | `9b1cedcb62b5f7ed0dd160255c1e0c4f960595e504fcda192f8972a6beca068a` |
| `latency_events.jsonl` | `7a9c448e930524baf40a96c41197c35f74a4ff36ce847aad3e3217c33f9d0351` |
| `latency_summary.json` | `25f23a782d3a0ce472fca2cd6f8fc7c0648f18797c6998a758912771d0885bc7` |
