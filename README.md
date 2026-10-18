# KBrag V6 多模态客服 Agent

这是一个面向产品手册问答场景的多模态客服 Agent。系统将普通客服问题与产品技术问题分流；技术问题通过产品路由、混合检索和有界 ReAct 流程查找手册证据，并返回正文与相关图片引用。

本仓库主要用于个人学习、实验记录和代码备份，按个人运行环境维护。

## 项目背景与个人定位

项目起点是朋友分享的一套已有客服问答代码和产品手册数据。我没有把它包装成“从零独立开发”，而是将它作为一个可复现、可分析、可优化的 RAG 工程，重点完成以下工作：

- 搭建本地运行环境，梳理 API、Agent、检索、模型适配和答案格式化调用链；
- 修复 RRF 未进入主检索路径、变量引用错误和默认产品范围未生效等问题；
- 将 Dense/Rerank 迁移到 Qwen3 8B，并重建、校验和归档索引；
- 验证文本、连续追问和图片输入链路，记录工具调用与耗时；
- 观察 Markdown 解析、Section、Chunk、Overlap 和 BM25 分词结果；
- 设计可复现的分词与检索对照实验，用 Recall、MRR、延迟和索引成本量化优化效果。

适合在简历和面试中使用的定位是：**多模态客服 Agent 复现与检索优化**。

## 核心架构

```text
POST /chat
  -> service / tech 分类
  -> service：客服模型直接回答，不提供检索工具
  -> tech：产品路由
           -> BM25 + Qwen3 Dense
           -> RRF 融合
           -> Qwen3 Rerank
           -> Parent Section
           -> 有界 ReAct + search_manual
  -> 统一答案与图片引用格式
```

主要模块：

- `api_server.py`：FastAPI 入口、鉴权、输入校验、分类、会话短历史和超时控制；
- `agent.py`：客服/技术分轨、有界 ReAct、工具调用和回答收束；
- `product_router.py`：产品识别与候选范围控制；
- `retrieval_engine.py`：BM25、Dense、RRF、Rerank 和 Parent Section 返回；
- `llm_router.py`：OpenAI-compatible 模型调用及 Anthropic 形状兼容；
- `submission_utils.py`：`[[PIC:...]]`、`<PIC>` 和图片数组格式化；
- `parse_manuals.py`：Markdown 解析、Section 构建和检索 Chunk 切分。

## 快速运行

建议使用 Python 3.10+。当前项目已在 Python 3.12.4 环境验证。

```bash
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env，填写自己的服务 Token 和上游 API Key，再启动服务。
# 新克隆且没有索引时，先按下方说明构建文本索引。
./run_api.sh
```

如果本机 8000 端口不可用：

```bash
PORT=8004 ./run_api.sh
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

对话请求：

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Authorization: Bearer $KAFU_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question":"椅子的扶手为什么会松动？","session_id":"demo-1"}'
```

运行单个问题：

```bash
python agent.py "椅子的扶手为什么会松动？"
```

## 索引与本地文件

Git 仓库不包含 `.env`、虚拟环境、临时日志，以及生成的 `.faiss`、`.pkl`、`.npy` 索引和向量缓存。本机已有文件继续使用，不需要重建。

新克隆后，先填写 `.env`，确认 `data/retrieval_chunks.json`、`data/section_chunks.json` 和 `data/catalog.json` 存在，再构建文本检索索引：

```bash
python build_retrieval_index.py
```

该命令在 `data/index/` 生成 `dense.faiss` 和 `retrieval_index.pkl`；默认向远程 Embedding 服务发送手册文本，可能产生调用费用。也可以从自己原来的运行目录复制这两个匹配的文件。服务不会自动重建缺失索引。

视觉检索默认关闭，普通文本与图片问答不需要构建独立视觉索引。只有要复现视觉 FAISS 实验时才运行 `python build_visual_index.py`；该命令会向远程 Embedding 服务发送手册图片。

视觉评测集中的 R004、R007、R010 三张照片仅保留在本机，不随仓库上传；金标和历史实验报告仍保留来源及结果。因此，新克隆目录不具备完整的原始 40 题视觉评测素材。`validate_delivery.py` 用于含本地索引的完整运行目录检查，不是新克隆源码的开箱验收。

## 验证命令

```bash
python validate_delivery.py
python validate_performance.py
python -m unittest discover -s tests -p "test_request_context.py" -v
python -m unittest discover -s tests -p "test_search_manual_trace.py" -v
python -m unittest discover -s tests -p "test_api_images.py" -v
./smoke_test.sh
```

启动API后访问 `http://127.0.0.1:8000/` 可使用同源图文问答网页；页面按 `<PIC>` 顺序插入 `data.images`，并显示聊天等待时间与后台延迟基线进度。

静态检查结果会随源码和本地运行文件变化，不在README复制快照。验证口径、最近一次检查结果及历史数据适用范围见 `项目技术架构说明文档/04_验证报告.md`。

## 文档导航

| 文档 | 用途 |
| --- | --- |
| `项目技术架构说明文档/01_API接口说明.md` | API 请求、响应和错误码 |
| `项目技术架构说明文档/02_源码运行说明.md` | 环境、目录、启动和复现步骤 |
| `项目技术架构说明文档/03_技术方案说明.md` | 架构、检索和图文链路 |
| `项目技术架构说明文档/04_验证报告.md` | 已验证事实、历史结果和风险边界 |
| `项目技术架构说明文档/05_项目深度解剖与面试答辩白皮书.md` | 源码学习、个人贡献和面试表达 |
| `项目技术架构说明文档/06_分词与检索优化实验设计.md` | Rerank候选池与Chunk参数实测结果，以及BM25分词后续方案 |
| `项目技术架构说明文档/07_客服Prompt事实边界优化记录.md` | 客服Prompt修改前后真实回答与逐题结论 |
| `项目技术架构说明文档/08_视觉FAISS四路检索实验与优化复盘.md` | 视觉索引、四路质量提升、阶段计时、P/M/S消融和阶段性结论 |
| `项目面试，运行问题说明文档/项目当前进展与下一步.md` | 当前唯一工作入口、三层测试边界和冻结待办 |

## 文档与范围边界

- README只保留项目入口、稳定边界和文档导航；动态进度与待办只在 `项目面试，运行问题说明文档/项目当前进展与下一步.md` 维护，API协议、运行步骤、技术方案、验证证据、面试材料和专项实验分别由 `项目技术架构说明文档/01` 至 `08` 承担。
- 当前项目不包含数据库或Redis长期会话记忆、在线自动写回知识库、已验证有效并默认启用的视觉FAISS、已完成的BM25分词优化指标或完整单元测试套件。
- 视觉FAISS四路检索代码、独立 `visual.faiss` 和40条固定视觉金标已完成，但功能开关仍默认关闭；只有通过预设质量/延迟门槛后才允许启用。40条由30张手册派生图、6张GPT Image写实图、1张用户实拍图和3张仅限本地实验的网络照片组成。

## 当前状态

- 文本索引、混合检索、图片API和最小网页均已具备；视觉FAISS为独立PoC，默认仍关闭。
- 正式文本检索默认批量发送已选的动态N条Dense输入，并保留 `serial` 回退；这不生成额外关键词。
- 当前唯一任务、已完成结果、证据边界和冻结待办统一见 `项目面试，运行问题说明文档/项目当前进展与下一步.md`，README不再复制动态清单。

## 结果口径

- 2026 年 6 月生成的 400 题路由、格式和历史成绩用于说明旧版本表现，不能直接作为当前 Qwen3 索引的优化结果；
- 分词异常已经观察到，但尚未证明造成 Recall 或最终答案下降；
- 已完成的Rerank候选池和Chunk参数实验数字来自冻结30题金标与原始记录；尚未运行的BM25分词和架构级Chunk实验继续只写方法与通过条件，不填估算收益；
- 视觉质量主结果来自40题、3轮配对实验；阶段计时只能把长尾定位到远程HTTP调用，不能继续断言公网传输或服务端内部计算各占多少；
- 当前目录含运行配置和完整数据，公开前应确认授权并完成密钥与数据脱敏。
