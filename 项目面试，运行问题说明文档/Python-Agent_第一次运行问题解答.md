# Python / Agent 第一次运行问题解答

> 依据：2026-09-19 亲手运行的终端 A/B 日志、`.run/hands_on.trace.jsonl` 和当前源码。  
> 结论标记：**已确认**表示能由本次日志/Trace/源码直接证明；**推断**表示符合调用链，但本次 Trace 没有进一步拆分内部耗时。

## 一、本次已经亲手验证了什么

这次不是“只看到一个答案”，而是已经完成了以下真实验证：

- Python 3.12.4 和项目虚拟环境可用；
- 依赖无冲突，静态交付检查通过；
- 成功回读 6,570 个 Retrieval Chunk、1,943 个 Section、39 个产品和 4,096 维 FAISS 索引；
- CLI 技术题发生 1 次 Tool Call、2 轮 ReAct；
- FastAPI 在 8004 启动，`/health` 正常；
- 错误 Bearer Token 返回 401，正确 Token 返回 200；
- 技术题进入 `tech`，客服题进入 `service`；
- 同一个 `session_id` 的追问带入了上一轮历史；
- 客服题 0 次工具调用、1 轮完成；
- Trace 能还原分类、预检索、Tool Call、最终生成和耗时。

## 二、问题 1：那条很长的 Python 命令有什么作用

原命令：

```powershell
.\.venv\Scripts\python.exe -B -c "from retrieval_engine import RetrievalEngine; e=RetrievalEngine(); e.ensure_index(); print('retrieval_chunks=', len(e.retrieval_chunks)); print('sections=', len(e.section_chunks)); print('products=', len(e.catalog)); print('faiss_ntotal=', e.dense_index.ntotal); print('faiss_dim=', e.dense_index.d)"
```

它是一条**只加载并检查现有索引的临时 Python 程序**，没有重建索引，也没有向模型提问。

逐段拆开：

```text
.\.venv\Scripts\python.exe
```

使用项目自己的虚拟环境解释器，避免误用系统 Python。

```text
-B
```

本次运行不生成 `__pycache__/*.pyc`，避免临时缓存污染交付目录。

```text
-c "..."
```

让 Python 直接执行引号内的小程序，不需要另建 `.py` 文件。

等价的多行 Python 是：

```python
from retrieval_engine import RetrievalEngine

e = RetrievalEngine()
e.ensure_index()

print("retrieval_chunks=", len(e.retrieval_chunks))
print("sections=", len(e.section_chunks))
print("products=", len(e.catalog))
print("faiss_ntotal=", e.dense_index.ntotal)
print("faiss_dim=", e.dense_index.d)
```

关键代码：

```python
def ensure_index(self) -> None:
    if self.dense_index is not None and self.bm25 is not None:
        return
    self.load_index()
```

`load_index()` 从：

```text
data/index/retrieval_index.pkl
data/index/dense.faiss
```

回读 Metadata、BM25 分词结果、Dense Vectors 和 FAISS Index。

各数字的含义：

| 输出 | 含义 |
| --- | --- |
| `retrieval_chunks=6570` | 用于召回定位的小 Chunk 数量 |
| `sections=1943` | 用于回填完整上下文的 Parent Section 数量 |
| `products=39` | 产品手册数量 |
| `faiss_ntotal=6570` | FAISS 中实际保存的向量数量，应与 Chunk 数一致 |
| `faiss_dim=4096` | 每条 Qwen3 Embedding 向量的维度 |

这条命令的价值是做**索引完整性冒烟检查**：文件存在不等于能正确加载；只有数量、向量数和维度都符合预期，才说明索引与 Metadata 至少在结构上匹配。

## 三、问题 2：为什么同类问题耗时差很多

### 3.1 先看本次有 Trace 的真实分解

第一次 API 技术题总耗时为 24.333 秒：

| 阶段 | 本次耗时 |
| --- | ---: |
| 三路分类 | 1.945 秒 |
| 产品路由 | 0.056 秒 |
| 预检索 | 10.258 秒 |
| └ Dense Embedding/Recall | 9.708 秒 |
| └ Rerank | 0.550 秒 |
| 第一轮 LLM（产生 Tool Call） | 1.299 秒 |
| 正式 `search_manual` | 9.462 秒 |
| 第二轮 LLM（最终回答） | 0.839 秒 |

同会话追问总耗时为 43.171 秒：

| 阶段 | 本次耗时 |
| --- | ---: |
| 三路分类 | 0.757 秒 |
| 产品路由 | 0.169 秒 |
| 预检索 | 9.091 秒 |
| └ Dense Embedding/Recall | 8.561 秒 |
| └ Rerank | 0.530 秒 |
| 第一轮 LLM | 0.794 秒 |
| 正式 `search_manual` | 30.876 秒 |
| 第二轮 LLM | 1.312 秒 |

### 3.2 视觉 FAISS 关闭后，为什么仍然慢

**已确认：仍然存在明显延迟，而且本次主要不是本地 FAISS 或主回答 LLM。**

关闭视觉 FAISS 只会使下面的分支直接回到文本检索：

```python
if not visual_retrieval_enabled() or (not images and not has_visual_intent(visual_query)):
    return self._search_text_manual(...)
```

文本检索本身仍然包含远程服务：

```text
查询文本 -> Qwen3-Embedding-8B 远程 Embedding
          -> 本地 FAISS 内积搜索
          -> BM25
          -> 本地 RRF
          -> Qwen3-Reranker-8B 远程 Rerank
```

其中 `_dense_recall()` 明确先请求远程查询向量：

```python
query_vector = self.client.embed_texts([query], self.embedding_model)[0]
query_array = self._l2_normalize(np.asarray([query_vector], dtype=np.float32))
```

本次第一次技术题的预检索中，Dense 阶段独占 9.708 秒，而 Rerank 只有 0.550 秒。正式 Tool Search 又花了 9.462 秒。第二次追问的正式 Tool Search 更出现了 30.876 秒长尾。

### 3.3 为什么会做两次检索

当前技术链路有两层：

1. **预检索**：Agent 正式决策前先做 Dense Top-30 + Rerank Top-5，作为定位 Hint。
2. **正式检索**：技术题要求模型必须显式调用一次 `search_manual`，正式链路会再做 BM25、Dense、RRF 和 Rerank，作为可审计证据。

源码中明确写着：

```python
# 系统预检索仅作为首轮定位参考；技术题仍需至少一次 search_manual 做正式确认。
formal_retrieval_confirmed = False

tool_choice = "required" if is_technical and not formal_retrieval_confirmed else None
```

因此，当前实现会为一个技术题至少发生一次预检索和一次正式检索。两次都可能调用远程 Embedding/Rerank，这就是即使视觉支路关闭仍然有明显延迟的核心结构原因。

### 3.4 15.7、19.8、8.8 秒为什么不固定

这三次 CLI 没有保存同等级的详细 Trace，因此不能精确断言每次慢在哪一步。但可以确定：

- 三次都为 1 次工具调用、2 轮；
- 本地 Jieba 首次初始化只用了约 0.35 秒，不足以解释十几秒差异；
- 查询 Embedding、Rerank、两轮 LLM 都是远程调用，会受到服务排队、网络、连接复用和上游负载影响；
- 不同问题形成的关键词、候选数量和文本长度也可能不同。

因此，**同样的工具调用次数不代表同样的耗时**。空气净化器问题的 8.8 秒只能说明该次远程链路较快，不能证明空气净化器检索结构天然比椅子快。

### 3.5 当前可以怎样准确描述延迟问题

> 视觉 FAISS 默认关闭后，文本链路仍有远程 Embedding、Rerank 和两轮 LLM 调用。实测首个 API 技术题 24.3 秒，其中预检索 Dense 9.7 秒、正式搜索 9.5 秒；同会话追问 43.2 秒，其中正式搜索出现 30.9 秒长尾。说明当前延迟主要来自远程文本检索服务以及预检索和正式检索的重复，而不是本地 FAISS 计算。

## 四、`<PIC>` 和 `['Manual03_21']` 是什么

这不是“图片输出失败”，而是项目约定的**图片引用协议**。

手册原文包含：

```markdown
长按控制面板上的睡眠 + 自动键 5 秒以上重置指示灯。
![Manual03_21](../手册/插图/Manual03_21.jpg)
```

解析后的 Section 保存为：

```json
{
  "text": "……控制面板上的滤网更换指示灯位置如下图所示。 <PIC>",
  "pics": ["Manual03_21"]
}
```

系统内部给模型看的形式是：

```text
[[PIC:Manual03_21]]
```

模型需要把这个锚点保留在正确的正文位置。回答完成后：

```python
def _resolve_answer_pics(answer: str) -> tuple[str, list[str]]:
    answer, inline_pics = extract_inline_pic_refs(answer)
    pic_count = answer.count("<PIC>")
    pics = inline_pics[:pic_count] if inline_pics else []
    ...
    return answer, pics
```

将它拆成：

```text
正文：……位置如下图所示。<PIC>
图片ID列表：['Manual03_21']
```

含义是：正文走到 `<PIC>` 的位置时，渲染或提交系统应当按顺序插入 `Manual03_21` 对应图片。

真实文件位于：

```text
手册/插图/Manual03_21.jpg
```

它展示的是空气净化器控制面板上的滤网更换指示灯位置。

CLI 只打印文本和图片 ID，不负责把 JPEG 渲染到终端；当前 `/chat` 也没有提供静态图片文件或 URL，而是通过最终答案字符串保留提交格式：

```text
正文<PIC>,["Manual03_21"]
```

所以更准确的结论是：

- 图片匹配和引用成功；
- CLI/API 返回的是图片锚点和 ID，不是图片二进制；
- 若要在普通网页中直接显示，需要前端把 ID 映射为图片 URL，后端还要提供受控的静态图片端点或文件服务。

同会话追问返回的：

```text
<PIC>,["Manual02_8"]
```

也是同一协议，表示在 `<PIC>` 位置插入人体工学椅安装图 `Manual02_8`。

## 五、问题 3：为什么输错命令后回复英文

你当时看到的是：

```text
> New-Item -ItemType Directory -Force -Path '.\.run' | Out-Null
```

关键不是命令本身写错，而是**输入位置错了**：行首是 Agent 交互模式的 `>`，不是 PowerShell 的 `PS F:\...>`。因此整条 PowerShell 命令被当成用户问题发送给 Agent。

CLI 调用 `run_agent(question, engine)` 时没有传 `question_id`：

```python
is_technical = question_id is None or question_id >= 64
```

所以 CLI 默认把它放进技术分支，仍会做预检索、强制调用 `search_manual`，最终耗时 21 秒。它并不会像 `/chat` 那样先做 service/tech 分类。

技术 Prompt 又要求回答语言与用户输入一致。你的输入几乎全是英文命令词，因此模型选择了英文。

想回到 PowerShell，应先输入：

```text
quit
```

看到：

```text
PS F:\BaiduNetdiskDownload\V6_submit_code>
```

再输入 `New-Item`。

### 是否应该改成中文提示

可以改善 CLI 体验，但不是核心业务错误。可选改法有两种：

1. CLI 检测明显的 PowerShell/Bash 命令，直接用中文提示“当前仍在 Agent 交互模式，请先输入 quit”，不发远程请求。
2. 在系统 Prompt 中规定无法识别时默认使用中文。

第一种更可靠，因为它还能避免一次无意义的 21 秒远程调用和费用。当前未修改源码，避免把一次操作失误扩大成未经评测的业务改动。

## 六、问题 4：为什么有时 200，有时 401

这与 GET/POST 本身没有直接关系，而与**访问的端点和 Bearer Token 是否正确**有关。

### 6.1 `/health` 为什么是 200

健康检查没有挂鉴权依赖：

```python
@app.get("/health")
async def health() -> dict[str, Any]:
    ...
```

因此本机直接 GET 即可返回 200。

### 6.2 第一次 `/chat` 为什么是 401

`/chat` 声明了鉴权：

```python
@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(auth)])
async def chat(...):
    ...
```

而你故意使用：

```powershell
Authorization = 'Bearer wrong-token'
```

`auth()` 中的判断是：

```python
if creds.credentials != EXPECTED_TOKEN:
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="invalid bearer token",
    )
```

所以 401 正是预期结果，证明鉴权生效，并不是故障。

### 6.3 后面的 `/chat` 为什么是 200

终端 A 设置了：

```powershell
$env:KAFU_API_TOKEN = 'local-first-run-token'
```

终端 B 使用了匹配的：

```powershell
Authorization = 'Bearer local-first-run-token'
```

鉴权通过，Agent 正常完成，所以本地 Uvicorn 记录：

```text
POST /chat HTTP/1.1 200 OK
```

### 6.4 要区分三层 HTTP 状态

| 层次 | 示例 | 含义 |
| --- | --- | --- |
| 客户端 → 本地 FastAPI | `127.0.0.1:8004 /chat 401/200` | 你自己的接口鉴权和业务结果 |
| FastAPI → DeepSeek | `api.deepseek.com ... 200` | 分类或主 LLM 请求成功 |
| FastAPI → SiliconFlow | Embedding/Rerank HTTP | 检索模型请求成功或失败 |

这些 200/401 不是同一台服务器给出的状态，不能混在一起解释。

## 七、问题 5：多条 DeepSeek POST 是 TCP 三次握手/四次挥手吗

**不是。**

日志：

```text
HTTP Request: POST https://api.deepseek.com/chat/completions "HTTP/1.1 200 OK"
```

表示一次**应用层 HTTP 请求已经收到 200 响应**。它没有显示 TCP SYN/SYN-ACK/ACK，也没有显示 FIN/ACK。

真实网络栈大致是：

```text
Agent 业务代码
  -> OpenAI SDK / httpx
  -> HTTPS
  -> TLS
  -> TCP
  -> IP
```

TCP 三次握手和 TLS 握手发生在更底层，通常不会由这条 `httpx` INFO 日志展示。HTTP 客户端还可能通过连接池复用已有 TCP/TLS 连接，因此也不能看到五条 HTTP POST 就断言发生了五次完整 TCP 建连和断连。

### 7.1 最前面的三条并发 POST 是什么

`/chat` 在 Agent 前先用三个不同 Prompt 做 service/tech 二分类：

```python
_BINARY_PROMPTS = {
    "service_guard": ...,
    "tech_recall": ...,
    "answer_source": ...,
}
```

三路通过线程池并发请求：

```python
with concurrent.futures.ThreadPoolExecutor(max_workers=len(_BINARY_PROMPTS)) as pool:
    futures = [
        pool.submit(_classify_one_prompt, name, prompt, question, classifier_timeout)
        for name, prompt in _BINARY_PROMPTS.items()
    ]
```

因此你会看到三条时间几乎相同的 POST：

```text
08:57:11.875
08:57:11.887
08:57:11.887
```

这是三次独立的 DeepSeek 分类请求，不是一次请求的 TCP 三次握手。

### 7.2 后面两条 POST 是什么

技术题需要两轮 LLM：

1. `08:57:23.919`：第一轮 LLM 返回结构化 `search_manual` Tool Call。
2. Python 宿主执行 `search_manual`，本次耗时 9.462 秒。
3. `08:57:34.644`：第二轮 LLM读取 Tool Result，生成最终回答。

所以完整时间线是：

```text
客户端 POST /chat
  -> DeepSeek 分类请求 × 3（并发）
  -> 文本预检索：Embedding + FAISS + Rerank
  -> DeepSeek 主模型第 1 次：决定调用 search_manual
  -> 正式检索：BM25 + Embedding + RRF + Rerank + Parent Section
  -> DeepSeek 主模型第 2 次：生成最终答案
  -> 本地 FastAPI 返回 200
```

### 7.3 为什么中间看不到 Embedding/Rerank 的相同日志

DeepSeek 分类和主模型走 OpenAI SDK/httpx，因此出现 `httpx2` INFO；文本 Embedding/Rerank 客户端主要使用 `requests`，默认没有打印同样格式的 INFO 日志。

“没有那行日志”不代表没有调用。Trace 已经记录预检索 Dense 9.708 秒、Rerank 0.550 秒和正式 Tool Search 9.462 秒。

### 7.4 客服题为什么只有 3 + 1 条

客服题日志为：

```text
三条并发 POST：三路分类
一条 POST：客服主模型直接回答
```

客服分支不提供 `search_manual`，因此没有第二轮 Tool Result 后生成：

```text
route=service
tool_calls=0
turns=1
api_elapsed=1.961
```

这正好证明了双轨路由确实生效。

## 八、日志中其他看似异常、实际正常的地方

### 8.1 FAISS 的 AVX512/AVX2 加载失败

日志先尝试专用优化模块：

```text
Could not load ... faiss.swigfaiss_avx512
Could not load ... faiss.swigfaiss_avx2
```

随后：

```text
Successfully loaded faiss.
RetrievalEngine 就绪 (0.3s)
```

说明它已经回退并成功加载通用 FAISS。这可能少一些 CPU 指令级优化，但不影响本次索引正确加载和功能运行，不是启动失败。

### 8.2 `$service | ConvertTo-Json` 为什么输出 `null`

你在创建 `$serviceBody` 和调用 `$service = Invoke-RestMethod ...` 之前，提前执行了一次：

```powershell
$service | ConvertTo-Json -Depth 8
```

当时 `$service` 尚未赋值，PowerShell 把它视为 `$null`，所以输出：

```text
null
```

后面赋值后再次执行已经正常返回，这不是 API 问题。

### 8.3 `answer_length` 为什么为空

Trace 的 `result` 实际包含：

```text
answer
pics
tool_calls
turns
```

没有单独保存 `answer_length`。原教程命令引用了不存在的字段，已修正为：

```powershell
[pscustomobject]@{
    tool_calls = $lastTrace.result.tool_calls
    turns = $lastTrace.result.turns
    answer_length = ($lastTrace.result.answer).Length
}
```

### 8.4 Event 表格为什么有空白

不同 Event 的字段不同：

- `pre_retrieval` 保存 `products` 和 `sections`；
- `tool_call` 保存 `name`、`elapsed`、输入和命中结果；
- `llm_call` Event 保存 Action/Text Preview；
- LLM 精确耗时保存在 `timings.llm_calls`。

因此统一 `Select-Object kind,name,turn,elapsed` 时出现空白是正常的，不代表步骤没有执行。

### 8.5 客服 Trace 中为什么仍有空的 `pre_retrieval` Event

客服分支确实没有检索。代码先跳过技术题预检索，但随后为了保持 Trace 结构统一，仍追加一个空事件：

```json
{
  "kind": "pre_retrieval",
  "products": [],
  "sections": []
}
```

判断是否真正做了检索，要看 `sections`、`timings.pre_retrieval` 和 Tool Calls，而不是只看是否存在这个 Event 名称。

### 8.6 为什么运行后再次执行 `validate_delivery.py` 会失败

你第一次运行前的检查已经通过。实际启动服务后，项目中出现了两类运行产物：

```text
__pycache__/*.pyc
.run/hands_on.trace.jsonl
```

交付检查刻意拒绝把字节码缓存和内部 Trace 放入最终提交包，所以现在会报告这些文件。这是“运行目录不再处于干净交付状态”，不是 Python 源码语法或 Agent 功能失败。

教程中的启动命令已改为：

```powershell
.\.venv\Scripts\python.exe -B -m uvicorn api_server:app --host 127.0.0.1 --port 8004 --workers 1
```

`-B` 会阻止之后继续写 `.pyc`。本次 Trace 是亲手运行的重要证据，当前保留；准备最终交付时再单独清理这些运行产物，不要在还未复盘前直接删除。

## 九、从这次运行得到的准确面试表述

> 我亲手跑通后发现，视觉 FAISS 关闭不代表链路没有检索延迟，因为文本 Dense 仍要远程生成 Query Embedding，之后还有远程 Rerank。一次技术请求还会先做 Dense 预检索，再由模型显式调用 `search_manual` 做正式 BM25/Dense/RRF/Rerank，因此存在重复远程调用。我的实测请求总耗时 24.3 秒，其中预检索 10.3 秒、正式搜索 9.5 秒；另一次追问的正式搜索出现 30.9 秒长尾。客服题则三路分类后直接一轮生成，0 次工具调用，总耗时约 2 秒。
>
> Tool Calling 过程中，模型第一轮只返回 `search_manual` 的结构化调用，真正执行检索的是 Python；检索结果作为 Tool Result 回填后，模型第二轮才生成答案。`<PIC>` 不是图片加载失败，而是正文中的图片插入锚点，旁边的 `Manual03_21` 是对应手册图片 ID，CLI 只展示协议，网页端还需要把 ID 映射成图片 URL 才能渲染。

这段表述完全来自你本次亲手运行的 Trace 和源码，不再只是阅读文档所得。
