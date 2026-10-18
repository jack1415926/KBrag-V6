# Python / Agent 第一次亲手运行教程（Windows）

> 目标：不用 Codex 代跑，由你亲自在 PowerShell 和 PyCharm 中完成一次“环境检查 → 本地索引加载 → CLI Agent → FastAPI → 技术题 Tool Calling → 客服题无工具 → Trace 查看 → 断点调试”。  
> 当前机器实测：项目虚拟环境为 Python 3.12.4，`pip check` 无依赖冲突；Windows 排除了 8000 端口，因此本教程统一使用 **8004**。

## 0. 这次实操完成后，你应该能说清什么

不要把“终端成功返回了一段答案”当成全部目标。完整跑完后，你应该能亲口解释：

1. Python 如何加载本地的 6,570 个 Retrieval Chunk、1,943 个 Section 和 4,096 维 FAISS 索引。
2. CLI 运行与 FastAPI 运行有什么区别。
3. `/chat` 怎样经过分类、`fake_qid=0/64`、`run_agent()` 和答案格式化。
4. 技术题为什么会调用 `search_manual`，客服题为什么不提供工具。
5. 模型怎样返回 Tool Call，Python 宿主怎样执行工具并回填 Tool Result。
6. BM25、Dense、RRF、Rerank 和 Parent Section 分别处在链路的哪一层。
7. 如何从日志和 Trace 证明一次请求实际走过哪些步骤。

## 1. 运行前边界

### 1.1 哪些步骤完全本地

以下步骤不会请求远程模型：

- Python 与依赖检查；
- `validate_delivery.py`；
- 加载现有 FAISS/BM25 索引；
- 查看 Section、Chunk 和索引数量；
- 阅读 Trace 文件和源码断点。

### 1.2 哪些步骤会访问远程服务

CLI Agent 和 `/chat` 会使用远程分类、LLM、Embedding 和 Rerank 服务。技术问题及部分检索候选会被发送给已配置的远程接口，且可能产生 API 费用。

本教程不会要求你打印任何 API Key。不要在截图、面试材料或聊天中展示 `config_runtime.py` 里的密钥。若默认凭据已经失效，用你自己的环境变量覆盖，不要把新密钥写进教程或提交记录。

## 2. 打开 PowerShell 并进入项目

建议使用 PowerShell 7。打开一个新终端，执行：

```powershell
Set-Location -LiteralPath 'F:\BaiduNetdiskDownload\V6_submit_code'
Get-Location
```

预期路径：

```text
F:\BaiduNetdiskDownload\V6_submit_code
```

后续所有命令都在这个目录执行。不要使用 `bash -lc`；当前 Windows 环境直接使用 PowerShell 和 `.venv\Scripts\python.exe`。

## 3. 第一步：确认 Python 和依赖

依次执行：

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip check
```

当前机器的预期输出是：

```text
Python 3.12.4
No broken requirements found.
```

如果 `.venv\Scripts\python.exe` 不存在，才需要重新创建环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

当前项目已经有可用 `.venv`，第一次实操无需重复安装。

## 4. 第二步：做一次完全本地的交付检查

执行：

```powershell
.\.venv\Scripts\python.exe -B validate_delivery.py
```

重点看最后的 `delivery_check`、`errors` 和 `warnings`。完全通过时应为：

```text
delivery_check=OK warnings=0
```

这个结果只能证明文件、语法、密钥散落检查和文档覆盖等静态条件通过，不能代替真实 Agent/远程 API 测试。2026-09-23当前工作目录仍会因 `.DS_Store` 和 `.run/hands_on.trace.jsonl` 返回FAILED；这是已知本地文件阻断，不要把教程中的通过示例误当成当前实测结果。

## 5. 第三步：亲手加载本地检索索引

复制执行：

```powershell
.\.venv\Scripts\python.exe -B -c "from retrieval_engine import RetrievalEngine; e=RetrievalEngine(); e.ensure_index(); print('retrieval_chunks=', len(e.retrieval_chunks)); print('sections=', len(e.section_chunks)); print('products=', len(e.catalog)); print('faiss_ntotal=', e.dense_index.ntotal); print('faiss_dim=', e.dense_index.d)"
```

当前数据的预期结果：

```text
retrieval_chunks= 6570
sections= 1943
products= 39
faiss_ntotal= 6570
faiss_dim= 4096
```

这一段很重要：它证明你不是现场重新构建索引，而是成功回读了预建的 Qwen3 文本向量索引及 Metadata。

## 6. 第四步：先用 CLI 跑一次 Agent

### 6.1 单问题模式

执行一个技术问题：

```powershell
.\.venv\Scripts\python.exe -B agent.py '椅子的扶手使用一段时间后为什么会松动？'
```

这一步会访问远程 LLM、Embedding 和 Rerank。正常情况下你会看到：

1. `索引加载完成: 6570 检索块, 39 产品`；
2. 模型与检索过程日志；
3. 最终答案；
4. 最后一行显示工具调用次数、Agent 轮数和耗时。

典型过程应当是第一轮产生 `search_manual` Tool Call，Python 执行检索，第二轮模型读取 Observation 后输出最终答案。实际轮数和耗时以你本次终端输出为准，不要预先编造。

### 6.2 交互模式

单题跑通后再试：

```powershell
.\.venv\Scripts\python.exe -B agent.py -i
```

出现 `>` 后可以依次输入：

```text
椅子的扶手为什么会松动？
空气净化器的滤网应该怎样清洁？
quit
```

看到行首 `>` 时，你仍在 Python Agent 的交互输入中；此处输入的任何内容都会被当成用户问题发送给 Agent。必须先输入 `quit` 回到 `PS ...>` 提示符，才能执行 `New-Item`、`Get-Content` 等 PowerShell 命令。

注意：CLI 的交互模式只是连续提问，不等同于 FastAPI 的 `session_id` 短历史。要验证同一会话追问，应继续完成后面的 API 实操。

## 7. 第五步：亲手启动 FastAPI

这一部分需要两个 PowerShell 窗口。

### 7.1 终端 A：启动服务

在项目目录执行：

```powershell
New-Item -ItemType Directory -Force -Path '.\.run' | Out-Null
$env:KAFU_API_TOKEN = 'local-first-run-token'
$env:CHAT_API_TRACE_PATH = '.run\hands_on.trace.jsonl'
.\.venv\Scripts\python.exe -B -m uvicorn api_server:app --host 127.0.0.1 --port 8004 --workers 1
```

这里手动设置了一个只用于本机练习的 Bearer Token，环境变量会覆盖配置文件默认值。不要把真实模型 API Key当作 `/chat` 的 Bearer Token。

正常启动时应看到类似：

```text
Uvicorn running on http://127.0.0.1:8004
KAFU_API_TOKEN 已配置
RetrievalEngine 就绪
```

索引预热是异步启动的，刚看到 Uvicorn 后可能还在加载索引；等几秒再进行健康检查。

### 7.2 为什么不用 8000

这台机器当前的 Windows 排除端口列表明确包含 8000。若使用 8000，可能出现绑定权限错误，这不是项目业务代码故障。8004 当前没有监听进程，也不在排除列表中。

## 8. 第六步：在终端 B 调用 API

重新打开一个 PowerShell 窗口，再进入项目：

```powershell
Set-Location -LiteralPath 'F:\BaiduNetdiskDownload\V6_submit_code'
```

### 8.1 健康检查

执行：

```powershell
Invoke-RestMethod -Method Get -Uri 'http://127.0.0.1:8004/health' | ConvertTo-Json -Depth 5
```

预期至少包含：

```json
{
  "status": "ok",
  "engine_ready": true,
  "background_agents": 0,
  "auth_configured": true
}
```

如果 `engine_ready` 暂时为 `false`，等待几秒后再请求一次。若长期为 `false`，查看终端 A 是否有“引擎预热失败”。

### 8.2 先验证错误鉴权

故意使用错误 Token：

```powershell
$badHeaders = @{ Authorization = 'Bearer wrong-token' }
try {
    Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8004/chat' -Headers $badHeaders -ContentType 'application/json; charset=utf-8' -Body '{"question":"测试"}'
} catch {
    $_.Exception.Response.StatusCode.value__
}
```

预期返回：

```text
401
```

这证明 Bearer 鉴权确实生效。

### 8.3 调用一个技术题

执行：

```powershell
$headers = @{ Authorization = 'Bearer local-first-run-token' }
$body = @{
    question = '椅子的扶手使用一段时间后为什么会松动？'
    session_id = 'hands-on-001'
} | ConvertTo-Json

$tech = Invoke-RestMethod -Method Post `
    -Uri 'http://127.0.0.1:8004/chat' `
    -Headers $headers `
    -ContentType 'application/json; charset=utf-8' `
    -Body $body

$tech | ConvertTo-Json -Depth 8
```

预期响应结构：

```text
code = 0
msg = success
data.answer 非空
data.session_id = hands-on-001
```

同时观察终端 A。重点不是逐字记日志，而是找这些事实：

- 请求被分类为 `tech`；
- Agent 进入技术分支；
- 至少发生一次正式 `search_manual`；
- 日志最终有 `tool_calls`、`agent_turns`、耗时和答案长度。

### 8.4 用同一个 session_id 追问

```powershell
$followupBody = @{
    question = '那应该怎么处理？'
    session_id = 'hands-on-001'
} | ConvertTo-Json

$followup = Invoke-RestMethod -Method Post `
    -Uri 'http://127.0.0.1:8004/chat' `
    -Headers $headers `
    -ContentType 'application/json; charset=utf-8' `
    -Body $followupBody

$followup | ConvertTo-Json -Depth 8
```

这一步验证的是进程内短会话历史。它不是数据库长期记忆：服务重启后历史会丢失，多 Worker 之间也不共享。

### 8.5 再调用一个客服题

```powershell
$serviceBody = @{
    question = '商品到货后有问题，应该如何申请售后？'
    session_id = 'hands-on-service-001'
} | ConvertTo-Json

$service = Invoke-RestMethod -Method Post `
    -Uri 'http://127.0.0.1:8004/chat' `
    -Headers $headers `
    -ContentType 'application/json; charset=utf-8' `
    -Body $serviceBody

$service | ConvertTo-Json -Depth 8
```

观察终端 A：客服题应被路由到 `service`，代码级不向模型提供 `search_manual`，因此正常结果应为 0 次工具调用。这样你就亲手验证了“客服直接回答、技术题走 RAG”的双轨分支。

## 9. 第七步：亲手查看 Trace

API 请求完成后，在终端 B 执行：

```powershell
$lastTrace = Get-Content -LiteralPath '.\.run\hands_on.trace.jsonl' -Tail 1 | ConvertFrom-Json
$lastTrace | Select-Object request_id, session_id, route, api_elapsed, session_history_turns
[pscustomobject]@{
    tool_calls = $lastTrace.result.tool_calls
    turns = $lastTrace.result.turns
    answer_length = ($lastTrace.result.answer).Length
}
$lastTrace.events | Select-Object kind, name, turn, elapsed
```

`result` 原始字段只有 `answer`、`pics`、`tool_calls` 和 `turns`，没有单独保存 `answer_length`，所以上面通过字符串的 `.Length` 现场计算。不同事件的字段也不完全相同：`pre_retrieval` 主要保存 Sections，`tool_call` 保存 name/elapsed，LLM 耗时位于 `$lastTrace.timings.llm_calls`。因此表格中的空白字段不等于执行失败。

如需查看最后一条完整 Trace：

```powershell
$lastTrace | ConvertTo-Json -Depth 12
```

然后重新发一次技术题，再执行相同命令。你应该能比较出：

- 技术题：`route=tech`，事件中有工具调用与检索证据；
- 客服题：`route=service`，没有 `search_manual`；
- Trace 中能看到 Agent 轮数、工具参数、结果摘要和各阶段耗时。

`.run` 是本地练习目录，其中可能包含问题和回答，不要随意对外发送完整 Trace。

`validate_delivery.py` 面向干净的最终交付包，会把 `.run/*.jsonl` 等运行期 Trace 判为不应提交的文件。本教程故意保留 Trace 供你学习，因此运行之后再次执行交付检查可能失败；这不代表 Agent 源码运行失败。完成复盘并准备重新打包时，再单独清理运行产物。`-B` 可以避免后续启动继续生成 Python 字节码缓存，但此前已经生成的 `__pycache__` 不会自动消失。

## 10. 第八步：在 PyCharm 中用断点走一遍

终端跑通后，再做一次真正有学习价值的 Debug。

### 10.1 配置解释器

在 PyCharm 中打开项目目录：

```text
F:\BaiduNetdiskDownload\V6_submit_code
```

把项目解释器设置为：

```text
F:\BaiduNetdiskDownload\V6_submit_code\.venv\Scripts\python.exe
```

### 10.2 建立 CLI Run Configuration

配置：

```text
Script path: F:\BaiduNetdiskDownload\V6_submit_code\agent.py
Parameters: 椅子的扶手使用一段时间后为什么会松动？
Working directory: F:\BaiduNetdiskDownload\V6_submit_code
```

### 10.3 推荐断点顺序

按顺序放断点，不要一次在所有函数里乱跳：

1. `agent.py::run_agent()`：观察 `question_id`、`is_technical`、`current_route`。
2. `agent.py` 中设置 `tool_choice` 的位置：第一次技术题应为 `required`。
3. `llm_router.py::_create_openai_message()`：观察发送给模型的 `messages`、`tools`、`tool_choice`，不要展开或截图 API Key。
4. `agent.py` 遍历 `response.content` 的位置：观察模型返回的 `block.type`、`block.name` 和 `block.input`。
5. `agent.py::_execute_tool_with_pics()`：确认真正执行工具的是 Python，而不是模型。
6. `retrieval_engine.py::search_manual()`：观察 keywords、original_query、products。
7. `retrieval_engine.py::_rrf_merge()`：观察多个 ranked list 如何合并；这是 RRF，不是 MRR。
8. `retrieval_engine.py::_build_results()`：比较命中的 `chunk["text"]` 与返回的 `section["text"]`，亲眼确认 Parent Section 回溯。
9. 回到 `run_agent()`：观察 `tool_result` 如何带着 `tool_use_id` 写回消息历史，进入下一轮生成最终答案。

第一次 Debug 建议使用 Step Over 为主；进入远程 SDK 内部通常只会增加噪声。

## 11. 你必须亲眼观察的变量

建议在纸上或笔记里记录本次真实值：

| 阶段 | 变量/结果 | 你本次观察值 |
| --- | --- | --- |
| 索引 | `len(retrieval_chunks)` | |
| 索引 | `dense_index.d` | |
| 路由 | `route` / `current_route.products` | |
| 工具选择 | 首轮 `tool_choice` | |
| Tool Call | `block.name` | |
| Tool Args | keywords / products / query | |
| RRF | ranked list 数量与候选数量 | |
| Parent 回溯 | 命中 Chunk 与返回 Section 的长度差异 | |
| 最终结果 | tool_calls / turns / elapsed | |
| API | HTTP 状态与 session_id | |

面试时最有价值的不是背诵预设数字，而是能说：“我在断点里看到模型返回了 `search_manual` 的 name、arguments 和 call ID；随后宿主程序执行检索，把带相同 ID 的 Tool Result 回填到消息历史。”

## 12. 常见错误与处理

### 12.1 8000 端口绑定失败

原因：本机 Windows 排除了 8000。直接使用教程中的 8004，不需要修改系统排除规则。

### 12.2 `/health` 能访问，但 `engine_ready=false`

先等几秒。若一直为 false，查看终端 A 的索引加载异常，并确认：

```powershell
Get-Item '.\data\index\dense.faiss'
Get-Item '.\data\index\retrieval_index.pkl'
```

### 12.3 `/chat` 返回本地 401

这是本项目 Bearer Token 不匹配。确认终端 A 和终端 B 都使用：

```text
local-first-run-token
```

模型服务返回的 401/403 是另一层问题，通常代表远程 API 凭据失效，不要把两种鉴权混为一谈。

### 12.4 远程调用出现 401/403

不要打印旧密钥。使用自己的密钥，以环境变量覆盖后重启终端 A：

```powershell
$env:SILICONFLOW_API_KEY = '<你的主模型API Key>'
$env:EMBEDDING_API_KEY = '<你的Embedding API Key>'
$env:RERANK_API_KEY = '<你的Rerank API Key>'
```

具体服务商可能要求不同 Key。只在当前终端设置，关闭终端后环境变量消失。

### 12.5 连接超时或网络错误

先区分本地服务与远程服务：

- `/health` 都打不开：本地 Uvicorn 没启动或端口不对。
- `/health` 正常、`/chat` 失败：继续看终端 A 的分类、LLM、Embedding 或 Rerank错误。
- HTTP 504：请求超过统一截止时间；Python 线程不会被强杀，而是通过 cancellation event 和下游超时协作退出。

### 12.6 中文请求乱码

优先使用本教程的 `Invoke-RestMethod` + PowerShell Hashtable + `ConvertTo-Json`，不要把复杂中文 JSON 硬塞进多层引号的命令。

## 13. 停止服务

回到终端 A，按：

```text
Ctrl+C
```

看到 Uvicorn shutdown 日志并回到 PowerShell 提示符后，服务才算停止。再验证：

```powershell
Get-NetTCPConnection -State Listen -LocalPort 8004 -ErrorAction SilentlyContinue
```

没有输出表示 8004 已不再监听。

## 14. 第一次实操的完成标准

只有下面全部完成，才建议把这次算成“我亲手跑过”：

- [ ] 自己进入项目目录并确认 Python 版本。
- [ ] 自己执行 `pip check` 和 `validate_delivery.py`。
- [ ] 自己加载索引并看到 6,570 / 1,943 / 39 / 4,096。
- [ ] 自己通过 CLI 跑通一个技术问题。
- [ ] 自己在 8004 启动 Uvicorn。
- [ ] 自己调用 `/health` 并看到 `engine_ready=true`。
- [ ] 自己验证错误 Token 返回 401。
- [ ] 自己调用技术题并观察到 `search_manual`。
- [ ] 自己调用客服题并确认没有工具调用。
- [ ] 自己用同一个 `session_id` 做一次追问。
- [ ] 自己查看最后一条 Trace。
- [ ] 自己在 PyCharm 断点中看过 Tool Call → 工具执行 → Tool Result 回填。
- [ ] 自己用 Ctrl+C 停止服务并确认 8004 不再监听。

## 15. 跑完后可以使用的面试表述

> 我后来专门在 Windows 环境从零走了一遍运行过程：先用项目虚拟环境完成依赖和静态检查，回读 6,570 条、4,096 维的预建 FAISS 索引；然后分别通过 CLI 和 Uvicorn/FastAPI 调用 Agent。我用技术题观察到首轮 `tool_choice=required`，模型返回 `search_manual` 的结构化 Tool Call，Python 宿主完成 BM25/Dense、RRF、Rerank 和 Parent Section 回溯，再把 Tool Result 回填给模型生成答案；我也用客服题验证了代码级不提供检索工具，并通过 Trace 对照了 route、tool_calls、turns 和耗时。FastAPI 不是我从零写的，但这条运行和调试链路是我自己亲手走通并理解的。

这段话只有在你完成上面的实操和断点观察后再使用。
