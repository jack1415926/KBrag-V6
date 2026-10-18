# Rerank候选池Baseline边界/C20/C10对照操作记录

## 1. 结论

在冻结的30题Section相关性标注集上，C20与Baseline三轮结果完全保持相同的Section Recall@5和MRR；C10仍为30/30命中，但问题86的正确Section从第1名降至第2名，使MRR绝对下降0.0167，超过预设0.01容忍线。

C20是本轮推荐的候选配置，但本轮没有修改现役默认值80。原因是C20的Rerank延迟下降明确，而完整检索P95受到远程Embedding长尾波动影响，没有显示稳定下降。Baseline虽然配置上限为80，但本轮每题实际候选数只有5至34条，因此下文称为“B34边界”，不把它表述为实际处理80条的固定组。

## 2. “30题”和“候选池上限”的关系

30题是30个独立查询；候选池上限表示每个查询最多向Reranker发送多少候选Chunk。候选池可以大于评测题数，两者不是同一个维度。

80只是现役配置上限，不要求每题凑满80条。本轮Baseline没有触发80条截断，实际边界为34，因此质量与延迟表使用“B34（配置80）”标识。

### 2.1 为什么本轮最大候选只有34

本轮为了只比较候选池大小，固定调用参数为：

```text
keywords=[]
semantic_query=原问题
original_query=原问题
products=[金标产品]
top_k=5
```

在 `retrieval_engine.py::search_manual()` 中，这种调用只产生两条主召回列表：BM25 Top-20和Dense Top-20。因为 `keywords=[]`，逐关键词的额外BM25/Dense Top-5列表为空。因此：

```text
理论去重上限 = 20 + 20 = 40
实际候选数 = BM25集合 ∪ Dense集合
             = BM25数量 + Dense数量 - 两路重合数量
```

30题实测候选数范围为5至34，平均26.43，中位数27.5。最大样本是问题358（Lawn Mower）：BM25和Dense各返回20条，其中6条重合，因此RRF去重后为 `20 + 20 - 6 = 34`。最小样本是问题90（人体工学椅），该产品在当前索引中只有5个Retrieval Chunk，因此两路召回和去重后都只有5条。

这不表示生产链路永远最多34条。真实Agent可能向 `search_manual` 传入多个LLM生成关键词，代码会为每个关键词追加BM25 Top-5和Dense Top-5，合并后的候选可以超过34，并继续受现役配置上限80约束。因此B34只是本次固定查询方法下的观测边界。

## 3. 固定条件

- 金标：`evaluation/section_relevance_gold_30.jsonl`
- 金标SHA-256：`1c5165bde70e03e5a086dd7684d5ce76ee6ab26a30eceab603ca29ca3d201899`
- 问题数：30，其中中文20题、英文10题
- 产品范围：使用金标中的正确产品，实验只衡量产品内Section检索，不衡量Product Router
- 查询：原问题同时作为统一语义查询和原始问题，不使用LLM临时生成关键词
- 最终结果：Top-5去重Parent Section
- 变量：仅改变 `rerank_candidate_pool`
- 组别：B34（代码配置80，本轮观测边界34）、C20、C10
- 重复：每组3轮，每轮30题；组别循环移位，问题顺序正反交替
- 总调用：270次真实Embedding+Rerank检索
- P95算法：Nearest Rank

## 4. 具体运行与手动复核步骤

### 4.1 进入项目并检查环境

```powershell
Set-Location 'F:\BaiduNetdiskDownload\V6_submit_code'
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -B -c "import os, config_runtime as c; c.apply_default_env(); print('embedding_model=', os.getenv('EMBEDDING_MODEL')); print('rerank_model=', os.getenv('RERANK_MODEL_ALIAS')); print('embedding_key_configured=', bool(os.getenv('EMBEDDING_API_KEY'))); print('rerank_key_configured=', bool(os.getenv('RERANK_API_KEY')))"
```

最后一条命令只检查密钥是否已配置，不打印密钥内容。

### 4.2 校验冻结金标

```powershell
.\.venv\Scripts\python.exe -B .\validate_section_gold.py
```

预期：

```text
section_gold=OK records=30 sha256=1c5165bde70e03e5a086dd7684d5ce76ee6ab26a30eceab603ca29ca3d201899
```

### 4.3 使用新目录重新运行

脚本拒绝覆盖已有结果。手动复核时使用带时间戳的新目录：

```powershell
$reviewDir = ".\experiments\rerank_pool_manual_$(Get-Date -Format yyyyMMdd_HHmmss)"
.\.venv\Scripts\python.exe -B .\run_rerank_pool_experiment.py `
  --pools 80,20,10 `
  --repeats 3 `
  --out-dir $reviewDir
```

实验会把问题和候选手册片段发送给当前配置的外部SiliconFlow Embedding/Rerank服务。本次已保存运行获得过用户明确授权；以后在其他数据或服务上复跑时仍应重新确认数据外发范围。

### 4.4 检查完整性和聚合结果

```powershell
$rows = Get-Content "$reviewDir\runs.jsonl" | ForEach-Object { $_ | ConvertFrom-Json }
"records=$($rows.Count)"
"errors=$(($rows | Where-Object { $_.error }).Count)"
"fallbacks=$(($rows | Where-Object { $_.rerank_error }).Count)"
Get-Content -Raw "$reviewDir\summary.json"
```

预期记录数为270，错误和回退均为0。远程服务延迟会随时间变化，所以手动复跑应重点检查Recall@5、MRR、候选数和失败数是否一致；不要要求耗时逐毫秒相同。

### 4.5 查看产生排序退化的问题86

原始记录中的 `repeat` 从0开始。查看第一轮三个配置：

```powershell
$rows | Where-Object { $_.repeat -eq 0 -and $_.id -eq 86 } |
  Select-Object pool, first_relevant_rank, candidate_count, top_sections |
  Format-List
```

本次预期：配置80/B34和C20的正确Section为第1名，C10降至第2名。

### 4.6 核对本次固定产物哈希

```powershell
Get-FileHash -Algorithm SHA256 `
  .\experiments\rerank_pool_20260908_r80_r20_r10\config.json, `
  .\experiments\rerank_pool_20260908_r80_r20_r10\runs.jsonl, `
  .\experiments\rerank_pool_20260908_r80_r20_r10\summary.json
```

## 5. 质量结果

| 配置 | 代码上限/实测最大 | 每轮命中 | Section Recall@5 | MRR | 相对Baseline |
| --- | --- | --- | ---: | ---: | --- |
| B34 | 80 / 34 | 30/30、30/30、30/30 | 100.00% | 0.9611 | baseline |
| C20 | 20 / 20 | 30/30、30/30、30/30 | 100.00% | 0.9611 | Recall和MRR不变 |
| C10 | 10 / 10 | 30/30、30/30、30/30 | 100.00% | 0.9444 | MRR绝对下降0.0167 |

B34和C20中，问题64的正确Section排第3，问题203的首个正确Section排第2，其余问题排第1。C10额外使问题86从第1降至第2；三轮均出现相同变化，说明这不是单次排序抖动。

## 6. 延迟与实际候选数

| 配置 | 实际候选中位数/最大值 | Rerank中位数 | Rerank P95 | 完整检索中位数 | 完整检索P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B34（配置80） | 27.5 / 34 | 1.632秒 | 3.349秒 | 2.364秒 | 8.703秒 |
| C20 | 20 / 20 | 1.291秒 | 1.664秒 | 1.933秒 | 11.489秒 |
| C10 | 10 / 10 | 0.874秒 | 1.665秒 | 1.430秒 | 4.650秒 |

相对B34：

- C20的Rerank中位数下降20.9%，Rerank P95下降50.3%，完整检索中位数下降18.2%；完整检索P95上升32.0%。
- C10的Rerank中位数下降46.4%，但MRR下降超过容忍线，因此不选。

完整检索P95与Rerank P95方向不一致。原始记录显示多次完整检索出现9至40秒长尾，而对应Rerank并未同步出现相同幅度的长尾，因此主要波动来自Embedding或网络阶段。基于本轮数据，只能可靠声称C20降低了Rerank自身延迟和完整检索中位数，不能声称完整检索P95得到改善。

## 7. 稳定性与边界

- 270/270次检索完成，失败0次，Rerank回退0次。
- C20与B34的正确Section排名三轮完全一致，但Top-5完整列表并非逐题完全相同；第一轮23/30题的Top-5集合相同。
- 本实验固定正确产品，因此不能代表端到端产品路由准确率。
- 本实验没有使用Agent临时生成的多关键词，Baseline实际候选最大34；若要证明“生产查询也可统一截断到20”仍安全，需要保存真实工具参数后增加一轮确认实验。
- 30题足以进行本轮小规模消融，但不能排除更大题集中的稀有退化。

## 8. 产物与校验值

| 文件 | SHA-256 |
| --- | --- |
| `config.json` | `597fb20248a87160c89b8ada96118f5665459a550c9e61168738dfc376f4201e` |
| `runs.jsonl` | `02bd001640d210427c1c456920fe99c74a2de8bd7833cae6cf009b234fd64f82` |
| `summary.json` | `ed23e3af2a23a038e4b7ac9cd31f1dbd064ce3902e3c4dff61b9ecc6defbcb4a` |

`runs.jsonl`保存每次调用的题号、Top-5 Section、正确Section排名、实际候选数、完整检索耗时、Rerank耗时和错误信息；`config.json`保存金标、索引和源码指纹；`summary.json`保存三组聚合结果。

原始JSON中的 `pool=80` 和汇总键 `R80` 保留实际执行参数，便于机械复算；人类可读结论将它标为B34，表示“配置上限80、当前方法实测边界34”。
