# 视觉FAISS建库与四路冒烟记录

## 1. 当前结论

2026-09-15已完成独立视觉索引构建：2,504张被Section引用且非noise的手册图片，使用 `Qwen/Qwen3-VL-Embedding-8B`生成4,096维向量，写入 `data/visual_index/visual.faiss`。索引回读为2,504条、4,096维，Metadata 2,504条，向量全部有限且L2范数最小/最大均为1.000000。

视觉功能仍保持 `VISUAL_RETRIEVAL_ENABLED=0`。V001只是一条链路冒烟：图片找图片为第1，融合Section为第1，VL Rerank无回退，耗时20.581秒；文本找图片未在Top-5命中。因此不能从该单例宣称视觉检索总体提升。

## 2. 数据范围

- 物理图片：2,605张；
- 被Section引用的唯一图片：2,563张；
- 排除noise：59张；
- 最终入库：2,504张；
- 无Caption但保留：1张；
- 未被Section引用而排除：42张；
- 一张图片关联多个Section：Metadata使用 `section_keys`列表保存。

## 3. 接口契约

使用Qwen官方公开样例图完成真实API探测，不包含项目数据：

- 文本Embedding：4,096维；
- 图片Embedding：4,096维；
- 图文混合Embedding：4,096维；
- 图片Query/图片与文本Document的VL Rerank：通过；
- 图文混合Query/Document的VL Rerank：通过。

结构化记录为 `validation_outputs/visual_api_probe.json`。

## 4. 构建过程与故障处理

1. 首次直接运行建库脚本时返回401。原因是该脚本没有像现有检索引擎一样调用 `apply_default_env()`，导致Embedding凭据为空；确认0张成功后停止并清除失败断点，修复为统一加载项目默认配置。
2. 正式构建以batch size 4运行。2,500张成功，一批4张图片收到远端500；断点续建只重试 `Manual10_55`至`Manual10_58`，随后达到2,504/2,504。
3. Windows在正式FAISS和Metadata写出后拒绝删除仍由memmap占用的临时矩阵。已修复为先flush并关闭 `_mmap`再删除；当前临时矩阵已在正式索引回读成功后移除。
4. 进度文件保留2,504个唯一成功图片ID和4条历史500错误记录；最终摘要为2,504完成、0最终失败。

## 5. 固定产物

| 文件 | SHA-256 |
| --- | --- |
| `visual.faiss` | `846f0dfd500938d5c18b42d438c6ae42400660e03ba2cdc81d37ff87f58ab309` |
| `visual_metadata.json` | `b66a745300b3b74c78b8e4834e01ebcc63d7b759c46c1719781e04be35d8f795` |
| 现役文本 `dense.faiss` | `55e82fa05bbf5514a16fab6638e49d832cdae3e044729edf81aff08078ce6238` |
| 现役文本 `retrieval_index.pkl` | `b636584692472a15fe287bdd8cd0f2687abb85f2fa638f6a6156935bcbe624a3` |

文本索引哈希仅用于证明本轮没有替换或重建现役文本索引。

## 6. 剩余工作

- 40条分来源视觉金标已完成：30张手册派生、6张GPT Image生成、1张用户实拍、3张仅限本地实验的网络照片；
- 在40条视觉金标上运行三轮四组对照；
- 运行30题文本回归和图片请求延迟门槛；
- 只有 `validate_visual_experiment.py` 输出 `visual_rollout_gate=OK`才允许默认启用。
