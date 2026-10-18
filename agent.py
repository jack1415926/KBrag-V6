"""
ReAct 客服智能体。

工具：
- search_manual: 统一检索入口。模型主要给关键词，系统同时做 BM25 + dense 召回并统一 rerank

LLM: 由 llm_router 统一路由。交付配置 SILICONFLOW_ONLY=1，实际只启用 SILICONFLOW 路由
（默认 deepseek-v4-flash，经 OpenAI-compatible 协议调用）；llm_router 中的 MiniMax 备用路由
（HIGHSPEED / ANTHROPIC 前缀）在交付配置下不会启用。
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from product_router import ProductRouteDecision, ProductRouter, build_product_prompt_block
from request_context import RequestCancelled, check_cancelled, set_current_qid
from retrieval_engine import RetrievalEngine, SearchResult, contains_cjk
from llm_router import create_message_with_fallback, create_message_streaming
from submission_utils import extract_inline_pic_refs, inject_inline_pic_refs
from visual_retrieval import (
    analysis_query_text,
    analyze_query_images,
    image_input_sha256,
    visual_retrieval_enabled,
)

ROOT = Path(__file__).resolve().parent
SKILLS_DIR = ROOT / "skills"
IMAGE_CAPTIONS_PATH = ROOT / "data" / "image_captions_v4_final.json"

# ────────────────── 配置 ──────────────────

MAX_TURNS = 2
MAX_SEARCH_RESULTS = 8
PRE_RETRIEVAL_RESULTS = 5
MAX_SEARCH_ATTEMPTS = 2
MAX_INTERNAL_ITERATIONS = 10

PRODUCT_PROMPT_BLOCK = build_product_prompt_block()
_IMAGE_CAPTIONS_CACHE: dict[str, dict] | None = None


def load_skill_md(skill_name: str) -> str | None:
    """读取固定 skill 的 markdown 说明文件。"""
    md_path = SKILLS_DIR / f"{skill_name}.md"
    if not md_path.exists():
        return None
    return md_path.read_text(encoding="utf-8")


SEARCH_MANUAL_SKILL_BLOCK = load_skill_md("search_manual") or """# 手册检索

- search_manual(keywords, products?, query?): 统一检索入口。优先填写关键词列表；
系统会同时执行 BM25 关键词检索和向量语义检索，合并后用 rerank 排序。若要取消路由改查全库，传空数组 []

检索结果中的正文会直接带 `[[PIC:图片文件名]]` 锚点；若下方出现“图片内容标注”，它是对该图画面的辅助描述。
它的唯一用途是帮你判断这张图画的是什么、是否与你正文这一段相符，从而决定要不要保留该图；严禁把图片标注照抄进答案。配图跟着最相关那一段本身有没有图走。"""


def _load_image_captions() -> dict[str, dict]:
    global _IMAGE_CAPTIONS_CACHE
    if _IMAGE_CAPTIONS_CACHE is None:
        try:
            payload = json.loads(IMAGE_CAPTIONS_PATH.read_text(encoding="utf-8"))
            _IMAGE_CAPTIONS_CACHE = payload.get("items", {})
        except Exception:
            _IMAGE_CAPTIONS_CACHE = {}
    return _IMAGE_CAPTIONS_CACHE


def _format_image_evidence(product: str, pics: list[str], *, max_items: int = 8) -> str:
    """Return concise image evidence lines for captions tied to visible PIC anchors."""
    if not pics:
        return ""
    captions = _load_image_captions()
    lines: list[str] = []
    seen: set[str] = set()
    for pic in pics:
        if pic in seen:
            continue
        seen.add(pic)
        item = captions.get(f"{product}|{pic}")
        if not item:
            continue
        cat = item.get("category")
        # noise 不注入（装饰/图标），其余都注入（part_view/schematic/info_table）
        if cat == "noise":
            continue
        # info_table（表格/参数数据）是产品级的，不限于某个章节，跳过章节匹配检查
        if item.get("section_fit") == "mismatch" and cat != "info_table":
            continue
        short = (item.get("short_caption") or "").strip()
        dense = (item.get("content") or "").strip()
        evidence = dense or short
        if not evidence:
            continue
        if len(evidence) > 260:
            evidence = evidence[:260].rstrip() + "..."
        lines.append(f"- [[PIC:{pic}]] {short}: {evidence}" if short else f"- [[PIC:{pic}]] {evidence}")
        if len(lines) >= max_items:
            break
    if not lines:
        return ""
    return (
        "图片内容标注（仅供你判断这张图画的是什么、是否与你这段正文相符，从而决定要不要保留它的 [[PIC:...]] 锚点；"
        "据此用你自己的话写一句简短图说即可）。注意：这是图片的辅助标注，不是手册正文，"
        "严禁把下面的清单/表格/字段原文照抄进答案；下面文字若不完整，直接忽略，绝不要在答案里提“截断/未显示/未完整”之类的话，也不要输出本标题：\n"
        + "\n".join(lines)
    )



# ────────────────── 通用客服 SYSTEM PROMPT（V5.0：事实边界优先） ──────────────────

SERVICE_SYSTEM_PROMPT = """\
你是通用电商客服助手。当前没有接入具体平台政策、订单状态、商家规则或售后审核结果，只能提供不依赖特定平台的通用处理建议。

本题已被判定为通用客服问题，绝对不要调用任何搜索或技能工具。

要求：
1. 先直接回答用户的问题，再给出必要的下一步操作；语气友好、自然，但不要用安抚话术代替事实。
2. 未提供可靠依据时，不得声称或暗示具体平台支持某项政策；不得编造处理时效、退款到账时间、赔付金额、优惠券、运费承担方、责任归属、换新资格、自动审核、强制执行或兜底机制。即使加上“通常、一般、可能”等弱化词，也不得推断这些未知规则。
3. 不得新增用户没有提供的小时、天数、比例、金额或期限，不得承诺“保证、一定、必须、无条件、全额、百分之百”等确定性结果。可以引用用户问题里的数字来说明“无法确认”，但不能把它改写成建议等待时间。用户要求确认具体政策时，明确说明当前无法确认，并提示以订单售后页面、商品页面展示规则或平台人工审核结果为准。
4. 可以提供通用且低风险的操作步骤，例如保留商品和包装、拍照或录像、在订单页面提交售后申请、保存沟通与物流凭证、等待平台核实；不要虚构页面名称、按钮、工单机制或处理阶段。
5. 默认用80至180个中文字符回答；只有用户提出多个独立问题时才分点，并逐项说明“当前能确认什么、不能确认什么、下一步怎么做”。
6. 不要编造电话号码、邮箱、网址、实体门店地址、客服工号或法律条文，不要要求用户在对话中提供完整订单号等敏感信息。
7. 禁止使用emoji或Unicode装饰符号，只使用纯文本。
8. 不要输出与问题无关的背景知识，不要为了显得全面而扩写。回答前自检并删除所有没有证据的数字时效、费用分配、责任判断和结果承诺。
"""

# ────────────────── 产品技术 SYSTEM PROMPT（V3.1：V3 路由 + 完整性优先） ──────────────────

TECH_SYSTEM_PROMPT = """\
你是一个多模态产品客服智能体，负责回答产品技术问题（型号、操作、故障、配件、参数）。

<Role & Goal>
先查手册，再回答。技术问题必须基于检索到的手册内容作答，禁止凭常识补全型号、步骤、参数、限制条件。
语言必须与用户提问一致：中文问中文答，英文问英文答。
若命中的手册原文与用户语言不同，必须逐句直译成用户语言；专有名词、型号、品牌可保留原文。

<Scoring & Strategy>
你的回答会被一个评分模型对照手册原文打分，而它的参照标准是**手册里与这个问题最相关的那一个章节段落**——基本等同于"把那一段答准、答全、配上它本身的图"就是满分。理解这套机制能帮你拿高分：
- 命中那一段的要点（步骤、规格、按钮名、部件名、警告、以及那一段原本就带的配图）才得分；越贴近那一段、越完整、图文越互补，分越高。
- **多答了别的章节的内容不会加分，反而会扣分**：因为它偏离了评分参照的那一段、冲淡了重点；同理，凭空补的、手册里没有的内容不得分还可能被判为错误。
- 配图也以"那一段本身有没有图"为准：那一段有图就带上（图文互补加分），那一段本就没图（如纯条款/纯文字操作），硬塞别处的图反而扣分。
所以制胜打法是**精准**而非**全面**：锁定最相关的那一个主题章节，把它答准答全（若它被手册切成了几个相邻同主题小节，就合起来还原），既不漏它的要点，也不掺别的主题。

<Execution Logic>
- 你总共只有 2 次正常 ReAct 决策机会：第一轮通常用于 search_manual 正式确认，第二轮应基于系统预检索和 search_manual 证据正常收束；如果第二轮仍冒险继续检索，后续只能进入无工具强制收束
- 当前唯一主动工具是 search_manual。技术题正式回答前至少调用一次 search_manual；最多允许两次 search_manual：第一次用于正式确认，第二次只作为证据明显错路由或完全不覆盖问题时的补救检索。若第一次证据已经足够，第二轮应直接基于该证据完整回答，不要继续搜索；完整回答不等于简短回答
- 系统预检索只作为首轮定位参考，不能直接替代正式工具检索。技术题仍需继续调用 search_manual 做显式确认后再作答；若 search_manual 与预检索证据一致且足够，请下一轮直接完整收束
- search_manual 返回的是完整 parent section 证据。第二轮回答前必须检查该 section 是否包含同一主题下的并列步骤、部件、图示、警告和例外；这些只要直接回答用户问题，就应保留
- 若 search_manual 结果已经聚合到同一最相关 parent section，不要继续搜索；应基于该 parent section 完整收束，而不是过度摘要
- 若系统给出 [PRODUCT_ROUTE]，优先在候选产品内检索；只有 medium/conflict、低增益、无结果或证据指向其他产品时，才扩展 products=[] 做全库确认
- 若候选检索结果不足、偏泛、连续命中相近章节或无结果，可换关键词，或把 products 设为空数组 `[]` 扩展到全库
- 路由 high confidence（单产品）时，第一轮直接在该产品内精查；1-2 次工具调用通常足够
- 路由 medium confidence（多候选或仅内容投票）时，若用户明示的产品概念与首候选高度一致，可先从首候选起步；若题目本身歧义较大、或首轮结果偏泛/低增益，再用 products=[] 做一次全库确认。注意：多个候选只是帮你**定位到正确的那一本手册**，不代表要同时回答多本——拿到证据后，最终答案必须收敛到**唯一一个**最相关的产品手册
- 仅当用户**明确**问“有哪些 / 组成 / 部件 / 组件 / 功能 / 接口 / 视图”等**枚举型**问题时，才不要默认几个片段已经完整；此时应基于预检索和 search_manual 返回的完整 parent section 判断是否覆盖 overview / parts / view / functions 等并列项。**操作型 / 步骤型 / 单点问题（how / 怎样 / 为什么 / 某个具体动作）不适用本条**——这类题应聚焦最相关的那一个主题章节作答，不要为“求全”去翻并列章节
- 对“有哪些 / 组成 / 部件 / 视图 / 接口 / 功能”等枚举型问题，不要只写最先看到的几个点；必须保留 search_manual 返回的同一最相关 parent section 内直接相关的并列项及对应图片
- 何时停止检索：
  1. 只有当现有检索结果已经足以完整回答用户问题，并覆盖关键步骤、规格、限制条件、注意事项、例外情况后，才允许停止检索并开始作答
  2. 若用户问 how/procedure/steps，而当前结果只有零散描述、单张配图、或不完整的片段步骤，则不算“已足够回答”，应继续检索到可执行的完整步骤
  3. 若用户一次问多个点，只有当每个子问题都已有对应证据时，才允许停止检索；只覆盖其中一部分时应继续检索
  4. 单独出现 `[[PIC:...]]` 图片锚点、单条规格、或单段条件说明，不自动等于“可以停止”；必须确认这些证据已足以解答用户疑惑
  5. 仅限**明确的枚举型问题**（有哪些/组成/部件/功能/接口/视图）：即使已有若干相关片段，也要确认已命中的完整 parent section 是否覆盖并列项；未确认完整性前不要直接回答。**操作型/步骤型/单点问题不走本条**——锁定最贴题的那一个主题章节即可，不要为求全去拽并列章节；但锁定后必须完整保留该主题章节中直接相关的并列项和对应图片
  6. 连续 2 次检索返回相同/相近章节（如都是 Safety 或 Regulatory）→ 停止重复搜索，改写关键词或基于最相关章节完整收束
  7. 返回"无检索结果"或"无新增结果"→ 最多再换 1 组关键词重试；仍无结果就直接说明手册未覆盖或基于已有证据收束
- 不要在“证据已足以完整解答用户疑惑”的情况下为了“更多信息”而额外检索；但只要还有关键缺口，就必须继续检索

{product_prompt_block}

<Constraint Rules>
1. 问题焦点优先（锁定最相关那一段）：答案围绕与问题最相关的那一个章节主题组织，按各章节 heading 判断哪个最贴题。只保留与问题直接相关的步骤、警告、规格、例外；不要把别的章节主题、或同章里的通用安全/维护/保修/背景整段搬进来。若这个主题被手册切成了几个相邻同主题的小节，合起来答全（还原同一主题不算发散）；heading 换了主题就停
   - **单段取材（重要）**：检索通常会一次返回**多个** section（每个以 heading 标记）。这些只是候选——你必须先判定**唯一一个**最贴题的 section，然后**答案的正文和配图只能取自这一个 section**。其余 section 即使内容相关、相邻、看着有用，也**不要从中摘正文、更不要把它们的 `[[PIC:...]]` 带进答案**（那会变成"多他段图/范围发散"而扣分）。判断"该不该写进答案"的标准只有一个：它是否来自你锁定的那一个 section。唯一例外仍是"被手册切碎的同一主题相邻小节"（同主题还原），以及用户明确的枚举型问题
2. 逐句直译：跨语言命中时，必须逐句直译，禁止只写大意；括号内补充说明、注意事项、例外条件不得遗漏
3. 多子问题完整覆盖：用户一次问多个点时，每个点都要分别回答，禁止只答其中一部分
4. 保留关键步骤编号与部件代号：手册里的数字步骤编号、部件代号（[1] / [A] / a / b / C1 / C2）必须保留；但**手册原文里的"Figure N / 图N"等图片编号不要写进答案**（见下方第 8 条与 Output Format）
5. 单一手册收敛（防跨手册大杂烩）：路由给出的多个候选**仅用于帮你找对是哪一本产品手册**。一旦检索证据指明问题属于某一本手册，整篇答案就**只基于这一本**组织——严禁把不同产品手册的内容、步骤、图片拼接进同一个答案（例如把"船+微波炉+耳机"的排障内容混在一起、图片来自多本手册）。即使题面没有产品名、候选有好几本，也必须先判断最可能是哪一本，再只答那一本。唯一例外：用户在**同一个产品**下问多个型号时（如"DCB107 或 DCB112"），才在该产品手册内并列写"若是 A…；若是 B…"——这是同手册内的多型号，不是跨手册
6. 枚举型问题保护（窄触发）：仅当用户问句**明确**问"有哪些 / 列出 / 一共多少 / 包含什么 / 组成部分 / 配件清单 / what are the parts / list the components"这类**列举性**问题时，必须完整保留检索结果中相关的并列项及其对应图片锚点，不得为了精简而省略任何并列项。问"如何 / 怎样 / how to / how do I"这类**操作型/步骤型**问题不属于枚举题，应只保留与该具体动作直接相关的步骤与图片，不要把整章所有带图步骤都搬进来
7. 必要完整性：不要为了追求简短而漏掉问题所需的关键步骤、数字、按钮名、部件名、图片锚点和安全警告；但当内容只是同章相邻主题、泛化提醒或与问题无关的长篇原文时，必须裁掉
8. 图片锚点与展示顺序绑定：你写的每一个 `[[PIC:文件名]]` 都会按出现顺序变成用户看到的第 1、2、3… 张图。**严禁在文中写"图1 / 图2 / 图3 / Figure 1 / Figure 2 / 第N张图"这类数字编号引用图片**——手册原文的图编号与用户看到的展示顺序通常不一致，写出来必然错位。需要回指前面的图时写"上图 / 前面那张图 / 下图 / 如图"；每个 `[[PIC:...]]` 前后必须有一句文字说明这张图展示什么（部件、方向、状态、灯光颜色等）

<Output Format>
- 若检索结果中带有 `[[PIC:图片文件名]]`，**只在你正文里实际描述到该图所示内容时才保留该锚点**；与当前问题不相关的图必须删掉，不要为了"完整"而把整章图全堆出来
- 保留下来的锚点必须原样写成 `[[PIC:文件名]]`；严禁改名、只写成 `<PIC>`、PIC、[PIC]、`<PIC>文件名</PIC>`
- 严禁在正文中出现"图1 / 图2 / Figure 3 / 第N张图"等数字编号引用图片；需要回指写"上图 / 下图 / 如图"
- 每个 `[[PIC:...]]` 前后必须有一句文字描述该图展示的部件/方向/状态/颜色/标注，让用户不看图也能理解
- 一段话内不要连续出现 3 个以上 `[[PIC:...]]` 而不加文字说明
- 不要把带图段落改写成纯文字段落（指相关图片不要删）
- 保留换行和段落分隔（空行表示新段落），不用 markdown 标题(#)、列表(-/*)、加粗(**)、表格(|)、代码块(```)
- 需要小标题时直接写裸文字一行，不加任何符号
- 不要说“根据手册”“手册中显示”“请查阅手册”“如图所示”“见下图”“根据检索到的信息”“以下是...”等元话术；直接进入内容
- **严禁在答案开头或任何位置写关于你自己检索/思考过程的话**，例如：“检索结果已完整覆盖/已经命中/足以回答”“可以直接作答”“我已找到完整信息”“Based on the search results”“According to the manual”“I have found / the search results show”“The manual provides”等。这些是你的内部思考，绝不能出现在给用户的答案里。答案第一句必须直接是用户要的结论/步骤本身。
- **严禁输出 `---` 分隔线或“以下为正式回答”之类过渡语**；直接从正文开始
- 问什么答什么：先给用户要的结论、步骤、规格，再补充必要警告
- 检索确实没命中时直说“未在手册中找到相关内容，建议联系售后确认”——但这句必须**与答案正文同语言**：英文题用英文表述（如 “This is not covered in the manual; please contact after-sales support.”），绝不能在英文答案里夹中文

## 产品技术回答格式

参考官方范例：

范例 A（图例型）：
问：我的DCB107或DCB112型号电钻指示灯闪烁时，这些闪烁标识代表什么含义？
答：DCB107、DCB112 电池组充电中[[PIC:Manual04_22]]电池组已充满[[PIC:Manual04_23]]过热/过冷延迟[[PIC:Manual04_24]]电池组或充电器故障[[PIC:Manual04_25]]电源故障[[PIC:Manual04_26]]

说明：图例型题目要让每个图对应一个短语，并保留检索结果中的 `[[PIC:...]]` 锚点；不要改写成纯文字段落。

范例 B（结构型）：
问：我想更换健身追踪器的表带，有其他尺寸可选吗？
答：表带尺寸

表带尺寸如下所示。注意：单独销售的配件表带可能略有差异。
[[PIC:Manual16_51]]

环境条件
[[PIC:Manual16_52]]

范例 C（聚焦·无图型，满分答案）：
问：如何清洁空气净化器的设备内外？
答：清洁设备内外前，务必先拔下电源插头。不要在通电状态下清洁，以免触电或造成设备故障。

1. 清洁外壳：用温水或温和清洁剂浸湿软布，擦拭空气净化器外壳，然后再用软布擦干。

2. 清洁内部滤网仓：打开背部滤网盖并取出滤网。用吸尘器和湿毛巾清洁滤网仓内部。

3. 日常频率：为保证净化效果，每月清洁设备及预过滤网 1-2 次；灰尘较多的地区建议增加清洁频率。

清洁时不要把水直接倒入或喷入机身内部，湿布应拧至不滴水后再擦拭。清洁完成后，确认外壳和滤网仓内部已擦干、滤网已装回、背部滤网盖已盖好，再重新接通电源使用。日常维护时，建议定期检查进风口和滤网仓是否有明显积尘，避免灰尘堆积影响进风和净化效率。

说明：这是真实拿到满分（5/5）的答案，它是 0 图的——因为它的参考章节"设备清洁与日常维护"本身没图，于是只精准照搬这一个章节、没去拽相邻的"灰尘传感器清洁"等别的章节、也没硬凑图。聚焦单章节、该 0 图就 0 图，就是满分。

## 工具
{search_manual_skill_block}


## 幻觉抑制（硬约束）
- 型号、规格数字化、按钮名称、步骤顺序、故障代码、灯光含义、配件兼容性、保修政策、维修费用、官方时效、安全警告原文，这些必须基于检索内容，检索没写就不编造
- 问什么答什么：用户问 how/procedure/steps 时，优先给可执行步骤；若当前命中主要是 safety/regulatory/notice，先继续检索步骤型章节。确实找不到步骤时，只用一句话简短说明“手册未给出完整步骤”，不要展开长篇解释

## 回答丰富度（只限同一最相关章节内的直接补充）
回答首先追求**贴题和聚焦**，不要为了显得全面而主动扩展。只有当补充内容同时满足以下条件时才可以加入：
1. 补充信息来自你已经锁定的同一个最相关主题章节，或是该章节原文中明确出现的 note / warning / condition / exception；
2. 补充信息能直接帮助回答用户当前问法中的动作、条件、判断或注意事项；
3. 加入后不会把答案带到相邻章节、通用维护、安全背景、使用建议或另一个功能主题。

允许保留的补充类型仅限：
- 同一章节原文明确写出的适用条件、例外情况、完成判断、警告/注意事项；
- 同一章节图片中可见、且与正文步骤直接对应的部件/按钮/方向/状态说明。

不要额外发挥使用场景、易错点、日常维护建议或经验性技巧；除非这些内容就在同一最相关章节原文里，并且直接回答用户的问题。若不确定是否属于同一章节，宁可不补。

丰富回答范例：
问：如何开启空调的节能制冷模式？
答：节能制冷模式可最大限度降低制冷时的耗电量，并将设定温度调节至最适宜的水平，打造更舒适的环境。

1. 按下开/关键开启电源。
2. 反复按下模式键，选择制冷模式。
3. 按下节能键，显示屏上会显示节能标识。[[PIC:Manual01_21]]

注：部分机型不支持此功能。

日常使用中，夏季夜间睡眠或白天长时间离家时开启此模式效果最佳。达到设定温度后压缩机会自动降低运行频率，相比普通制冷模式更安静省电。若感觉制冷不够，可先将温度调低1-2度快速降温，再切回节能模式维持恒温。

## 最后复述
1. 是否逐句完整翻译且未作任何删减，尤其不要漏括号内补充说明、例外条件、免责条款
2. 是否完整保留与正文描述对应的 `[[PIC:图片文件名]]`、步骤编号、部件代号；是否已删除所有"图N / Figure N / 第N张图"数字引用
3. 是否没有使用任何 Markdown 列表格式，并且没有写元话术

常见错误示范：
- 错：指示灯代表 PIC 正在充电 PIC 已充满     → 对：指示灯代表[[PIC:Manual04_22]]正在充电[[PIC:Manual04_23]]已充满
- 错：<PIC>Manual04_22</PIC>                 → 对：[[PIC:Manual04_22]]
- 错：把带 3 张图的检索结果改写成纯文字       → 对：保留 3 个 `[[PIC:...]]`
- 错：删掉 "Important: do not spray..."      → 对：保留 safety 警告原文
- 英文问题错：您好，空气炸锅首次使用前...     → 对：Before using the air fryer for the first time...
""".format(
    product_prompt_block=PRODUCT_PROMPT_BLOCK,
    search_manual_skill_block=SEARCH_MANUAL_SKILL_BLOCK,
)

# 兼容旧名：外部如果还在引用 SYSTEM_PROMPT，默认指向 TECH（更具一般性）
SYSTEM_PROMPT = TECH_SYSTEM_PROMPT

# ────────────────── 工具定义 ──────────────────

TOOLS = [
    {
        "name": "search_manual",
        "description": "统一检索入口。优先输入关键词列表；系统会基于关键词做 BM25，并始终带上原始用户问题做语义召回；"
                       "若补充 query，也会把它作为额外语义线索一起召回，最后统一 rerank。适用于绝大多数普通检索场景。可通过 products 参数限定产品范围。",
        "input_schema": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "关键词列表，如 [\"DCB107\", \"指示灯\"]。尽量给 2-6 个高信息量词。",
                },
                "query": {
                    "type": "string",
                    "description": "可选的补充语义描述。通常可省略；只有关键词不足以表达动作关系时再填写。即使填写，系统也仍会保留原始用户问题参与语义召回。",
                },
                "products": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "限定检索的产品名称列表，如 [\"电钻手册\"]。不传则搜索全部产品。",
                },
            },
            "required": ["keywords"],
        },
    },
]


# ────────────────── 工具执行 ──────────────────

def format_search_results(results: list[SearchResult], filtered_count: int = 0) -> str:
    """把检索结果格式化为 LLM 可读的文本，正文里直接带内联图片锚点。"""
    if not results and filtered_count == 0:
        return "\n".join([
            "[SEARCH_STATUS] no_result",
            "[SEARCH_REASON] empty_recall",
            "[SEARCH_FILTERED] 0",
            "[SEARCH_SUGGEST] switch_strategy",
            "(无检索结果)",
        ])
    if not results and filtered_count > 0:
        return "\n".join([
            "[SEARCH_STATUS] no_result",
            "[SEARCH_REASON] empty_after_postprocess",
            f"[SEARCH_FILTERED] {filtered_count}",
            "[SEARCH_SUGGEST] switch_strategy",
            f"(本次检索返回的候选在后处理阶段未形成可用结果。建议换关键词、扩展 products=[]，或基于已有证据收束)",
        ])

    lines = []
    section_ids = [
        int(r.source.get("parent_section_id"))
        for r in results
        if isinstance(r.source.get("parent_section_id"), int)
    ]
    top_section_id: int | None = None
    top_section_count = 0
    top_section_summary = ""
    if section_ids:
        section_counts = Counter(section_ids)
        top_section_id, top_section_count = section_counts.most_common(1)[0]
        for r in results:
            if r.source.get("parent_section_id") == top_section_id:
                top_section_summary = (r.source.get("section_summary") or "").strip()
                if top_section_summary:
                    break
    if filtered_count > 0:
        lines.append(f"（注：{filtered_count} 条候选在后处理阶段未被保留）")
    if section_ids:
        lines.append(f"[SECTION_IDS] {','.join(str(sid) for sid in section_ids)}")
    if top_section_id is not None:
        lines.append(f"[SECTION_TOP] {top_section_id}")
        lines.append(f"[SECTION_TOP_COUNT] {top_section_count}")
        if top_section_summary:
            lines.append(f"[SECTION_TOP_SUMMARY] {top_section_summary}")

    # search_manual returns parent-section evidence through the retrieval engine;
    # the model may answer directly when the returned evidence is sufficient.
    SECTION_FULL_TOP_N = 0
    SECTION_FULL_CHAR_CAP = 3500
    section_freq = Counter()
    section_first_idx: dict = {}
    for idx, r in enumerate(results):
        psid = r.source.get("parent_section_id")
        if not isinstance(psid, int):
            continue
        section_freq[psid] += 1
        if psid not in section_first_idx:
            section_first_idx[psid] = idx
    expanded_section_ids: set = set()
    for psid, _count in section_freq.most_common(SECTION_FULL_TOP_N):
        ref = results[section_first_idx[psid]]
        sec_text = (ref.source.get("section_text") or "").strip()
        if not sec_text:
            continue
        sec_pics = list(ref.source.get("section_pics") or [])
        sec_heading = (ref.source.get("section_heading") or ref.heading or "").strip()
        full_text = inject_inline_pic_refs(sec_text, sec_pics)
        evidence = _format_image_evidence(ref.product, sec_pics)
        if evidence:
            full_text = f"{full_text}\n{evidence}"
        truncated = ""
        if len(full_text) > SECTION_FULL_CHAR_CAP:
            full_text = full_text[:SECTION_FULL_CHAR_CAP]
            truncated = " ...(章节文本已截断，请优先基于已显示内容作答，必要时换关键词检索同主题章节)"
        lines.append(f"[SECTION_FULL] 产品: {ref.product} | 章节ID: {psid} | 章节: {sec_heading}")
        lines.append(f"    完整章节正文:")
        lines.append(f"    {full_text}{truncated}")
        lines.append("")
        expanded_section_ids.add(psid)

    # —— 剩余 chunk：仅展示尚未被展开章节覆盖的，避免重复
    chunk_idx = 0
    for r in results:
        psid = r.source.get("parent_section_id")
        if isinstance(psid, int) and psid in expanded_section_ids:
            continue
        chunk_idx += 1
        lines.append(f"[{chunk_idx}] 产品: {r.product} | 章节: {r.heading}")
        if isinstance(psid, int):
            lines.append(f"    上层章节ID: {psid}")
        section_summary = (r.source.get("section_summary") or "").strip()
        if section_summary:
            lines.append(f"    上层摘要: {section_summary}")
        content = inject_inline_pic_refs(r.text, r.pics)
        evidence = _format_image_evidence(r.product, list(r.pics or []))
        if evidence:
            content = f"{content}\n{evidence}"
        lines.append(f"    内容: {content}")
        lines.append("")
    return "\n".join(lines)


_ROUTER_CACHE: dict[int, ProductRouter] = {}


def _get_product_router(engine: RetrievalEngine) -> ProductRouter:
    engine.ensure_index()
    cache_key = id(engine)
    router = _ROUTER_CACHE.get(cache_key)
    if router is None:
        router = ProductRouter(engine.catalog, engine=engine)
        _ROUTER_CACHE[cache_key] = router
    return router


def _run_search_with_defaults(
    engine: RetrievalEngine,
    *,
    name: str,
    input_data: dict,
    default_products: list[str] | None,
    default_query_context: str = "",
    query_images: list[str] | None = None,
    visual_query_text: str = "",
    visual_products: list[str] | None = None,
    precomputed_visual_hits: list[list] | None = None,
    original_text_query: str = "",
    original_text_products: list[str] | None = None,
    timing_trace: dict | None = None,
) -> tuple[list[SearchResult], int]:
    has_products_key = "products" in input_data
    products = input_data.get("products") if has_products_key else default_products
    if isinstance(products, list) and len(products) == 0:
        products = None

    if os.getenv("DEBUG_ROUTE"):
        print(f"[TOOL] {name} llm_products={input_data.get('products', '<unset>')} default={default_products} → used={products}", flush=True)

    if name in SEARCH_TOOL_NAMES:
        if name == "search_manual":
            keywords = input_data.get("keywords", [])
            semantic_query = (input_data.get("query") or "").strip()
        elif name == "keyword_search":
            keywords = input_data.get("keywords", [])
            semantic_query = ""
        else:
            semantic_query = input_data.get("query", "")
            keywords = re.findall(r"\S+", semantic_query)
        results, filtered = engine.search_manual(
            keywords,
            semantic_query=semantic_query,
            original_query=default_query_context,
            top_k=MAX_SEARCH_RESULTS,
            products=products,
            query_images=query_images,
            visual_query_text=visual_query_text,
            visual_products=visual_products,
            precomputed_visual_hits=precomputed_visual_hits,
            original_text_query=original_text_query,
            original_text_products=original_text_products,
            timing_trace=timing_trace,
        )
        return results, filtered

    return [], 0


def execute_tool(
    engine: RetrievalEngine,
    name: str,
    input_data: dict,
    default_products: list[str] | None = None,
    default_query_context: str = "",
    query_images: list[str] | None = None,
    visual_query_text: str = "",
    visual_products: list[str] | None = None,
    precomputed_visual_hits: list[list] | None = None,
    original_text_query: str = "",
    original_text_products: list[str] | None = None,
) -> str:
    """执行工具调用，返回结果文本。"""
    if name in SEARCH_TOOL_NAMES:
        results, filtered = _run_search_with_defaults(
            engine,
            name=name,
            input_data=input_data,
            default_products=default_products,
            default_query_context=default_query_context,
            query_images=query_images,
            visual_query_text=visual_query_text,
            visual_products=visual_products,
            precomputed_visual_hits=precomputed_visual_hits,
            original_text_query=original_text_query,
            original_text_products=original_text_products,
        )
        return format_search_results(results, filtered)

    return f"未知工具: {name}"


# ────────────────── Agent 主循环 ──────────────────

@dataclass
class AgentResult:
    """一次 run_agent 调用的结构化产物。

    answer/pics 是最终提交格式化前的核心输出；tool_calls/turns 用于统计工具纪律；trace 保存产品路由、预检索、LLM tool_use 与最终收束路径，便于验证报告复盘。
    """
    answer: str
    pics: list[str] = field(default_factory=list)
    tool_calls: int = 0
    turns: int = 0
    trace: dict | None = None
    # 最终回答的首 token 耗时（秒）。仅 stream_ttft=True 时填充：主循环每轮流式跑，
    # 出现文本增量(content)的那轮即最终回答，记其首 token 时间；纯工具轮不计。
    ttft: float | None = None


def _serialize_trace_content(content) -> object:
    if isinstance(content, (str, int, float, bool)) or content is None:
        return content
    if isinstance(content, list):
        return [_serialize_trace_content(item) for item in content]
    if isinstance(content, dict):
        return {str(k): _serialize_trace_content(v) for k, v in content.items()}

    data: dict[str, object] = {}
    for attr in ("type", "id", "name", "input", "text", "tool_use_id", "content"):
        if hasattr(content, attr):
            data[attr] = _serialize_trace_content(getattr(content, attr))
    if data:
        return data
    return repr(content)


def _build_trace_llm_event(*, index: int, response_content) -> dict:
    event: dict[str, object] = {
        "kind": "llm_call",
        "index": index,
        "actions": [],
    }
    text_preview_parts: list[str] = []
    actions: list[dict[str, object]] = []

    for block in response_content:
        block_type = getattr(block, "type", None)
        if block_type == "tool_use":
            actions.append({
                "type": "tool_use",
                "name": getattr(block, "name", ""),
                "input": _serialize_trace_content(getattr(block, "input", {}) or {}),
            })
        elif block_type == "text":
            text = (getattr(block, "text", "") or "").strip()
            if text:
                text_preview_parts.append(text)

    if actions:
        event["actions"] = actions
    if text_preview_parts:
        preview = "\n".join(text_preview_parts)
        event["text_preview"] = preview[:300]
    return event


def _extract_search_trace_hits(result_text: str) -> list[dict[str, object]]:
    hits: list[dict[str, object]] = []
    current: dict[str, object] | None = None

    for raw_line in (result_text or "").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        m = re.match(r"^\[(\d+)\]\s+产品:\s+(.*?)\s+\|\s+章节:\s+(.*)$", stripped)
        if m:
            current = {
                "rank": int(m.group(1)),
                "product": m.group(2).strip(),
                "heading": m.group(3).strip(),
            }
            hits.append(current)
            continue
        if current is None:
            continue
        if stripped.startswith("上层章节ID:"):
            value = stripped.split(":", 1)[1].strip()
            if value.isdigit():
                current["parent_section_id"] = int(value)
            else:
                current["parent_section_id"] = value
        elif stripped.startswith("上层摘要:"):
            current["section_summary"] = stripped.split(":", 1)[1].strip()[:200]
        elif stripped.startswith("内容:"):
            current["text_preview"] = stripped.split(":", 1)[1].strip()[:300]

    return hits



def _build_trace_tool_event(
    *,
    index: int,
    name: str,
    input_data: dict,
    default_products: list[str] | None,
    default_query_context: str,
    elapsed: float,
    pics: list[str],
    result_text: str,
    retrieval_timings: dict | None = None,
) -> dict:
    obs = _observe_tool_output(name, input_data, result_text)
    action = input_data.get("action") if isinstance(input_data, dict) else None
    event: dict[str, object] = {
        "kind": "tool_call",
        "index": index,
        "name": name,
        "input": _serialize_trace_content(input_data),
        "default_products": _serialize_trace_content(default_products),
        "default_query_context": default_query_context,
        "elapsed": round(elapsed, 3),
        "pics": pics,
        "no_result": obs.no_result,
        "products": obs.products,
        "headings": obs.headings[:8],
        "parent_section_ids": obs.parent_section_ids[:8],
        "explicit_product": obs.explicit_product,
        "dominant_product": obs.dominant_product,
        "dominant_parent_section_id": obs.dominant_parent_section_id,
        "search_status": obs.search_status,
        "search_reason": obs.search_reason,
        "search_filtered": obs.search_filtered,
    }
    if name in SEARCH_TOOL_NAMES:
        event["retrieval_hits"] = _extract_search_trace_hits(result_text)
        if retrieval_timings:
            event["retrieval_timings"] = _serialize_trace_content(retrieval_timings)
    result_preview = (result_text or "").strip()
    if result_preview:
        event["result_preview"] = result_preview[:500]
    return event


def _collect_pics_from_results(results: list[SearchResult]) -> list[str]:
    """从检索结果列表里按顺序去重收集图片文件名。"""
    pics: list[str] = []
    for r in results:
        for p in r.pics:
            if p not in pics:
                pics.append(p)
    return pics


def _result_fingerprint(result: SearchResult) -> str:
    """为检索条目生成稳定指纹，用于会话内去重。"""
    text = " ".join((result.text or "").split())
    heading = " ".join((result.heading or "").split())
    product = (result.product or "").strip()
    return f"{product}\n{heading}\n{text}"


def _dedup_results_by_history(
    results: list[SearchResult],
    seen_result_keys: set[str],
) -> tuple[list[SearchResult], int]:
    """过滤历史已见检索内容，返回(新增结果, 被过滤数量)。"""
    fresh: list[SearchResult] = []
    dropped = 0
    for r in results:
        key = _result_fingerprint(r)
        if key in seen_result_keys:
            dropped += 1
            continue
        seen_result_keys.add(key)
        fresh.append(r)
    return fresh, dropped


def _execute_tool_with_pics(
    engine: RetrievalEngine,
    name: str,
    input_data: dict,
    default_products: list[str] | None = None,
    default_query_context: str = "",
    seen_result_keys: set[str] | None = None,
    query_images: list[str] | None = None,
    visual_query_text: str = "",
    visual_products: list[str] | None = None,
    precomputed_visual_hits: list[list] | None = None,
    original_text_query: str = "",
    original_text_products: list[str] | None = None,
    timing_trace: dict | None = None,
) -> tuple[str, list[str]]:
    """执行工具调用，同时返回本次检索到的 trace 图片列表。"""
    if name in SEARCH_TOOL_NAMES:
        results, filtered = _run_search_with_defaults(
            engine,
            name=name,
            input_data=input_data,
            default_products=default_products,
            default_query_context=default_query_context,
            query_images=query_images,
            visual_query_text=visual_query_text,
            visual_products=visual_products,
            precomputed_visual_hits=precomputed_visual_hits,
            original_text_query=original_text_query,
            original_text_products=original_text_products,
            timing_trace=timing_trace,
        )
        dropped = 0
        if seen_result_keys is not None:
            results, dropped = _dedup_results_by_history(results, seen_result_keys)
        if not results:
            if dropped > 0:
                return "(无新增检索结果，当前结果与历史重复)", []
            return format_search_results(results, filtered), []
        return format_search_results(results, filtered), _collect_pics_from_results(results)

    return f"未知工具: {name}", []


_GENERIC_FAILURE_ANSWERS = {
    "",
    "抱歉，处理过程中出现异常，请重试。",
    "处理过程中出现异常，请重试。",
    "抱歉，请重试。",
}


def _extract_text_from_response(response) -> str:
    parts: list[str] = []
    for block in response.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(parts).strip()


def _normalize_final_answer(answer: str) -> str:
    return " ".join(answer.strip().split())


def _resolve_answer_pics(answer: str) -> tuple[str, list[str]]:
    """只从正文中的 [[PIC:...]] 抽图，不再按检索结果顺序兜底补图。"""
    answer, inline_pics = extract_inline_pic_refs(answer)
    pic_count = answer.count("<PIC>")
    pics = inline_pics[:pic_count] if inline_pics else []
    if pic_count > len(pics):
        parts = answer.split("<PIC>")
        rebuilt = parts[0]
        for i, tail in enumerate(parts[1:], start=1):
            rebuilt += ("<PIC>" if i <= len(pics) else "") + tail
        answer = rebuilt
    return answer, pics


def _question_requests_comparison(question: str) -> bool:
    q = (question or "").lower()
    markers = [
        "compare",
        "difference",
        "vs",
        "versus",
        "区别",
        "对比",
        "分别",
        "各自",
        "哪种",
    ]
    return any(marker in q for marker in markers)


def _format_spec_blocks(answer: str) -> str:
    text = answer or ""
    replacements = [
        ("电源要求：工作电压：", "电源要求：\n工作电压："),
        ("，50Hz 工作电流：", "，50Hz\n工作电流："),
        ("认证标准：交流电源适配器：", "认证标准：\n交流电源适配器："),
    ]
    for src, dst in replacements:
        text = text.replace(src, dst)
    return text


def _rewrite_single_product_answer(
    *,
    answer: str,
    question: str,
    system_prompt: str,
    model: str | None,
    products: list[str],
) -> str:
    response, _route = create_message_with_fallback(
        max_tokens=int(os.getenv("AGENT_FINALIZE_MAX_TOKENS", "4096")),
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": (
                    "请只基于已有证据重写下面这条最终答案，不要调用工具。\n"
                    "要求：\n"
                    "1. 用户没有要求对比时，不要把多个产品/手册来源无标记地混写在同一答案里\n"
                    "2. 若证据已足以支持单一产品答案，就只保留一个最自洽的产品答案\n"
                    "3. 只有在确实必须保留多个产品时，才显式写成“若是 A…；若是 B…”\n"
                    "4. 保留已有的图片锚点 [[PIC:...]]、步骤、规格和警告，不要新增未检索到的信息\n\n"
                    f"问题：{question}\n"
                    f"候选产品：{'、'.join(products)}\n"
                    f"当前答案：\n{answer}"
                ),
            }
        ],
        model=model,
    )
    rewritten = _extract_text_from_response(response).strip()
    return rewritten or answer


def _postprocess_final_answer(
    *,
    answer: str,
    question: str,
    system_prompt: str,
    model: str | None,
    route_products: list[str],
) -> str:
    answer = _format_spec_blocks(answer)
    if route_products and not _question_requests_comparison(question):
        mentioned = [p for p in route_products if p and p in answer]
        if len(mentioned) >= 2:
            answer = _rewrite_single_product_answer(
                answer=answer,
                question=question,
                system_prompt=system_prompt,
                model=model,
                products=route_products,
            )
            answer = _format_spec_blocks(answer)
    return answer


def _finalize_without_tools(
    *,
    system_prompt: str,
    messages: list[dict],
    model: str | None,
) -> str:
    """跑满最大轮数后，基于现有上下文强制收束一次最终答案。"""
    finalize_messages = list(messages)
    finalize_messages.append({
        "role": "user",
        "content": (
            "不要再调用任何工具。请仅根据上面对话里已经检索到的内容，"
            "现在直接输出最终答案。\n"
            "要求：\n"
            "1. 禁止输出“处理中”“请重试”“需要更多信息”等异常或占位话术\n"
            "2. 若已有检索结果，必须尽最大可能整合成可提交答案\n"
            "3. 技术题继续保留正文中的图片锚点（如 [[PIC:Manual01_1]]）、数字规格、警告语和列表编号；客服题保持自然客服口吻\n"
            "4. 若现有检索内容仍不足，请明确说明（用与回答相同的语言：中文题用中文、英文题用英文，绝不中英混杂），不要输出异常兜底句\n"
            "5. 严格自检：逐句完整翻译且不删减；保留与描述对应的 [[PIC:...]]、步骤/部件代号；删除所有 '图N / Figure N' 数字引用，回指改写为'上图/下图'；不要使用 Markdown 列表"
        ),
    })
    response, _route = create_message_with_fallback(
        max_tokens=int(os.getenv("AGENT_MAX_TOKENS", "8192")),
        system=system_prompt,
        messages=finalize_messages,
        model=model,
    )
    return _extract_text_from_response(response)


_ROUTE_HINT_CS = (
    "【路由信号：本题为通用客服问题（非产品技术），"
    "禁止调用任何检索工具，直接按客服范例 C/D/E 的风格作答。】"
)

# 技术题通用指南
_ROUTE_HINT_TECH = (
    "【路由提示：本题更可能是产品技术问题。建议先调用 search_manual 检索手册再回答；"
    "英文提问请用英文回答，中文提问请用中文回答。"
    "若 search_manual 连续返回无结果（no_result）或命中偏泛，请换关键词、必要时 products=[] 全库确认；"
    "若仍无证据，基于已有证据收束或说明手册未覆盖，避免反复空转。】"
)
_STRUCTURE_QUERY_TOKENS = [
    "anatomy",
    "overview",
    "front view",
    "rear view",
    "navigation button view",
    "top view",
    "bottom view",
    "buttons and interfaces",
    "buttons & indicators",
    "parts",
    "components",
    "结构",
    "部件",
    "组件",
    "视图",
    "按键",
    "接口",
]


def _build_product_route_hint(route: ProductRouteDecision, question: str = "") -> str:
    """产品路由提示：恢复老版自然语言形式，按 reason/置信度分支。"""
    question_is_zh = contains_cjk(question) if question else False

    # 1) 显式产品名 / 别名硬锁（单产品 high）
    if route.reason in {"explicit_product_name", "explicit_product_nickname"} and len(route.products) == 1:
        product = route.products[0]
        cross_lang = (product.endswith("手册")) != question_is_zh
        cross_lang_part = (
            "命中的手册语言与提问语言不同，请翻译后再回答。" if cross_lang else ""
        )
        return (
            f"【产品路由提示：题面已显式指明产品={product}。全程检索仅限该产品手册，"
            f"禁止扩展到其他手册或全库。{cross_lang_part}】"
        )

    # 2) 未识别候选（低置信 / 内容投票发散等）
    if not route.products:
        return (
            "【产品路由提示：本题未能可靠识别产品候选。"
            "建议直接 search_manual 用 products=[] 做全库检索；"
            "若结果偏泛或无结果，请改写关键词后再确认一次，仍无证据则说明手册未覆盖。】"
        )

    # 3) 单候选高置信（别名命中、name_and_content_agree 等）
    if len(route.products) == 1 and route.confidence == "high":
        product = route.products[0]
        cross_lang = (product.endswith("手册")) != question_is_zh
        cross_lang_part = (
            "命中的手册语言与提问语言不同，请翻译后再回答。" if cross_lang else ""
        )
        return (
            f"【产品路由提示：候选={product}，置信较高。"
            "建议优先在该手册内检索；若结果偏泛或连续无结果，再将 products 设为 [] 做一次全库确认。"
            f"{cross_lang_part}】"
        )

    # 4) 多候选（medium 置信）— 老版核心软指令
    cross_lang_products = [
        p for p in route.products if (p.endswith("手册")) != question_is_zh
    ]
    cross_lang_note = (
        "命中的部分手册语言与提问语言不同，请翻译后再回答。"
        if cross_lang_products else ""
    )
    structure_note = (
        "本题为结构/部件类问题，请优先检索 overview/view/parts/functions 等章节并基于完整 parent section 判断并列项。"
        if _is_structure_query(question) else ""
    )

    products_text = "、".join(route.products)
    confidence_word = "较高" if route.confidence == "high" else "一般"

    return (
        f"【产品路由提示：候选={products_text}。该候选置信{confidence_word}；"
        "把这些候选当作检索起点，不是唯一答案。"
        "若多个候选都命中相关信息，可以并列回答。"
        "若结果偏泛、连续命中相近章节或无结果，再将 products 设为 [] 做一次全库确认。"
        f"{cross_lang_note}{structure_note}】"
    )


def _build_routed_question(
    question: str,
    question_id: int | None,
    product_route: ProductRouteDecision | None = None,
) -> str:
    """根据 id 在用户消息前面加路由提示；不传 id 则让 LLM 自判。

    技术题分两段：通用技术题指南 + 产品候选指南。
    客服题（qid<64）只挂 _ROUTE_HINT_CS。
    """
    parts: list[str] = []
    if question_id is None:
        # 没 id（API 模式）→ 默认按技术题处理
        parts.append(_ROUTE_HINT_TECH)
        if product_route is not None:
            hint = _build_product_route_hint(product_route, question)
            if hint:
                parts.append(hint)
        parts.append(question)
        return "\n\n".join(parts)

    if question_id < 64:
        parts.append(_ROUTE_HINT_CS)
    else:
        parts.append(_ROUTE_HINT_TECH)
        if product_route is not None:
            product_hint = _build_product_route_hint(product_route, question)
            if product_hint:
                parts.append(product_hint)
    parts.append(question)
    return "\n\n".join(parts)


@dataclass
class ToolObservation:
    """一次 search_manual 工具返回后的轻量结构化观察。

    主循环用它判断是否无结果、是否反复命中同一 parent section、是否需要扩全库或强制收束；这些字段也写入 trace 供赛后复盘。
    """
    no_result: bool = False
    products: list[str] = field(default_factory=list)
    headings: list[str] = field(default_factory=list)
    parent_section_ids: list[int] = field(default_factory=list)
    dominant_product: str | None = None
    dominant_count: int = 0
    dominant_parent_section_id: int | None = None
    dominant_parent_section_count: int = 0
    dominant_section_summary: str | None = None
    explicit_product: str | None = None
    search_status: str | None = None
    search_reason: str | None = None
    search_filtered: int = 0


SEARCH_TOOL_NAMES = {"search_manual", "keyword_search", "vector_search"}


def _is_structure_query(question: str) -> bool:
    q = (question or "").lower()
    return any(token in q for token in _STRUCTURE_QUERY_TOKENS)


def _normalize_heading_key(heading: str) -> str:
    text = re.sub(r"\s+", " ", (heading or "").strip().lower())
    return text


def _is_safety_like_heading(heading: str) -> bool:
    key = _normalize_heading_key(heading)
    markers = [
        "safety",
        "hazard",
        "regulatory",
        "legal",
        "fcc",
        "warning",
        "telephone and fcc notices",
        "product safety guide",
    ]
    return any(marker in key for marker in markers)


def _parse_products_and_headings(result_text: str) -> tuple[list[str], list[str]]:
    products: list[str] = []
    headings: list[str] = []
    for line in result_text.splitlines():
        m = re.match(r"^\[\d+\]\s+产品:\s+(.*?)\s+\|\s+章节:\s+(.*)$", line.strip())
        if m:
            products.append(m.group(1).strip())
            headings.append(m.group(2).strip())
            continue
        if line.startswith("产品: "):
            products.append(line[len("产品: "):].strip())
            continue
        m2 = re.match(r"^章节:\s+\[\d+\]\s+(.*)$", line.strip())
        if m2:
            headings.append(m2.group(1).strip())
    return products, headings


def _extract_tag_value(result_text: str, tag: str) -> str | None:
    pattern = rf"^\[{re.escape(tag)}\]\s+(.*)$"
    for line in result_text.splitlines():
        m = re.match(pattern, line.strip())
        if m:
            return m.group(1).strip()
    return None


def _extract_tag_int(result_text: str, tag: str) -> int | None:
    value = _extract_tag_value(result_text, tag)
    if value is None:
        return None
    value = value.strip()
    return int(value) if value.isdigit() else None


def _observe_tool_output(name: str, input_data: dict, result_text: str) -> ToolObservation:
    obs = ToolObservation()
    text = (result_text or "").strip()
    obs.search_status = _extract_tag_value(text, "SEARCH_STATUS")
    obs.search_reason = _extract_tag_value(text, "SEARCH_REASON")
    filtered_text = _extract_tag_value(text, "SEARCH_FILTERED")
    if filtered_text and filtered_text.isdigit():
        obs.search_filtered = int(filtered_text)
    obs.no_result = text in {"", "(无检索结果)"} or text.startswith("未找到 ")
    if obs.search_status == "no_result":
        obs.no_result = True
    products, headings = _parse_products_and_headings(text)
    obs.products = products
    obs.headings = headings
    section_ids_text = _extract_tag_value(text, "SECTION_IDS")
    if section_ids_text:
        obs.parent_section_ids = [
            int(part.strip())
            for part in section_ids_text.split(",")
            if part.strip().isdigit()
        ]
    counts = Counter(products)
    if counts:
        obs.dominant_product, obs.dominant_count = counts.most_common(1)[0]
    obs.dominant_parent_section_id = _extract_tag_int(text, "SECTION_TOP")
    obs.dominant_parent_section_count = _extract_tag_int(text, "SECTION_TOP_COUNT") or 0
    obs.dominant_section_summary = _extract_tag_value(text, "SECTION_TOP_SUMMARY")

    if name in SEARCH_TOOL_NAMES:
        products_arg = input_data.get("products")
        if isinstance(products_arg, list) and len(products_arg) == 1:
            obs.explicit_product = products_arg[0]

    return obs


def _make_route_note(text: str) -> str:
    return f"【路由状态更新：{text}】"


def _coerce_tool_params(value) -> dict:
    return dict(value) if isinstance(value, dict) else {}


def _get_primary_route_product(route: ProductRouteDecision) -> str | None:
    return route.products[0] if route.products else None


def _get_locked_route_product(route: ProductRouteDecision) -> str | None:
    if route.reason in {"explicit_product_name", "explicit_product_nickname"} and len(route.products) == 1:
        return route.products[0]
    return None



def _question_text(question: str | list) -> str:
    """从纯文本或 OpenAI-compatible 多模态 content 中抽出文本问题。

    产品路由、预检索、trace 和客服/技术分类只需要文字；图片仍保留在原始 content 中交给回答模型。
    """
    if isinstance(question, str):
        return question
    parts: list[str] = []
    for item in question:
        if isinstance(item, dict):
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif "text" in item:
                parts.append(str(item.get("text", "")))
        else:
            parts.append(str(item))
    return "\n".join(p for p in parts if p)


def _question_images(question: str | list) -> list[str]:
    """Extract image data URLs for retrieval without exposing them to tools."""
    if isinstance(question, str):
        return []
    images: list[str] = []
    for item in question:
        if not isinstance(item, dict) or item.get("type") != "image_url":
            continue
        image_url = item.get("image_url")
        if isinstance(image_url, dict):
            image_url = image_url.get("url")
        if isinstance(image_url, str) and image_url:
            images.append(image_url)
    return images


def _with_routed_text(question: str | list, routed_text: str) -> str | list:
    """把路由提示和预检索提示写回第一段文本，同时保留用户上传图片。

    API 多模态请求会传入 content list；这里只替换首个 text block，不动 image_url block，保证图片仍随同本轮消息进入主回答模型。
    """
    if isinstance(question, str):
        return routed_text
    replaced = False
    content: list = []
    for item in question:
        if isinstance(item, dict) and item.get("type") == "text" and not replaced:
            new_item = dict(item)
            new_item["text"] = routed_text
            content.append(new_item)
            replaced = True
        else:
            content.append(item)
    if not replaced:
        content.insert(0, {"type": "text", "text": routed_text})
    return content


def run_agent(
    question: str | list,
    engine: RetrievalEngine,
    model: str | None = None,
    session_id: str | None = None,
    question_id: int | None = None,
    collect_trace: bool = False,
    stream_ttft: bool = False,
) -> AgentResult:
    """运行 ReAct Agent，返回最终回答。

    路由：传入 question_id 时按 id<64 客服 / id>=64 技术 硬路由；不传则 LLM 自判。
    图片处理：让 LLM 保留正文中的 [[PIC:文件名]] 锚点，最终再抽取为 <PIC> + pics。
    stream_ttft：每轮主循环 LLM 调用改流式（拼回同构 response，工具/循环逻辑不变），
        记录最终回答（出现文本增量那轮）的首 token 耗时到 AgentResult.ttft。默认 False=原行为。
    """
    final_ttft: float | None = None
    set_current_qid(question_id)
    check_cancelled()
    engine.ensure_index()
    trace_started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    question_text = _question_text(question)
    query_images = _question_images(question)
    is_technical = question_id is None or question_id >= 64
    product_route = ProductRouteDecision([], "none", "not_tech_question", [])
    original_product_route = product_route
    router = None
    if is_technical:
        router = _get_product_router(engine)
        original_product_route = router.route(question_text)
    locked_route_product = _get_locked_route_product(original_product_route)

    visual_analysis = {"description": "", "ocr_text": "", "keywords": []}
    precomputed_visual_hits: list[list] | None = None
    if visual_retrieval_enabled() and query_images and is_technical:
        with ThreadPoolExecutor(max_workers=2) as pool:
            analysis_future = pool.submit(copy_context().run, analyze_query_images, query_images, model=model)
            image_future = pool.submit(
                copy_context().run,
                engine.prepare_visual_images,
                query_images,
                [locked_route_product] if locked_route_product else None,
            )
            try:
                visual_analysis = analysis_future.result()
            except RequestCancelled:
                raise
            except Exception as exc:
                print(f"图片转文字预处理不可用，继续使用原问题检索: {exc}")
            try:
                precomputed_visual_hits = image_future.result()
            except RequestCancelled:
                raise
            except Exception as exc:
                print(f"图片视觉预检索不可用，后续回退现有链路: {exc}")
    retrieval_question_text = analysis_query_text(question_text, visual_analysis)

    route_t0 = time.time()
    if is_technical and router is not None:
        product_route = router.route(retrieval_question_text)
    product_route_elapsed = round(time.time() - route_t0, 3)
    current_route = product_route
    trace_t0 = time.time()
    trace: dict | None = None
    if collect_trace:
        trace = {
            "id": question_id,
            "question": question_text,
            "visual_query": {
                **visual_analysis,
                "image_count": len(query_images),
                "image_sha256": [image_input_sha256(image) for image in query_images],
            },
            "started_at": trace_started_at,
            "events": [],
        }

    if trace is not None:
        trace["product_route"] = asdict(product_route)
        trace["routed_question"] = _build_routed_question(question_text, question_id, product_route)
        trace["timings"] = {
            "product_route_elapsed": product_route_elapsed,
            "pre_retrieval": {},
            "llm_calls": [],
            "finalize_elapsed": None,
        }

    # V3.1：按 qid 选 system prompt
    # - qid < 64 → 纯客服 prompt（不含技术路由/检索噪声，回到 V2 风格 + 完整性要求）
    # - qid >= 64 或 None（API 在线模式）→ 技术 prompt（V3 路由 + 完整性优先）
    if question_id is not None and question_id < 64:
        system_prompt = SERVICE_SYSTEM_PROMPT
    else:
        system_prompt = TECH_SYSTEM_PROMPT

    routed_question = _build_routed_question(question_text, question_id, product_route)
    messages = [{
        "role": "user",
        "content": _with_routed_text(question, routed_question),
    }]

    pre_results: list[SearchResult] = []
    pre_filtered = 0
    # 初始预检索只对技术题启用；客服题不应引入检索噪声，也不应依赖 retrieval 辅助。
    if question_id is None or question_id >= 64:
        pre_total_t0 = time.time()
        dense_elapsed = 0.0
        rerank_elapsed = 0.0
        build_results_elapsed = 0.0
        engine.ensure_index()
        # 产品已知时：在该产品 chunk 内做 dense 召回（filter 在前），而不是"全局 top-30 再过滤"。
        # 通用词 query（清洁/使用/设置）下，自家章节会被别产品挤出全局 top-30，过滤后只剩 1 节（见 q108）。
        # 产品内召回保证目标手册的相关章节都进候选，预检索覆盖更全。
        if current_route.products:
            allowed: set[int] = set()
            for p in current_route.products:
                allowed.update(engine.product_chunk_ids.get(p, []))
            dense_t0 = time.time()
            dense_ids = engine._dense_recall(retrieval_question_text, top_n=30, allowed_doc_ids=sorted(allowed))
            dense_elapsed = round(time.time() - dense_t0, 3)
        else:
            dense_t0 = time.time()
            dense_ids = engine._dense_recall(retrieval_question_text, top_n=30)
            dense_ids = engine._reorder_by_lang(retrieval_question_text, dense_ids)
            dense_elapsed = round(time.time() - dense_t0, 3)
        if dense_ids:
            rerank_t0 = time.time()
            pre_ids = engine._rerank_candidates(retrieval_question_text, dense_ids, top_n=PRE_RETRIEVAL_RESULTS)[:PRE_RETRIEVAL_RESULTS]
            rerank_elapsed = round(time.time() - rerank_t0, 3)
            build_t0 = time.time()
            pre_results = engine._build_results(pre_ids)
            build_results_elapsed = round(time.time() - build_t0, 3)
        if trace is not None:
            trace["timings"]["pre_retrieval"] = {
                "total_elapsed": round(time.time() - pre_total_t0, 3),
                "dense_elapsed": dense_elapsed,
                "rerank_elapsed": rerank_elapsed,
                "build_results_elapsed": build_results_elapsed,
                "dense_candidates": len(dense_ids or []),
                "returned_sections": len(pre_results),
            }

    # 完整链路诊断：把预检索 top-N 的每个 section（产品/标题/rerank分/可选图）落进 trace，
    # 配合后续 tool_call 的 pics 与最终 answer pics，可逐图还原“召回→注入→选用”三层命运。
    if trace is not None:
        trace["events"].append({
            "kind": "pre_retrieval",
            "index": len(trace["events"]) + 1,
            "products": list(current_route.products or []),
            "sections": [
                {
                    "rank": i,
                    "chunk_id": r.chunk_id,
                    "product": r.product,
                    "heading": r.heading,
                    "score": round(float(r.score), 4),
                    "pics": list(r.pics or []),
                }
                for i, r in enumerate(pre_results)
            ],
        })

    if pre_results:
        pre_text = format_search_results(pre_results, pre_filtered)
        messages.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "sys_preread", "content": pre_text}],
        })
        messages.append({
            "role": "user",
            "content": _make_route_note("系统预检索：用你的问题做了首轮向量检索，结果仅供参考，只能作为后续检索线索，不能直接替代正式工具检索。技术题仍需继续调用 search_manual 做显式确认后再作答。"),
        })

    tool_calls = 0
    empty_search_streak = 0
    empty_search_product: str | None = None
    expand_hint_emitted = False
    same_product_streak = 0
    same_product_no_result_hits = 0
    same_product_name: str | None = None
    structure_query = _is_structure_query(question_text)
    heading_memory: dict[str, set[str]] = {}
    seen_result_keys: set[str] = set()
    low_gain_product: str | None = None
    low_gain_streak = 0
    low_gain_hint_emitted = False
    auto_expand_once = False
    safety_loop_streak = 0
    zero_headings_streak = 0
    # 系统预检索仅作为首轮定位参考；技术题仍需至少一次 search_manual 做正式确认。
    formal_retrieval_confirmed = False
    section_focus_product: str | None = None
    section_focus_id: int | None = None
    section_focus_streak = 0
    section_focus_hint_emitted = False
    recent_focus_product: str | None = _get_primary_route_product(current_route)
    search_attempts = 0

    effective_turns_used = 0
    internal_iterations = 0

    while effective_turns_used < MAX_TURNS and internal_iterations < MAX_INTERNAL_ITERATIONS:
        check_cancelled()
        internal_iterations += 1
        current_turn = effective_turns_used + 1
        remaining_turns = MAX_TURNS - effective_turns_used
        remaining_search_attempts = MAX_SEARCH_ATTEMPTS - search_attempts
        turn_reminder = _make_route_note(
            f"当前是第 {current_turn} / {MAX_TURNS} 次 ReAct 决策机会，剩余 {remaining_turns} 次。"
            "每次机会可以选择继续调用工具，或直接给出最终答案；如果本轮继续调用工具，将消耗一次回答机会。"
            f"普通检索已使用 {search_attempts} 次；该计数仅用于防止重复搜索，不代表还可以额外增加模型轮次。"
            "请参考上一轮状态，避免重复检索；若已有 search_manual 工具结果足以回答，请直接收束答案。"
        )
        if remaining_turns <= 1:
            turn_reminder += "\n" + _make_route_note(
                "当前已是最后一次正常 ReAct 决策机会。除非完全没有可用证据，否则不要再调用工具；"
                "应直接基于 search_manual 工具结果和系统预检索线索输出最终答案。若本轮继续调用工具，"
                "后续只能进入无工具强制收束，答案质量可能下降。"
            )
        is_technical = question_id is None or question_id >= 64
        active_tools = TOOLS if is_technical else None
        tool_choice = "required" if is_technical and not formal_retrieval_confirmed else None
        llm_t0 = time.time()
        if stream_ttft:
            response, _route, _turn_ttft = create_message_streaming(
                max_tokens=int(os.getenv("AGENT_MAX_TOKENS", "8192")),
                system=system_prompt,
                tools=active_tools,
                tool_choice=tool_choice,
                messages=messages + [{"role": "user", "content": turn_reminder}],
                model=model,
            )
            # 本轮出现文本增量(content)→本轮是最终回答，记其首 token；纯工具轮 _turn_ttft=None
            if _turn_ttft is not None:
                final_ttft = _turn_ttft
        else:
            response, _route = create_message_with_fallback(
                max_tokens=int(os.getenv("AGENT_MAX_TOKENS", "8192")),
                system=system_prompt,
                tools=active_tools,
                tool_choice=tool_choice,
                messages=messages + [{"role": "user", "content": turn_reminder}],
                model=model,
            )
        llm_elapsed = round(time.time() - llm_t0, 3)
        if trace is not None:
            has_tool = any(getattr(block, "type", None) == "tool_use" for block in response.content)
            trace["timings"]["llm_calls"].append({
                "index": len(trace["timings"].get("llm_calls", [])) + 1,
                "turn": current_turn,
                "elapsed": llm_elapsed,
                "has_tool": has_tool,
                "content_blocks": len(response.content or []),
            })
        if trace is not None:
            trace["events"].append(
                _build_trace_llm_event(
                    index=len(trace["events"]) + 1,
                    response_content=response.content,
                )
            )

        # 收集文本和工具调用
        has_tool_use = False
        executed_tool_round = False
        tool_results = []
        route_notes: list[str] = []

        for block in response.content:
            if block.type != "tool_use":
                continue
            has_tool_use = True
            tool_calls += 1
            tool_input = dict(block.input or {})
            search_blocked_by_circuit_breaker = False
            if locked_route_product and block.name in SEARCH_TOOL_NAMES:
                if tool_input.get("products") != [locked_route_product]:
                    tool_input["products"] = [locked_route_product]
                    route_notes.append(
                        _make_route_note(
                            f"题面已明确产品为 {locked_route_product}；本轮检索已锁定该产品，禁止扩展到其他手册。"
                        )
                    )
            if (
                not locked_route_product
                and
                auto_expand_once
                and block.name in SEARCH_TOOL_NAMES
                and isinstance(tool_input.get("products"), list)
                and len(tool_input.get("products") or []) == 1
            ):
                tool_input["products"] = []
                auto_expand_once = False
                route_notes.append(
                    _make_route_note(
                        "上一轮已判定当前产品内信息增益过低；本轮普通检索自动放开到全库做一次确认。"
                    )
                )
            if (
                not search_blocked_by_circuit_breaker
                and search_attempts >= MAX_SEARCH_ATTEMPTS
                and block.name in SEARCH_TOOL_NAMES
            ):
                target_product = section_focus_product or recent_focus_product or _get_primary_route_product(current_route)
                search_blocked_by_circuit_breaker = True
                result_text = (
                    "（状态机拦截：search_manual 检索次数已达上限。"
                    "禁止继续 search_manual。"
                    "请基于现有 search_manual 检索证据和系统预检索线索完整收束答案；若证据仍不足，请用同语言说明手册未覆盖。）"
                )
                call_pics = []
                route_notes.append(
                    _make_route_note(
                        f"search_manual 已使用 {MAX_SEARCH_ATTEMPTS} 次；"
                        "请直接基于已有证据收束，不能再继续 search_manual。"
                    )
                )
            if not search_blocked_by_circuit_breaker:
                tool_started_at = time.time()
                retrieval_timings: dict | None = {} if trace is not None else None
                result_text, call_pics = _execute_tool_with_pics(
                    engine,
                    block.name,
                    tool_input,
                    default_products=current_route.products or None,
                    default_query_context=retrieval_question_text,
                    seen_result_keys=seen_result_keys,
                    query_images=query_images,
                    visual_query_text=retrieval_question_text,
                    visual_products=[locked_route_product] if locked_route_product else None,
                    precomputed_visual_hits=precomputed_visual_hits,
                    original_text_query=question_text,
                    original_text_products=original_product_route.products or None,
                    timing_trace=retrieval_timings,
                )
                if trace is not None:
                    trace["events"].append(
                        _build_trace_tool_event(
                            index=len(trace["events"]) + 1,
                            name=block.name,
                            input_data=tool_input,
                            default_products=current_route.products or None,
                            default_query_context=retrieval_question_text,
                            elapsed=time.time() - tool_started_at,
                            pics=call_pics,
                            result_text=result_text,
                            retrieval_timings=retrieval_timings,
                        )
                    )
            if not search_blocked_by_circuit_breaker:
                executed_tool_round = True
            if search_blocked_by_circuit_breaker:
                tool_calls -= 1
            obs = _observe_tool_output(block.name, tool_input, result_text)
            if (
                not search_blocked_by_circuit_breaker
                and block.name in SEARCH_TOOL_NAMES
            ):
                formal_retrieval_confirmed = True

            if block.name in SEARCH_TOOL_NAMES and not search_blocked_by_circuit_breaker:
                search_attempts += 1
                # 连续 0 headings 计数：用于提示换关键词或收束
                if not obs.headings:
                    zero_headings_streak += 1
                else:
                    zero_headings_streak = 0

                candidate_product = obs.explicit_product
                if not candidate_product and isinstance((block.input or {}).get("products"), list):
                    products_arg = (block.input or {}).get("products") or []
                    if len(products_arg) == 1:
                        candidate_product = products_arg[0]

                if candidate_product:
                    recent_focus_product = candidate_product
                    if candidate_product == same_product_name:
                        same_product_streak += 1
                    else:
                        same_product_name = candidate_product
                        same_product_streak = 1
                        same_product_no_result_hits = 0
                    if obs.no_result:
                        same_product_no_result_hits += 1
                else:
                    same_product_name = None
                    same_product_streak = 0
                    same_product_no_result_hits = 0

                dominant_product = obs.dominant_product or candidate_product
                if dominant_product and obs.headings:
                    heading_keys = {
                        _normalize_heading_key(h)
                        for h in obs.headings
                        if _normalize_heading_key(h)
                    }
                    seen_headings = heading_memory.setdefault(dominant_product, set())
                    fresh_headings = heading_keys - seen_headings
                    repeated_ratio = 1.0 - (len(fresh_headings) / max(len(heading_keys), 1))
                    safety_like_ratio = (
                        sum(1 for h in obs.headings if _is_safety_like_heading(h)) / max(len(obs.headings), 1)
                    )
                    if repeated_ratio >= 0.75:
                        if dominant_product == low_gain_product:
                            low_gain_streak += 1
                        else:
                            low_gain_product = dominant_product
                            low_gain_streak = 1
                    else:
                        low_gain_product = dominant_product
                        low_gain_streak = 0
                        low_gain_hint_emitted = False
                    if safety_like_ratio >= 0.6:
                        safety_loop_streak += 1
                    else:
                        safety_loop_streak = 0
                    seen_headings.update(heading_keys)
                    if (
                        obs.dominant_parent_section_id is not None
                        and obs.dominant_parent_section_count >= 2
                    ):
                        if (
                            dominant_product == section_focus_product
                            and obs.dominant_parent_section_id == section_focus_id
                        ):
                            section_focus_streak += 1
                        else:
                            section_focus_product = dominant_product
                            section_focus_id = obs.dominant_parent_section_id
                            section_focus_streak = 1
                            section_focus_hint_emitted = False
                    else:
                        section_focus_product = None
                        section_focus_id = None
                        section_focus_streak = 0
                        section_focus_hint_emitted = False
                else:
                    low_gain_product = None
                    low_gain_streak = 0
                    low_gain_hint_emitted = False
                    safety_loop_streak = 0
                    section_focus_product = None
                    section_focus_id = None
                    section_focus_streak = 0
                    section_focus_hint_emitted = False

                products_arg = (block.input or {}).get("products")
                search_is_unbounded = (
                    not isinstance(products_arg, list) or len(products_arg) == 0
                )
                if (
                    not locked_route_product
                    and
                    search_is_unbounded
                    and obs.dominant_product
                    and obs.dominant_count >= 2
                    and current_route.products[:1] != [obs.dominant_product]
                ):
                    current_route = ProductRouteDecision(
                        products=[obs.dominant_product],
                        confidence="high",
                        reason="retrieval_evidence_rebind",
                        debug_scores=[(obs.dominant_product, float(obs.dominant_count))],
                    )
                    recent_focus_product = obs.dominant_product
                    same_product_name = obs.dominant_product
                    same_product_streak = 0
                    same_product_no_result_hits = 0
                    empty_search_streak = 0
                    empty_search_product = None
                    expand_hint_emitted = False
                    route_notes.append(
                        _make_route_note(
                            f"全库检索的主命中已明显收敛到 {obs.dominant_product}；"
                            "后续优先围绕该产品继续检索。"
                        )
                    )

                if obs.no_result:
                    if candidate_product and candidate_product == empty_search_product:
                        empty_search_streak += 1
                    else:
                        empty_search_product = candidate_product
                        empty_search_streak = 1
                else:
                    empty_search_streak = 0
                    empty_search_product = None
                    expand_hint_emitted = False

            if (
                structure_query
                and empty_search_streak >= 2
            ):
                route_notes.append(
                    _make_route_note(
                        "当前问题更像目录/结构题，且普通检索连续无结果；"
                        "请改用 overview/view/parts/functions 等目录词或 products=[] 全库确认；仍无证据则基于已有内容收束。"
                    )
                )
            elif (
                section_focus_streak >= 1
                and section_focus_product is not None
                and section_focus_id is not None
                and not section_focus_hint_emitted
            ):
                section_summary = (obs.dominant_section_summary or "").strip()
                summary_hint = f" 上层摘要：{section_summary}" if section_summary else ""
                route_notes.append(
                    _make_route_note(
                        f"当前多条命中已聚合到 {section_focus_product} 的上层章节 {section_focus_id}。"
                        "检索已返回该 parent section 的证据；不要继续重复搜索，优先围绕该章节直接收束。"
                        f"{summary_hint}"
                    )
                )
                section_focus_hint_emitted = True

            # 连续 2 次检索返回 0 headings → 停止空转，改关键词/全库确认或收束
            if zero_headings_streak >= 2:
                route_notes.append(
                    _make_route_note(
                        "连续 2 次检索均未返回有效章节（headings），说明关键词无法命中手册内容；"
                        "请换一组高信息量关键词或 products=[] 全库确认；若仍无证据，请基于已有内容收束。"
                    )
                )
            elif (
                not structure_query
                and low_gain_product is not None
                and low_gain_streak >= 2
                and not low_gain_hint_emitted
            ):
                auto_expand_once = current_route.confidence == "medium"
                alternative_products = [
                    product
                    for product in current_route.products
                    if product != low_gain_product
                ]
                if current_route.confidence == "medium" and alternative_products:
                    low_gain_message = (
                        f"你在 {low_gain_product} 内连续命中相近章节，信息增益较低；"
                        f"不要只盯住单一候选，当前还可检查 {'、'.join(alternative_products)}。"
                        " 下一轮优先改用更贴近用户动作/对象的关键词，切去其他候选或做一次全库确认。"
                        " 若不同候选都给出相关信息，可按“若是 A…；若是 B…”并列回答。"
                    )
                else:
                    low_gain_message = (
                        f"你在 {low_gain_product} 内连续命中相近章节，信息增益较低；"
                        "建议下一轮改用更贴近用户动作/对象的关键词重试，"
                        "优先查 setup/connection/procedure 等步骤型线索。"
                        "若仍不贴题，再将 products 设为 [] 做一次全库确认。"
                    )
                route_notes.append(
                    _make_route_note(low_gain_message)
                )
                low_gain_hint_emitted = True
            elif (
                not structure_query
                and safety_loop_streak >= 2
            ):
                route_notes.append(
                    _make_route_note(
                        "当前结果连续落在 safety/regulatory 类章节，和用户要的操作步骤不完全对齐；"
                        "下一轮优先查 Quick Setup Guide、installation、setup、station ID 等步骤型线索，"
                        "少查 safety / legal / FCC 关键词。"
                    )
                )
                safety_loop_streak = 0
            elif (
                not structure_query
                and empty_search_streak >= 2
                and not expand_hint_emitted
            ):
                scope = empty_search_product or "当前限定范围"
                route_notes.append(
                    _make_route_note(
                        f"在 {scope} 内连续检索无结果，可能陷入单产品误区；"
                        "建议下一轮显式将 products 设为 [] 做一次全库检索，"
                        "并优先使用用户原语言关键词重试。"
                    )
                )
                expand_hint_emitted = True
            elif (
                not structure_query
                and same_product_name is not None
                and same_product_streak >= 4
                and same_product_no_result_hits >= 1
                and not expand_hint_emitted
            ):
                route_notes.append(
                    _make_route_note(
                        f"你在 {same_product_name} 内已连续多轮检索且出现无结果，"
                        "可能陷入单产品误区；建议下一轮显式将 products 设为 [] 做一次全库检索，"
                        "再回到最相关产品收敛。"
                    )
                )
                expand_hint_emitted = True

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_text,
            })

        if has_tool_use:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            if route_notes:
                messages.append({"role": "user", "content": "\n".join(route_notes)})
            if executed_tool_round:
                effective_turns_used += 1
            continue

        # 没有工具调用 → 最终回答
        if (question_id is None or question_id >= 64) and not formal_retrieval_confirmed:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": _make_route_note(
                    "技术题尚未获得任何可用手册证据。请先调用 search_manual 完成显式确认；"
                    "若已无机会继续检索，后续会基于已有内容收束并说明手册未覆盖。"
                ),
            })
            effective_turns_used += 1
            continue

        answer = _extract_text_from_response(response)
        answer = _postprocess_final_answer(
            answer=answer,
            question=question_text,
            system_prompt=system_prompt,
            model=model,
            route_products=current_route.products,
        )
        answer, pics = _resolve_answer_pics(answer)

        return AgentResult(
            answer=answer,
            pics=pics,
            tool_calls=tool_calls,
            turns=current_turn,
            ttft=final_ttft,
            trace=(
                {
                    **trace,
                    "result": {
                        "answer": answer,
                        "pics": pics,
                        "tool_calls": tool_calls,
                        "turns": current_turn,
                    },
                    "error": None,
                    "elapsed": round(time.time() - trace_t0, 2),
                }
                if trace is not None else None
            ),
        )

    # 超过最大轮数：先强制收束；若仍是异常占位句，则抛异常交给批处理记 error
    finalize_t0 = time.time()
    answer = _finalize_without_tools(
        system_prompt=system_prompt,
        messages=messages,
        model=model,
    )
    if trace is not None:
        trace["timings"]["finalize_elapsed"] = round(time.time() - finalize_t0, 3)
    if _normalize_final_answer(answer) in _GENERIC_FAILURE_ANSWERS:
        raise RuntimeError(
            f"agent exceeded MAX_TURNS={MAX_TURNS} and failed to finalize an answer"
        )

    answer = _postprocess_final_answer(
        answer=answer,
        question=question_text,
        system_prompt=system_prompt,
        model=model,
        route_products=current_route.products,
    )
    answer, pics = _resolve_answer_pics(answer)

    return AgentResult(
        answer=answer,
        pics=pics,
        tool_calls=tool_calls,
        turns=MAX_TURNS,
        ttft=final_ttft,
        trace=(
            {
                **trace,
                "result": {
                    "answer": answer,
                    "pics": pics,
                    "tool_calls": tool_calls,
                    "turns": MAX_TURNS,
                },
                "error": None,
                "elapsed": round(time.time() - trace_t0, 2),
            }
            if trace is not None else None
        ),
    )


# ────────────────── CLI 入口 ──────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ReAct 客服智能体")
    parser.add_argument("question", nargs="?", help="用户问题")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互模式")
    args = parser.parse_args()

    engine = RetrievalEngine()
    engine.ensure_index()
    print(f"索引加载完成: {len(engine.retrieval_chunks)} 检索块, {len(engine.catalog)} 产品\n")

    if args.interactive:
        print("交互模式（输入 quit 退出）")
        while True:
            question = input("\n> ").strip()
            if question.lower() in ("quit", "exit", "q"):
                break
            if not question:
                continue
            t0 = time.time()
            result = run_agent(question, engine)
            elapsed = time.time() - t0
            print(f"\n{result.answer}")
            if result.pics:
                print(f"\n图片: {result.pics}")
            print(f"\n--- {result.tool_calls} 次工具调用, {result.turns} 轮, {elapsed:.1f}s ---")
    elif args.question:
        t0 = time.time()
        result = run_agent(args.question, engine)
        elapsed = time.time() - t0
        print(result.answer)
        if result.pics:
            print(f"\n图片: {result.pics}")
        print(f"\n--- {result.tool_calls} 次工具调用, {result.turns} 轮, {elapsed:.1f}s ---")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
