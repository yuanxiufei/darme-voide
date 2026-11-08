"""外部剧本/编剧知识库 —— **本地只读检索层**（2026-11-04 接入）。

两根目录都是第三方资料、**都不入库**（见 `.gitignore` 的「第三方剧本/编剧知识库」一节）：

* ``METHODS``：``剧本知识库/全球热门与获奖影视剧本知识库/`` —— 5403 部（pdf 5160 / docx 163 /
  txt 77 / md 3）＋ 供应商**预建的 SQLite 全文索引**
  ``99-索引与检索规则/curated-screenplay-methods.sqlite``（``chunks`` 是 fts5 **trigram** 表，
  128166 个片段）。本模块**只读**它：不重建、不写入、不复制 —— 重建会让「索引里 5403 部」
  与「盘上到底几部」变成两本账。
* ``PLAYBOOK``：``corpus/playbook/`` —— 实测 **137 份 md / 2300+ 小节**，按来源目录分成两棵
  （``film-expert-bridge/``、``film-cinedance/``，白名单见 :data:`PLAYBOOK_DIRS`；**目录名即
  ``skill=`` 取值**）。里面**没有** ``SKILL.md``，本仓不加载它、只检索 —— 别与 ``app/skills/``
  混淆，那才是能注入的技能。
  **本仓自带、随仓库分发**（2026-11-04 从 gitignore 的 ``方法语料/`` 移入）：知识库页面与
  ``search_script_library`` 开箱可用，不再依赖作者机器上的外部安装。来源/版本/许可与
  前后端用法写在 ``corpus/playbook/README.md``（根目录直属 md 是元数据，不进索引、不计未索引数）。
  内容按用途分两半：**判据**已消化进本仓 ``app/skills/*/references/``（表演、镜头、审片、
  真人质感、声音、打戏六份，可直接出稿）；**原料**（招式库、造型库、特效库、分镜规范、
  情绪实例、技法资料库）留在原处供 ``search_script_library`` 检索。
  2026-11-04 创作者「有用的抄没用的删」，清掉 94 份课程机制件 —— 安装脚本／交互菜单
  （``catalog/*/选项.md``）／专家路由与项目模板（``ai-film-router`` 等 8 个 skill）／
  飞书下载入口／他人机器上的目录路径（不复述）／已抄完的源文件／课程文档三份，
  267 文件 4.6 MiB → 137 文件 1.48 MiB。按 ``#`` 小节切块、超过
  :data:`PLAYBOOK_SECTION_MAX` 再切窗，然后在**进程内**建命中表（毫秒级）。

三处「不装样子」的口径（与 ``agent/tools/corpus_tools.py`` 同源纪律）：

1. **旧盘路径必须重映射**：索引里的 ``path`` 是建库机器上的旧盘符绝对路径（形如 ``…\\电影剧本\\…``）⇒ 一律按
   :data:`LOCAL_MARKER` 截尾拼回本仓 :data:`LIBRARY_ROOT`，并**逐条**回报 ``onDisk``。
   搬过盘的机器上 ``onDisk=false`` 是明话，不许当成「找到了一条参考」。
2. **隔离目录不参与检索**：``90-隔离与待审/`` 是供应商的待审桶（**当前库内 0 条**，但那是数据
   现状、不是契约）⇒ 命中一律剔除，而不是指望它一直空着。
3. **trigram 的下限是 3 个字**：fts5 的 trigram 分词器不为 1–2 字建索引 ⇒「雨夜」这类 2 字词走
   ``MATCH`` **恒空**（不是"没有相关内容"）。故词长 < :data:`MIN_GRAM` 一律回落 ``LIKE``
   （128k 片段全扫实测 ≈0.3 s），并在回执里写明 ``ranking`` —— 两种排序**不可比**。

⚠️ 本模块属 **services 层**（只依赖 ``core``）：不许 import ``app.agent.*``。Agent 工具在
``app/agent/tools/library_tools.py``、命令行在 ``app/scripts/corpus/library.py``，**三者共用本模块**
（单一真源，不许各写一份打分口径）。
"""
from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from typing import Any

from app.core.config import PROJECT_ROOT
from app.services import library_overlay, script_catalog

__all__ = [
    "EXCERPT_CHARS",
    "EXCLUDED_PARTS",
    "LIBRARY_ROOT",
    "LOCAL_MARKER",
    "METHODS_DB",
    "MIN_GRAM",
    "PLAYBOOK_DIRS",
    "PLAYBOOK_NAV_DOCS",
    "PLAYBOOK_NAV_INTENT",
    "PLAYBOOK_NAV_WEIGHT",
    "EXTRA_DB",
    "PLAYBOOK_ROOT",
    "PLAYBOOK_SHELLS",
    "library_report",
    "library_tree",
    "list_documents",
    "open_document",
    "playbook_skills",
    "query_terms",
    "resolve_local",
    "search",
    "search_methods",
    "search_playbook",
]

#: 知识库根（第三方资料；``SCRIPT_LIBRARY_ROOT`` 可指到别处，测试夹具也用它）
LIBRARY_ROOT = Path(os.environ.get("SCRIPT_LIBRARY_ROOT")
                    or (PROJECT_ROOT / "剧本知识库" / "全球热门与获奖影视剧本知识库"))
#: 方法语料根（``SCRIPT_PLAYBOOK_ROOT`` 可把根指向外部副本）。到此为止搬了五次：
#: ``参考/`` → ``方法语料/AI STUDIO洋子-V3.3.1-热修版/`` → ``data/method-library/…``
#: → ``data/playbook/`` → **``corpus/playbook/``**。前三处的毛病各不一样：前两处被
#: gitignore（版权存疑、只读参考），第三处虽进了仓，名字却带空格 / 大写 / 中文版本号 ——
#: 与仓库其余目录（一律小写英文）不一致，路径里那句 ``AI STUDIO洋子…`` 每引用一次都要抄一遍。
#: 实际后果是**新克隆的仓库里 playbook 源为空** ⇒ 知识库页面与 ``search_script_library``
#: 一起降级成 ``available=false``，等于「资料在作者机器上、系统里没有」。2026-11-04 创作者
#: 裁定：有用的资料必须进系统（前后端 / data / 知识库任一），故落到此处随仓库分发。
#:
#: ⚠️ 2026-11-07 第五次搬家（``data/playbook/`` ⇒ ``corpus/playbook/``）：``data/`` 是
#: **运行时数据根**（能被 ``.data-root`` / ``DATA_ROOT`` 切到别的盘），而语料是本仓随仓库
#: 分发的**源资产** —— 既不随数据根走，也不该在「备份我的作品数据」时被一起卷走。两者同居
#: ``data/`` 会让「备份 data/」的含义变得不明确，故语料独立成顶层 ``corpus/``。
#: 来源、版本、许可与前后端用法写在 ``corpus/playbook/README.md``，不靠目录名携带版本。
PLAYBOOK_ROOT = Path(os.environ.get("SCRIPT_PLAYBOOK_ROOT")
                     or (PROJECT_ROOT / "corpus" / "playbook"))
#: 供应商预建的 fts5 索引（**唯一真源**）
METHODS_DB = LIBRARY_ROOT / "99-索引与检索规则" / "curated-screenplay-methods.sqlite"
#: 本仓自建的 **OCR sidecar**：主库里 3139 份扫描件没有正文块（`withText 2262/5403`），
#: 由 ``app/scripts/corpus/ocr_index.py`` 逐页 OCR 补进来。**不写供应商那份**（只读），
#: 也不改它的 schema —— 两库同构，检索时按秩合并，``from`` 字段标明这条出自哪一库。
#: 落点 ``indexes/``（2026-11-07 从 ``data/ocr/`` 迁出）：这两只与 ``library-scenes`` 都是
#: **可重建的派生索引**，与 ``data/``（不可重建、必须备份）分开 —— 备份作品数据不必带上
#: 这 2 GB；但重建要跑数小时，所以也别当「缓存」随手删（见 ``indexes/`` 目录名而非 cache）。
OCR_DB = Path(os.environ.get("SCRIPT_LIBRARY_OCR_DB")
              or (PROJECT_ROOT / "indexes" / "library-ocr.sqlite"))
#: 本仓第三个库（``app/scripts/corpus/lib_index.py``）：补**主库压根没登记**的格式。
#: OCR sidecar 只认「主库登记了但没正文」的扫描件，而 ``.doc``（Word 97-2003，CSV 里
#: 873 份标「格式暂不支持全文索引」）主库里连行都没有 ⇒ 走不到那条线。
#: 同构 schema、同一条纪律：供应商那份只读，补出来的东西只进本仓文件。
EXTRA_DB = Path(os.environ.get("SCRIPT_LIBRARY_EXTRA_DB")
                or (PROJECT_ROOT / "indexes" / "library-extra.sqlite"))

#: 库里路径的锚点：它**右边**就是相对本仓 ``LIBRARY_ROOT`` 的路径
LOCAL_MARKER = "全球热门与获奖影视剧本知识库"
#: 供应商的待审桶（目录名出现在路径里即剔除）
EXCLUDED_PARTS = ("90-隔离与待审",)
#: fts5 trigram 的最小可检索长度（1–2 字没有索引 ⇒ 必须回退 LIKE）
MIN_GRAM = 3
#: sidecar 里文档的状态字面量，与 ``app/scripts/corpus/ocr_index.py`` 的 ``STATUS_*`` **逐字同源**。
#: 这里只抄字面、不 import 那个脚本 —— services 层不许依赖 scripts 层（分层纪律）。三档语义：
#:
#: * :data:`OCR_STATUS_DONE` 整本跑完，块可信；
#: * :data:`OCR_STATUS_PARTIAL` 中途停了（进程被杀 / ``--pages=N`` 试跑）：**块已经写进
#:   fts5 了**，但只覆盖前几页 ⇒ 默认不进检索，否则一份 920 页的书会以「全书可查」的样子出现；
#: * :data:`OCR_STATUS_QUARANTINE` 跑完了但质量门禁判废，块一个没写、等人工 ``--redo``。
OCR_STATUS_DONE = "ocr"
OCR_STATUS_PARTIAL = "partial"
OCR_STATUS_QUARANTINE = "quarantine"
#: 片段的字符上限（与 prompt 语料同一口径：参考是给模型看的，过长会挤占上下文）
EXCERPT_CHARS = 400
#: 参考文本的**索引白名单**：只有这些子树算「方法正文」。**每一项就是 ``skill=`` 的取值**
#: （:func:`playbook_skills` 可列出全部可用名 —— 不让调用方瞎猜 ``film-acting`` 这种词）。
#: 2026-11-04 扁平化：原结构是单壳 ``("packs",)`` ＋ ``packs/<包名>/references/library|modules``，
#: 最深一条文件路径有 **6 层目录**（``packs/film-expert-bridge/references/library/combat/战场场景库/x.md``）。
#: 现在**包名直接在一级**，``references/`` 与 ``library/`` 两个零信息壳已删 —— 前者还与
#: ``app/skills/*/references/``（判据层，会被直接引用）**同名不同义**，是全链路最易误导人的一层。
#: 写**显式名单**而不是「扫根下所有目录」：后者省事，但会把以后丢进根目录的别家资料一起卷进索引。
#: 新增一棵包必须在此登记；漏登记的 md 不会静默消失 —— :func:`_playbook_report` 的
#: ``unindexedFiles`` 会如实报「盘上有、没进索引」。
PLAYBOOK_DIRS = ("film-expert-bridge", "film-cinedance")
#: 外壳名，**只用于归一 ``skill=`` 入参，不是盘上目录**。``packs`` 是扁平化前那层壳，``skills``
#: 是更早的名字；调用方手里的提示词、旧会话记录、上游文档都还在用这两种形状，按新结构拒收
#: 等于把一次纯目录调整变成对外破坏。
#: ⚠️ 不要把它并回 :data:`PLAYBOOK_DIRS` 当「旧名 ＋ 新名」用：``PLAYBOOK_DIRS`` 现在装的
#: **就是技能名**，:func:`_skill_arg` 若拿它当前缀表，``film-expert-bridge/x`` 会被当成
#: 「壳/技能名」剥掉而返回 ``x`` —— 过滤全失效且不报错。
PLAYBOOK_SHELLS = ("packs", "skills")

#: 根目录直属、允许存在且**不算「漏索引」**的元数据文件名。写成点名名单而不是"排除根目录的
#: 所有 md"：后者一行代码更省事，但代价是根目录真漏放一份方法正文时永远不会报出来 ——
#: 静默缩小范围的守卫比没有守卫更危险。
PLAYBOOK_ROOT_FILES = ("README.md",)
#: 目录白名单挡不住同名文件：包目录里可能有 ``AGENTS.md``（素材包自己的使用规约），
#: 与根目录规约同性质 —— 按**文件名**再收一刀。当前盘上 0 份（2026-11-04 清掉
#: ``ai-film-router/references/project-rules/`` 那份之后）。``SKILL.md`` 不动，那才是方法正文。
PLAYBOOK_SKIP_NAMES = ("AGENTS.md",)
#: 「导航型」文档：正文主体是「关键词 → 文件」的表，自己**不含方法内容**。它们命中时按
#: :data:`PLAYBOOK_NAV_WEIGHT` 打折 —— **不排除**（用户想找目录时它得在结果里），但绝不能因
#: 为「一份索引把整个族的名词抄了一遍」就排到真资料前面。
#:
#: 实测症状（2026-11-06，`general/通用资料索引.md` 加入后）：搜「造型」第一条是
#: 「4. 角色造型（7 份）」、搜「四宫格」第一条是「12. 两个有名字没文件的术语」—— 点开只有表，
#: 一句方法都读不到，却和真资料同分并列。加索引本是为了让人**找得到**，结果先把检索顶掉了。
#:
#: 写成点名名单、不写成「文件名含『索引』」的规则：后者会误伤名字里带「索引」的真资料，
#: 也会让以后新增的索引文档**静默漏网**。漏网要在测试里报出来（见 `script_library_test.py`
#: 的「导航型名单」两条），所以名单必须是显式的。
PLAYBOOK_NAV_DOCS = ("通用资料索引", "打戏系统资料索引", "00-技能库索引")
#: 导航型文档的分数折扣。取 0.3 的算法：真资料单词命中通常 6 分上下（标题 5 ＋ 正文 1~6），
#: 索引靠「把族里所有名词抄一遍」能到 10~11 分 ⇒ 乘 0.3 落到 3~3.3，稳定排在真资料之后。
PLAYBOOK_NAV_WEIGHT = 0.3
#: 用户**明确在找目录**时（查询里带这些词）不给折扣 —— 否则「打戏索引」这种意图会被真资料顶掉。
PLAYBOOK_NAV_INTENT = ("索引", "目录", "index")
#: 小节正文超过这个长度就切窗（实测上千节里超长的只是少数几节）。不切的话，
#: 6000 字里任意一处命中都会让整节排前面，而返回的 excerpt 可能离检索词半页远。
PLAYBOOK_SECTION_MAX = 1500
#: 切窗宽度（按句界对齐拼窗；单句本身就超长才硬切）
PLAYBOOK_WINDOW = 1200
#: 相邻窗的重叠比例（百分比）。口径抄 RAGFlow ``chunker/token.go::mergeUnits``
#: （``threshold = target*(100-overlap)/100`` 预留 overlap 位、新窗无条件前置前窗尾巴）：
#: 不重叠时跨窗前后的那半句只剩一半，模型看到「而这一段的重点在于」这种残句就会误读。
PLAYBOOK_WINDOW_OVERLAP = 15
#: 跨源融合：RRF（reciprocal rank fusion），口径抄 txtai ``embeddings/search/hybrid.py::rrf``
#: —— ``score = Σ 1/(k + rank)``。两源分数量纲不可比（methods 是 bm25 负值、playbook 是词频
#: 计数、LIKE 甚至没有分数），只有**秩**可比。``k=1`` 与 txtai 一致；标准文献的 k=60 会把两源
#: 差异压到小数点后三位、等于没融合。
RRF_K = 1
#: 各源融合权重。**一律等权**：没有评测集就调权重等于拍脑袋；要偏置得先有「谁的答案更常被
#: 采纳」的数据。留这张表是为了让偏置**有地方可调**且一眼看见「当前是等权」。
SOURCE_WEIGHTS: dict[str, float] = {"methods": 1.0, "playbook": 1.0,
                                    #: 覆盖层里**改过的正文**（第四路）。权重与另两源相同：
                                    #: 它是「用户亲手改过的正文」，不该因为出自覆盖层而打折。
                                    "edits": 1.0}

_TERM_SPLIT = re.compile(r"[\s,，、;；|/]+")
_WHITESPACE = re.compile(r"\s+")
_HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*$", re.MULTILINE)

_NO_DB_HINT = (
    "剧本知识库未安装（或路径不对）：把「剧本知识库/全球热门与获奖影视剧本知识库/」放在仓库根，"
    "或用环境变量 SCRIPT_LIBRARY_ROOT 指向它；该目录是第三方资料、已 gitignore，不入库。"
)
_NO_PLAYBOOK_HINT = (
    "参考方法文本缺失（或路径不对）：本仓自带「corpus/playbook/」（随仓库分发），"
    "缺失时先确认该目录是否被清掉；"
    "或用环境变量 SCRIPT_PLAYBOOK_ROOT 指向外部副本。"
)
_RESULT_NOTE = (
    "两类来源分数不可比：methods=bm25（已取负号、大者更相关），playbook=标题/正文命中加权。"
    "参考只给写法与结构，不要搬运原文句子进剧本或提示词（合规红线）。"
)


# ===========================================================================
# 路径：旧盘 → 本仓
# ===========================================================================


def resolve_local(raw_path: str, root: Path | None = None) -> Path | None:
    """库里那条建库机器的旧盘符路径（形如 ``…\\电影剧本\\…``）→ 本仓路径；无锚点返回 ``None``（**不猜**）。"""
    text = (raw_path or "").replace("/", "\\")
    index = text.find(LOCAL_MARKER)
    if index < 0:
        return None
    tail = text[index + len(LOCAL_MARKER):].lstrip("\\")
    if not tail:
        return None
    return (root or LIBRARY_ROOT) / tail.replace("\\", "/")


def _on_disk(raw_path: str) -> tuple[bool, str]:
    """``(onDisk, relativePath)``：相对路径统一正斜杠（跨平台可比对）。

    ``library-extra.sqlite``（含本仓上传件）里存的是**库根下的相对路径**
    （``00-本仓补录/x.txt``），没有旧盘符那段 ``…\\知识库`` 锚点 ⇒ :func:`resolve_local`
    按设计返回 ``None``（不猜）。这里补一条明确可判的旁路：**本来就是相对路径**（无盘符、
    无 UNC 前缀、不以 ``/`` 开头）就当它是库根下的路径。不补这条，上传件的回执会一路
    自相矛盾 —— 清单里 ``onDisk=false``、``hardDeletable=false``、删不动也打不开。
    """
    local = resolve_local(raw_path)
    if local is None:
        text = (raw_path or "").strip()
        if not text or text.startswith("\\\\") or text.startswith("/") or ":" in text[:3]:
            return False, ""
        local = Path(LIBRARY_ROOT) / text.replace("\\", "/")
    root = Path(LIBRARY_ROOT)
    try:
        rel = local.relative_to(root).as_posix()
    except ValueError:  # 夹具把 root 换到别处时不炸，直接给绝对路径
        rel = local.as_posix()
    return local.exists(), rel


# ===========================================================================
# 检索词
# ===========================================================================


def query_terms(query: str) -> list[str]:
    """按空白/常见分隔符切词（中文不切：整串当一个词，trigram 与 LIKE 都吃得下）。

    顺手清掉 BOM / 零宽字符：PowerShell ``Set-Content -Encoding UTF8`` 写 ``--q-file``
    时会带 ``\\ufeff``，不清掉会让第一个词变成「\\ufeff微表情」⇒ 明明有内容却零命中。
    """
    cleaned = (query or "").strip().strip("\ufeff\u200b\u200c\u200d")
    return [term for term in _TERM_SPLIT.split(cleaned) if term]


def _flatten(text: str) -> str:
    return _WHITESPACE.sub(" ", text or "").strip()


def _window(text: str, term: str, chars: int = 200) -> str:
    """LIKE 路线没有 ``snippet()`` ⇒ 自己截命中窗口并标出命中词。"""
    flat = _flatten(text)
    position = flat.find(term)
    if position < 0:
        return flat[:chars]
    start = max(0, position - chars // 3)
    window = flat[start:start + chars].replace(term, f"【{term}】", 1)
    return ("…" if start > 0 else "") + window + ("…" if start + chars < len(flat) else "")


def _best_window(text: str, terms: list[str], chars: int = 200) -> str:
    """挑一个「覆盖检索词最多」的窗口，并把窗内命中的词都标出来。

    与 :func:`_window` 的区别：那个只认第一个词出现的位置，多词查询（「三幕 激励事件」）
    会把 excerpt 指到跟第二个词无关的地方；这里逐词算位置，取覆盖词数最多的窗。
    """
    flat = _flatten(text)
    if not flat:
        return ""
    hits: list[int] = []
    for term in terms:
        position = flat.find(term)
        while position >= 0 and len(hits) < 40:
            hits.append(position)
            position = flat.find(term, position + len(term))
    if not hits:
        return flat[:chars]
    hits.sort()

    def covered(start: int) -> tuple[int, int]:
        #: 同覆盖数时取**更靠前**的窗（读者从前往后找更顺 ⇒ 用 -start 让 max 选小 start）
        return (sum(1 for term in terms if term in flat[start:start + chars]), -start)

    best = max({max(0, position - chars // 3) for position in hits}, key=covered)
    window = flat[best:best + chars]
    for term in sorted(terms, key=len, reverse=True):
        if term in window:
            window = window.replace(term, f"【{term}】", 1)
    return ("…" if best > 0 else "") + window + ("…" if best + chars < len(flat) else "")


def _like(term: str) -> str:
    """转义 ``%``/``_`` ⇒ 检索词里的这些字符是**字面量**（配 ``ESCAPE '\\'``）。"""
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _connect(db: Path) -> sqlite3.Connection | None:
    if not db.exists():
        return None
    try:
        #: 只读打开：索引是第三方只读资产，任何写都说明我们走错了路
        return sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    except sqlite3.Error:
        return None


def _path_filters(path_contains: str | None) -> tuple[list[str], list[str]]:
    """``path`` 的过滤条件（隔离目录 ＋ 调用方限定），返回 ``(sql 片段, 参数)``。

    **这里必须用 ``instr``，不能用 ``LIKE``** —— 这三处 SQL 都打在 fts5 虚拟表
    （``chunks``）上，而 ``path`` 是 ``UNINDEXED`` 列。fts5 的 LIKE 优化在这块会塌，
    实测（三库 269 万块）：``path LIKE '%doc%'`` = **0 条**，而 ``instr(path,'doc')>0``
    = 30923 条；同一个库上 ``path LIKE '%Chinatown%'`` 又是 184 条、与 ``instr`` 一致 ——
    塌掉的是**长度恰为 3 的纯 ASCII** pattern（正好卡在 trigram 的最小 token 长度上），
    ``'%电影%'``（2 字）不走该优化所以也是对的。症状因此极其隐蔽：筛「Chinatown」正常、
    筛「BBC」「S02」「doc」静默返回空，看不出是过滤写错了。``instr`` 是字面子串，
    不吃通配符、也不进 tokenizer，两张表上都对。
    """
    clauses: list[str] = []
    params: list[str] = []
    for part in EXCLUDED_PARTS:
        clauses.append("instr(path, ?) = 0")
        params.append(part)
    if path_contains:
        clauses.append("instr(path, ?) > 0")
        params.append(path_contains)
    return clauses, params


# ===========================================================================
# 来源一：剧本/编剧方法库（SQLite fts5）
# ===========================================================================


def _filter_allowed(filters: dict[str, Any] | None) -> tuple[set[str] | None, dict[str, Any] | None]:
    """目录过滤 → ``(允许的相对路径集合, 回执)``。

    三态必须分清（这是本函数存在的全部理由）：

    * ``(None, None)`` —— 调用方没给条件，**不做任何过滤**；
    * ``(set(), {...applied: True})`` —— 条件给了、目录也在，但**没有作品满足** ⇒ 真结论，如实说；
    * ``(None, {...applied: False})`` —— 条件给了、**目录 CSV 不在** ⇒ 过滤**没有生效**。
      绝不能退化成空集：那会让「目录没装」看起来像「库里没有这部」，是本仓最忌的谎。
    """
    if not filters:
        return None, None
    criteria = {key: value for key, value in filters.items()
                if value not in (None, "", [], (), {})}
    if not criteria:
        return None, None
    allowed = script_catalog.select(LIBRARY_ROOT, marker=LOCAL_MARKER, **criteria)
    if allowed is None:
        return None, {
            "applied": False,
            "criteria": criteria,
            "hint": ("目录 CSV 未安装 ⇒ 这些过滤条件**没有生效**，"
                     "返回结果未按国家/类型/奖项/权限筛过。"),
        }
    return allowed, {"applied": True, "criteria": criteria, "allowed": len(allowed)}


def search_methods(query: str, *, limit: int = 5, path_contains: str | None = None,
                   filters: dict[str, Any] | None = None,
                   include_partial: bool = False,
                   include_buried: bool = False) -> dict[str, Any]:
    """在 fts5 索引上检索：≥3 字走 ``MATCH + bm25``，否则回落 ``LIKE``。

    **查两块库**：供应商的 ``curated-*``（2262 份有正文）与本仓的 OCR sidecar
    （``OCR_DB``，补扫描件）。两库同构但语料不同 ⇒ 各自取窗口后**按秩融合**，
    每条带 ``from: curated|ocr``；两库合计命中数在 ``matched``，分列在 ``parts``。

    **未跑完的文档默认不进检索**：sidecar 里 ``status='partial'``（只扫了前几页）与
    ``'quarantine'``（质量门禁判废）两类文档的块**已经躺在 fts5 里**（``_save`` 只对
    quarantine 不写块，partial 照样写）。库里实测 80 个块有 49 个来自 partial，其中一份
    920 页的书只扫了 2 页 —— 不排除的话，那 2 页会以「全书可查」的样子出现在结果里，
    模型引用时无从判断覆盖率。真要看半成品给 ``include_partial=True``，此时每条结果带
    ``coverage={pages, chunkPages}``（总页数 / 真有正文的页数），把「2/920」明说出来。

    ``filters`` 走目录层（国家/一级分类/二级分类/资料类型/权限/电影节）。**过滤发生在
    SQL 取样之后**：目录不在 sqlite 里（``documents`` 表没有这些列）⇒ 只能先按 bm25 取一个
    **窗口**（``limit*40``，上限 600）再在内存里筛。窗口不够时会明说 ``exhausted``，
    而不是悄悄少给几条。
    """
    terms = query_terms(query)
    if not terms:
        return {"available": True, "results": [], "hint": "检索词为空。",
                "matched": None, "ranking": None}

    allowed, filter_info = _filter_allowed(filters)
    if filter_info and filter_info["applied"] and not allowed:
        return {"available": True, "results": [], "matched": 0, "ranking": None,
                "hint": "目录里没有同时满足这些条件的作品（是筛选条件太窄，不是检索没命中）。",
                "filters": filter_info}

    clauses, params = _path_filters(path_contains)
    if not include_partial:
        #: ``NOT IN`` 而不是 ``JOIN``：fts5 表 JOIN 普通表会让 bm25 排序失去索引优势，
        #: 而这里只需要排除「明确标了未完成」的那几十份（子查询结果集极小）。
        #: 用 NOT IN 还有个好处：``document_id`` 在主库有孤儿时**不会**被误杀。
        clauses.append("chunks.document_id NOT IN "
                       "(SELECT id FROM documents WHERE status IN (?, ?))")
        params.extend((OCR_STATUS_PARTIAL, OCR_STATUS_QUARANTINE))
    where = " AND ".join(["1=1", *clauses])
    use_fts = all(len(term) >= MIN_GRAM for term in terms)

    #: 已删进回收站的**不该被搜到** —— 这是「删除」在检索上的唯一落点，
    #: 在 SQL 里加 ``path NOT IN`` 会让 fts5 的 bm25 失去索引优势（``path`` 是 UNINDEXED 列），
    #: 所以改成取窗口后在内存里剔。``hidden`` 如实记剔掉了几条，别让 ``matched`` 虚高。
    buried = {} if include_buried else library_overlay.buried_map()
    hidden = 0

    parts: list[tuple[str, dict[str, Any]]] = []
    for label, db in (("curated", Path(METHODS_DB)), ("ocr", Path(OCR_DB)),
                      ("extra", Path(EXTRA_DB))):
        conn = _connect(db)
        if conn is None:
            continue
        try:
            if use_fts:
                outcome = _methods_fts(conn, terms, where=where, params=params, limit=limit,
                                       allowed=allowed)
            else:
                outcome = _methods_like(conn, terms, where=where, params=params, limit=limit,
                                        allowed=allowed)
            if include_partial:
                _fill_partial_coverage(conn, outcome.get("results") or [])
        except sqlite3.Error as exc:  # noqa: BLE001 —— 索引坏了要说出来，不许吞成「没找到」
            outcome = {"available": True, "results": [], "matched": None, "ranking": None,
                       "hint": f"{label} 索引读取失败：{exc}"}
        finally:
            conn.close()
        if not outcome.get("available"):
            continue
        rows = outcome.get("results") or []
        gone_paths = buried.get(label, ())
        if gone_paths:
            kept = [row for row in rows if row.get("path") not in gone_paths]
            hidden += len(rows) - len(kept)
            rows = kept
            outcome["results"] = rows
        for row in rows:
            row["from"] = label
        parts.append((label, outcome))

    #: 第四路：**改过的正文**。不并进来的话「改完就搜不到」—— 用户只会认为保存失败了。
    #: 它在覆盖层自己的 fts5 上命中，所以与三个原库同构、可进同一个 RRF 融合。
    if not include_buried:
        edited_hits = library_overlay.search_edits(terms, limit=limit)
        rows = edited_hits.get("results") or []
        if rows:
            #: 改过的块也要跟着「删除」一起消失：它记的是 ``editSource`` 那个库，
            #: 而墓碑记在原库名下 ⇒ 三个集合并起来判。
            all_gone = set().union(*buried.values()) if buried else set()
            if all_gone:
                rows = [row for row in rows if row.get("path") not in all_gone]
                edited_hits["results"] = rows
            for row in rows:
                row["from"] = "edits"
        if rows:
            parts.append(("edits", edited_hits))

    if not parts:
        return {"available": False, "results": [], "hint": _NO_DB_HINT,
                "matched": None, "ranking": None}

    merged = _merge_parts(parts)[:limit]
    matchables = [part.get("matched") for _, part in parts if part.get("matched") is not None]
    outcome = {
        "available": True,
        "results": merged,
        "matched": int(sum(matchables)) if matchables else None,
        "ranking": next((part.get("ranking") for _, part in parts if part.get("ranking")), None),
        "hint": " ".join(dict.fromkeys(str(part["hint"]) for _, part in parts if part.get("hint"))),
        "parts": {
            label: {"ranking": part.get("ranking"), "matched": part.get("matched"),
                    "count": len(part.get("results") or [])}
            for label, part in parts
        },
    }
    if hidden:
        #: 删掉的那些**不该悄悄少几条**：用户刚删了一份又搜，得能对上号。
        outcome["hidden"] = {"count": hidden, "reason": "这些命中属于已删除（回收站）的文档。"}
    if len(parts) > 1:
        outcome["hint"] = (outcome["hint"] + " " if outcome["hint"] else "") + \
            "多库语料不同（curated=供应商抽取，ocr=本仓补的扫描件，" \
            "extra=本仓补的主库没登记的格式，edits=知识库里改过的正文）" \
            "⇒ 用 from 区分，分数不可比。"
    if include_partial and any(row.get("coverage") for row in merged):
        #: 只有 caller 主动要半成品时才会走到这里。必须说出来：``coverage`` 的
        #: ``chunkPages`` 远小于 ``pages`` 时，这条引用只代表那几页，不代表全书。
        outcome["hint"] = (outcome["hint"] + " " if outcome["hint"] else "") + \
            "结果含**没跑完的 OCR 文档**：coverage.pages 是总分页、coverage.chunkPages 是" \
            "真有正文的页数，别把片段当成整本可用。"
    windowed = next(((label, part["window"]) for label, part in parts if part.get("window")), None)
    if windowed:
        #: 取样窗只对**带目录过滤**的那一库有意义（过滤是内存里筛的）；两库都带时以
        #: curated 为准并在 note 里说明，分列计数仍在 ``parts`` —— 别把两个窗口加成一个。
        label, window = windowed
        outcome["window"] = {**window, "of": label}
        if len(parts) > 1:
            outcome["window"]["note"] = ("窗口只统计 curated 库；ocr 库的取样见 parts.ocr，"
                                         "两库窗口不可相加。")
    if filter_info:
        outcome["filters"] = filter_info
    return outcome


def _merge_parts(parts: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    """两库结果按**秩**融合（RRF，``k=RRF_K``）；去重键 ``(path, page, chunkNo)``。

    为什么不比分数：两库的 bm25 是对**各自语料**归一出来的，量纲不可比（同一段文字在
    5403 份库和 644 份库里的 idf 完全不同）⇒ 只能比秩。去重是为防「重跑同一份文档」时
    两库各留一份残留，正常情况不会重叠。
    """
    seen: set[tuple[Any, Any, Any]] = set()
    scored: list[tuple[float, dict[str, Any]]] = []
    for _label, outcome in parts:
        for rank, row in enumerate(outcome.get("results") or [], start=1):
            key = (row.get("path"), row.get("page"), row.get("chunkNo"))
            if key in seen:
                continue
            seen.add(key)
            scored.append((1.0 / (RRF_K + rank), row))
    scored.sort(key=lambda pair: -pair[0])
    return [row for _, row in scored]


def _window_size(limit: int, allowed: set[str] | None) -> int:
    """取样窗口：不带目录过滤就只要 ``limit`` 行；带过滤必须放大。

    为什么放大而不是在 SQL 里过滤：国家/类型/奖项**不在** sqlite 的 ``documents`` 表里
    （只有 path/name/page/status），想过滤只能先在内存里比对路径集合。窗口上限 600 是
    「取到够用就停」的折中；真到上限会在回执里报 ``exhausted=false``，不装作已经筛完。
    """
    return limit if allowed is None else min(max(limit * 40, 200), 600)


def _methods_fts(conn: sqlite3.Connection, terms: list[str], *, where: str,
                 params: list[str], limit: int,
                 allowed: set[str] | None = None) -> dict[str, Any]:
    #: 词都加双引号 ⇒ 词里的 ``AND`` / ``-`` / 括号是字面量，不是 FTS 语法
    match = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
    window = _window_size(limit, allowed)
    rows = conn.execute(
        "SELECT path, name, page, chunk_no, "
        "snippet(chunks, 5, '【', '】', '…', 40) AS excerpt, bm25(chunks) AS rank "
        f"FROM chunks WHERE chunks MATCH ? AND {where} ORDER BY rank LIMIT ?",
        [match, *params, window],
    ).fetchall()
    matched = conn.execute(
        f"SELECT count(*) FROM chunks WHERE chunks MATCH ? AND {where}", [match, *params]
    ).fetchone()[0]
    results = []
    examined = 0
    for path, name, page, chunk_no, excerpt, rank in rows:
        on_disk, rel = _on_disk(path)
        examined += 1
        if allowed is not None and rel not in allowed:
            continue
        results.append({
            "source": "methods",
            "title": name or "",
            "path": rel,
            "page": page,
            "chunkNo": chunk_no,
            #: bm25 是负值且越小越相关 ⇒ 取负号才对得起「分数大 = 更相关」
            "score": round(-float(rank), 4),
            "excerpt": _flatten(excerpt)[:EXCERPT_CHARS],
            "onDisk": on_disk,
        })
        if len(results) >= limit:
            break
    hint = "" if results else "换更具体的长词（≥3 字，如「三幕结构」「激励事件」）。"
    outcome: dict[str, Any] = {"available": True, "results": results,
                               "matched": int(matched), "ranking": "bm25", "hint": hint}
    if allowed is not None:
        #: 过滤在内存里做 ⇒ 必须交代「筛了多少行」，否则「命中 300 段只给 2 段」无法解释
        outcome["window"] = {"examined": examined, "size": window,
                             "exhausted": len(rows) < window}
    return outcome


def _methods_like(conn: sqlite3.Connection, terms: list[str], *, where: str,
                  params: list[str], limit: int,
                  allowed: set[str] | None = None) -> dict[str, Any]:
    """短词路线：先要求**全部命中**，空则放宽到**任一命中**（并如实标注）。"""
    for joined in ("AND", "OR"):
        cond = f" {joined} ".join(["text LIKE ? ESCAPE '\\'"] * len(terms))
        window = _window_size(limit, allowed)
        rows = conn.execute(
            f"SELECT path, name, page, chunk_no, text FROM chunks "
            f"WHERE {cond} AND {where} LIMIT ?",
            [_like(term) for term in terms] + params + [window * 4],
        ).fetchall()
        if not rows:
            continue
        scored = []
        examined = 0
        for path, name, page, chunk_no, text in rows:
            on_disk, rel = _on_disk(path)
            examined += 1
            if allowed is not None and rel not in allowed:
                continue
            flat = _flatten(text)
            scored.append({
                "source": "methods",
                "title": name or "",
                "path": rel,
                "page": page,
                "chunkNo": chunk_no,
                "score": float(sum(flat.count(term) for term in terms)),
                "excerpt": _window(text, terms[0])[:EXCERPT_CHARS],
                "onDisk": on_disk,
            })
        scored.sort(key=lambda item: (-item["score"], item["path"], str(item["chunkNo"])))
        outcome: dict[str, Any] = {
            "available": True,
            "results": scored[:limit],
            #: LIKE 没有 bm25，也就不做全量 count（全扫两次 ≈0.6 s，不值得）⇒ 如实给 None
            "matched": None,
            "ranking": "like" if joined == "AND" else "like-or",
            "hint": "" if joined == "AND" else "这些词不是同段共现，只是任一词命中。",
        }
        if allowed is not None:
            outcome["window"] = {"examined": examined, "size": window * 4,
                                 "exhausted": len(rows) < window * 4}
        return outcome
    return {"available": True, "results": [], "matched": None, "ranking": "like",
            "hint": "库里没有同时含这些词的片段。"}


def _fill_partial_coverage(conn: sqlite3.Connection,
                           results: list[dict[str, Any]]) -> None:
    """给结果里的**半成品**补 ``coverage``：这份文档总页数 vs 真有正文的页数。

    只在 ``include_partial=True`` 时调用 —— 默认路线已把半成品排除，不必白花一次扫描。
    ``document_id`` 在两库各自编号、互不相通，所以这里**用 path 对齐**（结果里的
    ``path`` 已由 :func:`_on_disk` 重映射成本仓相对路径，故同样过一遍 :func:`_rel_of`）。
    """
    if not results:
        return
    try:
        rows = conn.execute(
            "SELECT id, path, pages FROM documents WHERE status = ?",
            (OCR_STATUS_PARTIAL,)).fetchall()
        if not rows:
            return
        chunk_pages = {doc_id: int(count) for doc_id, count in conn.execute(
            "SELECT document_id, count(DISTINCT page) FROM chunks GROUP BY document_id")}
    except sqlite3.Error:  # pragma: no cover —— 老 sidecar 没有 status 列时不必标
        return
    by_rel: dict[str, tuple[int, int]] = {}
    for src_id, raw, pages in rows:
        rel = _rel_of(raw)
        if rel:
            by_rel[rel] = (src_id, int(pages or 0))
    for row in results:
        hit = by_rel.get(str(row.get("path") or ""))
        if hit is None:
            continue
        src_id, total = hit
        row["coverage"] = {"pages": total, "chunkPages": chunk_pages.get(src_id, 0)}


# ===========================================================================
# 来源二：参考方法文本（md 小节）
# ===========================================================================

#: 进程内缓存：``sig`` = 全部 md 的 (路径, mtime_ns) —— 变了才重读（223 份 ≈ 2 MB）
_PLAYBOOK: dict[str, Any] = {"sig": None, "sections": []}


def _split_sections(text: str) -> list[tuple[str, str]]:
    """按 ``#`` 标题切小节；首个标题之前的前言也算一节（有些 SKILL.md 开头不写标题）。"""
    marks = list(_HEADING.finditer(text))
    if not marks:
        return [("（全文）", text)]
    sections: list[tuple[str, str]] = []
    if marks[0].start() > 0 and len(text[:marks[0].start()].strip()) >= 20:
        sections.append(("（前言）", text[:marks[0].start()]))
    for index, mark in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(text)
        sections.append((mark.group(2).strip(), text[mark.end():end]))
    return sections


def _skill_of(rel_path: str) -> str:
    """``film-expert-bridge/combat/x.md`` → ``film-expert-bridge``；根直属文件 → 文件名。

    2026-11-04 扁平化后第一段**就是**技能名：原来的 ``packs/`` 壳与 ``references/library/``
    两级零信息目录已删，所以不再需要「认壳、再取第二段」那层判断。少一个判断也少一个静默
    错位的可能 —— 从前把壳名写死成字面量，会让所有小节的 ``skill`` 都变成壳名，检索照跑、
    结果照出，只是 ``skill=`` 过滤全失效。
    """
    return rel_path.split("/")[0] if rel_path else ""


#: 句界分隔符（分隔符留在前一片）。与 RAGFlow 的生产口径 ``DEFAULT_DELIMITER``
#: （``\n!?;。；！？``，见其 ``token.go`` L404-407 注释）逐字一致 —— **有意不含英文句点**：``Mr.`` / ``e.g.`` 会被切碎，
#: 而本库正文以中文为主，英文句点出现在缩写里的概率远高于真句末。
_SENTENCE_SPLIT = re.compile(r"(?<=[\n!?;。；！？])")


def _split_oversized(text: str, width: int) -> list[str]:
    """把超宽的一段按句界切成 <=width 的片；单句仍超宽才硬切（硬切是兜底不是常态）。"""
    if len(text) <= width:
        return [text]
    pieces: list[str] = []
    pending = ""
    for piece in (unit for unit in _SENTENCE_SPLIT.split(text) if unit):
        if not piece.strip():
            #: 纯空白片（连续分隔符产生）挂到**下一片**前面，保证拼回去逐字无损
            pending += piece
            continue
        piece = pending + piece
        pending = ""
        if len(piece) > width:
            pieces.extend(piece[index:index + width] for index in range(0, len(piece), width))
        else:
            pieces.append(piece)
    if pending:
        if pieces:
            pieces[-1] += pending
        else:
            pieces.append(pending)
    return pieces


def _overlap_tail(text: str, pct: int) -> str:
    """取 ``text`` 末尾 ``pct``%（按字，不按 token：本库正文以中文为主、1 字≈1 token）。"""
    if pct <= 0 or not text:
        return ""
    cut = int(len(text) * (100 - pct) / 100.0)
    return text[cut:] if cut < len(text) else ""


def _fit_prefix(overlap: str, current: str, width: int) -> str:
    """把重叠前缀接到新窗前；装不下就从左逐字裁前缀，宁可少重叠也不超宽。"""
    if not overlap:
        return current
    if len(overlap) + len(current) <= width:
        return overlap + current
    for drop in range(1, len(overlap) + 1):
        suffix = overlap[drop:]
        if len(suffix) + len(current) <= width:
            return suffix + current
    return current


def _windows(body: str, *, limit: int = PLAYBOOK_SECTION_MAX,
             width: int = PLAYBOOK_WINDOW,
             overlap: int = PLAYBOOK_WINDOW_OVERLAP) -> list[str]:
    """短节原样返回；长节按句界拼窗、相邻窗重叠 ``overlap``%（尾部过短的窗并入前窗）。

    三步对应 RAGFlow ``token.go::mergeUnits`` 的契约：超宽**单位**先按句界展开（硬切兜底）→
    拼窗的阈值按 overlap 缩小（``width*(100-pct)/100``，给重叠预留位置）→ 开新窗时无条件前置
    前窗尾巴。末窗过短并入前窗与之同理（Haystack ``document_splitter._concatenate_units`` 也是
    并进前一片，但它的 ``split_threshold`` 按 units 计、默认 0 即关闭）；本仓按字数的 ``<20%``
    窗宽**默认生效** —— 丢弃等于让这段文字不可检索，宁可让末窗略微超宽。
    """
    if len(body) <= limit:
        return [body]
    units: list[str] = []
    for para in re.split(r"\n\s*\n", body):
        para = para.strip()
        if para:
            units.extend(_split_oversized(para, width))

    threshold = width * (100 - overlap) / 100.0  # 留出重叠位，避免「拼满再叠」必然超宽
    pieces: list[str] = []
    buffer = ""
    for unit in units:
        if not buffer:
            buffer = unit
            continue
        #: ``+2`` 是 ``\n\n`` 连接符的字数；判据与 RAGFlow 的「投影合并 > target 就开新窗」同义
        if len(buffer) > threshold or len(buffer) + len(unit) + 2 > width:
            pieces.append(buffer)
            buffer = _fit_prefix(_overlap_tail(buffer, overlap), unit, width)
        else:
            buffer = f"{buffer}\n\n{unit}"
    if buffer:
        pieces.append(buffer)
    if len(pieces) > 1 and len(pieces[-1]) < width * 0.2:
        pieces[-2] = f"{pieces[-2]}\n\n{pieces[-1]}"
        pieces.pop()
    return [piece for piece in pieces if piece.strip()]


def _playbook_files(root: Path) -> list[Path]:
    """只走白名单子树（见 :data:`PLAYBOOK_DIRS`）；不 rglob 整根 —— 整根会把根目录元数据
    与以后可能出现的别家目录一起卷进索引。"""
    files: list[Path] = []
    for name in PLAYBOOK_DIRS:
        base = root / name
        if base.is_dir():
            files.extend(item for item in base.rglob("*.md")
                         if item.name not in PLAYBOOK_SKIP_NAMES)
    return sorted(files)


def _playbook_sections() -> list[dict[str, Any]]:
    root = Path(PLAYBOOK_ROOT)
    files = _playbook_files(root) if root.exists() else []
    sig = tuple((str(item), item.stat().st_mtime_ns) for item in files)
    if _PLAYBOOK["sig"] == sig:
        return _PLAYBOOK["sections"]
    sections: list[dict[str, Any]] = []
    for item in files:
        try:
            text = item.read_text(encoding="utf-8", errors="replace")
        except OSError:  # noqa: PERF203 —— 单份读失败不该让整库不可检索
            continue
        rel = item.relative_to(root).as_posix()
        skill = _skill_of(rel)
        for heading, body in _split_sections(text):
            body = body.strip()
            if len(body) < 20:  # 只有标题没内容的小节不进索引（检索到它等于没检索）
                continue
            pieces = _windows(body)
            for index, piece in enumerate(pieces):
                sections.append({"path": rel, "skill": skill, "heading": heading,
                                 "text": piece, "part": index, "parts": len(pieces)})
    _PLAYBOOK.update(sig=sig, sections=sections)
    return sections


def playbook_skills() -> dict[str, int]:
    """技能名 → 小节数（``/facets`` 与 Agent 自查用：别让调用方猜技能名）。"""
    counts: dict[str, int] = {}
    for section in _playbook_sections():
        counts[section["skill"]] = counts.get(section["skill"], 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _skill_arg(skill: str | None) -> str:
    """归一素材包名：``film-expert-bridge``、``film-expert-bridge/``、``packs/film-expert-bridge``
    都收成 ``film-expert-bridge``。

    前缀表取 :data:`PLAYBOOK_SHELLS` 而**不是** :data:`PLAYBOOK_DIRS`：后者装的是技能名本身，
    拿它当壳去剥，``film-expert-bridge/x`` 会被剥成 ``x``，过滤全失效且不报错。
    旧写法 ``packs/xxx``、``skills/xxx`` 仍认 —— 调用方手里的提示词、旧会话记录、上游文档
    都还在用这两种形状，按新结构拒收等于把一次纯目录调整变成对外破坏。
    """
    text = (skill or "").strip().strip("/")
    for prefix in PLAYBOOK_SHELLS:
        if text.startswith(prefix + "/"):
            text = text[len(prefix) + 1:]
            break
    return text.split("/")[0]


def search_playbook(query: str, *, limit: int = 5,
                    path_contains: str | None = None,
                    skill: str | None = None) -> dict[str, Any]:
    """在方法小节上打分：标题命中 5 分，正文按出现次数封顶 6 分；命中词数算覆盖率。

    ``skill`` 是 playbook 唯一的过滤维度（现存 ``film-cinedance`` / ``film-expert-bridge``，
    见 :func:`playbook_skills` —— 名字传错不会静默放宽，而是回 ``matched=0`` ＋
    ``filters.applied=false`` ＋ 可用名单）。长节已按 :data:`PLAYBOOK_SECTION_MAX` 切窗 ——
    同一节的多个窗会分别出现，用回执里的 ``part``/``parts`` 区分（单窗不写这两格）。
    """
    if not Path(PLAYBOOK_ROOT).exists():
        return {"available": False, "results": [], "hint": _NO_PLAYBOOK_HINT,
                "matched": None, "ranking": None}

    wanted = _skill_arg(skill)
    filters = None
    if wanted:
        known = playbook_skills()
        if wanted not in known:
            return {"available": True, "results": [], "matched": 0, "ranking": "gram",
                    "filters": {"applied": False, "criteria": {"skill": wanted},
                                "hint": (f"没有名为「{wanted}」的技能目录 —— 条件**没生效**，"
                                         f"可用：{'、'.join(known) or '（空）'}")}}

    terms = query_terms(query)
    if not terms:
        return {"available": True, "results": [], "hint": "检索词为空。",
                "matched": None, "ranking": None}

    #: 查询本身就带「索引／目录」时不打折 —— 这是「我想找目录」的明确意图，
    #: 此时真资料分数再高也不是他要的（见 :data:`PLAYBOOK_NAV_INTENT`）。
    wants_nav = any(word in (query or "") for word in PLAYBOOK_NAV_INTENT)

    scored: list[dict[str, Any]] = []
    for section in _playbook_sections():
        path = section["path"]
        if any(part in path for part in EXCLUDED_PARTS):
            continue
        if wanted and section["skill"] != wanted:
            continue
        if path_contains and path_contains not in path:
            continue
        heading = section["heading"]
        body = section["text"]
        score = 0.0
        #: 覆盖率按**词**算，不按"命中次数"算 —— 一个词同时出现在标题和正文里只算 1 个词
        #: （早先按次数算，coverage 会算出 1.333 这种数，覆盖率就成了假指标）
        hit_terms = 0
        for term in terms:
            hit = False
            if term in heading:
                score += 5.0
                hit = True
            count = body.count(term)
            if count:
                score += min(count, 6)
                hit = True
            if hit:
                hit_terms += 1
        if score <= 0:
            continue
        #: 导航型文档降权（不排除）。判据写在 :data:`PLAYBOOK_NAV_DOCS` 上，不按「词出现在
        #: 哪个文件」猜 —— 索引的职责就是把全族名词抄一遍，比词频永远赢，比内容永远输。
        if not wants_nav and any(name in Path(path).name for name in PLAYBOOK_NAV_DOCS):
            score *= PLAYBOOK_NAV_WEIGHT
        row = {
            "source": "playbook",
            "title": heading or Path(path).stem,
            "path": path,
            "skill": section["skill"],
            "page": None,
            "chunkNo": None,
            "score": round(score, 2),
            "coverage": round(hit_terms / len(terms), 3),
            "excerpt": _best_window(body, terms, EXCERPT_CHARS)[:EXCERPT_CHARS],
            "onDisk": True,
        }
        if section["parts"] > 1:
            row["part"] = section["part"] + 1
            row["parts"] = section["parts"]
        scored.append(row)
    #: 先看覆盖率再看分数：多词查询里「三个词都沾一点」比「一个词出现 6 次」更该排前面
    #: （单词语句覆盖率恒为 1 ⇒ 自动退化成按分数排）
    scored.sort(key=lambda item: (-item["coverage"], -item["score"], item["path"],
                                  item.get("part", 1)))
    if scored:
        hint = ""
    elif wanted:
        hint = f"技能「{wanted}」下没命中——换个说法，或去掉 skill 全库找。"
    else:
        hint = "换个说法试试（这些方法文本用词偏专业，如「激励事件」「分场」）。"
    outcome: dict[str, Any] = {"available": True, "results": scored[:limit],
                               "matched": len(scored), "ranking": "gram", "hint": hint}
    if wanted:
        outcome["filters"] = {"applied": True, "criteria": {"skill": wanted},
                              "matched": len(scored)}
    return outcome


# ===========================================================================
# 盘点（「库里到底有多少」）与统一入口
# ===========================================================================

#: 盘点缓存：键是库路径（夹具会换路径）⇒ 值里带 (mtime, size) 判失效
_METHODS_REPORT: dict[str, dict[str, Any]] = {}


def _methods_report() -> dict[str, Any] | None:
    db = Path(METHODS_DB)
    conn = _connect(db)
    if conn is None:
        return None
    try:
        stat = db.stat()
        key = db.as_posix()
        cached = _METHODS_REPORT.get(key)
        if cached and cached["stamp"] == (stat.st_mtime_ns, stat.st_size):
            return cached["report"]
        documents = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
        chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        #: **登记了 ≠ 有正文**：本库实测 5403 条登记里只有 2262 条真被抽出了文字
        #: （其余多是扫描件无文本层）⇒ 不报这个数，模型会把「扫不到」当成「库里没有」。
        with_text = conn.execute("SELECT count(DISTINCT document_id) FROM chunks").fetchone()[0]
        #: 逐条 .exists()：5403 条实测 ≈0.06 s（NTFS 元数据命中）⇒ 每次盘点都真查
        missing = 0
        for (raw,) in conn.execute("SELECT path FROM documents"):
            local = resolve_local(raw)
            if local is None or not local.exists():
                missing += 1
        report = {
            "documents": int(documents),
            "chunks": int(chunks),
            "withText": int(with_text),
            "coverage": round(with_text / documents, 4) if documents else 0.0,
            "missingFiles": int(missing),
            "db": db.name,
            "root": db.parent.parent.as_posix(),
        }
        _METHODS_REPORT[key] = {"stamp": (stat.st_mtime_ns, stat.st_size), "report": report}
        return report
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def _playbook_report() -> dict[str, Any] | None:
    root = Path(PLAYBOOK_ROOT)
    if not root.exists():
        return None
    sections = _playbook_sections()
    files = {section["path"] for section in sections}
    indexed_paths = {str(root / rel) for rel in files}
    #: 白名单外的 md **照样数出来**（不静默缩小范围）。唯一的例外是
    #: :data:`PLAYBOOK_ROOT_FILES` 里点名的根目录元数据（``README.md`` 记来源/版本/许可）——
    #: 写成"排除根目录所有 md"省事，但那样「根目录漏放了一份正文」这类真问题就永远沉默了。
    #: 只报"索引了 137 份"而不说"盘上还有几份没索引"，下一个人会以为索引漏了。
    on_disk = {str(item) for item in root.rglob("*.md")
               if not (item.parent == root and item.name in PLAYBOOK_ROOT_FILES)}
    skills: dict[str, dict[str, int]] = {}
    for section in sections:
        row = skills.setdefault(section["skill"], {"files": 0, "sections": 0})
        row["sections"] += 1
    for rel in files:
        name = _skill_of(rel)
        skills[name]["files"] += 1
    return {
        "files": len(files),
        "sections": len(sections),
        "chars": sum(len(section["text"]) for section in sections),
        "root": root.as_posix(),
        "dirs": list(PLAYBOOK_DIRS),
        "skills": dict(sorted(skills.items(), key=lambda item: (-item[1]["sections"],
                                                                 item[0]))),
        "unindexedFiles": len(on_disk - indexed_paths),
        "note": ("只索引 dirs 点名的来源目录；根目录的 README 属元数据不计入，其余 md 照报（见 unindexedFiles）。"),
    }


def _ocr_report() -> dict[str, Any] | None:
    """OCR sidecar 的进度：没有这个库就 ``None``（**不编 0**，与 methods 同一条纪律）。

    ``documents`` 是已 OCR 的份数、``pages`` 是累计页数 —— 这两项直接对得上
    ``ocr_index.py --stats`` 里的「还剩多少」，所以别在这里另算一套口径。
    """
    db = Path(OCR_DB)
    conn = _connect(db)
    if conn is None:
        return None
    try:
        documents = conn.execute(
            "SELECT count(*) FROM documents WHERE status = 'ocr'").fetchone()[0]
        partial = conn.execute(
            "SELECT count(*) FROM documents WHERE status <> 'ocr'").fetchone()[0]
        chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        #: 只算**整本完成**的页数：``--pages=N`` 试跑的半成品（partial）页数不计入，
        #: 否则跑 6 页的样本会把自己报成 586 页，进度条与 ``ocr_index --stats`` 打架。
        pages = conn.execute(
            "SELECT coalesce(sum(pages), 0) FROM documents WHERE status = 'ocr'"
        ).fetchone()[0]
        try:
            meta = dict(conn.execute("SELECT k, v FROM meta").fetchall())
        except sqlite3.Error:
            meta = {}
        stat = db.stat()
        return {
            "documents": int(documents),
            "partial": int(partial),
            "chunks": int(chunks),
            "pages": int(pages),
            "bytes": int(stat.st_size),
            "updatedAt": int(stat.st_mtime),
            "engine": meta.get("engine"),
            "dpi": meta.get("dpi"),
            "db": db.name,
        }
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def _extra_report() -> dict[str, Any] | None:
    """第三个库（``library-extra.sqlite``）的盘点；没有这个库就 ``None``（不编 0）。

    与 ``_ocr_report`` 的口径差别：这边**份数就是份数**，不区分「整本 / 半本」——
    ``lib_index`` 是按文件一次性抽完的，没有分页增量跑这回事；``quarantine`` 单列，
    因为那批是「抽到了但可打印率不达标」，属**已知缺口**而不是已完成。
    """
    db = Path(EXTRA_DB)
    conn = _connect(db)
    if conn is None:
        return None
    try:
        rows = dict(conn.execute("SELECT status, count(*) FROM documents GROUP BY status").fetchall())
        chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        by_ext = dict(conn.execute(
            "SELECT extension, count(*) FROM documents GROUP BY extension").fetchall())
        try:
            meta = dict(conn.execute("SELECT k, v FROM meta").fetchall())
        except sqlite3.Error:
            meta = {}
        stat = db.stat()
        return {
            "documents": int(sum(rows.values())),
            "byStatus": {str(k): int(v) for k, v in rows.items()},
            "byExtension": {str(k): int(v) for k, v in by_ext.items()},
            "chunks": int(chunks),
            "bytes": int(stat.st_size),
            "updatedAt": int(stat.st_mtime),
            "engines": meta.get("engine"),
            "db": db.name,
        }
    except sqlite3.Error:
        return None
    finally:
        conn.close()


#: 目录树 / 文档清单缓存：键是 (主库, sidecar) 两条路径，值带两侧 (mtime,size) 判失效。
#: 全扫 5403 条 + 128166 块实测 ≈0.09 s，看着不贵；但前端浏览会连着点目录，且这两个函数
#: 是**同一份扫描**的两种切法 ⇒ 扫一次存起来，别让「点一下目录」变成一次全表扫。
_TREE: dict[str, dict[str, Any]] = {}

#: 目录层级上限：一级=资料大类、二级=题材/合集、三级=单本书/单剧本
MAX_DEPTH = 3


def _rel_of(raw: str) -> str:
    """旧盘符绝对路径（形如 ``…\\全球热门与获奖影视剧本知识库\\08-…\\x.pdf``）→ ``08-…/x.pdf``。"""
    return script_catalog.normalize(raw, LOCAL_MARKER) or ""


def _stamp(*paths: Path) -> tuple:
    out: list[tuple[int, int] | None] = []
    for path in paths:
        try:
            stat = path.stat()
            out.append((stat.st_mtime_ns, stat.st_size))
        except OSError:
            out.append(None)
    return tuple(out)


def _scan_documents(*, include_buried: bool = False) -> dict[str, Any]:
    """主库 + OCR sidecar **一次扫全**，返回 ``{"available", "docs", "hint"}``。

    每条 doc 的字段与前端的文档表格一一对应：``name/path/size/pages/withText/ocr/partial``。
    ``withText`` 与 ``ocr`` 实测互斥（主库有正文的不会进 sidecar），但仍按两个布尔分别记，
    因为「两者皆假」才是真正待补的那批 —— 合成一个枚举就没法如实报 coverAge 了。

    ``ocr`` **只认 ``status='ocr'``**：曾经这里写成「sidecar 里有这条 path 就算已补」，
    于是 ``--pages=12`` 试跑出来的半成品（库里实测 14 份，含 920 页的书只扫了 2 页）
    被算成已完成 ⇒ 目录树报 20 份、:func:`_ocr_report` 报 6 份、``pending`` 少算 14 份。
    现在 ``partial`` 单独一档但仍然**计入 pending**（它确实还没做完），三处口径才对得上。

    ``include_buried=False``（默认）时**已删的不出现在任何读口**：树、清单、检索全都看不见它，
    这正是「删除」对用户的意义。回收站要看得见 ⇒ 那一处显式传 ``True``，并靠每条上的
    ``source``／``buried`` 字段把它标出来；**不是**另建一套扫描。
    """
    methods, sidecar, extra = Path(METHODS_DB), Path(OCR_DB), Path(EXTRA_DB)
    key = (f"{methods.as_posix()}|{sidecar.as_posix()}|{extra.as_posix()}"
           f"|{'buried' if include_buried else 'live'}")
    #: 第四库（覆盖层）也要进戳：删一份 / 改一块之后，树与清单必须当场变，
    #: 不能等三库本身的 mtime 变了才刷新 —— 覆盖层**永远不改那三个库**。
    stamp = _stamp(methods, sidecar, extra) + (library_overlay.stamp(),)
    cached = _TREE.get(key)
    if cached and cached["stamp"] == stamp:
        return cached["scan"]

    conn = _connect(methods)
    if conn is None:
        return {"available": False, "docs": [], "hint": "方法库未安装（METHODS_DB 不存在）。"}
    try:
        rows = conn.execute(
            "SELECT id, path, name, size, pages FROM documents").fetchall()
        have = {doc_id for (doc_id,) in
                conn.execute("SELECT DISTINCT document_id FROM chunks")}
    except sqlite3.Error as exc:
        return {"available": False, "docs": [], "hint": f"方法库读不了：{exc}"}
    finally:
        conn.close()

    ocr_paths: set[str] = set()
    partial_paths: set[str] = set()
    quarantine_paths: set[str] = set()
    side = _connect(sidecar)
    if side is not None:
        try:
            for raw, status in side.execute("SELECT path, status FROM documents"):
                if status == OCR_STATUS_DONE:
                    ocr_paths.add(raw)
                elif status == OCR_STATUS_QUARANTINE:
                    quarantine_paths.add(raw)
                else:
                    #: 没有 status 或未知状态的旧行一律按半成品算 —— 宁可多算待做，
                    #: 也不能把「只扫了几页」的文档报成「已补 OCR」。
                    partial_paths.add(raw)
        except sqlite3.Error:
            ocr_paths, partial_paths, quarantine_paths = set(), set(), set()
        finally:
            side.close()

    #: 第三个库（``lib_index`` 补的 ``.doc`` 等）：主库 ``documents`` 里**连行都没有**，
    #: 不并进来 ⇒ 目录树与文档清单看不见这批文件，"已补正文"更无从谈起。
    #: 它的 ``path`` 本来就是相对库根，不用过 ``_rel_of``。
    extra_docs: dict[str, dict[str, Any]] = {}
    side_extra = _connect(extra)
    if side_extra is not None:
        try:
            for rel, name, size, pages, status in side_extra.execute(
                    "SELECT path, name, size, pages, status FROM documents"):
                if not rel or any(part in rel for part in EXCLUDED_PARTS):
                    continue
                extra_docs[rel] = {"name": name or rel.rsplit("/", 1)[-1], "rel": rel,
                                   "size": int(size or 0), "pages": int(pages or 0),
                                   "status": status}
        except sqlite3.Error:
            extra_docs = {}
        finally:
            side_extra.close()

    #: 覆盖层的两份名单**一次取走**：``buried`` 决定看不看得见、``edited`` 只是打标记。
    #: 逐条查库会让一次目录树扫描变成几千次小查询（三库实测 6283 份）。
    buried = library_overlay.buried_map()
    edited = library_overlay.edited_paths()

    docs: list[dict[str, Any]] = []
    for doc_id, raw, name, size, pages in rows:
        rel = _rel_of(raw)
        if not rel or any(part in rel for part in EXCLUDED_PARTS):
            continue
        added = extra_docs.pop(rel, None)
        #: 一份文档归哪个库，决定「能不能删/改」以及墓碑记在谁名下 —— 三库各自一个命名空间，
        #: 不能只用 path 判（``extra`` 里的 ``A/x.doc`` 与主库的 ``A/x.doc`` 是两件东西）。
        source = ("extra" if added else
                  "ocr" if (raw in ocr_paths or raw in partial_paths or raw in quarantine_paths)
                  else "curated")
        gone = rel in buried.get(source, ())
        if gone and not include_buried:
            continue
        docs.append({
            "name": name or rel.rsplit("/", 1)[-1],
            "path": rel,
            "raw": raw,
            "size": int(size or 0),
            "pages": int(pages or 0),
            "withText": doc_id in have or bool(added and added["status"] == "extra"),
            #: 只有 sidecar 里 ``status='ocr'``（整本跑完）才算「已补 OCR」
            "ocr": raw in ocr_paths,
            #: 有 sidecar 记录但只做了前几页 ⇒ 仍属待做，单独暴露以便前端区分
            "partial": raw in partial_paths,
            #: 跑完但被判废、等人工 ``--redo``（不写块，自动重跑也不会捡它）
            "quarantine": raw in quarantine_paths,
            #: 主库没登记、由本仓 ``lib_index`` 补进 ``library-extra.sqlite`` 的那批
            "extra": bool(added),
            #: 这份归哪个库 —— 写操作（改/删）必须带对，否则墓碑会记到隔壁库名下
            "source": source,
            #: 有覆盖层正文（界面上要打「已修改」，否则用户看不出自己改过什么）
            "edited": rel in edited.get(source, ()),
            #: 已删进回收站（只在 ``include_buried=True`` 时可能出现）
            "buried": gone,
        })
    for rel, row in sorted(extra_docs.items()):
        gone = rel in buried.get("extra", ())
        if gone and not include_buried:
            continue
        docs.append({
            "name": row["name"], "path": rel, "raw": rel,
            "size": row["size"], "pages": row["pages"],
            "withText": row["status"] == "extra",
            "ocr": False, "partial": False,
            "quarantine": row["status"] == OCR_STATUS_QUARANTINE,
            "extra": True,
            "source": "extra",
            "edited": rel in edited.get("extra", ()),
            "buried": gone,
        })
    scan = {"available": True, "docs": docs, "hint": None}
    _TREE[key] = {"stamp": stamp, "scan": scan}
    return scan


def library_tree(*, depth: int = 2) -> dict[str, Any]:
    """目录树 ＋ 每个目录的正文覆盖（``documents/withText/ocr/pending``）。

    前端左侧浏览、Agent 自查「哪块还没正文」都读它。``pending`` 是**两者皆假**的那批
    （供应商没抽出文字、本仓也还没 OCR），不是「文档数 − 有正文」那种含糊减法。

    分组**永远是目录**：文档直接躺在这一层（路径段数 ≤ ``depth``）时取它的**上级目录**，
    否则截前 ``depth`` 段。不这么写，凡是在该层直接放文件的目录都会退化成
    「一份文件一个组」——前端就会把书当成文件夹点（实测夹具里 2 段路径即触发）。
    """
    scan = _scan_documents()
    depth = max(1, min(int(depth or 2), MAX_DEPTH))
    if not scan["available"]:
        return {"available": False, "hint": scan["hint"], "depth": depth,
                "totals": None, "groups": []}

    groups: dict[str, dict[str, Any]] = {}
    #: ``ocrPartial`` 是 ``pending`` 的**子集**（有 sidecar 记录但没跑完的那批），
    #: 单独列出来只为让「为什么它还挂在待做里」有话说；求总量时别再把它加一遍。
    totals = {"documents": 0, "withText": 0, "ocr": 0, "ocrPartial": 0,
              "pending": 0, "bytes": 0, "pages": 0}
    for doc in scan["docs"]:
        parts = doc["path"].split("/")
        label = "/".join(parts[:depth]) if len(parts) > depth else "/".join(parts[:-1])
        group = groups.setdefault(label, {
            #: 库根下的散文件（没有上级目录）归到根桶，`category=""` 即全部 —— 不编假目录名
            "path": label, "name": label.split("/")[-1] or "(库根)",
            "documents": 0, "withText": 0, "ocr": 0, "ocrPartial": 0,
            "pending": 0, "bytes": 0, "pages": 0,
        })
        pending = not doc["withText"] and not doc["ocr"]
        for bucket in (group, totals):
            bucket["documents"] += 1
            bucket["bytes"] += doc["size"]
            bucket["pages"] += doc["pages"]
            bucket["withText"] += 1 if doc["withText"] else 0
            bucket["ocr"] += 1 if doc["ocr"] else 0
            bucket["ocrPartial"] += 1 if doc["partial"] else 0
            bucket["pending"] += 1 if pending else 0

    ordered = sorted(groups.values(), key=lambda item: (-item["documents"], item["path"]))
    return {"available": True, "hint": None, "depth": depth,
            "totals": totals, "groups": ordered}


def list_documents(*, category: str | None = None, q: str = "", only: str = "all",
                   offset: int = 0, limit: int = 30) -> dict[str, Any]:
    """某目录（或以词搜名）下的文档清单，分页 —— 前端知识库页面的「浏览」态。

    ``only``：``all`` / ``withText``（有正文）/ ``pending``（待 OCR）/ ``ocr``（本仓已补整本）
    / ``partial``（补了一半）/ ``extra``（主库没登记、本仓 ``lib_index`` 补进来的格式，
    如 ``.doc``）/ ``edited``（改过的）/ ``buried``（回收站）。``pending`` **含** ``partial``
    （它们确实还没做完），与 :func:`library_tree` 的 ``totals.pending`` 同一口径 ⇒ 两个口子的
    数字不会打架。``buried`` 是唯一会把已删的扫进来的档位（其余一律看不见删掉的）。
    ``category`` 是 ``library_tree`` 里的 ``path`` 前缀（含子目录，前缀匹配）。
    """
    scan = _scan_documents(include_buried=(only == "buried"))
    if not scan["available"]:
        return {"available": False, "hint": scan["hint"], "total": 0, "items": []}

    needle = (q or "").strip().lower()
    prefix = (category or "").strip().replace("\\", "/").strip("/")
    picked: list[dict[str, Any]] = []
    for doc in scan["docs"]:
        if prefix and not (doc["path"] == prefix or doc["path"].startswith(prefix + "/")):
            continue
        if needle and needle not in doc["name"].lower():
            continue
        if only == "withText" and not doc["withText"]:
            continue
        if only == "ocr" and not doc["ocr"]:
            continue
        if only == "partial" and not doc["partial"]:
            continue
        if only == "extra" and not doc.get("extra"):
            continue
        if only == "edited" and not doc.get("edited"):
            continue
        if only == "buried" and not doc.get("buried"):
            continue
        if only == "pending" and (doc["withText"] or doc["ocr"]):
            continue
        picked.append(doc)

    picked.sort(key=lambda doc: (not doc["withText"], not doc["ocr"], doc["path"]))
    offset = max(0, int(offset or 0))
    limit = max(1, min(int(limit or 30), 200))
    window = picked[offset:offset + limit]
    return {
        "available": True,
        "hint": None,
        "category": prefix or None,
        "only": only,
        "total": len(picked),
        "offset": offset,
        "limit": limit,
        "items": [{
            "name": doc["name"],
            "path": doc["path"],
            "size": doc["size"],
            "pages": doc["pages"],
            "withText": doc["withText"],
            "ocr": doc["ocr"],
            "partial": doc["partial"],
            "extra": bool(doc.get("extra")),
            #: 界面上要直接看出「已修改 / 在回收站 / 归哪个库」—— 三个字段都来自同一次扫描，
            #: 不给的话前端只能逐条再问 ``/document/state``（6283 份就是 6283 个请求）。
            "source": doc.get("source"),
            "edited": bool(doc.get("edited")),
            "buried": bool(doc.get("buried")),
        } for doc in window],
    }


def library_report() -> dict[str, Any]:
    """五个来源的盘点（methods / ocr / extra / playbook / catalog）；缺哪个就 ``None``（**不编 0**）。"""
    return {
        "methods": _methods_report(),
        "ocr": _ocr_report(),
        "extra": _extra_report(),
        "playbook": _playbook_report(),
        "catalog": script_catalog.report(LIBRARY_ROOT, marker=LOCAL_MARKER),
    }


def _playbook_target(raw: str) -> tuple[Path, str] | None:
    """把路径解析成 playbook 里的一份 md（不是方法正文就返回 ``None``，交回原路）。

    认三种写法：检索结果里的相对路径（``film-expert-bridge/…``）、带 ``corpus/playbook/``
    前缀的、以及绝对路径。目录穿越（``../``、解析后跑出根外）在这里就拒 —— 收窄后必须仍在
    :data:`PLAYBOOK_DIRS` 白名单子树内，否则这个口子就成了任意文件读取。

    ⚠️ 旧前缀 ``data/playbook/`` 也认（2026-11-07 迁到 ``corpus/playbook/`` 之前生成的检索
    回执、前端缓存、用户书签里都是旧写法）。**兼容只是读旧值，产出的一律是新路径**。
    """
    text = (raw or "").strip().replace("\\", "/")
    if not text:
        return None
    base = Path(PLAYBOOK_ROOT).resolve()
    marker = "/corpus/playbook/"
    cut = text.rfind(marker)
    if cut >= 0:
        text = text[cut + len(marker):]
    else:
        for prefix in ("corpus/playbook/", "data/playbook/"):
            if text.startswith(prefix):
                text = text[len(prefix):]
                break
    text = text.lstrip("/")
    if text.startswith("./"):
        text = text[2:]
    if not text or ".." in text.split("/"):
        return None
    try:
        file = (base / text).resolve()
        rel = file.relative_to(base).as_posix()
    except (OSError, ValueError):
        return None
    if not rel.endswith(".md") or rel.split("/")[0] not in PLAYBOOK_DIRS:
        return None
    return (file, rel) if file.is_file() else None


def _open_playbook(file: Path, rel: str, *, offset: int, limit: int) -> dict[str, Any]:
    """把一份方法 md 按 ``#`` 小节分页返回（结构对齐 :func:`_open_on`，前端不必分叉）。

    **不截断正文**：这里读的是方法原文（一节通常几百字），截了就等于让人去猜后半句。
    ``page`` 恒为 ``None`` —— md 没有页码，硬编 0 会让前端显示「第 0 页」。
    """
    try:
        text = file.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"available": True, "origin": "playbook", "document": None,
                "chunks": [], "total": 0, "offset": offset,
                "hint": f"方法正文读取失败：{exc}"}
    sections = [(heading, body.strip()) for heading, body in _split_sections(text)
                if body.strip()]
    total = len(sections)
    window = sections[offset:offset + limit]
    chunks = [{"page": None, "chunkNo": offset + index + 1, "chars": len(body),
               "truncated": False, "heading": heading or "", "text": body}
              for index, (heading, body) in enumerate(window)]
    outcome: dict[str, Any] = {
        "available": True,
        "origin": "playbook",
        "document": {
            "path": rel,
            "title": file.stem,
            "extension": "md",
            "pages": None,
            "status": "md",
            "coverage": None,
            "onDisk": True,
        },
        "catalog": None,
        "chunks": chunks,
        "total": total,
        "offset": offset,
        "hasMore": offset + len(chunks) < total,
    }
    if total == 0:
        outcome["hint"] = "这份方法正文只有标题、没有可读的小节。"
    return outcome


def open_document(path: str, *, offset: int = 0, limit: int = 20,
                  chars: int = 2000) -> dict[str, Any]:
    """按检索结果里的相对路径**翻到原文**：文档头 + 该文档的正文块（分页）。

    为什么必须有这个口子：``search`` 只给 400 字片段，模型引用时得能自己再往下看（页号、
    块号都在），否则「有引用」只是一句空话。这也让 CLI/HTP 能拿同一份数据核账。

    ``path`` 接受检索结果里的相对路径，也接受建库机器旧盘符前缀的原始路径（统一走
    :func:`script_catalog.normalize`）。

    **两份来源都认**：先看是不是 playbook 的方法 md（:func:`_playbook_target`，命中就直接
    读文件，``origin="playbook"``）；不是才走下面两库。少了这一支，前端能从
    ``/search?source=playbook`` 搜到方法条目、点进去却永远「没有正文记录」——
    搜得到打不开，等于只做了一半。

    两库都试：供应商库里 **登记了但没正文** 的 3139 份扫描件，正文在本仓 OCR sidecar 里
    ⇒ 主库 0 块时自动落到 sidecar，回执的 ``origin`` 说明这段正文出自哪一库。
    """
    target = _playbook_target(path)
    if target is not None:
        offset = max(0, int(offset or 0))
        limit = max(1, min(int(limit or 20), 100))
        return _open_playbook(*target, offset=offset, limit=limit)

    rel = script_catalog.normalize(path, LOCAL_MARKER)
    if not rel:
        return {"available": True, "hint": "路径解析不出相对路径（缺库目录锚点）。",
                "document": None, "chunks": [], "total": 0}
    offset = max(0, int(offset or 0))
    limit = max(1, min(int(limit or 20), 100))

    fallback: dict[str, Any] | None = None
    opened = False
    for label, db in (("curated", Path(METHODS_DB)), ("ocr", Path(OCR_DB)),
                      ("extra", Path(EXTRA_DB))):
        conn = _connect(db)
        if conn is None:
            continue
        opened = True
        try:
            outcome = _open_on(conn, rel, offset=offset, limit=limit, chars=chars,
                               origin=label)
        except sqlite3.Error as exc:  # noqa: BLE001 —— 读坏了要说出来
            outcome = {"available": True, "document": None, "chunks": [], "total": 0,
                       "hint": f"{label} 索引读取失败：{exc}"}
        finally:
            conn.close()
        if int(outcome.get("total") or 0) > 0:
            return outcome
        fallback = fallback or outcome
    if fallback is not None:
        return fallback
    return {"available": False, "hint": _NO_DB_HINT, "document": None,
            "chunks": [], "total": 0}


def _open_on(conn: sqlite3.Connection, rel: str, *, offset: int, limit: int,
             chars: int, origin: str) -> dict[str, Any]:
    """在**单个库**上翻原文（``open_document`` 的两库循环调用它）。"""
    #: ``replace(path, '\', '/')`` 是必须的：三个库的 ``path`` 写法不同 —— 供应商库与
    #: OCR sidecar 存的是 Windows 原始路径（建库机器的旧盘符绝对路径，走 ``_rel_of`` 归一），而
    #: ``library-extra.sqlite``（含本仓上传件）存的是**正斜杠的相对路径**。只把查询词换成
    #: 反斜杠（``_like(rel.replace("/", "\\"))``）在 extra 上恒不命中 ⇒ 症状正是
    #: 「清单里看得到、点进去说没有这份、也改不了」。
    row = conn.execute(
        "SELECT id, path, name, extension, pages, status FROM documents "
        "WHERE replace(path, '\\', '/') LIKE ? ESCAPE '\\' ORDER BY id LIMIT 1",
        [_like(rel)],
    ).fetchone()
    if row is None:
        entry = script_catalog.lookup(LIBRARY_ROOT, rel, LOCAL_MARKER)
        return {
            "available": True,
            "document": None,
            "chunks": [],
            "total": 0,
            "origin": origin,
            "catalog": entry.as_dict() if entry else None,
            "hint": ("索引里没有这份文件的正文记录"
                     + ("（目录 CSV 里有，说明登记过但没抽出文字，多为扫描件）"
                        if entry else "（目录里也没有）")),
        }
    doc_id, raw_path, name, extension, pages, status = row
    total = conn.execute("SELECT count(*) FROM chunks WHERE document_id = ?",
                         [doc_id]).fetchone()[0]
    #: 半成品（只扫了前几页）要能自证覆盖率：``pages=920`` 配 ``total=2`` 若不说清，
    #: 读起来像「抽取失败」，实际是「跑到第 2 页就断了、剩下 918 页还没有」。
    coverage = None
    if status == OCR_STATUS_PARTIAL:
        chunk_pages = conn.execute(
            "SELECT count(DISTINCT page) FROM chunks WHERE document_id = ?",
            [doc_id]).fetchone()[0]
        coverage = {"pages": int(pages or 0), "chunkPages": int(chunk_pages or 0)}
    rows = conn.execute(
        "SELECT page, chunk_no, text FROM chunks WHERE document_id = ? "
        "ORDER BY chunk_no LIMIT ? OFFSET ?",
        [doc_id, limit, offset],
    ).fetchall()
    on_disk, rel_path = _on_disk(raw_path)
    entry = script_catalog.lookup(LIBRARY_ROOT, raw_path, LOCAL_MARKER)
    #: 覆盖层：同一块若被改过，这里读到的是**改后的正文**（原库一个字都不动）。
    #: 键用调用方传进来的 ``rel``，与 ``_scan_documents`` 里算的 ``rel`` 同一形态，
    #: 否则「清单里标了已修改、点进去还是原文」这种对不上的症状会很难查。
    overlay = library_overlay.edits_of(origin, rel)
    meta = overlay.pop(library_overlay.META_CHUNK, None)
    chunks = []
    for page, chunk_no, text in rows:
        body = _flatten(text)
        replaced = overlay.get(int(chunk_no))
        if replaced is not None:
            body = _flatten(replaced["text"])
        item = {
            "page": page,
            "chunkNo": chunk_no,
            "chars": len(body),
            "truncated": len(body) > chars,
            "text": body[:chars],
            #: 这一块的正文是不是覆盖层给的 —— 界面据此打「已修改」，不靠猜
            "edited": replaced is not None,
        }
        if replaced is not None:
            item["note"] = replaced.get("note")
            item["original"] = _flatten(text)[:chars]
        chunks.append(item)
    outcome: dict[str, Any] = {
        "available": True,
        "origin": origin,
        "document": {
            "path": rel_path or rel,
            "title": (meta or {}).get("text") if meta else (name or ""),
            "name": name or "",
            "extension": extension or "",
            "pages": int(pages or 0),
            "status": status or "",
            "coverage": coverage,
            "onDisk": on_disk,
            #: 写操作的三个开关（前端据此显示或隐藏按钮，而不是点了才报错）
            "source": origin,
            "editable": origin in library_overlay.WRITABLE_SOURCES,
            "edited": bool(meta) or bool(overlay),
            "buried": library_overlay.is_buried(origin, rel),
        },
        "catalog": entry.as_dict() if entry else None,
        "chunks": chunks,
        "total": int(total),
        "offset": offset,
        "hasMore": offset + len(chunks) < int(total),
    }
    #: 编辑块号落不到原库任何一块上 ⇒ 原库被重建过、块变少了，这些改动**再也对不上正文**。
    #: 必须报出来，否则用户以为自己改的那段还在（覆盖层不会自动清理，也不敢自动删）。
    #: 判据用**库里真实的块号区间**（``min/max``）而不是 ``0–total-1``：``extra`` 的块号从
    #: 1 起，按 ``total`` 推会把最后一块的编辑误报成 stale。
    span = conn.execute("SELECT min(chunk_no), max(chunk_no) FROM chunks WHERE document_id = ?",
                        [doc_id]).fetchone()
    lo, hi = (None, None) if span is None else span
    orphan = sorted(no for no in overlay
                    if lo is None or not int(lo) <= no <= int(hi))
    if orphan:
        outcome["staleEdits"] = orphan[:50]
        outcome["hint"] = (f"有 {len(orphan)} 处编辑指向原库里已不存在的块（原库可能重建过），"
                           "那些改动暂时读不到；可撤销或重新改一次。")
    if int(total) == 0:
        #: 登记了 ≠ 有正文：真库里 5403 条登记有 3139 条是这个状态（扫描件无文本层）。
        #: 不说这一句，模型只会得到「0 块」然后自己去编。
        outcome["hint"] = ("索引里登记了这份文件，但**没有正文块**"
                           "（多为扫描件无文本层，或抽取失败）—— 不是路径错。"
                           "可跑 app/scripts/corpus/ocr_index.py 把它 OCR 进来。")
    return outcome


# ===========================================================================
# 写：编辑 / 删除 / 还原（一律落覆盖层，不碰那三个只读库）
# ===========================================================================


def _connect_write(db: Path) -> sqlite3.Connection | None:
    """写连接。三个索引库平时只读；**只有删除本仓自己上传的件**才需要写 ``extra``。"""
    if not db.exists():
        return None
    try:
        conn = sqlite3.connect(db.as_posix(), timeout=60)
        conn.execute("PRAGMA busy_timeout=60000")
        return conn
    except sqlite3.Error:
        return None


def _db_of(source: str) -> Path:
    return {"curated": Path(METHODS_DB), "ocr": Path(OCR_DB),
            "extra": Path(EXTRA_DB)}.get(source, Path(METHODS_DB))


def _locate(path: str) -> dict[str, Any] | None:
    """查一份文档落在哪个库、几块、盘上在哪 —— 所有写操作的**第一步**。

    顺序与 :func:`open_document` 一致（curated → ocr → extra）：同一个相对路径在两个库里
    都出现时，读的是前面那个，写也必须写到同一个，否则「点进去看得到、改了却报没有这份」。
    """
    rel = script_catalog.normalize(path, LOCAL_MARKER)
    if not rel:
        return None
    for label in ("curated", "ocr", "extra"):
        conn = _connect(_db_of(label))
        if conn is None:
            continue
        try:
            #: 分隔符必须在 SQL 里归一（理由同 ``_open_on``）：``extra`` 存的相对路径是
            #: 正斜杠，查询词换成反斜杠在它上面恒不命中。
            row = conn.execute(
                "SELECT id, path, name, extension, pages, status FROM documents "
                "WHERE replace(path, '\\', '/') LIKE ? ESCAPE '\\' ORDER BY id LIMIT 1",
                [_like(rel)]).fetchone()
            if row is None:
                continue
            #: 块号**起点不是 0**：供应商库从 0 起，``lib_index._chunks`` 从 **1** 起。
            #: 只取 ``count(*)`` 会让「合法范围」写成 ``0–n-1``（对 extra 是错的：最后一块
            #: 报越界、第 0 块明明不存在却放行），所以三个数一起取回来，判越界按真号判。
            span = conn.execute(
                "SELECT count(*), min(chunk_no), max(chunk_no) FROM chunks WHERE document_id = ?",
                [row[0]]).fetchone()
        except sqlite3.Error:
            continue
        finally:
            conn.close()
        doc_id, raw_path, name, extension, pages, status = row
        on_disk, disk_rel = _on_disk(raw_path)
        local = Path(LIBRARY_ROOT) / (disk_rel or rel)
        return {"source": label, "rel": rel, "id": int(doc_id), "raw": raw_path,
                "name": name or rel.rsplit("/", 1)[-1], "extension": extension or "",
                "pages": int(pages or 0), "chunks": int(span[0] or 0),
                "chunkMin": None if span[1] is None else int(span[1]),
                "chunkMax": None if span[2] is None else int(span[2]),
                "status": status or "", "onDisk": on_disk, "local": local}
    return None


def document_state(path: str) -> dict[str, Any]:
    """写出按钮该不该亮：``{found, source, editable, deletable, hardDeletable, buried, chunks}``。

    界面**先问再画**，比「点了才报错」体验好得多，也避免了给只读件放一个必失败的按钮。
    """
    found = _locate(path)
    if not found:
        return {"found": False, "editable": False, "deletable": False,
                "hardDeletable": False, "buried": False, "chunks": 0,
                "reason": "索引里没有这份文件（可能是刚上传还没入库，或路径带错）。"}
    upload = library_overlay.upload_of(found["rel"])
    return {
        "found": True,
        "source": found["source"],
        "path": found["rel"],
        "name": found["name"],
        "chunks": found["chunks"],
        "onDisk": found["onDisk"],
        "editable": True,
        "deletable": True,
        #: 物理删只对「本仓上传、且确实还在盘上」的件开放：其余都是别处产出的，
        #: 删了文件也没用（下次重建又回来），只能靠墓碑隐藏。
        "hardDeletable": found["source"] == "extra" and bool(upload) and found["onDisk"],
        "uploaded": bool(upload),
        "buried": library_overlay.is_buried(found["source"], found["rel"]),
        "edits": len(library_overlay.edits_of(found["source"], found["rel"])),
    }


def write_chunk(path: str, chunk_no: int, text: str, *, note: str | None = None) -> dict[str, Any]:
    """改一块正文（``chunk_no = -1`` 改标题）。返回 ``{ok, …}``，失败带人话 ``reason``。"""
    found = _locate(path)
    if not found:
        return {"ok": False, "reason": "索引里没有这份文件（可能是刚上传还没入库，或路径带错）。"}
    try:
        chunk_no = int(chunk_no)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "chunkNo 必须是整数（改标题传 -1）。"}
    page = 0
    if chunk_no != library_overlay.META_CHUNK:
        #: 判越界**以库里真有这一块为准**，不按 ``0–n-1`` 推算：块号起点因库而异
        #: （供应商库 0 起、``extra`` 1 起），推算要么放过不存在的第 0 块、要么把最后一块
        #: 判成越界 —— 后者对用户就是「明明有这块却改不了」。
        conn = _connect(_db_of(found["source"]))
        row = None
        try:
            if conn is not None:
                row = conn.execute(
                    "SELECT page FROM chunks WHERE document_id = ? AND chunk_no = ?",
                    [found["id"], chunk_no]).fetchone()
        except sqlite3.Error:
            row = None
        finally:
            if conn is not None:
                conn.close()
        if conn is not None and row is None:
            if found["chunks"] == 0:
                return {"ok": False,
                        "reason": "这份目前一块正文都没有（空文档/扫描件），先补正文再改。"}
            return {"ok": False,
                    "reason": f"块号越界：这份共 {found['chunks']} 块，"
                              f"块号 {found['chunkMin']}–{found['chunkMax']}。"}
        page = int(row[0] or 0) if row else 0
    saved = library_overlay.save_edit(found["source"], found["rel"], chunk_no, text, note=note,
                                      name=found["name"], page=page)
    reset_cache()
    return {"ok": True, "saved": saved, "source": found["source"], "path": found["rel"],
            "chunks": found["chunks"],
            "hint": "已保存到覆盖层：检索与翻页立刻按改后的正文走，原库文件一个字没动。"}


def undo_chunk(path: str, chunk_no: int) -> dict[str, Any]:
    """撤销某块的编辑（回到原库那份正文）。"""
    found = _locate(path)
    if not found:
        return {"ok": False, "reason": "索引里没有这份文件。"}
    out = library_overlay.drop_edit(found["source"], found["rel"], int(chunk_no))
    reset_cache()
    return {"ok": True, **out, "hint": "已还原成原库正文。" if out["dropped"] else "这块本来就没改过。"}


def bury_document(path: str, *, hard: bool = False,
                  reason: str | None = None) -> dict[str, Any]:
    """删除一份。默认**进回收站**（可还原）；``hard=True`` 只对「本仓上传的件」有效。

    为什么默认不物理删：三库里 4894 + 3139 份是供应商/OCR 的产出，盘上文件是**它们的真源**。
    删文件既不可逆、也会让下一次重建把它又变回来 —— 用户会觉得"删了又自己回来了"。
    """
    found = _locate(path)
    if not found:
        return {"ok": False, "reason": "索引里没有这份文件。"}
    if hard:
        upload = library_overlay.upload_of(found["rel"])
        if found["source"] != "extra" or not upload:
            return {"ok": False,
                    "reason": "只有本仓自己上传的件能彻底删除；库里原有的件一律进回收站（可还原）。"}
        conn = _connect_write(_db_of("extra"))
        if conn is None:
            return {"ok": False, "reason": "补录库打不开，没能删除。"}
        try:
            with conn:
                conn.execute("DELETE FROM chunks WHERE document_id = ?", [found["id"]])
                conn.execute("DELETE FROM documents WHERE id = ?", [found["id"]])
        except sqlite3.Error as exc:  # noqa: BLE001
            return {"ok": False, "reason": f"补录库删除失败：{exc}"}
        finally:
            conn.close()
        removed = False
        try:
            local = Path(found["local"])
            if local.exists():
                local.unlink()
                removed = True
        except OSError as exc:  # noqa: BLE001
            return {"ok": False, "mode": "hard", "indexDeleted": True,
                    "reason": f"索引记录已删，但盘上文件删不掉（{exc}）；请手动处理。"}
        library_overlay.drop_edit(found["source"], found["rel"], library_overlay.META_CHUNK)
        library_overlay.raise_tomb(found["source"], found["rel"])
        reset_cache()
        return {"ok": True, "mode": "hard", "path": found["rel"], "fileRemoved": removed,
                "hint": "已彻底删除（索引记录 + 盘上文件）。"}
    out = library_overlay.bury(found["source"], found["rel"], reason=reason)
    reset_cache()
    return {"ok": True, "mode": "tombstone", **out,
            "hint": "已移入回收站：清单/检索/目录树都不再出现，随时可还原；原文件还在盘上。"}


def raise_document(path: str) -> dict[str, Any]:
    """把回收站里的一份拿回来。"""
    found = _locate(path)
    rel = found["rel"] if found else script_catalog.normalize(path, LOCAL_MARKER)
    if not rel:
        return {"ok": False, "reason": "路径解析不出相对路径。"}
    sources = [found["source"]] if found else list(library_overlay.WRITABLE_SOURCES)
    for source in sources:
        out = library_overlay.raise_tomb(source, rel)
        if out["restored"]:
            reset_cache()
            return {"ok": True, **out, "hint": "已还原到清单里。"}
    reset_cache()
    return {"ok": False, "reason": "回收站里没有这份（它本来就在清单里）。"}


def overlay_report() -> dict[str, Any]:
    """回收站与已改清单（前端「回收站」抽屉读这一个口子）。"""
    data = library_overlay.report()
    data["writableSources"] = list(library_overlay.WRITABLE_SOURCES)
    return data


def reset_cache() -> None:
    """清空进程内缓存（测试/换库时用；正常运行不需要）。"""
    _PLAYBOOK.update(sig=None, sections=[])
    _METHODS_REPORT.clear()
    _TREE.clear()


def _fuse(results: list[dict[str, Any]], *, k: int = RRF_K) -> list[dict[str, Any]]:
    """按「源内秩」重排（RRF），给每条补 ``fusion``。

    只依赖秩 ⇒ 单源调用是单调变换、顺序不变，所以可以无条件调用，不必分支判「是不是多源」。
    排序是**稳定**的：同分（两源各自的第 1 名）保持调用方给的源顺序，不会每跑一次就换一次。
    """
    rank_of: dict[str, int] = {}
    fused: list[dict[str, Any]] = []
    for row in results:
        source = str(row.get("source") or "")
        rank = rank_of.get(source, 0)
        rank_of[source] = rank + 1
        weight = SOURCE_WEIGHTS.get(source, 1.0)
        fused.append({**row, "fusion": round(weight / (k + rank + 1), 6)})
    fused.sort(key=lambda row: -row["fusion"])
    return fused


def search(query: str, *, source: str = "all", limit: int = 5,
           path_contains: str | None = None,
           filters: dict[str, Any] | None = None,
           skill: str | None = None,
           include_partial: bool = False) -> dict[str, Any]:
    """统一入口：``source`` ∈ ``all`` / ``methods`` / ``playbook``。

    返回 ``available`` / ``results`` / ``corpus`` / ``sources`` / ``hint`` / ``note``。
    ``sources`` 逐源给 ``available``、``ranking``、``matched``、``count``（这一源实际返回几条），
    这样模型能分清「一共命中 300 段、这里给了 5 段」和「库里就 5 段」。

    两组过滤维度**分开**，因为它们的适用范围是硬边界：

    * ``filters``（``country`` / ``genre`` / ``category`` / ``docType`` / ``permission`` /
      ``festival``）只对 ``methods`` 生效；
    * ``skill``（``film-acting`` …）只对 ``playbook`` 生效。

    给错源的那一组会被如实标成 ``filtersApplied=false`` 并附一句 hint —— 不许静默忽略。

    ``include_partial`` 透给 :func:`search_methods`：默认 **不** 返回只跑了几页的 OCR
    半成品，打开后每条带 ``coverage``（总页数 / 真有正文的页数）。
    """
    mode = (source or "all").strip().lower()
    if mode not in ("all", "methods", "playbook"):
        raise ValueError(f"未知来源：{source}（可用：all / methods / playbook）")
    limit = max(1, min(int(limit or 5), 10))
    skill_name = _skill_arg(skill)

    results: list[dict[str, Any]] = []
    sources: dict[str, Any] = {}
    hints: list[str] = []
    applied: dict[str, Any] | None = None
    for name, function in (("methods", search_methods), ("playbook", search_playbook)):
        if mode not in ("all", name):
            continue
        if name == "methods":
            #: ``include_partial`` 只对 methods 有意义（playbook 没有 OCR 半成品这回事）
            outcome = function(query, limit=limit, path_contains=path_contains,
                               filters=filters or None,
                               include_partial=include_partial)
        elif name == "playbook":
            outcome = function(query, limit=limit, path_contains=path_contains,
                               skill=skill_name or None)
        else:
            outcome = function(query, limit=limit, path_contains=path_contains)
        sources[name] = {
            "available": bool(outcome.get("available")),
            "ranking": outcome.get("ranking"),
            "matched": outcome.get("matched"),
            "count": len(outcome.get("results") or []),
        }
        if outcome.get("filters"):
            sources[name]["filters"] = outcome["filters"]
            if name == "methods":
                #: 顶层 ``filters`` 只代表主库（methods）那一组；playbook 的 skill 过滤
                #: 留 ``sources.playbook.filters`` —— 两组维度适用范围不同，不许合成一份。
                applied = outcome["filters"]
            if not outcome["filters"].get("applied") and outcome["filters"].get("hint"):
                hints.append(str(outcome["filters"]["hint"]))
        if outcome.get("window"):
            sources[name]["window"] = outcome["window"]
        if outcome.get("parts"):
            #: methods 由「供应商库 ＋ OCR sidecar」两块拼成 ⇒ 逐库报秩与命中数，
            #: 否则「OCR 补进来的那 644 份到底有没有参与」在回执里看不出来。
            sources[name]["parts"] = outcome["parts"]
        if filters and name == "methods":
            #: 逐源交代「这一源到底有没有被目录条件筛过」——``all`` 模式下 playbook 不受过滤，
            #: 不说清楚就会出现「问韩国，却返回一堆没有国别的专项方法文本」这种误导。
            sources[name]["filtersApplied"] = bool((outcome.get("filters") or {}).get("applied"))
        if name == "playbook":
            sources[name]["filtersApplied"] = bool((outcome.get("filters") or {}).get("applied"))
        if outcome.get("hint"):
            hints.append(str(outcome["hint"]))
        results.extend(outcome.get("results") or [])

    results = _fuse(results)  # 跨源按秩融合；单源时顺序不变（RRF 是单调变换）

    if filters and "playbook" in sources:
        hints.append("playbook 源没有国家/类型/奖项这些维度（只有 skill 维度），未参与过滤。")
    if skill_name and "methods" in sources:
        sources["methods"]["filtersApplied"] = False
        hints.append(f"skill={skill_name} 只对 playbook 生效，methods 源未参与过滤。")

    available = any(item["available"] for item in sources.values())
    report = library_report()
    corpus = {name: report.get(name) for name in sources}
    return {
        "available": available,
        "query": query,
        "source": mode,
        "results": results,
        "sources": sources,
        "filters": applied,
        #: 两源以上才谈融合；单源给 ``None`` 而不是伪造一个「融合过」的交代
        "fusion": ({"method": "rrf", "k": RRF_K, "weights": dict(SOURCE_WEIGHTS),
                    "note": "score 是各源原分数（量纲不可比），排序依据是 fusion"}
                   if len(sources) > 1 else None),
        "corpus": corpus,
        #: 两源可能给同一句提示（空检索词）⇒ 去重，但保持出现顺序
        "hint": " ".join(dict.fromkeys(hints)),
        "note": _RESULT_NOTE,
    }
