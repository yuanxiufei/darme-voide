#!/usr/bin/env python3
"""OCR / 文字层的**清洗与质量门禁** —— 让「乱码」根本进不了索引。

为什么要有这一层
================
``ocr_index.py`` 原先只有一道门槛 ``MIN_PAGE_CHARS = 12``：只要够 12 个字就写块。
对扫描件来说这条几乎等于没有 —— 实测最坏的形态是**引擎把整页识别成一排符号**
（``|||| ||| ```` `────`）或**同一行重复几十遍**，字数轻松过 12，于是脏文本直接进
fts5，检索时命中一堆读不出意思的片段，而且**看不出来是脏的**（块里没有"质量"字段）。

本模块两条职责，都是纯函数（不 import fitz / numpy / rapidocr，便于单测与离线复核）：

1. **清洗**（只做有把握的、可解释的替换）：控制字符、单字孤行、跨页页眉页脚；
2. **门禁**（``ok`` / ``suspect`` / ``reject`` 三档）：给每页一个判定与理由，
   ``reject`` 的页**不写块**，改写入 ``page_audit`` 表等人工复核。

为什么清洗只做这三件
====================
碎行合并是本批语料（1978–2009 铅印期刊、复印剧本）的**主要噪声**：PyMuPDF 按字形
簇出块 ⇒ ``布``、``、``、`,` 各自占一行，拼回正文时被切块逻辑当成正常换行，导致
块边界落在半个词中间、``snippet()`` 给出断句。其余"看起来脏"的东西（缺字、错字、
竖排倒序）属于**识别精度**问题，靠规则改只会越改越错，必须留给门禁去标记。

阈值为什么这么宽
================
这批语料里英文原版剧本、竖排目录、表格页都是合法形态：（英文书 CJK 占比 0、
目录页每行两三个字）。所以判据只盯**结构上不可能正常**的信号 —— 替换字符、
私用区字符、控制字符、同一字符连续 8 次、整页去重字符数 < 8。宁可漏报（进
``suspect`` 让人看），不可误杀（把好页标成 ``reject`` 就等于**主动删内容**）。
"""
from __future__ import annotations

import re
import unicodedata

__all__ = [
    "DEFAULT_RUNNING_HEAD_MAX",
    "JUDGEMENTS",
    "clean_document",
    "clean_page",
    "judge_page",
    "merge_broken_lines",
    "page_metrics",
    "reflow_lines",
    "reflow_page",
    "split_lines",
]

#: 三档判定。「ok」进库；「suspect」进库但记审计（供人工抽查）；「reject」不写块。
JUDGEMENTS = ("ok", "suspect", "reject")

#: 页眉/页脚行最长多少个字才可能被判成"重复的版头"。
#: 取 24：实测「·232·」(5)、「世界电影 1987 年第 3 期」(17)、「实惠图书之家独家制作」(10)
#: 都在 20 字内；期刊正文行常 30-40 字，取 40 会把**正文最后一行**误删。
#: ⚠️ 本模块初版就是 40，实测把剧本正文的收尾行当页脚删掉了（
#: 「“你们要的那批货，我带来了。”凌云把皮箱放下去，声音压得很低。」34 字）。
DEFAULT_RUNNING_HEAD_MAX = 24
#: **带句末标点**的行一律不当版头。版头是「·232·」「刊名 1987 年第 3 期」这种名词短语，
#: 不会出现句号/问号/感叹号；正文行几乎必然带一个 —— 这是与长度上限并行的第二道防线。
#: ⚠️ 只认**句末**标点，不认逗号/冒号：实测水印长行是
#: 「· ((&)实惠图书之家独家制作 联系方式 旺旺：…联系QQ：957719200」——它带全角冒号，
#: 若把冒号也算"像正文"，这种每页都有的水印就永远删不掉，会污染 BM25 与检索。
_PROSE_PUNCT = re.compile(r"[。！？…]")
#: 判定为「页眉页脚」需要在多少比例的行位（页首/页尾）里重复出现
RUNNING_HEAD_MIN_PAGES = 3
RUNNING_HEAD_SHARE = 0.5
#: **长版头**（超过 ``DEFAULT_RUNNING_HEAD_MAX`` 字）另走一条更高的门槛。
#: 真实水印会带联系方式、商号、电话，动辄 40+ 字（实测 45 字），按短行那套 24 字上限
#: 就永远删不掉。但长行的误删代价更大（那是整句正文），所以要求它在 **80%** 的页上
#: 逐字重复才删 —— 正文不会跨页重复，水印行会。
RUNNING_HEAD_LONG_SHARE = 0.8
#: 长版头的长度上限：超过它就不是"版头"而是整段（比如整页重复的引用块），不碰
RUNNING_HEAD_LONG_MAX = 200
#: 页首/页尾各看几行
RUNNING_HEAD_WINDOW = 2
#: 「版头装饰残留」的长度上限：水印常印成「花纹 + 商号」，OCR 只认出花纹那半截
#: （实测 ``· ((&)`` 6 字、``一((+/`` 5 字），单独成行、紧贴水印行。
_DECORATION_MAX = 8
#: 装饰残留里允许出现的"真字符"（字母/数字/CJK）个数上限。装饰行的本质是**纯符号**，
#: 偶尔有一个字是 OCR 把花纹认成了「一」「丨」；超过 1 个就是有人在写字，不是装饰。
_DECORATION_CHARS = 1
_DECORATION = re.compile(r"[0-9A-Za-z\u4e00-\u9fff]")

#: 结构上不可能出现在正常正文里的字符：替换符、控制字符（除 \n\r\t）、私用区
_HARD_JUNK = re.compile(r"[\ufffd\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f\ue000-\uf8ff]")
#: 同一非文字字符连续出现（OCR 把插图/表格线识别成一排符号的典型形态）。
#: ⚠️ 必须是非捕获组：带括号量化时 ``findall`` 只回**最后一个单字符**，那样
#: ``len(run)`` 恒为 1，"这一排符号有多长"就永远量不出来（占比判据直接失效）。
_JUNK_RUN = re.compile(r"[^\w\s\u4e00-\u9fff]{8,}")
#: 符号连续段占全页比例 ≥ 这个值 ⇒ 整页判废
JUNK_RUN_SHARE_REJECT = 0.15
#: 单个连续段长度 ≥ 这个值 ⇒ 无论占比都判废（那是整块插图/表格被认成符号）
JUNK_RUN_MAX_REJECT = 200
#: 一个"有效字符"：CJK、拉丁字母、数字、常见中英标点。
#: ⚠️ 不能用 ``\w``：Python 的 ``\w`` 覆盖**所有** Unicode 字母（含希腊/西里尔/私用区
#: 之外的杂符），拿它算"有效占比"恒等于 1，指标就是废的（本模块初版踩过）。
_VALID = re.compile(
    r"[0-9A-Za-z\u4e00-\u9fff\s"
    r"\u3000-\u303f"      # CJK 标点
    r"\uff01-\uff65"      # 全角标点/全角字母数字
    r"\u2000-\u206f"      # 通用标点（弯引号、破折号、省略号）
    r"\u0021-\u002f\u003a-\u0040\u005b-\u0060\u007b-\u007e]"  # ASCII 标点
)
_CJK = re.compile(r"[\u4e00-\u9fff]")
_CTRL = re.compile(r"[^\S\n]+")
_BLANK_LINES = re.compile(r"\n{3,}")

#: 判定阈值（含边界含义写清，改之前先读上面「阈值为什么这么宽」）
FFFD_REJECT = 0.02          #: 替换字符占比 > 2% ⇒ 必坏
VALID_REJECT = 0.60         #: 有效字符占比 < 60% ⇒ 整页是符号噪声
VALID_SUSPECT = 0.90        #: < 90% ⇒ 可能有符号混入
#: 整页不同字符少于 4 个 ⇒ 崩坏（同一字刷满一页）。
#: ⚠️ 门槛压到 4 而不是 8：中文短句重复（``"第一页正文。" * 40`` 去重 6 个）是**合法**
#: 文本形态，8 会把它判成 reject ⇒ 直接删内容。真正该拦的"刷屏"另有 ``DISTINCT_SUSPECT``
#: 那条（去重比 < 12%）兜着，标记为 suspect 供人工看，而不是直接丢。
DISTINCT_REJECT = 4
DISTINCT_SUSPECT = 0.12     #: 去重比 < 12% ⇒ 大量重复（同一行刷屏）
#: 「去重比」这条判据**只对中文页成立**：一个 1500 字的英文页去重字符只有 30–60 个
#: （26 个字母天然反复出现），去重比恒在 2–6%，永远低于 ``DISTINCT_SUSPECT`` ⇒
#: 每一页都被标存疑。2026-11-04 批次实证：《Monsters, Inc.》小说改编本 170 页**页页存疑**，
#: 库里 4000+ 条存疑里 98% 是这一条，把 ``--audit`` 真正要人工看的坏页埋掉了。
#: 故这条判据要求本页 CJK 占比 ≥ 这个值才算数；另一条「整页只有 <4 个不同字符」不设门槛
#: —— 拉丁文页面照样拦得住（``"aaa bbb" * 300`` 只剩 4 个字符时仍然 reject）。
DISTINCT_CJK_MIN = 0.3
#: 「去重字符数」类判据的字数下限。⚠️ 没有这一条会把**短页**全部误杀：
#: 实测一条 ``"第一页正文。" * 40`` 的合法测试正文去重只有 6 个字符（中文短句重复），
#: 而它完全正常。整页刷屏的坏样本动辄几百字，60 字的下限足够把两者分开。
DISTINCT_MIN_CHARS = 60
LINE_MEAN_SUSPECT = 3.0     #: 行均长度 < 3 字 ⇒ 碎行/竖排，读序可疑


def split_lines(text: str) -> list[str]:
    return [line.strip() for line in (text or "").replace("\r\n", "\n").split("\n")]


# ===========================================================================
# 清洗
# ===========================================================================


def _normalize(text: str) -> str:
    """控制字符、全角空格、零宽字符、连续空行 —— 只做**可逆可解释**的规整。"""
    body = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    body = body.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    body = body.replace("\ufeff", "").replace("\u3000", " ")
    body = "".join(ch for ch in body if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    body = _CTRL.sub(" ", body)
    lines = [line.strip() for line in body.split("\n")]
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def _is_fragment(line: str) -> bool:
    """孤字 / 纯标点行 —— 排版上不可能单独成立的碎块。

    两条：① 长度 1；② 整行没有任何字母/数字/CJK（全是标点符号）且不超过 3 字。

    ⚠️ 只认**极短**的：``布``、``、``、``。``、``,`` 这批是实测最典型的碎行。
    两个字的行坚决不动 —— 中文两字成词（``雨水``/``忽然``/``他说``），合并会把
    两句并成一句，属于**制造错误**而不是修复。
    """
    stripped = line.strip()
    if len(stripped) == 1:
        return True
    return bool(stripped) and len(stripped) <= 3 and not re.search(r"[\w\u4e00-\u9fff]", stripped)


def merge_broken_lines(lines: list[str]) -> list[str]:
    """把**孤字行与纯标点行**并回上一行；上一行是空行时向下并到下一行。

    两块语料的成因不同但症状一样：PyMuPDF 按字形簇出块（铅印期刊）、OCR 把
    标点单列成行。合并方向按中文排版惯例：标点跟随**前**文（``布``+``、`` ⇒
    ``布、``），所以优先并回上一行；页首那种「上一行是空行」的情况才向后并。
    """
    merged: list[str] = []
    carry = ""  #: 还没找到落点的碎行（出现在页首/段首）
    for raw in lines:
        line = raw.strip()
        if not line:
            if carry:
                merged.append(carry)
                carry = ""
            merged.append("")
            continue
        if carry:
            #: 前面攒着碎行 ⇒ 本行是它的正文宿主（``、`` + ``他走了`` ⇒ ``、他走了``）
            line = carry + line
            carry = ""
        if _is_fragment(line):
            if merged and merged[-1].strip():
                merged[-1] = merged[-1].rstrip() + line
            else:
                carry = line
            continue
        merged.append(line)
    if carry:
        merged.append(carry)
    return merged


#: 行尾出现这些 ⇒ 这一句已经说完，不应与下一行合并
_CLOSERS = "。！？；：…—\"'”’」』）】》〉"
#: 行首出现这些 ⇒ 它是上一句的续接（标点被切到了下一行开头）
_OPENERS = "。，、；：！？…—\"'”’）】》」』〉,.;:!?)]}"
#: 触发「整页重排」的行均字数上限。正常中文排版 30+ 字/行、双栏 20+ 字/行；
#: 实测被字形簇切碎的期刊页是 9 字/行（697 字 / 77 行），落差极大。
REFLOW_TRIGGER = 16.0
#: 触发重排的最少行数（太短的页没有"被切碎"这回事）
REFLOW_MIN_LINES = 8


def _needs_reflow(lines: list[str]) -> bool:
    body = [line for line in lines if line.strip()]
    if len(body) < REFLOW_MIN_LINES:
        return False
    total = sum(len(line) for line in body)
    return total / len(body) < REFLOW_TRIGGER


def reflow_lines(lines: list[str]) -> list[str]:
    """把**被切碎的整页**按句子接回去（仅在整页明显碎掉时才启动）。

    为什么必须做：PyMuPDF 对来源是「扫描件 + OCR 文字层」的 PDF 会按**字形簇**出行，
    实测《世界电影》一页 697 字被切成 77 行（9 字/行），句子碎成 ``在诸如断代`` /
    ``、国别`` / ``、样式`` 三段。不接回去的话，块边界与 ``snippet()`` 全部落在半个词上，
    检索命中后读出来是断句。

    安全边界（三条同时成立才会合并，任何一条不成立就另起一行）：

    1. 上一行**不以句末标点结尾**（``。！？；：…"」』）``）—— 句子没说完才可能是续接；
    2. 本行**不以续接标点开头**时仍可合并（中文句子中间换行不需要标点），
       但若上一行已收尾就另起；
    3. 整页行均 < ``REFLOW_TRIGGER`` 字 —— **正常排版的页一律不动**，这是与
       「短行就合并」的关键区别：日记体、诗歌、剧本对白在正常行均下不会被搅乱。

    已知风险：双栏页若左栏末行恰好在句子中间被切断，会与右栏首行串行。此时**行尾
    无句末标点**这个前提成立，无法从纯文本层区分（要区分得拿坐标做栏切分）。概率上
    栏末多排到句末，实测的期刊页也未出现串行；真串了由 ``judge_page`` 与 ``suspect``
    标记兜住，宁可让人看，不静默造错。
    """
    out: list[str] = []
    buffer = ""
    for line in lines:
        text = line.strip()
        if not text:
            if buffer:
                out.append(buffer)
                buffer = ""
            out.append("")
            continue
        if not buffer:
            buffer = text
            continue
        tail = buffer.rstrip()[-1] if buffer.strip() else ""
        head = text[0]
        if tail in _CLOSERS and head not in _OPENERS:
            out.append(buffer)
            buffer = text
            continue
        #: 中英混排：两侧都是 ASCII 字母/数字才补空格，中文直接拼
        glue = " " if (tail.isascii() and tail.isalnum()
                       and head.isascii() and head.isalnum()) else ""
        buffer = buffer.rstrip() + glue + text
    if buffer:
        out.append(buffer)
    return out


def reflow_page(text: str) -> str:
    """只做「整页被切碎时重排」这一步（页眉剔除之后才能调，见 ``clean_document``）。"""
    lines = split_lines(text)
    if _needs_reflow(lines):
        lines = reflow_lines(lines)
    return "\n".join(lines).strip()


def clean_page(text: str, *, reflow: bool = True) -> str:
    """单页清洗：规整 → 碎行合并 → （整页被切碎时）重排。

    不含跨页页眉剔除 —— 那个要整份文档的统计才能判（``drop_running_heads``）。
    ⚠️ 整份文档**不能**直接用本函数的结果去剔页眉，必须走 ``clean_document``：
    重排会把页尾水印拼进上一行正文（见那里的说明）。
    """
    lines = merge_broken_lines(split_lines(_normalize(text)))
    if reflow and _needs_reflow(lines):
        lines = reflow_lines(lines)
    return "\n".join(lines).strip()


#: 版头归一用的键长。取**尾部**而不是整行，是因为实测同一本书的水印长这样：
#: 「· ((&)实惠图书之家独家制作 …联系QQ：957719200」与「一((+/实惠图书之家独家制作 …」——
#: 差异全在前缀（OCR 把装饰符认成了 `((&` / `一((+/`），正文部分逐字相同。
HEAD_KEY_TAIL = 24


def _norm_head(line: str) -> str:
    """版头比对键：数字一律折成 0（页码才能跨页对上）、去掉一切符号与空白、取尾部。

    三步各解决一个实测问题：

    1. ``\\d -> 0``：``·232·`` 与 ``·233·`` 必须归一成同一个键，否则每页的页码都被
       当成"新行"，跨页重复统计永远凑不够数；
    2. 去符号：装饰符在不同页会被 OCR 认成不同东西（``·`` / ``一`` / ``((&``）；
    3. 取尾 ``HEAD_KEY_TAIL`` 字：差异集中在前缀，尾部才是稳定的正文部分。
    """
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", re.sub(r"\d", "0", line))
    return text[-HEAD_KEY_TAIL:]


def _body_of(line: str) -> str:
    """去掉行首的装饰符号，得到"这一行实际在说什么"。

    ``_PROSE_PUNCT`` 原本直接判整行，实测被一个真实形态绕过去了：水印框被 OCR 认成
    两个句号 ⇒ 整行是 ``。。实惠图书之家独家制作 …联系QQ：957719200`` ⇒「含句读 ⇒
    是正文」⇒ 十页里九页的水印都删了，偏偏这一页放行、留在 fts5 里。
    **行首的符号不构成正文**（正文不会以句号开头），剥掉再判；句中和行尾的句读照旧
    当作正文证据。
    """
    return re.sub(r"^[^\w\u4e00-\u9fff]+", "", line or "")


def _is_decoration(line: str) -> bool:
    """紧贴版头的那半截「花纹」——纯符号行，删掉零信息损失。

    三条同时成立才算：≤ ``_DECORATION_MAX`` 字、真字符（字母/数字/CJK）≤ 1 个、
    且至少含一个符号字符。后两条各防一种误删：

    * 纯数字行（``9``、``232``）—— 短但那是页码/编号，交给它们自己的版头判据；
    * **含句末标点的行** —— ``一。`` 这种是正文，不是花纹。
    """
    body = (line or "").strip()
    if not body or len(body) > _DECORATION_MAX:
        return False
    if _PROSE_PUNCT.search(body):
        return False
    if len(_DECORATION.findall(body)) > _DECORATION_CHARS:
        return False
    return any(not ch.isalnum() and not ch.isspace() for ch in body)


def drop_running_heads(pages: list[str], *,
                       max_chars: int = DEFAULT_RUNNING_HEAD_MAX) -> tuple[list[str], list[str]]:
    """剔除跨页重复的页眉页脚（``·232·``、``世界电影 1987 年第 3 期``、水印商号）。

    判据（全部基于**这份文档自己**的统计，不用外部词表 —— 写死"实惠图书之家"这种
    商号名只对这批语料有效，换一批书就失效）：

    1. 出现在页首 ``RUNNING_HEAD_WINDOW`` 行或页尾同样几行；
    2. 同一归一形在多页重复（比例门槛按行分成两档，见下）；
    3. **不含句末标点**（``。！？…``）—— 正文行几乎必然带一个。

    两档比例门槛：

    * 短行（≤ ``max_chars`` 字）：≥ ``RUNNING_HEAD_SHARE``（50%）的页重复即可；
    * 长行（> ``max_chars`` 且 ≤ ``RUNNING_HEAD_LONG_MAX``）：要 ≥ ``RUNNING_HEAD_LONG_SHARE``
      （80%）。原因是实测水印长这样：「· ((&)实惠图书之家独家制作 联系方式 旺旺：…联系QQ：957719200」
      （45 字），按 24 字上限会被直接放行，于是每页都进一次索引、把商号抬成高频词。
      而长行误删的代价是整句正文，所以门槛拉到 80% —— 正文不可能跨页逐字重复。

    返回 ``(清洗后的页, 被删的行)``。
    """
    if len(pages) < RUNNING_HEAD_MIN_PAGES:
        return list(pages), []
    short_counts: dict[str, int] = {}
    long_counts: dict[str, int] = {}
    per_page: list[list[tuple[int, str]]] = []
    for text in pages:
        lines = split_lines(text)
        edges: list[tuple[int, str]] = []
        for index in list(range(min(RUNNING_HEAD_WINDOW, len(lines)))):
            edges.append((index, lines[index]))
        for offset in range(1, min(RUNNING_HEAD_WINDOW, len(lines)) + 1):
            edges.append((len(lines) - offset, lines[len(lines) - offset]))
        kept: list[tuple[int, str]] = []
        seen: set[str] = set()
        for index, line in edges:
            body = line.strip()
            if (not body or len(body) > RUNNING_HEAD_LONG_MAX
                    or _PROSE_PUNCT.search(_body_of(body))):
                continue
            key = _norm_head(body)
            if not key or key in seen:
                continue
            seen.add(key)
            kept.append((index, body))
            bucket = short_counts if len(body) <= max_chars else long_counts
            bucket[key] = bucket.get(key, 0) + 1
        per_page.append(kept)

    total = len(pages)
    short_thr = max(RUNNING_HEAD_MIN_PAGES, int(total * RUNNING_HEAD_SHARE))
    long_thr = max(RUNNING_HEAD_MIN_PAGES, int(total * RUNNING_HEAD_LONG_SHARE + 0.999))
    heads = {key for key, count in short_counts.items() if count >= short_thr}
    heads |= {key for key, count in long_counts.items() if count >= long_thr}
    if not heads:
        return list(pages), []

    cleaned: list[str] = []
    dropped: list[str] = []
    for text, edges in zip(pages, per_page):
        lines = split_lines(text)
        kill: set[int] = set()
        for index, body in edges:
            if _norm_head(body) in heads and 0 <= index < len(lines):
                kill.add(index)
                dropped.append(body)
                #: 水印常印成「花纹 + 商号」两行，花纹那半截被 OCR 认成一串符号
                #: （实测 ``· ((&)`` / ``一((+/``）留在页尾 ⇒ 索引里多一条纯符号噪声。
                for neighbor in (index - 1, index + 1):
                    if (0 <= neighbor < len(lines) and neighbor not in kill
                            and _is_decoration(lines[neighbor])):
                        kill.add(neighbor)
                        dropped.append(lines[neighbor].strip())
        cleaned.append("\n".join(line for index, line in enumerate(lines)
                                 if index not in kill).strip())
    return cleaned, dropped


def clean_document(pages: list[str], *,
                   max_head_chars: int = DEFAULT_RUNNING_HEAD_MAX) -> dict:
    """整份文档清洗：规整+碎行合并 → 跨页剔页眉页脚 → **最后**才整页重排。

    三步顺序不能换（实测踩过，症状极隐蔽）：

    * 重排若排在剔版头之前，页尾水印会被当成本页正文的续接拼到上一行末尾
      （上一行不以句末标点结尾就合并）⇒ 水印行长度从 45 字变成 46/76/136/189 字，
      **每页都不一样** ⇒ ① 归一形各不相同，跨页重复统计凑不够数；② 部分超
      ``RUNNING_HEAD_LONG_MAX`` 直接跳过。结果就是「剔除 0 行」、水印照旧进 fts5，
      而检查时只看到"剔了别的东西"，极难定位。
    * 合并（``merge_broken_lines``）只把孤字/纯标点并回上一行，水印是整行文字，
      在这一步仍是独立一行 ⇒ 此时它的键与长度都是稳定形态。
    """
    merged = [clean_page(text, reflow=False) for text in pages]
    cleaned, dropped = drop_running_heads(merged, max_chars=max_head_chars)
    return {"pages": [reflow_page(text) for text in cleaned], "droppedHeads": dropped}


# ===========================================================================
# 门禁
# ===========================================================================


def page_metrics(text: str) -> dict:
    """量化一页的可疑程度（纯计量，不作判定）。"""
    body = text or ""
    chars = len(body.strip())
    if not chars:
        return {"chars": 0, "lines": 0, "lineMean": 0.0, "distinct": 0, "distinctRatio": 0.0,
                "cjkRatio": 0.0, "validRatio": 0.0, "junkRatio": 0.0, "junkRuns": 0,
                "junkRunChars": 0, "maxRun": 0, "junkRunShare": 0.0}
    lines = [line for line in split_lines(body) if line]
    countable = len(body.replace("\n", ""))
    runs = _JUNK_RUN.findall(body)
    run_chars = sum(len(run) for run in runs)
    return {
        "chars": chars,
        "lines": len(lines),
        "lineMean": round(countable / max(1, len(lines)), 2),
        "distinct": len(set(body)),
        "distinctRatio": round(len(set(body)) / max(1, countable), 3),
        #: 两个比例都按**去掉换行后的字数**算，否则长页会因为换行多而虚高
        "cjkRatio": round(len(_CJK.findall(body)) / max(1, countable), 3),
        "validRatio": round(len(_VALID.findall(body)) / max(1, countable), 3),
        "junkRatio": round(len(_HARD_JUNK.findall(body)) / max(1, countable), 4),
        "junkRuns": len(runs),
        #: ⚠️ 光有「几处」会判错：764 字的正常页里混进一处表格线/破折号，和整页被
        #: 认成一排 ``||||`` 是两回事。**占比**才是能不能用的判据，所以要长度。
        "junkRunChars": run_chars,
        "maxRun": max((len(run) for run in runs), default=0),
        "junkRunShare": round(run_chars / max(1, countable), 4),
    }


def judge_page(text: str, *, allow_cjk_free: bool = True) -> dict:
    """给一页定性：``{"verdict": ok|suspect|reject, "reasons": [...], "metrics": {...}}``。

    ``allow_cjk_free`` 默认 True：英文原版剧本、纯数字表格页 CJK 占比为 0 是**合法**的，
    中文占比只作报告用，**不参与判定**（否则整批英文剧本会被误杀）。
    """
    metrics = page_metrics(text)
    reasons: list[str] = []
    reject = False
    if metrics["chars"] == 0:
        return {"verdict": "ok", "reasons": ["空白页"], "metrics": metrics}

    if metrics["junkRatio"] > FFFD_REJECT:
        reject = True
        reasons.append(f"非法字符占比 {metrics['junkRatio']:.2%}")
    elif metrics["junkRatio"] > 0:
        reasons.append(f"含非法字符 {metrics['junkRatio']:.2%}")
    if metrics["junkRuns"] and metrics["maxRun"] >= JUNK_RUN_MAX_REJECT:
        #: 一整段被认成符号（插图/表格整块）—— 与占比无关，这段里面的字全没了
        reject = True
        reasons.append(f"符号连续段最长 {metrics['maxRun']} 字（整段被识别成符号）")
    elif metrics["junkRuns"] and metrics["junkRunShare"] >= JUNK_RUN_SHARE_REJECT:
        reject = True
        reasons.append(f"符号连续段占 {metrics['junkRunShare']:.0%}"
                       f"（{metrics['junkRuns']} 处）")
    elif metrics["junkRuns"]:
        #: 长页里零星几处表格线/破折号：**不判废**。实测 764 字的正常评论页因为
        #: 一处 8 字符号段被整页丢弃，764 字正文一个字都没进 fts5 —— 假阳性丢内容
        #: 比留一点噪声严重得多。记进预览，让人在 --audit 里能看见。
        reasons.append(f"局部符号连续段 {metrics['junkRuns']} 处"
                       f"（共 {metrics['junkRunChars']} 字，不判废）")
    if metrics["validRatio"] < VALID_REJECT:
        reject = True
        reasons.append(f"有效字符占比 {metrics['validRatio']:.0%}")
    elif metrics["validRatio"] < VALID_SUSPECT:
        reasons.append(f"有效字符占比偏低 {metrics['validRatio']:.0%}")
    long_enough = metrics["chars"] >= DISTINCT_MIN_CHARS
    if long_enough and metrics["distinct"] < DISTINCT_REJECT:
        reject = True
        reasons.append(f"整页只有 {metrics['distinct']} 个不同字符")
    elif (long_enough and metrics["cjkRatio"] >= DISTINCT_CJK_MIN
          and metrics["distinctRatio"] < DISTINCT_SUSPECT):
        #: CJK 门槛见 ``DISTINCT_CJK_MIN``：拉丁文页面的去重比天然个位数，
        #: 不设门槛就是「每本英文书整份进 --audit」。
        reasons.append(f"去重比 {metrics['distinctRatio']:.0%}（疑大量重复）")
    if metrics["lineMean"] < LINE_MEAN_SUSPECT:
        reasons.append(f"行均 {metrics['lineMean']} 字（碎行或竖排，读序可疑）")

    if reject:
        return {"verdict": "reject", "reasons": reasons, "metrics": metrics}
    return {"verdict": "suspect" if reasons else "ok", "reasons": reasons, "metrics": metrics}
