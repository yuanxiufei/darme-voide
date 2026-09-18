"""**占位符解析**（把 ``<location>L1</location>`` 解析成**具体内容** ✓ 零依赖 ✓）。

## 为什么需要它（本项目原来只有"剥标签" ✗）

``prompt_utils.strip_video_prompt_tags`` 做的是**剥掉标签、留下中间内容** ✓ ——
于是 ``<location>L1</location>`` 变成裸的 ``L1`` ✗，``<role>R5</role>`` 变成 ``R5`` ✗。
可扩散模型**不认识** ``L1``/``R5`` 这种内部编号 ✗ ⇒ 它们成了**无意义字符**（占注意力 ✓ 还可能被当成文字生成 ✗）。

参考项目（``reference/short-drama-agent`` 的 ``tagged-storyboard-format.md``）给出的正解是 **解析**：
先把 Location/Role/Prop/Clue 的**编号 → 具体描述**建表 ✓，再把占位符**替换成自然语言** ✓，
并且**最终提示词里不留 XML 标签** ✓（除非目标接口明确要求 ✓）。

本模块就做这件事 ✓，并额外把**四类"看不出来的坑"**如实报出来 ✗：

1. **未映射的编号** ⇒ 逐个列出 ✓（不静默留个 ``L9`` 在提示词里 ✗）；
2. **残留标签** ⇒ 输出后自查一遍 ✓（有就报 ✓ —— 送进模型才知道就晚了 ✗）；
3. **时长换算** ⇒ ``<duration-ms>6000</duration-ms>`` 一律走**统一模板** ✓（毫秒/秒两套写法混用最乱 ✗）；
4. **编号裸奔** ⇒ 若替换后仍出现形如 ``L1``/``R5`` 的**裸编号**（映射表里没有的 ✓）也报出来 ✓。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = ["PlaceholderMaps", "ResolveResult", "duration_text", "find_bare_ids",
           "resolve_placeholders"]

#: 占位符种类（与参考项目一致 ✓）；``duration-ms`` 单独处理 ✓（它是**数值**不是编号 ✓）
KINDS: tuple[str, ...] = ("location", "role", "prop", "clue")

#: ``<location>L1</location>`` —— 用**反向引用**保证开闭标签同名 ✓（写错时不误吞文本 ✓）
_TAG_RE = re.compile(r"<(?P<kind>[a-z][a-z0-9-]*)>(?P<body>[^<]*)</(?P=kind)>")

#: ⚠️ **残留检测**必须用**宽松**模式 ✓：只认上面那个严格正则的话 ✓，
#: ``<location>L1</role>``（闭合写错 ✓）两边都匹配不上 ⇒ 畸形标签会被当成「没有残留」✗✗
#: （自检 ⑩ 当场抓到 ✓）。**能匹配"任何像标签的东西"** 才是合格的体检 ✓。
_ANY_TAG_RE = re.compile(r"</?[a-z][a-z0-9-]*\s*/?>", re.IGNORECASE)

#: 裸编号：``L1`` / ``R5`` / ``P3`` / ``C2`` ✓（映射表没覆盖到就会残留 ⇒ 要报 ✓）
_BARE_ID_RE = re.compile(r"(?<![A-Za-z0-9])([LRPC])(\d{1,3})(?![A-Za-z0-9])")

#: 各前缀 → 种类 ✓（``L``=location ✓、``R``=role ✓、``P``=prop ✓、``C``=clue ✓）
_PREFIX_TO_KIND: dict[str, str] = {"l": "location", "r": "role", "p": "prop", "c": "clue"}


@dataclass(frozen=True)
class PlaceholderMaps:
    """编号 → **具体描述** ✓（描述里不要再放编号 ✗ —— 那就等于没解析 ✓）。"""

    locations: dict[str, str] = field(default_factory=dict)
    roles: dict[str, str] = field(default_factory=dict)
    props: dict[str, str] = field(default_factory=dict)
    clues: dict[str, str] = field(default_factory=dict)

    def table_for(self, kind: str) -> dict[str, str]:
        return {"location": self.locations, "role": self.roles,
                "prop": self.props, "clue": self.clues}.get(kind, {})

    def lookup(self, kind: str, value: str) -> str | None:
        table = self.table_for(kind)
        text = str(value).strip()
        if text in table:
            return str(table[text])
        # 容忍大小写（`l1` / `L1` ✓）—— 但**不**去猜前缀含义 ✗
        for key, mapped in table.items():
            if str(key).strip().lower() == text.lower():
                return str(mapped)
        return None

    def to_dict(self) -> dict[str, Any]:
        return {"locations": dict(self.locations), "roles": dict(self.roles),
                "props": dict(self.props), "clues": dict(self.clues),
                "sizes": {kind: len(self.table_for(kind)) for kind in KINDS}}


@dataclass
class ResolveResult:
    """解析结果 ✓（``text`` 可直接送模型 ✓，其余是**给人看的诊断** ✓）。"""

    text: str = ""
    used: list[dict[str, str]] = field(default_factory=list)
    unresolved: list[dict[str, str]] = field(default_factory=list)
    durationsMs: list[int] = field(default_factory=list)
    residualTags: list[str] = field(default_factory=list)
    bareIds: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """**可以放心送模型** ⇔ 没有未映射编号、没有残留标签 ✓（裸编号只算警告 ✓）。"""
        return not self.unresolved and not self.residualTags

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "ok": self.ok, "used": self.used,
                "unresolved": self.unresolved, "durationsMs": self.durationsMs,
                "residualTags": self.residualTags, "bareIds": self.bareIds,
                "problems": self.problems}


def duration_text(ms: int, *, template: str = "目标时长{ms}毫秒") -> str:
    """毫秒 → 统一的中文时长写法 ✓（**一处定义** ✓ —— 免得"6000"和"6 秒"混着写 ✓）。"""
    return str(template).format(ms=int(ms), seconds=round(int(ms) / 1000.0, 3))


def find_bare_ids(text: str) -> list[str]:
    """找出**裸编号** ✓（``L1``/``R5``/``P3``/``C2`` 这种内部编号本身不该出现在提示词里 ✗）。"""
    return sorted({f"{match.group(1)}{match.group(2)}" for match in _BARE_ID_RE.finditer(text or "")})


def resolve_placeholders(text: str, maps: PlaceholderMaps, *,
                         duration_template: str = "目标时长{ms}毫秒",
                         keep_unmapped: bool = True,
                         flag_bare_ids: bool = True) -> ResolveResult:
    """把占位符替换成**具体内容** ✓。

    ``keep_unmapped=True``（默认 ✓）⇒ 未映射的编号**去掉标签但保留原文** ✓
    （至少不会把 XML 送进模型 ✗），同时在 ``unresolved`` 里逐条报出 ✓。
    """
    result = ResolveResult()
    source = str(text or "")

    def _replace(match: re.Match[str]) -> str:
        kind = match.group("kind").lower()
        body = match.group("body").strip()
        if kind == "duration-ms":
            try:
                ms = int(float(body))
            except (TypeError, ValueError):
                result.problems.append(f"<duration-ms> 不是数字：{body!r} ✓（已原样保留 ✓）")
                return body
            result.durationsMs.append(ms)
            return duration_text(ms, template=duration_template)
        if kind not in KINDS:
            result.problems.append(f"未知占位符种类 <{kind}> ✓（已原样保留内容 ✓）")
            return body
        mapped = maps.lookup(kind, body)
        if mapped is None:
            result.unresolved.append({"kind": kind, "value": body})
            return body if keep_unmapped else ""
        result.used.append({"kind": kind, "value": body, "text": mapped})
        return mapped

    resolved = _TAG_RE.sub(_replace, source)

    # 输出的**自查**（送进模型前就要知道 ✓）—— 用**宽松**模式 ✓（括写错的也要抓到 ✓）
    result.residualTags = sorted({match.group(0) for match in _ANY_TAG_RE.finditer(resolved)})
    if result.residualTags:
        result.problems.append(
            f"输出里仍有 {len(result.residualTags)} 个类标签片段 ✗ ⇒ 扩散模型不认识它们 ✓"
            f"（例如 {result.residualTags[:3]} ✓ —— 常见原因是**开闭标签写错/嵌套/跨行** ✓）")
    if flag_bare_ids:
        result.bareIds = find_bare_ids(resolved)
        if result.bareIds:
            result.problems.append(
                f"提示词里还有**裸编号** {result.bareIds[:4]} ✗ ⇒ 模型不认识这些内部编号 ✓"
                f"（要么补进映射表 ✓，要么从提示词里删掉 ✓）")
    if result.unresolved:
        result.problems.append(
            f"{len(result.unresolved)} 个编号没有映射 ✗ ⇒ 已保留原文但**没有具体内容** ✓"
            f"（例如 {result.unresolved[:2]} ✓）")

    # 收尾：压缩空白 ✓（替换进来的是长句 ✓，容易留下多余空格/空行 ✓）
    result.text = re.sub(r"[ \t]+", " ", resolved)
    result.text = re.sub(r"\s*\n\s*", "\n", result.text).strip()
    return result
