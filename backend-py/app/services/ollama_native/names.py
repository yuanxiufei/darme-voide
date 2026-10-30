"""模型名的解析与规范化 —— 逐函数移植 ``reference/ollama/types/model/name.go``。

ollama 的模型名允许 12 种写法（``llama3`` / ``user/llama3`` /
``localhost:5000/ns/llama3:v2`` …），落盘时又必须变成
``manifests/<host>/<namespace>/<model>/<tag>`` 四段路径。这套解析是 pull / push /
create / show / delete 的**共同前置**，做错任何一个 corner case 都会让模型「找得到名字却读不到文件」。

⚠️ 移植时的两个容易错的点（都按 Go 原文保留）：

1. 切分方向一律**从右往左**（``cutPromised`` 用 ``LastIndex``）。所以
   ``a/b/c`` 会被切成 host=``a`` / namespace=``b`` / model=``c``，
   而不是「第一个 / 之前是 host」。
2. 空段不是报错而是填 ``MissingPart``（``!MISSING!``）—— 让
   ``!``/``!`` 这种非法字符在 ``IsValid`` 阶段自然暴露，而不是在解析阶段就抛异常。
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_HOST = "registry.ollama.ai"
DEFAULT_NAMESPACE = "library"
DEFAULT_TAG = "latest"
DEFAULT_PROTOCOL_SCHEME = "https"

#: 由分隔符「承诺存在」但实际缺失的段。取值与 Go 侧一致（便于照抄日志与测试）。
MISSING_PART = "!MISSING!"


class NameError_(ValueError):
    """名字不合法（未完全限定）。命名上带下划线是为了不和内置 ``NameError`` 撞名。"""


@dataclass(frozen=True)
class Name:
    """结构化模型名。字段**不保证合法**，用 :meth:`is_valid` 判断。

    ``frozen=True`` 不只是「值对象」的洁癖：上游把 ``Name`` 当 **map 的键**
    （``map[model.Name]*Manifest``），``all_manifests()`` 同样需要它可哈希。
    """

    host: str = ""
    namespace: str = ""
    model: str = ""
    tag: str = ""
    protocol_scheme: str = ""

    # ── 构造 ────────────────────────────────────────────────────────────
    @classmethod
    def default(cls) -> "Name":
        return cls(
            host=DEFAULT_HOST,
            namespace=DEFAULT_NAMESPACE,
            tag=DEFAULT_TAG,
            protocol_scheme=DEFAULT_PROTOCOL_SCHEME,
        )

    # ── 渲染 ────────────────────────────────────────────────────────────
    def __str__(self) -> str:
        parts: list[str] = []
        if self.host:
            parts.append(self.host + "/")
        if self.namespace:
            parts.append(self.namespace + "/")
        parts.append(self.model)
        if self.tag:
            parts.append(":" + self.tag)
        return "".join(parts)

    def display_shortest(self) -> str:
        """尽量短的显示形式（``llama3:latest`` / ``user/x:tag`` / ``host/ns/x:tag``）。"""
        out: list[str] = []
        if self.host.lower() != DEFAULT_HOST.lower():
            out.append(f"{self.host}/{self.namespace}/")
        elif self.namespace.lower() != DEFAULT_NAMESPACE.lower():
            out.append(f"{self.namespace}/")
        out.append(f"{self.model}:{self.tag}")
        return "".join(out)

    # ── 校验 ────────────────────────────────────────────────────────────
    def is_fully_qualified(self) -> bool:
        for kind, part in (
            ("host", self.host),
            ("namespace", self.namespace),
            ("model", self.model),
            ("tag", self.tag),
        ):
            if not is_valid_part(kind, part):
                return False
        return True

    def is_valid(self) -> bool:
        # 上游已移除 digest 校验，这里只判四段
        return self.is_fully_qualified()

    def require_valid(self) -> "Name":
        if not self.is_valid():
            raise NameError_(f"unqualified name: {self}")
        return self

    def equal_fold(self, other: "Name") -> bool:
        return (
            self.host.lower() == other.host.lower()
            and self.namespace.lower() == other.namespace.lower()
            and self.model.lower() == other.model.lower()
            and self.tag.lower() == other.tag.lower()
        )

    # ── 落盘 ────────────────────────────────────────────────────────────
    def filepath(self) -> str:
        """``{host}/{namespace}/{model}/{tag}``（相对 ``manifests/`` 的路径，正斜杠）。"""
        self.require_valid()
        return f"{self.host}/{self.namespace}/{self.model}/{self.tag}"

    def parts(self) -> tuple[str, str, str, str]:
        return self.host, self.namespace, self.model, self.tag

    def base_url(self) -> str:
        return f"{self.protocol_scheme}://{self.host}"


def default_name() -> Name:
    return Name.default()


def merge(primary: Name, fallback: Name) -> Name:
    """合并：``primary`` 里有值的取它，空段取 ``fallback`` 的（对应 Go 的 ``cmp.Or`` 链）。"""
    return Name(
        host=primary.host or fallback.host,
        namespace=primary.namespace or fallback.namespace,
        model=primary.model or fallback.model,
        tag=primary.tag or fallback.tag,
        protocol_scheme=primary.protocol_scheme or fallback.protocol_scheme,
    )


def _cut_last(text: str, separator: str) -> tuple[str, str, bool]:
    index = text.rfind(separator)
    if index >= 0:
        return text[:index], text[index + len(separator) :], True
    return text, "", False


def _cut_promised(text: str, separator: str) -> tuple[str, str, bool]:
    """从右侧切一段；切出来的空段填 ``MISSING_PART``（让它在校验阶段暴露）。"""
    before, after, ok = _cut_last(text, separator)
    if not ok:
        return before, after, False
    return before or MISSING_PART, after or MISSING_PART, True


def parse_name_bare(text: str) -> Name:
    """不叠加默认值的解析（``llama3`` 解析完 tag 就是空的）。"""
    rest = text
    tag = ""

    # "/" 不可能是 tag 的合法字符 ⇒ 用「最后一个 : 是否在最后一个 / 之后」判断有没有 tag
    if rest.rfind(":") > rest.rfind("/"):
        rest, tag, _ = _cut_promised(rest, ":")

    rest, model, promised = _cut_promised(rest, "/")
    if not promised:
        return Name(model=rest, tag=tag)

    rest, namespace, promised = _cut_promised(rest, "/")
    if not promised:
        return Name(namespace=rest, model=model, tag=tag)

    if "://" in rest:
        scheme, _, host = rest.partition("://")
    else:
        scheme, host = "", rest
    return Name(
        host=host,
        namespace=namespace,
        model=model,
        tag=tag,
        protocol_scheme=scheme,
    )


def parse_name(text: str) -> Name:
    """解析并补齐默认值（host=registry.ollama.ai / nd=library / tag=latest）。"""
    return merge(parse_name_bare(text), default_name())


def parse_name_from_filepath(path: str) -> Name:
    """把 ``host/ns/model/tag`` 四段路径解析回 ``Name``（段数不对返回空 Name）。"""
    parts = path.replace("\\", "/").split("/")
    if len(parts) != 4:
        return Name()
    name = Name(host=parts[0], namespace=parts[1], model=parts[2], tag=parts[3])
    if not name.is_fully_qualified():
        return Name()
    return name


def is_valid_namespace(text: str) -> bool:
    return is_valid_part("namespace", text)


def is_valid_part(kind: str, text: str) -> bool:
    """逐字符校验（规则见 Go 的 ``isValidPart``）。

    长度一律按 **UTF-8 字节数** 算，与 Go 的 ``len(s)`` 一致。
    """
    length = len(text.encode("utf-8"))
    if kind == "host":
        if not 1 <= length <= 350:
            return False
    elif not 1 <= length <= 80:
        return False

    for index, char in enumerate(text):
        if index == 0:
            if not _is_alphanumeric_or_underscore(char):
                return False
            continue
        if char in ("_", "-"):
            continue
        if char == ".":
            if kind == "namespace":
                return False
            continue
        if char == ":":
            if kind not in ("host", "digest"):
                return False
            continue
        if not _is_alphanumeric_or_underscore(char):
            return False
    return True


def _is_alphanumeric_or_underscore(char: str) -> bool:
    return (
        ("a" <= char <= "z")
        or ("A" <= char <= "Z")
        or ("0" <= char <= "9")
        or char == "_"
    )


__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_NAMESPACE",
    "DEFAULT_PROTOCOL_SCHEME",
    "DEFAULT_TAG",
    "MISSING_PART",
    "Name",
    "NameError_",
    "default_name",
    "is_valid_namespace",
    "is_valid_part",
    "merge",
    "parse_name",
    "parse_name_bare",
    "parse_name_from_filepath",
]
