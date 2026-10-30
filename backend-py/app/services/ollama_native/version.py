"""**版本号** —— 上游 ``version/version.go`` 的对等实现 + 一处本仓扩展。

上游只有一个可被 ``-ldflags -X`` 注入的变量；服务端另有一个 ``.ollama-version`` 文件
（安装包写入，``ollama serve`` 启动时读它做自检）。这里两样都给。

扩展部分（上游没有）：:func:`compare` —— 本仓要通过 HTTP 跟外部 ``ollama serve`` 对话，
版本低于某条线时接口形状会不一样（如 ``/api/chat`` 的 ``thinking`` 字段、harmony 工具调用），
判定「对端够不够新」必须有可比函数，否则只能字符串比大小。
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = ["Version", "compare", "load_from_file", "parse_version"]

#: 上游默认值 ``"0.0.0"``；发布构建由 ldflags 注入，本仓用常量代替。
Version = "0.0.0"

_VERSION_FILE = ".ollama-version"
_SEMVER = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?$")


def parse_version(text: str) -> tuple[int, int, int, tuple[str, ...]] | None:
    """解析 ``v1.2.3-rc1`` 这类串；不合法返回 ``None``（不猜、不抛）。"""
    match = _SEMVER.match(text.strip())
    if not match:
        return None
    major, minor, patch = (int(match.group(i)) for i in (1, 2, 3))
    prerelease = tuple((match.group(4) or "").split(".")) if match.group(4) else ()
    return major, minor, patch, prerelease


def compare(left: str, right: str) -> int:
    """语义化比较：``<0`` 左小、``0`` 相等、``>0`` 左大。

    预发布版本比同号正式版**小**（``1.2.3-rc1 < 1.2.3``），这是 semver 的硬规定 ——
    按字符串比会把 ``1.2.3-rc1`` 判成「比 1.2.3 新」，正好反了。
    """
    a = parse_version(left)
    b = parse_version(right)
    if a is None or b is None:
        raise ValueError(f"无法解析版本号：{left!r} / {right!r}")

    if a[:3] != b[:3]:
        return -1 if a[:3] < b[:3] else 1
    if a[3] == b[3]:
        return 0
    if not a[3]:
        return 1            # 正式版 > 预发布
    if not b[3]:
        return -1
    return -1 if a[3] < b[3] else 1


def load_from_file(root: str | Path | None = None) -> str:
    """读 ``.ollama-version``（安装包写在 ollama 主目录里）。读不到就返回 :data:`Version`。"""
    path = Path(root) if root is not None else Path.home() / ".ollama"
    if path.is_dir() or not path.suffix:
        path = path / _VERSION_FILE
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return Version
    return text or Version
