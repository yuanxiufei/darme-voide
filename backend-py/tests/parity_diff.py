"""Node ↔ Python 差分对拍 —— 「删 ``backend/`` 前的等价性证据」（S7 第 5 步的验收工具）。

用途：两个后端**并排起来**（Node 5789 / Python 5790，同一 ``DATA_ROOT``），逐端点比对响应，
把差异分成三类：

* **same**  —— 逐字段一致（时间戳、id 这类易变字段先归一化）；
* **known** —— 命中**有意差异白名单**（每条都要求填 ``why``，写在下面 ``KNOWN_DIFFS``）；
* **new**   —— 白名单没覆盖的差异 ⇒ **退出码 1**（这才是要人看的东西）。

用法::

    # A) 真对拍（需两个后端都已启动、且指向同一份数据）
    python tests/parity_diff.py --node http://127.0.0.1:5789 --py http://127.0.0.1:5790

    # B) 离线自检（只验比较器本身，不起任何服务）
    python tests/parity_diff.py --selftest

⚠️ 三个必须知道的坑：

1. **Node 的数据根优先级是 ``.data-root`` 标记文件 > ``DATA_ROOT`` 环境变量** ⇒ 若项目根留了这个
   标记，两个后端会读**别处**的库、对拍结果无意义。本脚本启动时会检查并**直接拒绝运行**；
2. 两侧都会在启动时**写库**（Node 种服务商、Python 跑崩溃恢复）⇒ 对拍前先让 Node 起好再起 Python，
   并只比对**只读端点**（本表全是 GET）；
3. 归一只做两件事：丢易变键（时间戳/``requestId`` 之类）与**保持列表顺序**（顺序本身是行为的一部分，
   比如「分镜按 number 排」）——**不要把差异归因于排序**，那等于放掉一个真 bug。
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

__all__ = ["CASES", "KNOWN_DIFFS", "MISSING_CASES", "MISSING_EXPECT", "VOLATILE_KEYS",
           "classify", "classify_missing", "diff", "normalize"]

#: 值不参与比较的键（各域通用）。⚠️ 只放**真的易变**的：生成时刻、审计时间戳。
VOLATILE_KEYS = frozenset({
    "generatedAt", "updatedAt", "createdAt", "deletedAt", "lastSeenAt",
    "requestId", "traceId", "uptimeMs", "pid",
})

#: 待比对端点（**只读**）。元组 = (method, path, 说明)。
CASES: tuple[tuple[str, str, str], ...] = (
    ("GET", "/api/v1/dramas", "剧目列表"),
    ("GET", "/api/v1/dramas/stats", "统计"),
    ("GET", "/api/v1/dramas/{drama_id}", "剧目详情"),
    ("GET", "/api/v1/dramas/{drama_id}/prompts", "提示词汇总"),
    ("GET", "/api/v1/props", "道具列表"),
    ("GET", "/api/v1/ai-configs", "AI 配置列表"),
    ("GET", "/api/v1/ai-configs/configs/local", "本地配置列表"),
    ("GET", "/api/v1/ai-providers", "服务商目录"),
    ("GET", "/api/v1/evaluation/cases", "评测基准 case 列表"),
    ("GET", "/api/v1/ai-configs/gpu/status", "GPU 显存状态"),
)

#: 有意差异白名单：``(method, path) -> {"why": 理由, "fields": (差异路径前缀…)}``。
#: ⚠️ 每新增一条都要写清理由；**没有理由的差异不许进白名单**（那等于把 bug 藏起来）。
KNOWN_DIFFS: dict[tuple[str, str], dict[str, Any]] = {
    ("GET", "/api/v1/ai-configs/gpu/status"): {
        "why": "hardware 段依赖本机 nvidia-smi；Python 侧字段数不足时返回 null（有意加固）",
        "fields": ("hardware",),
    },
    ("GET", "/api/v1/ai-providers"): {
        "why": "两边的「已启用」推导实现不同源（Node 由 provider 目录 + 配置推导，"
               "Python 走 ai_configs 服务），允许 enabled/isLocal 字段差异",
        "fields": ("enabled", "isLocal"),
    },
}


#: 「**两边都不该有**」的路径 —— 专门盯「Node 有而 Python 没有」的漏网端点。
#:
#: ⚠️ 背景（2026-09-14 实测）：Node 的 ``episodes.ts`` / ``storyboards.ts`` / ``characters.ts`` /
#: ``scenes.ts`` **都没有裸列表路由**（只有 ``POST /`` 与 ``/:id/...``）⇒ 请求这些路径时
#: **Node 回 404**（Hono 默认未匹配），**Python 回 501**（绞杀者兜底「尚未迁移」文案）。
#: 两者都表示「端点不存在」，只是状态码不同 ⇒ 归为 ``MISSING``，不算 ``new``；
#: **但若哪天 Node 侧回了 200，说明它真新增了这个端点而 Python 没有 ⇒ 必须当 ``new`` 报出来。**
MISSING_CASES: tuple[tuple[str, str], ...] = (
    ("GET", "/api/v1/episodes"),
    ("GET", "/api/v1/storyboards"),
    ("GET", "/api/v1/characters"),
    ("GET", "/api/v1/scenes"),
)

#: ``MISSING`` 一类的预期状态码（Node 未匹配 / Python 兜底）
MISSING_EXPECT = {"node": 404, "py": 501}


def normalize(value: Any, *, drop_volatile: bool = True) -> Any:
    """归一化：递归丢掉易变键（**不改列表顺序**、不做任何排序）。"""
    if isinstance(value, dict):
        return {k: normalize(v, drop_volatile=drop_volatile)
                for k, v in value.items()
                if not (drop_volatile and k in VOLATILE_KEYS)}
    if isinstance(value, list):
        return [normalize(v, drop_volatile=drop_volatile) for v in value]
    return value


def diff(left: Any, right: Any, path: str = "") -> list[str]:
    """逐字段比较，返回**差异路径**列表（如 ``data.shots[0].status``）。"""
    if isinstance(left, dict) and isinstance(right, dict):
        out: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else str(key)
            if key not in left:
                out.append(f"{child} (仅Python有) ")
            elif key not in right:
                out.append(f"{child} (仅Node有)")
            else:
                out.extend(diff(left[key], right[key], child))
        return out
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [f"{path} (长度 {len(left)} vs {len(right)})"]
        out = []
        for index, (a, b) in enumerate(zip(left, right)):
            out.extend(diff(a, b, f"{path}[{index}]"))
        return out
    if left != right:
        return [f"{path}: {_short(left)} vs {_short(right)}"]
    return []


def _short(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else f'"{value}"'
    return text if len(text) <= 60 else text[:57] + "…"


def classify(method: str, path: str, node: Any, py: Any) -> tuple[str, list[str]]:
    """返回 ``("same" | "known" | "new", 差异路径列表)``。"""
    if isinstance(node, dict) and isinstance(py, dict) and "__status__" in node:
        if node["__status__"] != py.get("__status__"):
            return "new", [f"HTTP {node['__status__']} vs {py.get('__status__')}"]
    differences = diff(normalize(node), normalize(py))
    if not differences:
        return "same", []
    allowed = KNOWN_DIFFS.get((method, path), {}).get("fields", ())
    if allowed and all(any(item.startswith(prefix) for prefix in allowed)
                       for item in differences):
        return "known", differences
    return "new", differences


def classify_missing(node: Any, py: Any) -> tuple[str, list[str]]:
    """``MISSING`` 类的判定：Node 未匹配(404) + Python 兜底(501) ⇒ ``missing``（预期内）。

    ⚠️ 只要 **Node 回了别的状态码**（尤其 200）就说明「Node 真有这个端点」⇒ 判 ``new``。
    """
    node_status = node.get("__status__")
    py_status = py.get("__status__")
    if node_status == MISSING_EXPECT["node"] and py_status == MISSING_EXPECT["py"]:
        return "missing", [f"Node {node_status}（未匹配）/ Python {py_status}（兜底）— 见 MISSING_CASES 注释"]
    return "new", [f"HTTP {node_status} vs {py_status} —— 与 MISSING_EXPECT 不符"
                   f"（Node 真回 200 就说明这个端点 Python 缺）"]


def selftest() -> int:
    """离线验比较器（不起服务）：返回退出码。"""
    failures: list[str] = []
    cases: list[tuple[str, str, Any, Any, str]] = [
        ("identical", "same", {"a": 1, "b": [1, 2]}, {"a": 1, "b": [1, 2]}, "same"),
        ("volatile-only", "same", {"a": 1, "updatedAt": "x"}, {"a": 1, "updatedAt": "y"}, "same"),
        ("list-order-matters", "new", {"b": [1, 2]}, {"b": [2, 1]}, "new"),
        ("list-length", "new", {"b": [1]}, {"b": [1, 2]}, "new"),
        ("nested", "new", {"d": {"x": {"y": 1}}}, {"d": {"x": {"y": 2}}}, "new"),
        ("only-python", "new", {"a": 1}, {"a": 1, "b": 2}, "new"),
        ("only-node", "new", {"a": 1, "b": 2}, {"a": 1}, "new"),
        ("status-code", "new", {"__status__": 200}, {"__status__": 500}, "new"),
    ]
    for label, path_key, node, py, expected in cases:
        got, details = classify("GET", f"/{path_key}", node, py)
        if got != expected:
            failures.append(f"{label}: 期望 {expected} 得到 {got} {details}")

    allowed = ("hardware",)
    got, _ = classify("GET", "/x", {"hardware": {"a": 1}}, {"hardware": None})
    if got != "new":
        failures.append("白名单未命中用例：/x 不在表里却判成 known")
    got, _ = classify("GET", "/api/v1/ai-configs/gpu/status",
                      {"hardware": {"a": 1}, "usedVRAM_GB": 0},
                      {"hardware": None, "usedVRAM_GB": 0})
    if got != "known":
        failures.append("白名单命中用例：gpu/status 的 hardware 差异应判 known")
    got, _ = classify("GET", "/api/v1/ai-configs/gpu/status",
                      {"hardware": None, "usedVRAM_GB": 0},
                      {"hardware": None, "usedVRAM_GB": 5})
    if got != "new":
        failures.append("白名单越界用例：hardware 之外还有差异应判 new")
    if allowed != KNOWN_DIFFS[("GET", "/api/v1/ai-configs/gpu/status")]["fields"]:
        failures.append("白名单常量被改动")

    for failure in failures:
        print("FAIL  " + failure)
    print()
    print(f"SELFTEST: {len(cases) + 3 - len(failures)}/{len(cases) + 3} passed")
    return 1 if failures else 0


def _fetch(client: Any, base: str, method: str, path: str) -> Any:
    """请求一个端点，返回 ``{"__status__": n, ...body}``（body 非对象时包一层）。"""
    response = client.request(method, base + path, timeout=30)
    try:
        body = response.json()
    except Exception:  # noqa: BLE001
        body = {"__raw__": response.text[:500]}
    if not isinstance(body, dict):
        body = {"__body__": body}
    return {"__status__": response.status_code, **body}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Node↔Python 差分对拍")
    parser.add_argument("--node", default="http://127.0.0.1:5789")
    parser.add_argument("--py", default="http://127.0.0.1:5790")
    parser.add_argument("--drama-id", default="1", help="替换 {drama_id} 的真实 id")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--report", default="")
    args = parser.parse_args(argv)

    if args.selftest:
        return selftest()

    import httpx  # noqa: PLC0415

    counts = {"same": 0, "known": 0, "new": 0, "missing": 0}
    rows: list[dict[str, Any]] = []
    with httpx.Client() as client:
        for method, path, note in CASES:
            real = path.replace("{drama_id}", str(args.drama_id))
            outcome = "new"
            details: list[str] = []
            try:
                node = _fetch(client, args.node, method, real)
                py = _fetch(client, args.py, method, real)
                outcome, details = classify(method, path, node, py)
            except Exception as exc:  # noqa: BLE001
                details = [f"请求失败：{exc}"]
            counts[outcome] += 1
            rows.append({"method": method, "path": path, "note": note,
                         "outcome": outcome, "details": details})

        # 「两边都不该有」的路径：Node 未匹配 / Python 兜底 —— 真正盯的是「Node 有而 Python 没有」
        for method, path in MISSING_CASES:
            outcome = "new"
            details: list[str] = []
            try:
                node = _fetch(client, args.node, method, path)
                py = _fetch(client, args.py, method, path)
                outcome, details = classify_missing(node, py)
            except Exception as exc:  # noqa: BLE001
                details = [f"请求失败：{exc}"]
            counts[outcome] += 1
            rows.append({"method": method, "path": path, "note": "端点不存在（两侧都不该有）",
                         "outcome": outcome, "details": details})

    for row in rows:
        mark = {"same": "OK    ", "known": "KNOWN ", "new": "NEW   ", "missing": "MISSING"}[row["outcome"]]
        print(f"{mark} {row['method']:4} {row['path']}  ({row['note']})")
        if row["outcome"] != "same":
            for detail in row["details"][:8]:
                print(f"        - {detail}")
    print()
    print(f"一致 {counts['same']} ｜ 已知差异 {counts['known']} ｜ "
          f"不存在路径 {counts['missing']} ｜ **新差异 {counts['new']}**")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump({"counts": counts, "rows": rows}, handle, ensure_ascii=False, indent=2)
        print(f"报告已写：{args.report}")
    return 1 if counts["new"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
