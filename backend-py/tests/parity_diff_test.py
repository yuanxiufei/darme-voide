"""S7 自检：差分对拍**比较器**（``tests/parity_diff.py``）。

对拍本身要两个后端都在跑（Node 5789 / Python 5790），所以那部分**不在这里跑**；这里锁的是
比较器的**判定语义**与**两张表的自洽**——这些错了，「一致/已知/新差异」的结论就不可信：

1. 归一化只丢**易变键**，**绝不排序**（列表顺序是行为的一部分，放掉它就等于放掉一个真 bug）；
2. 差异路径精确到字段（``data.shots[0].status`` 这种），便于定位；
3. 白名单**只覆盖它列出的字段前缀**，越界必须判 ``new``；
4. 两张表自洽：白名单的每条都能对上 ``CASES`` 里的真实端点，且**每条都写了理由**。

运行::

    ./.venv/Scripts/python.exe tests/parity_diff_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="pdiff_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from parity_diff import (  # noqa: E402
    CASES,
    strip_timestamps,
    KNOWN_DIFFS,
    MISSING_CASES,
    MISSING_EXPECT,
    VOLATILE_KEYS,
    classify,
    classify_missing,
    diff,
    normalize,
)

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def main() -> int:  # noqa: C901
    # ── 归一化 ──
    check("归一化: 丢易变键（时间戳/requestId），保留其余",
          normalize({"a": 1, "updatedAt": "x"}) == {"a": 1}
          and all(k not in normalize({k: 1}) for k in VOLATILE_KEYS))
    check("归一化: **不排序**（列表顺序原样保留）",
          normalize({"b": [3, 1, 2]}) == {"b": [3, 1, 2]})
    check("归一化: 嵌套 dict/list 递归生效",
          normalize({"d": {"c": [{"createdAt": 1, "n": 2}]}}) == {"d": {"c": [{"n": 2}]}})

    # ── 差异路径 ──
    check("差异: 路径精确到字段与下标",
          diff({"d": {"shots": [{"s": 1}]}}, {"d": {"shots": [{"s": 2}]}})
          == ["d.shots[0].s: 1 vs 2"],
          diff({"d": {"shots": [{"s": 1}]}}, {"d": {"shots": [{"s": 2}]}}))
    check("差异: 长度不同只报一条（不逐元素刷屏）",
          diff({"b": [1]}, {"b": [1, 2]}) == ["b (长度 1 vs 2)"], diff({"b": [1]}, {"b": [1, 2]}))
    check("差异: 单侧缺失会标注方向",
          diff({"a": 1}, {"a": 1, "b": 2}) == ["b (仅Python有) "]
          and diff({"a": 1, "b": 2}, {"a": 1}) == ["b (仅Node有)"],
          (diff({"a": 1}, {"a": 1, "b": 2}), diff({"a": 1, "b": 2}, {"a": 1})))

    # ── 判定三态 ──
    check("判定: 完全一致 -> same",
          classify("GET", "/x", {"a": 1}, {"a": 1})[0] == "same")
    check("判定: 只差易变键 -> same（否则每次对拍都红）",
          classify("GET", "/x", {"a": 1, "createdAt": "t1"},
                   {"a": 1, "createdAt": "t2"})[0] == "same")
    check("判定: HTTP 状态码不同 -> new（最硬的差异）",
          classify("GET", "/x", {"__status__": 200, "a": 1},
                   {"__status__": 500, "a": 1})[0] == "new")
    check("判定: 白名单命中（列出的字段前缀）-> known",
          classify("GET", "/api/v1/ai-configs/gpu/status",
                   {"hardware": {"gpuName": "x"}, "usedVRAM_GB": 0},
                   {"hardware": None, "usedVRAM_GB": 0})[0] == "known")
    check("判定: 白名单**越界**（另有字段不同）-> new（白名单不能当遮羞布）",
          classify("GET", "/api/v1/ai-configs/gpu/status",
                   {"hardware": None, "usedVRAM_GB": 0},
                   {"hardware": None, "usedVRAM_GB": 7})[0] == "new")
    check("判定: 不在白名单里的端点有差异 -> new",
          classify("GET", "/api/v1/dramas", {"a": 1}, {"a": 2})[0] == "new")

    # ── MISSING 类（「两边都不该有」的路径）──
    check("MISSING: Node 未匹配(404) + Python 兜底(501) -> missing（预期内，不是 new）",
          classify_missing({"__status__": 404}, {"__status__": 501})[0] == "missing")
    check("MISSING: **Node 回 200** -> new（说明它真有这个端点而 Python 缺 —— 这才是要报的）",
          classify_missing({"__status__": 200, "data": []}, {"__status__": 501})[0] == "new")
    check("MISSING: 状态码组合不符预期（如 Python 也 404）-> new（口径变了要人看）",
          classify_missing({"__status__": 404}, {"__status__": 404})[0] == "new")
    check("MISSING: 预期常量是 404/501（改成别的等于悄悄放宽口径）",
          MISSING_EXPECT == {"node": 404, "py": 501}, MISSING_EXPECT)
    check("MISSING: 全是 GET 且与 CASES 不重叠（同一个路径不能既是比对项又是缺失项）",
          all(m == "GET" for m, _p in MISSING_CASES)
          and not ({(m, p) for m, p in MISSING_CASES} & {(m, p) for m, p, _n in CASES}),
          MISSING_CASES)

    # ── 两张表自洽 ──
    table = {(method, path) for method, path, _note in CASES}
    check("表: CASES 全是 **GET**（只读对拍，避免对拍本身改数据）",
          all(method == "GET" for method, _p, _n in CASES), table)
    check("表: 无重复端点", len(table) == len(CASES), len(CASES))
    check("表: 白名单每条都对应 CASES 里的真实端点（防拼写漂移）",
          set(KNOWN_DIFFS) <= table, set(KNOWN_DIFFS) - table)
    check("表: 白名单每条都写了理由与字段前缀（不许留空手套白狼的条目）",
          all(entry.get("why") and entry.get("fields") for entry in KNOWN_DIFFS.values()),
          KNOWN_DIFFS)

    # ── S7 新增覆盖：只读 GET 补齐 + 文本响应的处理策略 ──
    check("表: S7 新增的只读 GET 已纳入（rhythm / qc-report?format=json）",
          any("/rhythm" in path for _m, path, _n in CASES)
          and any("qc-report?format=json" in path for _m, path, _n in CASES),
          [path for _m, path, _n in CASES][-3:])
    check("表: **文本响应故意不入表**（contact-sheet / qc-report?format=html —— 噪声大、JSON 用例已覆盖数据等价）",
          not any("contact-sheet" in path or "format=html" in path for _m, path, _n in CASES),
          [path for _m, path, _n in CASES])
    check("文本响应: ISO 时间戳被抹成 `<ts>`（否则毫秒差会伪装成新差异）",
          strip_timestamps("x 2026-09-15T06:24:43.123Z y") == "x <ts> y"
          and strip_timestamps("无时间戳") == "无时间戳")

    failed_items = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed_items)}/{len(_RESULTS)} passed")
    return 1 if failed_items else 0


if __name__ == "__main__":
    raise SystemExit(main())
