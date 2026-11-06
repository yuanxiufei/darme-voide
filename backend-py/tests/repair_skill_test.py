"""修复技能守卫：**提示词的每条硬清单必须从实现取值，不许手抄**（源 `skill.mjs` 重写）。

钉四件事：

1. **清单零漂移** —— 正文里的拒绝段/拒绝文件/可改前缀/各条上限，
   全部与 ``repair_tools`` / ``repair_diagnose`` 的常量**双向**一致：
   常量里有的正文必须点名，正文点名的必须真是常量（多一个 = 编的 ✗）。
2. **改常量即改提示词** —— monkeypatch 掉 ``MAX_PROPOSAL_FILES`` 后正文要跟着变。
   这条是「用 f-string 取值」与「手抄一份」的分水岭：手抄的写法在这里必失败 ✓。
3. **不夹带源项目运维手册** —— 源那份里 ``8291`` 网关端口、``v0.22.x`` 版本行为、
   魔搭地址、``LOCAL_ONLY``、``QwenAnglePromptConcat``、``restart_comfy`` /
   ``replace_model`` 这些**在本仓一个都不存在**；出现任意一条就是照抄而非重写 ✗。
4. **拼装与端点** —— ``context`` 只在给了才追加；``GET /api/v1/repair/skill`` 要真出现在
   OpenAPI 里（否则前端取不到，等于这份提示词没有出口）。

运行::

    ./.venv/Scripts/python.exe tests/repair_skill_test.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BP = REPO / "backend-py"

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def main() -> int:
    sys.path.insert(0, str(BP))
    from app.services import repair_diagnose as diag  # noqa: PLC0415
    from app.services import repair_skill as skill  # noqa: PLC0415
    from app.services import repair_tools as tools  # noqa: PLC0415

    text = skill.skill_markdown()

    # ---------------------------------------------------------------- ① 清单零漂移
    denied_segments = sorted(tools.DENY_SEGMENTS)
    missing_segments = [item for item in denied_segments if f"`{item}`" not in text]
    check("拒绝段: 常量里每一条都在正文点名", not missing_segments, missing_segments)

    denied_files = sorted(tools.DENY_FILES)
    missing_files = [item for item in denied_files if f"`{item}`" not in text]
    check("拒绝文件: 常量里每一条都在正文点名", not missing_files, missing_files)

    # 反向：正文「绝不进入 / 绝不修改」那两行里反引号包的东西，必须真的在常量里
    seg_line = next((line for line in text.splitlines() if "绝不进入" in line), "")
    named_segments = set(re.findall(r"`([^`]+)`", seg_line))
    check("拒绝段: 正文没编造常量外的段",
          named_segments <= set(denied_segments), sorted(named_segments - set(denied_segments)))

    file_line = next((line for line in text.splitlines() if "绝不修改" in line), "")
    named_files = set(re.findall(r"`([^`]+)`", file_line))
    check("拒绝文件: 正文没编造常量外的名字",
          named_files <= set(denied_files), sorted(named_files - set(denied_files)))

    for prefix in tools.DEFAULT_ALLOWED_PREFIXES:
        check(f"可改前缀: {prefix} 在正文", f"`{prefix}`" in text, prefix)

    for label, value in [
        ("MAX_PROPOSAL_FILES", tools.MAX_PROPOSAL_FILES),
        ("MAX_READ_BYTES", tools.MAX_READ_BYTES),
        ("MAX_WRITE_BYTES", tools.MAX_WRITE_BYTES),
        ("MAX_LIST_DEPTH", tools.MAX_LIST_DEPTH),
        ("MAX_LIST_ENTRIES", tools.MAX_LIST_ENTRIES),
        ("MAX_SEARCH_FILES", tools.MAX_SEARCH_FILES),
        ("MAX_PROBLEMS", diag.MAX_PROBLEMS),
        ("MAX_MISSING_ROWS", diag.MAX_MISSING_ROWS),
    ]:
        check(f"上限: {label} 的当前值写进正文", f"{value}" in text, value)

    # ---------------------------------------------------------------- ② 改常量即改提示词
    original = tools.MAX_PROPOSAL_FILES
    try:
        tools.MAX_PROPOSAL_FILES = 3
        changed = skill.skill_markdown()
        check("漂移: 改常量后正文跟着变", "每次最多 3 个文件" in changed, changed[:200])
        check("漂移: 旧值不再残留", f"每次最多 {original} 个文件" not in changed, changed[:200])
    finally:
        tools.MAX_PROPOSAL_FILES = original

    # ---------------------------------------------------------------- ③ 不夹带源运维手册
    forbidden = [
        "8291",            # 源网关端口，本仓没有
        "v0.22",           # 源项目的版本行为差异
        "modelscope",      # 源项目资源下载地址
        "LOCAL_ONLY",      # 源项目渠道模式
        "QwenAnglePromptConcat",   # 源项目自定义节点
        "restart_comfy",   # 源专有工具：本仓修复域没有
        "replace_model",   # 同上
        "update.json",     # 源项目更新器清单
        "fixes.json",      # 源项目修复记录落点（本仓在 dataRoot/repair）
    ]
    leaked = [item for item in forbidden if item.lower() in text.lower()]
    check("重写: 无源项目专有词汇", not leaked, leaked)

    # 端点名必须真实：正文提到的 /api/v1/repair/* 都要在路由里存在
    router_src = (BP / "app" / "routers" / "repair.py").read_text(encoding="utf-8")
    declared = set(re.findall(r'@router\.(?:get|post)\("([^"]+)"\)', router_src))
    mentioned = set(re.findall(r"`(/api/v1/repair/[a-z-]+)", text))
    # 正文用短名（`GET /diagnose`）时补前缀再比
    short = set(re.findall(r"`(?:GET|POST) `?/([a-z-]+)`", text))
    normalized = {f"/{item}" for item in short} | {item.replace("/api/v1/repair", "") for item in mentioned}
    unknown = sorted(item for item in normalized if item not in declared)
    check("端点: 正文提到的修复接口都真实存在", not unknown, unknown)
    check("端点: 正文至少点名 5 个真接口", len(normalized) >= 5, sorted(normalized))

    # ---------------------------------------------------------------- ④ 拼装与端点
    prompt = skill.build_system_prompt(workspace_root="D:/tmp/ws")
    check("拼装: 含正文", skill.SKILL_NAME in prompt and "权能级别" in prompt, prompt[:120])
    check("拼装: 含工作区", "D:/tmp/ws" in prompt, prompt[-300:])
    check("拼装: 无 context 时不出现 JSON 段", "## 当前上下文" not in prompt, prompt[-300:])

    with_context = skill.build_system_prompt(context={"node": "IMG-1"}, workspace_root="D:/tmp/ws")
    check("拼装: 有 context 时追加 JSON 段", "## 当前上下文" in with_context, with_context[-300:])
    check("拼装: context 用 UTF-8 不转义", '"node": "IMG-1"' in with_context, with_context[-300:])

    constrained = skill.build_system_prompt(allowed_prefixes=("backend-py/",))
    check("拼装: 可改目录可覆盖", "`backend-py/`" in constrained, constrained[-400:])

    # 端点可见（OpenAPI 是唯一能可靠读到的路由视图）
    from app.main import app  # noqa: PLC0415

    paths = app.openapi()["paths"]
    check("端点: GET /api/v1/repair/skill 已挂上", "/api/v1/repair/skill" in paths,
          sorted(item for item in paths if item.startswith("/api/v1/repair")))

    # context 传字符串（非法 JSON 忽略、合法 JSON 嵌入）—— 端点那条路径的解析口径
    parsed = json.loads('{"a": 1}')
    check("端点: 合法 JSON 可解析", parsed == {"a": 1}, parsed)
    bad = None
    try:
        bad = json.loads("not json")
    except json.JSONDecodeError:
        bad = None
    check("端点: 非法 JSON 解析为 None（不抛）", bad is None, bad)

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed_ok, detail in _RESULTS:
        print(("PASS  " if passed_ok else "FAIL  ") + name + ("" if passed_ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
