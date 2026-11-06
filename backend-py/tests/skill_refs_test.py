"""``services/skill_refs`` 守卫 —— 只读、有边界、三档回执不许混。

为什么单独立一套：这个模块是**唯一**能把 ``skills/**/references/`` 里的文件内容
交给模型的口子。判据一旦松掉（比如把「字符串以 ``references/`` 开头」当成安全判据），
它就变成了任意文件读取 —— 而这类错误在功能测试里完全不显形（读得出来就叫"通过"）。
所以这里主要测**拒绝**：穿越、绝对路径、非白名单后缀、超限。

跑法::

    .venv/Scripts/python.exe -X utf8 tests/skill_refs_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_PY))
sys.stdout.reconfigure(encoding="utf-8")

_PASSED = 0
_FAILED = 0


def check(name: str, condition: object, detail: object = None) -> None:
    global _PASSED, _FAILED
    if condition:
        _PASSED += 1
        print(f"  PASS  {name}")
    else:
        _FAILED += 1
        print(f"  FAIL  {name}")
        if detail is not None:
            print(f"        <<< {detail!r}")


def build_tree(root: Path) -> None:
    """造一个最小技能库：一个带附件的 skill、一个只有 SKILL.md 的 skill。"""
    with_refs = root / "demo-lib" / "with-refs"
    (with_refs / "references" / "sub").mkdir(parents=True)
    (root / "demo-lib" / "library.yaml").write_text("name: demo-lib\n", encoding="utf-8")
    (with_refs / "SKILL.md").write_text("# demo\n", encoding="utf-8")
    (with_refs / "references" / "a.md").write_text(
        "# 标题\nline2\nline3\n", encoding="utf-8")
    (with_refs / "references" / "sub" / "b.md").write_text(
        "打戏节奏：先给全景再切手部。\n", encoding="utf-8")
    (with_refs / "references" / "pic.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    #: 超上限：列得出、读不了（不许整份塞进上下文）
    (with_refs / "references" / "big.md").write_text(
        "x" * (512 * 1024 + 10), encoding="utf-8")

    no_refs = root / "demo-lib" / "no-refs"
    no_refs.mkdir(parents=True)
    (no_refs / "SKILL.md").write_text("# 只有正文\n", encoding="utf-8")


def raises(fn, *args, **kwargs) -> str:
    """调用并返回错误消息（没抛就返回空串）。"""
    try:
        fn(*args, **kwargs)
    except ValueError as error:
        return str(error)
    return ""


def case_list() -> None:
    print("\n[列目录] 三档回执：有附件 / 空 / 根本没有")
    from app.services import skill_refs  # noqa: PLC0415

    listed = skill_refs.list_refs("demo-lib/with-refs")
    paths = [item["path"] for item in listed["files"]]
    check("available=True 且只列文本附件（png 不进清单）",
          listed["available"] is True and "references/a.md" in paths
          and "references/pic.png" not in paths, paths)
    check("子目录也列（rglob）", "references/sub/b.md" in paths, paths)
    check("每份带 firstLine（模型凭它决定读哪份，不用逐份试读）",
          any(item["firstLine"].startswith("# 标题") for item in listed["files"]),
          [item["firstLine"] for item in listed["files"]])
    check("超上限的文件**照样列出**但 readable=False（否则模型以为它不存在）",
          any(item["path"] == "references/big.md" and item["readable"] is False
              for item in listed["files"]), paths)

    missing = skill_refs.list_refs("demo-lib/no-refs")
    check("没有 references/ ⇒ available=False ＋ hint（不是「空目录」）",
          missing["available"] is False and missing["files"] == []
          and missing["hint"] != "", missing)

    prefixed = skill_refs.list_refs("demo-lib/with-refs", prefix="references/sub/")
    check("prefix 只列该子目录",
          [item["path"] for item in prefixed["files"]] == ["references/sub/b.md"],
          prefixed["files"])
    empty = skill_refs.list_refs("demo-lib/with-refs", prefix="references/nope/")
    check("prefix 下没有 ⇒ files=[] ＋ hint（与 available=False 分得开）",
          empty["files"] == [] and empty["available"] is True and empty["hint"] != "",
          empty)


def case_read() -> None:
    print("\n[读一份] 分页与截断必须如实")
    from app.services import skill_refs  # noqa: PLC0415

    got = skill_refs.read_ref("demo-lib/with-refs", "references/a.md")
    check("整份读回 3 行、truncated=False",
          got["content"] == "# 标题\nline2\nline3" and got["totalLines"] == 3
          and got["truncated"] is False, got)

    page = skill_refs.read_ref("demo-lib/with-refs", "references/a.md", offset=2, limit=1)
    check("分页：offset=2 从第 2 行起、truncated=True 如实报「没看完」",
          page["content"] == "line2" and page["truncated"] is True
          and page["offset"] == 2 and page["totalLines"] == 3, page)

    check("limit 超上限会被夹到 MAX_LINES（不许一次把整本塞进来）",
          skill_refs.read_ref("demo-lib/with-refs", "references/a.md",
                              limit=99999)["limit"] == skill_refs.MAX_LINES)


def case_reject() -> None:
    print("\n[拒绝] 越界、绝对路径、非白名单、超限")
    from app.services import skill_refs  # noqa: PLC0415

    check("skill 名含 .. ⇒ 拒", raises(skill_refs.list_refs, "demo-lib/../demo-lib") != "")
    check("skill 不存在 ⇒ 拒（不返回空清单冒充成功）",
          "没有这个 skill" in raises(skill_refs.list_refs, "demo-lib/nope"))
    check("附件路径含 .. ⇒ 拒（挡住 references/../SKILL.md）",
          raises(skill_refs.read_ref, "demo-lib/with-refs", "../SKILL.md") != "")
    check("多级 .. ⇒ 拒", raises(
        skill_refs.read_ref, "demo-lib/with-refs", "references/sub/../../SKILL.md") != "")
    check("绝对路径 ⇒ 拒（Windows 盘符与 POSIX 都算）", raises(
        skill_refs.read_ref, "demo-lib/with-refs", "C:/Windows/win.ini") != ""
        and raises(skill_refs.read_ref, "demo-lib/with-refs", "/etc/passwd") != "")
    check("非白名单后缀 ⇒ 拒并说明走媒体接口",
          "媒体接口" in raises(skill_refs.read_ref, "demo-lib/with-refs",
                               "references/pic.png"))
    check("超字节上限 ⇒ 拒（提示改用检索，不是「读一半」）",
          "超过单份上限" in raises(skill_refs.read_ref, "demo-lib/with-refs",
                                    "references/big.md"))
    check("读不存在的附件 ⇒ 拒",
          "没有这个附件" in raises(skill_refs.read_ref, "demo-lib/with-refs",
                                    "references/none.md"))


def case_tools() -> None:
    print("\n[工具层] 只转发，错误变成 error 字段（不许抛穿 agent 循环）")
    import asyncio

    from app.agent.tools.skill_ref_tools import create_skill_ref_tools

    tools = create_skill_ref_tools()
    check("两个工具的 id 与注册表键一致",
          set(tools) == {"list_skill_refs", "read_skill_ref"}
          and all(key == tool.id for key, tool in tools.items()), list(tools))

    listed = asyncio.run(tools["list_skill_refs"].execute(
        {"skill": "demo-lib/with-refs", "prefix": "references/sub/"}))
    check("list 工具走通", listed["total"] == 1, listed)

    bad = asyncio.run(tools["read_skill_ref"].execute(
        {"skill": "demo-lib/with-refs", "path": "../SKILL.md"}))
    check("越界请求在工具层变成 error 字段（不抛异常）",
          "error" in bad and bad["error"], bad)

    tool_bad = asyncio.run(tools["list_skill_refs"].execute({"skill": ""}))
    check("空 skill ⇒ error 字段", "error" in tool_bad, tool_bad)


def main() -> int:
    global _PASSED, _FAILED
    print("=" * 66)
    print("skill_refs 附件读取守卫")
    print("=" * 66)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "skills"
        build_tree(root)
        previous = os.environ.get("SKILLS_DIR")
        os.environ["SKILLS_DIR"] = str(root)
        try:
            case_list()
            case_read()
            case_reject()
            case_tools()
        finally:
            if previous is None:
                os.environ.pop("SKILLS_DIR", None)
            else:
                os.environ["SKILLS_DIR"] = previous

    print(f"\nSUMMARY: {_PASSED}/{_PASSED + _FAILED} passed")
    return 0 if _FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
