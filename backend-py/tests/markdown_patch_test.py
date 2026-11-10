"""增量变更集（P1-2）的守卫测试。

被测：``app/services/markdown_patch.py`` ＋ ``script_tools.edit_script`` 真落库。

三处**"跑得通但其实是假的"**是本文件的靶子：

* ``oldText`` 命中 0 次也照样替换（Python 的 ``str.replace`` 找不到就原样返回，
  不报错）⇒ 模型以为改了，剧本一个字没动；
* 命中 ≥2 次随便挑第一处改 ⇒ 把台词 A 的修改打进了台词 B；
* 半批生效：第 1 条改了、第 2 条没过 ⇒ 模型看到"失败"重送整批，第 1 条被**再改一遍**
  （两个 oldText 现已都不存在 ⇒ 第二轮整批失败，模型陷入死循环）。

跑法：``.venv/Scripts/python.exe tests/markdown_patch_test.py``。
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="markdown_patch_"))
os.environ.setdefault("SKILLS_DIR", str(Path(os.environ["DATA_ROOT"]) / "skills"))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.core.db import engine  # noqa: E402
from app.core.models import episodes  # noqa: E402
from app.main import app  # noqa: E402
from app.services import markdown_patch as mp  # noqa: E402
from app.agent.tools.script_tools import create_script_tools  # noqa: E402

_failures: list[str] = []
_passed = 0


def check(label: str, ok: bool, detail: str = "") -> bool:
    global _passed
    if ok:
        _passed += 1
        print(f"  ✓ {label}")
        return True
    _failures.append(f"{label}{(' —— ' + detail) if detail else ''}")
    print(f"  ✗ {label}{(' —— ' + detail) if detail else ''}")
    return False


def _call(tools: dict, tool_id: str, arguments: dict | None = None):
    return asyncio.run(tools[tool_id].run(arguments))


SCRIPT = (
    "第01场 夜 内 东家别墅\n"
    "凌云：（压低声音）别动。\n"
    "若云：（后退半步）你要做什么？\n"
    "凌云：先走。\n"
)


# ---------------------------------------------------------------------------
# 一、命中计数
# ---------------------------------------------------------------------------

def case_count() -> None:
    print("\n[1] 命中计数（重叠不算、空串单判）")
    text = "abcabc"
    check("出现 2 次", mp.count_occurrences(text, "abc") == 2, str(mp.count_occurrences(text, "abc")))
    check("出现 0 次", mp.count_occurrences(text, "zzz") == 0)
    check("重叠不重复计（aaa 里的 aa 算 1 次）",
          mp.count_occurrences("aaa", "aa") == 1, str(mp.count_occurrences("aaa", "aa")))
    check("空 needle ⇒ 0（不返回 len+1 那个没用的数）",
          mp.count_occurrences("abc", "") == 0)
    check("跨段命中", mp.count_occurrences("a\nb\na", "a") == 2)


# ---------------------------------------------------------------------------
# 二、整批原子
# ---------------------------------------------------------------------------

def case_apply() -> None:
    print("\n[2] 整批原子（任一条不过，一条都不落）")
    single = mp.apply_edits("甲乙丙", [mp.TextEdit(old_text="乙", new_text="丁")])
    check("单条改动生效", single.ok and single.text == "甲丁丙", single.text)
    check("回执带字数变化", single.applied[0]["charsBefore"] == 3 and len(single.text) == 3)

    seq = mp.apply_edits("甲乙丙", [
        mp.TextEdit(old_text="乙", new_text="乙乙"),
        mp.TextEdit(old_text="乙乙", new_text="戊"),   # 只有看到上一条的结果才命中
    ])
    check("按顺序应用（后一条看到前一条的结果）", seq.ok and seq.text == "甲戊丙", seq.text)

    two = mp.apply_edits(SCRIPT, [
        mp.TextEdit(old_text="别动。", new_text="别出声。", reason="台词更狠"),
        mp.TextEdit(old_text="先走。", new_text="先走，别回头。", reason="补钩子"),
    ])
    check("两条都改成功", two.ok and "别出声。" in two.text and "别回头" in two.text)
    check("两条都记进 applied（带 reason）",
          [item["reason"] for item in two.applied] == ["台词更狠", "补钩子"], str(two.applied))

    bad = mp.apply_edits(SCRIPT, [
        mp.TextEdit(old_text="别动。", new_text="别出声。"),
        mp.TextEdit(old_text="这句话原本不存在", new_text="X"),
    ])
    check("第 2 条不命中 ⇒ 整批失败", (not bad.ok) and bad.failed_index == 1,
          f"ok={bad.ok} index={bad.failed_index}")
    check("**原文一个字节都没动**", bad.text == SCRIPT, bad.text[:30])
    check("失败时回执不谎报已改动", bad.applied == [], str(bad.applied))

    ambiguous = mp.apply_edits(SCRIPT, [mp.TextEdit(old_text="凌云：", new_text="他：")])
    check("命中 2 次 ⇒ 拒（不猜是哪一处）", (not ambiguous.ok) and ambiguous.failed_index == 0)
    check("歧义句子给出命中次数", "2 次" in ambiguous.error, ambiguous.error)
    check("歧义句子教它扩上下文", "扩到" in ambiguous.error or "唯一" in ambiguous.error,
          ambiguous.error)

    missing = mp.apply_edits(SCRIPT, [mp.TextEdit(old_text="完全找不到的一句", new_text="X")])
    check("命中 0 次 ⇒ 拒", not missing.ok)
    check("0 次句子说'先重新读一遍'", "重新读一遍" in missing.error, missing.error)
    check("0 次句子原样带上它送的片段", "完全找不到的一句" in missing.error, missing.error)

    empty_old = mp.apply_edits(SCRIPT, [mp.TextEdit(old_text="", new_text="插入")])
    check("空 oldText ⇒ 拒（不做'任意位置插入'的猜测）", not empty_old.ok)
    check("空 oldText 的句子解释原因", "唯一出现" in empty_old.error, empty_old.error)

    noop = mp.apply_edits(SCRIPT, [mp.TextEdit(old_text="别动。", new_text="别动。")])
    check("oldText 与 newText 相同 ⇒ 允许但显式警告",
          noop.ok and bool(noop.warnings) and noop.applied[0]["noop"] is True)

    nothing = mp.apply_edits(SCRIPT, [])
    check("空数组 ⇒ ok 但提醒没改东西", nothing.ok and bool(nothing.warnings) and nothing.text == SCRIPT)


# ---------------------------------------------------------------------------
# 三、模型送来的形状
# ---------------------------------------------------------------------------

def case_coerce() -> None:
    print("\n[3] 输入归一（模型会照参考实现的 camelCase 写）")
    edits, error = mp.coerce_edits([{"oldText": "甲", "newText": "乙"}])
    check("camelCase 认", not error and edits[0].old_text == "甲" and edits[0].new_text == "乙")
    edits, error = mp.coerce_edits([{"old_text": "甲", "new_text": "乙", "note": "理由"}])
    check("snake_case 认，note 当 reason", not error and edits[0].reason == "理由")

    _, error = mp.coerce_edits([{"oldText": "甲"}])
    check("缺 newText ⇒ 点名下标", "edits[0]" in error and "newText" in error, error)
    _, error = mp.coerce_edits([{"oldText": "a", "newText": "b"}, {"oldText": "c"}])
    check("第二条缺字段 ⇒ 报 edits[1]", "edits[1]" in error, error)

    _, error = mp.coerce_edits("不是数组")
    check("非数组 ⇒ 给出正确写法示例", "edits: [" in error, error)
    _, error = mp.coerce_edits([1, 2])
    check("数组元素不是对象 ⇒ 点名下标", "edits[0]" in error and "对象" in error, error)
    _, error = mp.coerce_edits([{"oldText": 1, "newText": "b"}])
    check("oldText 不是字符串 ⇒ 报类型", "字符串" in error, error)
    edits, error = mp.coerce_edits(None)
    check("None ⇒ 空数组无错", not error and edits == [])


# ---------------------------------------------------------------------------
# 四、工具真落库
# ---------------------------------------------------------------------------

def case_tool() -> None:
    print("\n[4] edit_script 工具（原子性必须落到数据库上）")
    client = TestClient(app)
    drama_id = client.post("/api/v1/dramas", json={"title": "SMOKE 增量改写"}).json()["data"]["id"]
    episode_id = client.get(f"/api/v1/dramas/{drama_id}").json()["data"]["episodes"][0]["id"]
    with engine.begin() as conn:
        conn.execute(episodes.update().where(episodes.c.id == episode_id)
                     .values(script_content=SCRIPT, script_hash="oldhash"))

    import app.agent.tools.script_tools as st

    original_refresh = st.refresh_episode_script_hash
    refresh_calls: list[int] = []

    def _fake_refresh(conn, ep_id):
        refresh_calls.append(ep_id)
        return "hash"

    st.refresh_episode_script_hash = _fake_refresh  # type: ignore[assignment]
    try:
        tools = create_script_tools(episode_id)

        ok = _call(tools, "edit_script", {"edits": [
            {"oldText": "别动。", "newText": "别出声。", "reason": "更狠"},
            {"oldText": "先走。", "newText": "先走，别回头。", "reason": "补钩子"},
        ]})
        with engine.begin() as conn:
            row = conn.execute(select(episodes).where(episodes.c.id == episode_id)).first()
        check("成功时真写库", "别出声。" in row.script_content and "别回头" in row.script_content,
              row.script_content[:40])
        check("成功时刷新指纹", refresh_calls == [episode_id], str(refresh_calls))
        check("回执给人话与条数", ok.get("appliedCount") == 2 and "已应用" in ok.get("message", ""), ok)
        check("回执带字数字段", ok.get("word_count") == len(row.script_content))

        baseline = row.script_content
        refresh_calls.clear()
        fail = _call(tools, "edit_script", {"edits": [
            {"oldText": "别出声。", "newText": "别动。"},
            {"oldText": "这句不存在", "newText": "X"},
        ]})
        with engine.begin() as conn:
            after = conn.execute(select(episodes).where(episodes.c.id == episode_id)).first()
        check("失败时不写库（第 1 条也没落）", after.script_content == baseline,
              after.script_content[:40])
        check("失败时不刷新指纹", refresh_calls == [], str(refresh_calls))
        check("失败回执带 failedIndex", fail.get("failedIndex") == 1, fail)
        check("失败回执明说'没有修改'", "没有" in fail.get("note", ""), fail.get("note", ""))
        check("失败回执是模型可读的整句", "edits[1]" in fail.get("error", ""), fail.get("error", ""))

        empty = _call(tools, "edit_script", {"edits": []})
        check("空数组直接拒", "error" in empty and empty.get("appliedCount") == 0, empty)
        shape = _call(tools, "edit_script", {"edits": "乱写"})
        check("形状不对直接拒", "error" in shape, shape)

        missing_ep = _call(create_script_tools(999999), "edit_script",
                           {"edits": [{"oldText": "a", "newText": "b"}]})
        check("剧集不存在 ⇒ 带 id 的报错",
              missing_ep.get("error") == "Episode not found (id=999999)", missing_ep)
    finally:
        st.refresh_episode_script_hash = original_refresh  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 五、接线
# ---------------------------------------------------------------------------

def case_wiring() -> None:
    print("\n[5] 接线（工具进宿主白名单、提示词提到它、渲染不虚构）")
    from app.services.agent_registry import HOST_TOOL_NAMES
    from app.services.agent_prompts import SCRIPT_REWRITER

    check("edit_script 在宿主工具白名单里", "edit_script" in HOST_TOOL_NAMES)
    check("编剧提示词提到 edit_script", "edit_script" in SCRIPT_REWRITER)
    check("提示词同时保留整篇覆盖（两条路都要在）", "save_script" in SCRIPT_REWRITER)
    check("提示词强调 oldText 要逐字复制",
          "逐字" in SCRIPT_REWRITER or "原样" in SCRIPT_REWRITER, SCRIPT_REWRITER[:80])

    result = mp.apply_edits("甲乙", [mp.TextEdit(old_text="甲", new_text="甲甲")])
    check("渲染人话带方向符号与字数", "+1" in mp.render_result(result), mp.render_result(result))
    failed = mp.apply_edits("甲乙", [mp.TextEdit(old_text="丙", new_text="丁")])
    check("渲染失败句以'未改动任何内容'开头",
          mp.render_result(failed).startswith("未改动任何内容"), mp.render_result(failed))
    check("单条描述带 reason",
          "更狠" in mp.describe_edit(mp.TextEdit("a", "b", "更狠")))


def main() -> int:
    case_count()
    case_apply()
    case_coerce()
    case_tool()
    case_wiring()

    total = _passed + len(_failures)
    print(f"\nSUMMARY: {_passed}/{total} passed")
    if _failures:
        print("失败项：")
        for item in _failures:
            print(f"  - {item}")
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    sys.exit(main())
