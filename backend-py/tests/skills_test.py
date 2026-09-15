"""S6 自检：Skill 解析器 + 加载器（``agents/skill-parser.ts`` 172 行 + ``skills.ts`` 218 行）。

这两块的核心不是「解析 frontmatter」（那很短），而是**几条判据**：

1. **`userConfigured` 的判据**：只要 DB 里能解析出**配置项**，即使用户把 skill **全部取消勾选**，
   也**不回退默认绑定** —— 判反了就会把「一个都不注入」反向执行成「注入全部默认 skill」
   （前端勾选框与面板文案都承诺「仅勾选的会注入」）；
2. **体量预算闸**：按优先级累加，超出预算者**跳过并写进注入文本**（可诊断，不是静默丢弃）；
3. **外来工具只认 `hub_` 前缀**：正文里大量反引号 token 是字段名/文件名，宽泛猜会全判成缺失；
4. **`allowed-tools` 要兼容三种写法**：YAML 数组 / 逗号纯量字符串（YAML **不按逗号切分**）/ 换行列表。

前半段用**临时 SKILL.md**（隔离、可造边界），后半段对**真实 ``skills/`` 目录**跑集成。

运行::

    ./.venv/Scripts/python.exe tests/skills_test.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="skills_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.config import PROJECT_ROOT  # noqa: E402
from app.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models import dramas  # noqa: E402
from app.services.agents import skill_parser as sp  # noqa: E402
from app.services.agents import skills as sk  # noqa: E402
from app.services.agents import runtime as rt  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


SKILL_TEMPLATE = """---
name: {name}
description: {description}
preconditions:
  - 剧本已就位
  - 角色名单已确认
protocol:
  - shots_planned
agents: [storyboard_breaker]
priority: {priority}
allowed-tools: {tools}
---

# {name} 正文

先调用 `hub_list_capabilities`，再按 shot_type 组织镜头。
"""


def _make_skills_dir(root: Path) -> None:
    """造一个隔离的 skills/ 目录：两个 core skill + 一个外部库容器（无 SKILL.md）。"""
    for name, priority, tools in (("alpha", 20, "[hub_read, hub_write]"),
                                  ("beta", 5, "hub_read, hub_query_dag_result")):
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "SKILL.md").write_text(
            SKILL_TEMPLATE.format(name=name, description=f"{name} 说明", priority=priority,
                                  tools=tools),
            encoding="utf-8",
        )
    # 外部库容器：顶层没有 SKILL.md，子目录才有 ⇒ 不进 core，但进 list_skill_ids
    nested = root / "vendorlib" / "installed" / "gamma"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "SKILL.md").write_text("---\nname: gamma\n---\n\n正文\n", encoding="utf-8")


def main() -> int:  # noqa: C901
    # ================= skill_parser：归一化 =================
    check("归一: to_str_array 兼容 省略/数组/换行列表 三种写法",
          sp.to_str_array(None) == []
          and sp.to_str_array([" a ", "b", ""]) == ["a", "b"]
          and sp.to_str_array("- 甲\n- 乙\n") == ["甲", "乙"]
          and sp.to_str_array(3) == ["3"],
          [sp.to_str_array(None), sp.to_str_array([" a ", "b", ""]),
           sp.to_str_array("- 甲\n- 乙\n")])
    check("归一: to_tool_list 兼容 YAML 数组 / 逗号纯量 / 换行列表 / 单元素方括号",
          sp.to_tool_list(["hub_read", "hub_write"]) == ["hub_read", "hub_write"]
          and sp.to_tool_list("hub_read, hub_write") == ["hub_read", "hub_write"]
          and sp.to_tool_list("- hub_read\n- hub_write") == ["hub_read", "hub_write"]
          and sp.to_tool_list("[hub_read]") == ["hub_read"],
          [sp.to_tool_list("hub_read, hub_write"), sp.to_tool_list("[hub_read]")])
    # ⚠️ 记录一个**保真的粗糙点**：清洗顺序是「去括号 → 去首尾引号 → trim」，
    #    所以 `[ "a", "b" ]` 这种「括号 + 引号 + 空格」混写会留下脏引号
    #    （TS 的顺序与正则完全一致 ⇒ 同样脏）。不修：它只影响从没在真实 frontmatter 里
    #    出现过的写法，而修了就跟 TS 不一致了。
    check("归一: `[ \"a\", \"b\" ]` 混写会留脏引号（与 TS 同款粗糙，**刻意不修**）",
          sp.to_tool_list('[ "hub_read", "hub_write" ]') == ['"hub_read', 'hub_write"'],
          sp.to_tool_list('[ "hub_read", "hub_write" ]'))
    check("归一: to_tool_list **去重保序**（`Set` 语义）",
          sp.to_tool_list("a, b, a") == ["a", "b"], sp.to_tool_list("a, b, a"))
    check("归一: priority 非法/非正 -> 100，合法值转 int",
          [sp.normalize_priority(x) for x in (None, "abc", 0, -3, "20", 20.9)] == [100, 100, 100, 100, 20, 20],
          [sp.normalize_priority(x) for x in (None, "abc", 0, -3, "20", 20.9)])
    check("外来工具: 只认 `hub_` 前缀（字段名/文件名不会被误判）",
          sp.extract_foreign_tool_refs(
              "调用 `hub_list_capabilities` 与 `hub_query_dag_result`，注意 `shot_type`、`read`、`bgm_url`"
          ) == ["hub_list_capabilities", "hub_query_dag_result"],
          sp.extract_foreign_tool_refs("`hub_x1` `shot_type` `read` `bgm_url`"))
    check("外来工具: 结果**去重 + 排序**",
          sp.extract_foreign_tool_refs("`hub_b` `hub_a` `hub_b`") == ["hub_a", "hub_b"])

    # ================= skill_parser：解析与渲染 =================
    parsed = sp.parse_skill(SKILL_TEMPLATE.format(
        name="alpha", description=" 说明 ", priority=7, tools="[hub_read]"), "fallback-id")
    check("解析: 元数据逐字段正确（含 description trim / priority / allowed-tools）",
          parsed.metadata.name == "alpha" and parsed.metadata.description == "说明"
          and parsed.metadata.preconditions == ["剧本已就位", "角色名单已确认"]
          and parsed.metadata.protocol == ["shots_planned"]
          and parsed.metadata.agents == ["storyboard_breaker"]
          and parsed.metadata.priority == 7 and parsed.metadata.allowed_tools == ["hub_read"],
          parsed.metadata)
    check("解析: body **不含 frontmatter**、且已 trim",
          parsed.body.startswith("# alpha 正文") and "---" not in parsed.body.split("\n")[0],
          parsed.body[:20])
    check("解析: 同时产出正文里的外来工具引用", parsed.foreign_tool_refs == ["hub_list_capabilities"])

    no_front = sp.parse_skill("只有正文", "fallback-id")
    check("解析: 没有 frontmatter -> 正文原样、name 回退 id、priority 100",
          no_front.body == "只有正文" and no_front.metadata.name == "fallback-id"
          and no_front.metadata.priority == 100 and no_front.metadata.agents == [])
    broken = sp.parse_skill("---\nname: [未闭合\n---\n正文可用", "fid")
    check("解析: frontmatter 坏 YAML -> **降级为空元数据**，正文照常可用",
          broken.metadata.name == "fid" and broken.body == "正文可用", broken.metadata.name)
    blank_name = sp.parse_skill("---\nname: '   '\n---\n正文", "fid")
    check("解析: name 只有空白 -> 回退 fallbackId", blank_name.metadata.name == "fid")

    rendered = sp.render_skill(parsed)
    check("渲染: 正文 + 前置契约 + 输出协议 三段，`\\n\\n` 连接、标题逐字一致",
          rendered.startswith(parsed.body + "\n\n## 前置契约（执行前必须满足）")
          and "以下条件不满足时，停下并向用户说明缺失项，不要臆造数据继续执行：" in rendered
          and "- 剧本已就位" in rendered
          and "## 输出契约" not in rendered
          and "\n\n## 输出协议字段\n" in rendered
          and "在收尾的 YAML 协议块中，除 status / summary 外，还必须额外汇报以下字段：" in rendered
          and "- shots_planned" in rendered,
          rendered[:60])
    bare = sp.render_skill(sp.parse_skill("正文而已", "id"))
    check("渲染: 无前置契约/协议时**只有正文**（不多出空段）", bare == "正文而已", bare)

    # ================= skills：隔离目录 =================
    tmp_root = Path(tempfile.mkdtemp(prefix="skills_dir_"))
    _make_skills_dir(tmp_root)
    os.environ["SKILLS_DIR"] = str(tmp_root)
    sk._parsed_cache.clear()  # noqa: SLF001 —— 缓存不认目录，换目录要清（与 TS 同性质）

    check("目录: SKILLS_DIR 覆盖生效", sk.skills_dir() == tmp_root)
    check("目录: core 只含**顶层有 SKILL.md** 的目录（外部库容器不算）",
          sk.list_core_skill_ids() == ["alpha", "beta"], sk.list_core_skill_ids())
    check("目录: list_skill_ids 含嵌套的外部库 skill",
          set(sk.list_skill_ids()) == {"alpha", "beta", "vendorlib/installed/gamma"},
          sk.list_skill_ids())
    check("默认绑定: 按 frontmatter agents 命中，并按 **priority 升序**（beta=5 先于 alpha=20）",
          sk.resolve_default_skills("storyboard_breaker") == ["beta", "alpha"],
          sk.resolve_default_skills("storyboard_breaker"))
    check("默认绑定: 空/未知 agent -> 空列表",
          sk.resolve_default_skills("") == [] and sk.resolve_default_skills("nope") == [])

    injected = sk.load_agent_skills("storyboard_breaker")
    check("注入: 默认绑定注入，格式 `## Available Skills` 开头、`---` 结尾",
          isinstance(injected, str) and injected.startswith("## Available Skills")
          and injected.rstrip("\n").endswith("---"), (injected or "")[:40])
    check("注入: 两个 skill 的正文都在（beta 在前）",
          "# beta 正文" in injected and "# alpha 正文" in injected
          and injected.index("# beta 正文") < injected.index("# alpha 正文"))

    # DB 配置优先 + 几条判据
    only_beta = sk.load_agent_skills("storyboard_breaker",
                                     '[{"id": "beta", "enabled": true}]')
    check("注入: **DB 配置优先**（只注入 beta，不再带默认的 alpha）",
          "# beta 正文" in only_beta and "# alpha 正文" not in only_beta)
    check("注入: **全部取消勾选 -> 不注入任何 skill（绝不回退默认）**",
          sk.load_agent_skills("storyboard_breaker",
                               '[{"id": "beta", "enabled": false}]') is None)
    check("注入: DB 是空数组 -> 回退默认绑定（与没配过等价）",
          sk.load_agent_skills("storyboard_breaker", "[]") == injected)
    check("注入: DB 是坏 JSON -> 回退默认绑定",
          sk.load_agent_skills("storyboard_breaker", "{坏") == injected)
    missing = sk.load_agent_skills("storyboard_breaker",
                                   '[{"id": "beta"}, {"id": "不存在"}]')
    check("注入: 文件缺失 -> **跳过并写进通知**（不是静默丢弃）",
          isinstance(missing, str) and "> 未注入：不存在（缺失）" in missing,
          (missing or "")[:60])
    # ⚠️ 保真记录：`if (!sections.length) return null` **排在通知之前** ⇒
    #    「绑定全缺失」时连「未注入」通知都拿不到（返回 None）。原 TS 同样如此，不修。
    check("注入: **全部缺失 -> None**（通知也随之丢失，与 TS 的早退语义一致）",
          sk.load_agent_skills("storyboard_breaker", '[{"id": "不存在"}]') is None)

    check("配置解析: 非数组/无 id/坏 JSON 分别过滤或空",
          sk.parse_skills_config(None) == []
          and sk.parse_skills_config("{坏") == []
          and sk.parse_skills_config('{"a":1}') == []
          and [b.id for b in sk.parse_skills_config('["x", {"v":1}, {"id":"ok"}]')] == ["ok"])
    check("配置解析: enabled 缺省为真、priority 非数字为 0",
          [(b.enabled, b.priority)
           for b in sk.parse_skills_config('[{"id":"x"}, {"id":"y","enabled":false,"priority":3}]')]
          == [(True, 0), (False, 3)])

    # 体量预算
    os.environ["AGENT_SKILL_BUDGET"] = "1"
    check("预算: 上限 1 字符 -> 全部超预算 -> **没有任何 skill 注入 => None**",
          sk.load_agent_skills("storyboard_breaker") is None)
    # ⚠️ 原 TS 注释写「设为 0 关闭限制」，但实现是 `raw > 0 ? raw : 60000` ⇒ **0 根本走不到**，
    #    `SKILL_CHAR_BUDGET > 0` 那道闸因此永远为真（形同虚设）。保真不修，但别信注释。
    os.environ["AGENT_SKILL_BUDGET"] = "0"
    check("预算: env=0 **并不关闭限制**（回落 60000 —— 原注释与实现不符，属保真点）",
          sk.skill_char_budget() == sk.SKILL_CHAR_BUDGET_DEFAULT)
    beta_len = len(sp.render_skill(sk.load_skill("beta")))
    os.environ["AGENT_SKILL_BUDGET"] = str(beta_len + 10)
    limited = sk.load_agent_skills("storyboard_breaker")
    check("预算: 恰好只装得下第一个 -> 部分注入 + 标注超预算",
          isinstance(limited, str) and "# beta 正文" in limited
          and "# alpha 正文" not in limited
          and "（超预算）" in limited
          and f"（文件缺失或超出 {beta_len + 10} 字符预算）" in limited,
          (limited or "")[:80])
    os.environ["AGENT_SKILL_BUDGET"] = "abc"
    check("预算: 非法值 -> 回落默认 60000", sk.skill_char_budget() == sk.SKILL_CHAR_BUDGET_DEFAULT)
    os.environ.pop("AGENT_SKILL_BUDGET", None)

    # ================= skills：真实目录 =================
    os.environ.pop("SKILLS_DIR", None)
    sk._parsed_cache.clear()  # noqa: SLF001
    check("真实目录: 定位到 `backend-py/skills`（2026-09-15 并入后端；删 backend/ 不受影响）",
          sk.skills_dir() == PROJECT_ROOT / "backend-py" / "skills"
          and sk.skills_dir().is_dir())
    core = sk.list_core_skill_ids()
    check("真实目录: core 非空、已排序、每个都真有 SKILL.md",
          len(core) >= 5 and core == sorted(core)
          and all((sk.skills_dir() / cid / "SKILL.md").exists() for cid in core), core)
    check("真实目录: list_skill_ids 是 core 的超集（含外部库嵌套）",
          set(core) <= set(sk.list_skill_ids()), sk.list_skill_ids())
    real_defaults = sk.resolve_default_skills("storyboard_breaker")
    check("真实目录: 分镜 Agent 的默认绑定非空且都是 core skill",
          len(real_defaults) >= 1 and set(real_defaults) <= set(core), real_defaults)
    check("真实目录: 绑定顺序按 (priority, id) 稳定可复现",
          real_defaults == sk.resolve_default_skills("storyboard_breaker")
          and [sk.load_skill(i).metadata.priority for i in real_defaults]
          == sorted(sk.load_skill(i).metadata.priority for i in real_defaults),
          [sk.load_skill(i).metadata.priority for i in real_defaults])
    real_injected = sk.load_agent_skills("storyboard_breaker")
    check("真实目录: 注入文本非空且含 `## Available Skills`",
          isinstance(real_injected, str) and "## Available Skills" in real_injected
          and len(real_injected) > 500, len(real_injected or ""))

    # ================= 接上运行时 / defaults 端点 =================
    client = TestClient(app)
    client.post("/api/v1/ai-configs", json={
        "service_type": "text", "provider": "openai", "base_url": "https://api.example.com",
        "api_key": "k", "model": ["m"], "is_active": True,
    })
    drama_id = client.post("/api/v1/dramas", json={"title": "SMOKE skills"}).json()["data"]["id"]

    with engine.begin() as conn:
        built = rt.build_agent_config(conn, "storyboard_breaker", 1, drama_id)
    check("运行时: instructions 里**真的带上了 skill 段**（此前恒为 None）",
          "## Available Skills" in built.instructions
          and built.instructions.index("## Available Skills")
          < built.instructions.index("输出协议"),
          built.instructions[-80:])

    class FakeDriver:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def __call__(self, **kwargs) -> dict:
            self.calls.append(kwargs)
            return {"text": "ok", "tool_calls": [], "tool_results": [], "steps": []}

    async def _no_sleep(_seconds: float) -> None:
        return None

    driver = FakeDriver()
    with engine.begin() as conn:
        asyncio.run(rt.run_agent_with_retry(conn, "storyboard_breaker", 1, drama_id, "hi",
                                            generate=driver, sleep=_no_sleep))
    check("运行时: 传给驱动的指令也含 skill 段（组装路径与 build_agent_config 一致）",
          "## Available Skills" in driver.calls[0]["instructions"])

    defaults = client.get("/api/v1/agent-configs/defaults")
    body = defaults.json()
    check("端点: GET /agent-configs/defaults **已本地化**（6 个 Agent、四字段）",
          defaults.status_code == 200 and len(body["data"]) == 6
          and all(set(item) == {"agent_type", "name", "instructions", "skills"}
                  for item in body["data"]),
          defaults.status_code)
    check("端点: 每项提示词非空、skills 与 resolve_default_skills 一致",
          all(item["instructions"].strip() for item in body["data"])
          and all(item["skills"] == sk.resolve_default_skills(item["agent_type"])
                  for item in body["data"]),
          [item["skills"] for item in body["data"]])
    check("端点: 顺序即 VALID_AGENT_TYPES（与 agent_registry 同源）",
          [item["agent_type"] for item in body["data"]] == list(rt.VALID_AGENT_TYPES))

    # ================= 汇总 =================
    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
