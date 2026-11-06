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

from app.core.config import PROJECT_ROOT  # noqa: E402
from app.core.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.core.models import dramas  # noqa: E402
from app.services import skill_parser as sp  # noqa: E402（⚠️ 2026-10-30：`app/agent/skill_parser.py` 转发壳已删）
from app.agent import skills as sk  # noqa: E402
from app.agent import runtime as rt  # noqa: E402
from app.services import agent_registry as ar  # noqa: E402
from app.agent.tool import ToolRegistry as _ToolRegistry  # noqa: E402
from app.routers import skills as skr  # noqa: E402（库声明解析的单一真源，别在测试里重写一遍规则）

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


def _binding_problems(
    skill_ids: list[str],
    core_ids: set[str],
    valid_types: set[str],
    load=None,
) -> tuple[list[tuple[str, list[str]]], list[tuple[str, list[str]]]]:
    """两类**静默不注入**的 `agents:` 声明，返回 ``(未知类型, 库内声明)``。

    1. 名字不在 ``VALID_AGENT_TYPES`` —— 拼错一个字母，该 Agent 就少一段上下文，
       加载器不会报错（它只做 `agent_type in metadata.agents` 的成员判断）；
    2. 声明在**库内** skill 上 —— 默认绑定只扫 ``list_core_skill_ids()``，库内 `agents:` 是空头支票。

    抽成函数是为了**能被反验**：两条判据既要抓得住坏数据，也不能见谁都喊。
    """
    loader = load or sk.load_skill
    unknown: list[tuple[str, list[str]]] = []
    vendor_declared: list[tuple[str, list[str]]] = []
    for skill_id in skill_ids:
        parsed = loader(skill_id)
        if parsed is None:
            continue
        agents = list(getattr(parsed.metadata, "agents", None) or [])
        if not agents:
            continue
        bad = [a for a in agents if a not in valid_types]
        if bad:
            unknown.append((skill_id, bad))
        if skill_id not in core_ids:
            vendor_declared.append((skill_id, agents))
    return unknown, vendor_declared


def _library_id_problems() -> list[tuple[str, str]]:
    """库声明的 ``name``（＝前端分组键 ``lib:<id>``）必须两两唯一、且能安全拼进那个键。

    判据来自「分组键唯一」这个事实：前端是
    ``vendorLibs.find(l => `lib:${l.id}` === type)`` —— **只取第一个**；而 ``inGroup`` 用
    ``lib:${s.source}`` 比对，两个库声明同一个 ``name`` 时两边都成立 ⇒ 两库 skill 在侧栏
    **并成一组**，组名与描述只显示第一个库的。全程**不报错、不告警**，只表现为
    「这个库里混进了别的库的 skill」。``:`` 与 ``/`` 会破坏 ``lib:`` 前缀解析，一并拒。

    解析规则走 ``routers.skills`` 的真源（含「声明缺失 ⇒ 回退目录名」「声明损坏 ⇒ 回退
    目录名」两条回退），**不在测试里重写一遍** —— 重写的那份迟早与真源漂移。它与
    ``app/core/config.skills_dir()`` 同源（``SKILLS_DIR`` 可覆盖）⇒ 隔离测试靠切环境变量。
    """
    base = sk.skills_dir()
    if not base.is_dir():
        return []
    problems: list[tuple[str, str]] = []
    seen: dict[str, str] = {}
    for entry in sorted(base.iterdir()):
        if not (entry.is_dir() and (entry / skr.LIBRARY_FILE).exists()):
            continue
        lib_id = skr._read_library(entry.name)["id"]  # noqa: SLF001
        if not lib_id or any(ch in lib_id for ch in ":/\\"):
            problems.append((entry.name, f"库标识 {lib_id!r} 不能用作 lib:<id> 分组键"))
        if lib_id in seen:
            problems.append((entry.name, f"库标识 {lib_id!r} 与 {seen[lib_id]} 重复 ⇒ 侧栏并组"))
        seen.setdefault(lib_id, entry.name)
    return problems


def _container_problems() -> list[tuple[str, str]]:
    """顶层目录的两种退化形态 —— 都让 skill 在侧栏**找不到入口**，且全程不报错。

    判定依据全在「谁给 skill 编出侧栏分组」这一条链上：

    * 侧栏选项 = ``['all', ...agentDefs, ...vendorLibs.map(`lib:${l.id}`)]``（``skills.vue``），
      而 ``vendorLibs`` 来自 ``_list_libraries()``，它**只收带 ``library.yaml`` 的目录**；
    * 每个 skill 的 ``source`` 由 ``_classify()`` 给：顶层目录有 ``SKILL.md`` ⇒ ``core``；
      有 ``library.yaml`` ⇒ ``vendor`` + 声明里的 id；**两者都没有就兜底 ``vendor`` + 目录名**。

    于是两种形态都成了「有数据、没入口」：

    ① **容器目录漏写 ``library.yaml``**（自身无 ``SKILL.md``、下面却挂着 skill）：兜底把它判成
       ``vendor``/目录名，但 ``vendorLibs`` 里没有这一项 ⇒ 侧栏压根没有这个分组，那些 skill
       只在「全部」里露头，按库浏览永远找不到。（``skills.vue`` 里还有一份 ``valid`` 白名单，
       所以就算硬拼出 ``lib:<目录名>`` 也会被重置掉。）
    ② **顶层目录同时有 ``SKILL.md`` 与 ``library.yaml``**：``_classify`` 先判 ``core``
       （``"/" not in skill_id`` 与「顶层有 SKILL.md」两条都命中），而侧栏又会为它渲染一个
       ``lib:<id>`` 分组；``inGroup`` 要求 ``category === 'vendor'`` ⇒ **那个分组点开永远是空的**。

    与「库标识重复」是同一个病：前端分组由**两处独立推导**拼出来，任一处漏项都不报错，
    只表现为「界面上少了东西 / 空了一块」。
    """
    base = sk.skills_dir()
    if not base.is_dir():
        return []
    problems: list[tuple[str, str]] = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        own = (entry / "SKILL.md").exists()
        declared = (entry / skr.LIBRARY_FILE).exists()
        nested = [p for p in entry.rglob("SKILL.md") if p.parent != entry]
        if own and declared:
            problems.append((
                entry.name,
                f"顶层目录既有 SKILL.md 又有 {skr.LIBRARY_FILE} ⇒ 子 skill 全判 core，"
                f"侧栏的 lib:<id> 分组恒空",
            ))
        elif not own and nested and not declared:
            problems.append((
                entry.name,
                f"容器目录未写 {skr.LIBRARY_FILE} ⇒ 其下 {len(nested)} 个 skill "
                f"只在「全部」可见，按库浏览无入口",
            ))
    return problems


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
    # `allowed-tools` 的**变体键**：只读那一个名字会静默丢掉声明（见 to_allowed_tools 的 docstring）
    check("归一: to_allowed_tools 收 `allowed-tools` 与变体键（`allowed-tools-speech`）",
          sp.to_allowed_tools({"allowed-tools": "hub_read, hub_write",
                               "allowed-tools-speech": "hub_voice_prepare"})
          == ["hub_read", "hub_write", "hub_voice_prepare"],
          sp.to_allowed_tools({"allowed-tools": "hub_read", "allowed-tools-speech": "hub_voice_prepare"}))
    check("归一: to_allowed_tools 跨键**去重**（同名两边都写只留一份）",
          sp.to_allowed_tools({"allowed-tools": "hub_read",
                               "allowed-tools-speech": "hub_read, hub_tts"})
          == ["hub_read", "hub_tts"],
          sp.to_allowed_tools({"allowed-tools": "hub_read", "allowed-tools-speech": "hub_read, hub_tts"}))
    check("归一: to_allowed_tools 不认识的键不收（`allowed-skills` 不是工具声明）",
          sp.to_allowed_tools({"allowed-skills": "hub_read", "allowed-tools": "hub_write"})
          == ["hub_write"],
          sp.to_allowed_tools({"allowed-skills": "hub_read", "allowed-tools": "hub_write"}))
    _variant = sp.parse_skill(
        "---\nname: v\nallowed-tools-speech: hub_voice_prepare\n---\n正文没提任何工具\n", "v")
    check("解析: 只有变体键声明 ⇒ 也进 allowed_tools（否则 missingTools 恒空、徽标不亮）",
          _variant.metadata.allowed_tools == ["hub_voice_prepare"]
          and _variant.foreign_tool_refs == [],
          (_variant.metadata.allowed_tools, _variant.foreign_tool_refs))
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
    check("真实目录: 定位到 `backend-py/app/skills`（2026-09-15 并入 app/；删 backend/ 不受影响）",
          sk.skills_dir() == PROJECT_ROOT / "backend-py" / "app" / "skills"
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
    # ⚠️ 两类「静默不注入」：都**不报错、不告警**，只是那次注入悄悄少了一段。
    #    2026-11-04 实测踩到 —— wuxia-action 两份 skill 同时犯了这两条：
    #      agents: [screenwriter, storyboard_breaker]   # screenwriter 不是合法类型
    #      而它俩本身在库内（wuxia-action/…）           # 库内声明 agents 一律无效
    #    两条错叠在一起，症状一模一样（注入里没有它们），且没有任何地方会喊。
    _valid_types = set(rt.VALID_AGENT_TYPES)
    _unknown, _vendor_declared = _binding_problems(sk.list_skill_ids(), set(core), _valid_types)
    check("真实目录: 每份 SKILL.md 的 `agents:` 都在 VALID_AGENT_TYPES 里（写错名字=静默不注入）",
          not _unknown, _unknown)
    check("真实目录: 库内 skill（非 core）不得声明 `agents:`（默认绑定只扫 core，声明即空头支票）",
          not _vendor_declared, _vendor_declared)
    check("真实目录: `agents:` 的合法取值是真源 rt.VALID_AGENT_TYPES（不是手抄清单）",
          _valid_types == set(rt.VALID_AGENT_TYPES) and len(_valid_types) >= 5, sorted(_valid_types))
    # 工具声明：变体键只有 `to_allowed_tools` 收得下。漏读 ⇒ required_tools 少一项 ⇒
    # missingTools 少一项 ⇒ 前端「缺 N 个工具」徽标不亮；而正文从没提过那个工具，
    # `foreign_tool_refs` 也补不上 ⇒ 「依赖本项目未提供的工具」这件事只剩静默。
    _voice = sk.skills_dir() / "production-tools" / "voice-clone" / "SKILL.md"
    _voice_tools = (sp.parse_skill(_voice.read_text(encoding="utf-8"), "voice-clone")
                    .metadata.allowed_tools if _voice.exists() else [])
    check("真实目录: `voice-clone` 的 `hub_voice_prepare` 进了 allowed_tools（它只写在 `allowed-tools-speech` 里）",
          "hub_voice_prepare" in _voice_tools, _voice_tools)
    # 宿主工具清单 = `/skills` 判 missingTools 的唯一依据，它必须与**真实工厂产出**严格相等：
    # 少一个 ⇒ 有 skill 声明那个工具时徽标报「缺」，而那工具其实存在（假警报）；
    # 多一个 ⇒ 徽标装死，绑定后调用落空（假绿灯）。
    # 旧守卫只对着**冻结的 TS 快照**比对，快照定格在 Node 后端删除那一刻 ⇒ 漏了 3 个真实工具。
    _live_tools = ar.live_host_tool_names()
    check("真实目录: HOST_TOOL_NAMES 与真实工具工厂**严格相等**（双向零差集）",
          _live_tools == ar.HOST_TOOL_NAMES, sorted(_live_tools ^ ar.HOST_TOOL_NAMES))
    check("真实目录: `search_script_library` / `list_skill_refs` / `read_skill_ref` 在清单里（2026-11-04 新增，旧静态清单漏掉）",
          {"search_script_library", "list_skill_refs", "read_skill_ref"} <= ar.HOST_TOOL_NAMES,
          sorted(ar.HOST_TOOL_NAMES))
    # live 若读工厂 dict 的**键**而不是 `Tool.id`，「真实工具集」就与「模型能调用的名字」分家：
    # 键=id 时看不出差别，一旦某工厂沿用 TS 的 camelCase 键，live 会凭空多一个模型永远调不到的
    # 名字 ⇒ HOST_TOOL_NAMES 守卫误红、missingTools 跟着说错。这里把两侧钉死在同一来源上。
    _live_objs = ar.live_tool_objects()
    _ids = [t.id for t in _live_objs]
    _reg = _ToolRegistry({f"k{i}": t for i, t in enumerate(_live_objs)})
    _openai_names = {entry["function"]["name"] for entry in _reg.as_openai_tools()}
    check("真实目录: live_host_tool_names 与驱动侧 `function.name` **同源**（都取 Tool.id，不取工厂键）",
          set(_live_tools) == _openai_names, sorted(set(_live_tools) ^ _openai_names))
    check("真实目录: 宿主工具集中无重名 `Tool.id`（重名会让注册表静默覆盖、一个工具凭空消失）",
          len(_ids) == len(set(_ids)), len(_ids) - len(set(_ids)))
    check("真实目录: 两个 subagent 专属工具真的由工厂产出（常量只是声明，不是代答）",
          ar.SUBAGENT_ONLY_TOOL_NAMES <= set(_live_tools),
          sorted(ar.SUBAGENT_ONLY_TOOL_NAMES - set(_live_tools)))
    # Agent 名册：`AGENT_PHASES` 是「挂在流水线哪个阶段」的**展示表**（orchestrator 不在其中），
    # `VALID_AGENT_TYPES` 才是能运行的 Agent 全集。拿展示表当名册 ⇒ 默认绑定为
    # `agents: [orchestrator]` 的 skill 在本页永远「未绑定」，而运行期照注入 —— 整族静默。
    check("agent 名册: `AGENT_PHASES` 确实是 `VALID_AGENT_TYPES` 的**真子集**（差额＝不参与阶段展示的 Agent）",
          set(ar.AGENT_PHASES) < set(ar.VALID_AGENT_TYPES)
          and set(ar.VALID_AGENT_TYPES) - set(ar.AGENT_PHASES) == {"orchestrator"},
          sorted(set(ar.VALID_AGENT_TYPES) - set(ar.AGENT_PHASES)))
    with engine.connect() as _conn:
        _seen: list[str] = []
        _orig_resolve = skr.resolve_default_skills
        try:
            skr.resolve_default_skills = lambda a: (_seen.append(a), [])[1]  # type: ignore[assignment]
            skr._load_skill_bindings(_conn)  # noqa: SLF001 —— 判据要盯的就是这个函数
        finally:
            skr.resolve_default_skills = _orig_resolve  # type: ignore[assignment]
    check("agent 名册: `/skills` 反向绑定按 `VALID_AGENT_TYPES` 枚举（orchestrator 也被问过）",
          set(_seen) == set(ar.VALID_AGENT_TYPES), sorted(set(ar.VALID_AGENT_TYPES) ^ set(_seen)))
    _meta = TestClient(app).get("/api/v1/skills/meta").json()["data"]
    _meta_types = [a["type"] for a in _meta["agents"]]
    check("agent 名册: `/skills/meta` 的 agents 覆盖全部 Agent（顺序＝AGENT_SPECS 声明序）",
          tuple(_meta_types) == tuple(ar.VALID_AGENT_TYPES), _meta_types)
    _orch = next((a for a in _meta["agents"] if a["type"] == "orchestrator"), None)
    check("agent 名册: orchestrator 在侧栏有分组（phase 为空串，label 走出厂名）",
          _orch is not None and _orch["phase"] == "" and _orch["label"] == ar.get_default_name("orchestrator"),
          _orch)
    # 反验：判据不许见谁都喊 —— 合格数据必须一声不响
    _ok_unknown, _ok_vendor = _binding_problems(
        ["good-core"], {"good-core"}, {"storyboard_breaker"},
        load=lambda _sid: sp.parse_skill("---\nname: x\nagents: [storyboard_breaker]\n---\n正文", "x"))
    check("反验: 合法绑定（core + 真类型）⇒ 两条判据都不报",
          not _ok_unknown and not _ok_vendor, (_ok_unknown, _ok_vendor))
    _bad_unknown, _bad_vendor = _binding_problems(
        ["lib/two-ways-wrong"], set(), {"storyboard_breaker"},
        load=lambda _sid: sp.parse_skill("---\nname: x\nagents: [screenwriter]\n---\n正文", "x"))
    check("反验: 库内 + 写错类型 ⇒ 两条判据同时报（这正是 2026-11-04 那份 skill 的形态）",
          _bad_unknown and _bad_vendor, (_bad_unknown, _bad_vendor))

    # ---- 库声明：`name` 是前端分组键 `lib:<id>`，重复即静默并组（不报错、只是两个库混一起）
    _lib_problems = _library_id_problems()
    check("真实目录: 每个库声明的 `name` 唯一且可作 `lib:<id>` 分组键（重复 ⇒ 侧栏静默并组）",
          not _lib_problems, _lib_problems)
    _lib_count = sum(
        1 for e in sk.skills_dir().iterdir()
        if e.is_dir() and (e / skr.LIBRARY_FILE).exists()
    )
    check("真实目录: 确实扫到了多个库声明（判据别是空跑）", _lib_count >= 2, _lib_count)
    # 反验：四个坏/好法各造一棵最小树（切 `SKILLS_DIR` 隔离，跑完必须恢复）
    _rev = Path(tempfile.mkdtemp(prefix="lib_ids_"))
    for _name in ("one", "two"):
        (_rev / _name).mkdir(parents=True, exist_ok=True)
        (_rev / _name / skr.LIBRARY_FILE).write_text("name: same\n", encoding="utf-8")
    _prev_dir = os.environ.get("SKILLS_DIR")
    os.environ["SKILLS_DIR"] = str(_rev)
    try:
        check("反验: 两个库声明同一 name ⇒ 报重复",
              len(_library_id_problems()) == 1, _library_id_problems())
        (_rev / "two" / skr.LIBRARY_FILE).write_text("name: two\n", encoding="utf-8")
        check("反验: name 两两不同 ⇒ 一声不响", not _library_id_problems(), _library_id_problems())
        (_rev / "two" / skr.LIBRARY_FILE).write_text("name: 'a:b'\n", encoding="utf-8")
        check("反验: name 含冒号 ⇒ 报不可作 `lib:<id>` 键",
              len(_library_id_problems()) == 1, _library_id_problems())
        (_rev / "two" / skr.LIBRARY_FILE).write_text("label: 没写 name\n", encoding="utf-8")
        check("反验: 声明没有 name ⇒ 回退目录名，仍不报（设计允许与目录名解耦，不是缺省错误）",
              not _library_id_problems(), _library_id_problems())
    finally:
        if _prev_dir is None:
            os.environ.pop("SKILLS_DIR", None)
        else:
            os.environ["SKILLS_DIR"] = _prev_dir

    # ---- 容器形态：侧栏分组由「声明文件 + 目录形态」两处独立推导，任一处漏项都只有症状
    _container = _container_problems()
    check("真实目录: 顶层目录形态自洽（漏写 library.yaml ⇒ skill 无侧栏入口；两者齐备 ⇒ 库分组恒空）",
          not _container, _container)
    _top_dirs = [e for e in sk.skills_dir().iterdir() if e.is_dir()]
    check("真实目录: 顶层目录确实被扫到（判据别是空跑）", len(_top_dirs) >= 10, len(_top_dirs))
    # 反验：五种形态（切 `SKILLS_DIR` 隔离，跑完必须恢复）
    _rev2 = Path(tempfile.mkdtemp(prefix="containers_"))
    _prev_dir2 = os.environ.get("SKILLS_DIR")
    os.environ["SKILLS_DIR"] = str(_rev2)
    try:
        (_rev2 / "nocfg" / "sub").mkdir(parents=True)
        (_rev2 / "nocfg" / "sub" / "SKILL.md").write_text("# 子\n", encoding="utf-8")
        check("反验: 容器漏写 library.yaml ⇒ 报「无侧栏入口」并点出子 skill 数",
              len(_container_problems()) == 1 and "1 个 skill" in _container_problems()[0][1],
              _container_problems())
        (_rev2 / "nocfg" / skr.LIBRARY_FILE).write_text("name: nocfg\n", encoding="utf-8")
        check("反验: 补上 library.yaml ⇒ 一声不响",
              not _container_problems(), _container_problems())

        (_rev2 / "both" / "sub").mkdir(parents=True)
        (_rev2 / "both" / "SKILL.md").write_text("# 自身\n", encoding="utf-8")
        (_rev2 / "both" / "sub" / "SKILL.md").write_text("# 子\n", encoding="utf-8")
        (_rev2 / "both" / skr.LIBRARY_FILE).write_text("name: both\n", encoding="utf-8")
        check("反验: 既是自有 skill 又声明为库 ⇒ 报「库分组恒空」",
              len(_container_problems()) == 1 and "恒空" in _container_problems()[0][1],
              _container_problems())
        (_rev2 / "both" / skr.LIBRARY_FILE).unlink()
        check("反验: 删掉声明后是合法的「带子目录的自有 skill」⇒ 不报",
              not _container_problems(), _container_problems())

        (_rev2 / "empty").mkdir(parents=True)
        (_rev2 / "empty" / skr.LIBRARY_FILE).write_text("name: empty\n", encoding="utf-8")
        check("反验: 空库（有声明、还没装 skill）⇒ 不报（建库未装是合法中间态）",
              not _container_problems(), _container_problems())
    finally:
        if _prev_dir2 is None:
            os.environ.pop("SKILLS_DIR", None)
        else:
            os.environ["SKILLS_DIR"] = _prev_dir2

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
    # ⚠️ Windows 中文控制台是 **GBK**（py3.14 及以前不会自动 UTF-8 ✗）⇒ 不加这行，裸跑第一条 `print` 就 `UnicodeEncodeError` 崩 ✗（守口径见 `check_cli_encoding.py` ✓）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
