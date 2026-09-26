"""路由一致性守卫 —— 找绞杀者迁移特有的一类**静默失效**。

问题：Python 侧注册了参数路由（如 ``GET /agent-configs/{id}``）后，Node 侧同前缀、
同方法、同段数的**未迁移**路径（如 ``GET /agent-configs/defaults``）会被参数路由**抢走**，
永远到不了兜底委派 —— 不报错、不告警，只返回一个看起来合理的 404，
于是**悄悄丢掉了 Node 的实现**（前端表现为「功能莫名失效」）。

判据是**行为**而不是内省（FastAPI 的 ``include_router`` 会包一层 ``_IncludedRouter``，
拿不到 ``.path``）：对每条「Node 有、Python 没注册」的路径真发一次请求，
**必须落到兜底**（``PROXY_TO_NODE=0`` 时表现为 501）。任何其它状态码都说明被吞了。

三类输出：
* ``SHADOW``  —— 未迁移端点被已注册路由吞掉（**致命**，退码 1）
* ``EXTRA``   —— Python 注册了 Node 根本没有的路径（多半是路径写错）
* 其余为正常：要么是已迁移，要么正确走兜底委派

用法：``python tests/route_parity_test.py``
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

# 关键：必须在导入 app 之前指到临时数据根，避免碰真实库；且关掉反代以便用 501 判定
os.environ["DATA_ROOT"] = tempfile.mkdtemp(prefix="parity_")
os.environ["PROXY_TO_NODE"] = "0"

REPO = Path(__file__).resolve().parents[2]

#: TS 源码根：**真源码优先**（迁移期），删掉 ``backend/`` 后**自动回退到 Python 快照**
#: （``tests/frozen_ts_source.py``，由 ``tests/freeze_ts_snapshot.py`` 在删库前生成）。
#: ⚠️ ``PARITY_USE_FROZEN=1`` 可**强制**用快照 —— 用来在真源码还在时就验证「删库后守卫照样能跑」。
_TS_SRC = REPO / "backend" / "src"

if os.environ.get("PARITY_USE_FROZEN") == "1" or not _TS_SRC.is_dir():
    # 快照**物化到临时目录**再当普通目录读 ⇒ 下面 15 处 `_SRC_ROOT / ...` 一行都不用改。
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from frozen_ts import snapshot_root  # noqa: E402

    _SRC_ROOT = snapshot_root()
else:
    _SRC_ROOT = _TS_SRC
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

#: Node 侧「路由文件 → Python 前缀」。只需列**已开始迁移**的域：
#: 整段没迁的域由兜底接管，不存在遮蔽问题。
MIGRATED: dict[str, str] = {
    "dramas.ts": "/api/v1/dramas",
    "episodes.ts": "/api/v1/episodes",
    "characters.ts": "/api/v1/characters",
    "scenes.ts": "/api/v1/scenes",
    "props.ts": "/api/v1/props",
    "storyboards.ts": "/api/v1/storyboards",
    "videos.ts": "/api/v1/videos",
    # ⚠️ Node 是 `api.route('/compose', compose)`，而 compose.ts 的子路径自带 /storyboards、
    #    /episodes ⇒ 真实路径是 /api/v1/compose/episodes/:id/compose-all（看着别扭但保真）
    "compose.ts": "/api/v1/compose",
    "agent.ts": "/api/v1/agent",
    "auto-pipeline.ts": "/api/v1/auto-pipeline",
    "localModels.ts": "/api/v1/local-models",
    "mcp.ts": "/api/v1/mcp",
    # evaluation.ts：5 端点里已迁 2 个（cases / evaluate），另 3 个（optimize/scheduler/run）
    # 依赖未迁件 ⇒ 由 catch-all 兜底（守卫会把它们算进「未注册」，正好当待办清单用）
    "evaluation.ts": "/api/v1/evaluation",
    "merge.ts": "/api/v1/merge",
    "grid.ts": "/api/v1/grid",
    "images.ts": "/api/v1/images",
    "webhooks.ts": "/api/v1/webhooks",
    "visual-graph.ts": "/api/v1/visual-graph",
    "aiVoices.ts": "/api/v1/ai-voices",
    # ⚠️ Node 是 `api.route('/preset/framework', presetFramework)`（路径里就有两段）
    "preset-framework.ts": "/api/v1/preset/framework",
    "presets.ts": "/api/v1/presets",
    "app-settings.ts": "/api/v1/app-settings",
    "agentConfigs.ts": "/api/v1/agent-configs",
    "aiConfigs.ts": "/api/v1/ai-configs",
    "skills.ts": "/api/v1/skills",
    "upload.ts": "/api/v1/upload",
    "export.ts": "/api/v1/export",
    "style-profiles.ts": "/api/v1/style-profiles",
    "usage.ts": "/api/v1/usage",
    "storage.ts": "/api/v1/storage",
    "asset-versions.ts": "/api/v1/asset-versions",
    "traces.ts": "/api/v1/traces",
    "generations.ts": "/api/v1/generations",
    "characterLibrary.ts": "/api/v1/character-library",
    "sceneLibrary.ts": "/api/v1/scene-library",
    "weaponLibrary.ts": "/api/v1/weapon-library",
    "costumeLibrary.ts": "/api/v1/costume-library",
}

#: ⚠️ 正则要写成「**裸标识符** + `.方法('字面量')`」：
#: * 早先只写 `\.(get|post|...)\(` ⇒ 把 ``resp.headers.get('content-length')``、
#:   ``groups.get('x')`` 这类**属性访问**也当成端点（假阳性 ⇒ 计数虚高）；
#: * 但**不能**只锚 `app.` —— 仓里还有 ``router.get(...)``（53 处）与
#:   ``aiProviders.post(...)`` 等别的路由变量名（锚死会漏掉一半端点）。
#: 所以用「前面不是 `.` 或 `$`」来区分「模块级路由变量」与「属性上的同名方法」。
_TS_ROUTE = re.compile(
    r"(?<![\w.$])[A-Za-z_$][\w$]*\.(get|post|put|patch|delete)\(\s*'([^']+)'"
)
_PARAM = re.compile(r"\{[^}]+\}")

#: Python 侧有意多出来的路径（不算 EXTRA）：
#: * ``/health`` 在 Node 的 ``index.ts`` 里定义，不在 routes 目录 ⇒ 不在本脚本的扫描范围；
#: * ``/ai-providers`` 与 ``/ai-configs`` **同源一个文件**（``aiConfigs.ts`` 里 ``aiProviders``
#:   是独立导出的 Hono app），一个文件只映射一个前缀，所以它得走白名单。
_EXTRA_ALLOWLIST = {("GET", "/api/v1/health"), ("GET", "/api/v1/ai-providers"),
                    # 2026-09-24 自研接线 ✓：导演稿分段工具 ✓、超清计划 ✓、混合加载 ✓（Node 侧从来没有 ✓）
                    ("POST", "/api/v1/prompts/segments"),
                    ("POST", "/api/v1/engine/upscale-plan"),
                    ("POST", "/api/v1/engine/hybrid-merge")}


def _node_path(prefix: str, ts_path: str) -> str:
    """``'/:id/foo'`` → ``'/api/v1/x/{id}/foo'``（与 FastAPI 的参数写法对齐）。"""
    parts = [
        f"{{{p[1:]}}}" if p.startswith(":") else p
        for p in ts_path.strip("/").split("/")
        if p
    ]
    return f"{prefix}/{'/'.join(parts)}" if parts else prefix


def _concrete(path: str) -> str:
    """把 ``{id}`` 换成具体值，才能真发请求。"""
    return _PARAM.sub("1", path)


def _pattern(path: str) -> str:
    """参数名归一化 —— **这是比对的关键**。

    Node 侧写 ``:id`` ⇒ 折成 ``{id}``；而 Python 侧我按语义命名（``{drama_id}``）。
    不比参数名的话，**每一条已迁移的详情路由都会被误报成「被吞」**。
    """
    return _PARAM.sub("*", path)


def _registry_drift() -> list[str]:
    """校验 ``services/agent_registry.py`` 与 TS 源不漂移。

    那个文件是 ``agents/index.ts``（``AGENT_PHASES`` / ``DEFAULT_PROMPTS[].name``）与
    ``tools/*.ts``（``tool.id``）的**镜像副本** —— Python 侧无法执行 Mastra 工厂，只能静态镜像。
    若 Node 侧改名/加工具而没同步，症状是**静默的**：`/skills` 的 ``missingTools`` 会误报，
    ``/skills/meta`` 的 Agent 标签会显示错名。所以这里直接从 TS 源码抽取并比对。
    """
    from app.services.agent_registry import (
        AGENT_DEFAULT_NAMES,
        AGENT_PHASES,
        HOST_TOOL_NAMES,
    )

    agents_dir = _SRC_ROOT / "agents"
    problems: list[str] = []

    # ① 宿主工具名 = tools/*.ts 与 subagent.ts 里的 `id: 'xxx'` 并集
    tool_ids: set[str] = set()
    for path in [*sorted((agents_dir / "tools").glob("*.ts")), agents_dir / "subagent.ts"]:
        tool_ids |= set(
            re.findall(r"^\s*id:\s*'([a-z0-9_]+)'", path.read_text(encoding="utf-8"), re.M)
        )
    ts_only = sorted(tool_ids - set(HOST_TOOL_NAMES))
    py_only = sorted(set(HOST_TOOL_NAMES) - tool_ids)
    if ts_only or py_only:
        problems.append(f"宿主工具名漂移：TS 独有 {ts_only} ／ Python 独有 {py_only}")

    index_src = (agents_dir / "index.ts").read_text(encoding="utf-8")

    # ② AGENT_PHASES
    block = re.search(r"AGENT_PHASES[^{]*\{(.*?)\n\}", index_src, re.S)
    ts_phases = dict(re.findall(r"^\s*([a-z_]+):\s*'([^']+)'", block.group(1), re.M)) if block else {}
    if ts_phases and ts_phases != AGENT_PHASES:
        problems.append(f"AGENT_PHASES 漂移：TS={ts_phases} ／ Python={AGENT_PHASES}")
    elif not ts_phases:
        problems.append("AGENT_PHASES 未能从 index.ts 解析出来（正则需要跟着改）")

    # ③ Agent 显示名：DEFAULT_PROMPTS 里每个 `key: { name: '...' }`
    ts_names = dict(
        re.findall(r"^\s{2}([a-z_]+):\s*\{\s*\n\s*name:\s*'([^']+)'", index_src, re.M)
    )
    if ts_names and ts_names != AGENT_DEFAULT_NAMES:
        problems.append(f"Agent 显示名漂移：TS={ts_names} ／ Python={AGENT_DEFAULT_NAMES}")
    elif not ts_names:
        problems.append("DEFAULT_PROMPTS 的 name 未能从 index.ts 解析出来（正则需要跟着改）")

    return problems


def _extract_ts_string_constants(src: str) -> dict[str, str]:
    """从 TS 源抽取 ``const NAME = '...'`` 常量（含**跨行拼接**与反引号模板）。

    ⚠️ 这两个形态都必须吃住，否则会漏比对或误报：

    * 跨行拼接：``const A = 'x, ' +\\n  'y'`` ⇒ 把各段**原样拼接**（不能做空白归一，会改坏字符串）
    * 反引号模板：``const A = `${B}, z` `` ⇒ 用已解析到的 ``B`` 代入 ``${B}``
    """
    out: dict[str, str] = {}
    lines = src.splitlines()
    index = 0
    while index < len(lines):
        match = re.match(r"^(?:export )?const ([A-Z][A-Z0-9_]*) =\s*(.*)$", lines[index])
        if not match:
            index += 1
            continue
        name, chunk = match.group(1), match.group(2)
        # 续行条件（三种形态都要吃住）：
        #   a) 值写在**下一行**（`= \n  '...'`）⇒ chunk 里还没有任何完整字面量
        #   b) 跨行拼接 （行尾 `+`）
        #   c) 引号/反引号未闭合
        # ⚠️ 但遇到**下一条声明**就停 —— 否则数组类常量（如 GRID_ANGLES）会把后续声明一起吞掉，
        # 导致那些常量根本没机会被解析（静默漏比误报更糟）。
        while (
            not re.search(r"'[^']*'|`[^`]*`", chunk)
            or chunk.count("'") % 2 == 1
            or chunk.count("`") % 2 == 1
            or chunk.rstrip().endswith("+")
        ):
            index += 1
            if index >= len(lines):
                break
            if re.match(r"^(?:export )?(?:const|function|/\*)", lines[index]):
                break
            chunk += "\n" + lines[index]
        index += 1

        quoted = re.findall(r"'([^']*)'", chunk)
        if quoted:
            value = "".join(quoted)  # 原样拼接，勿归一空白
        else:
            template = re.search(r"`([^`]*)`", chunk, re.S)
            value = template.group(1) if template else None
        if value is None:
            continue
        out[name] = re.sub(
            r"\$\{([A-Z_]+)\}", lambda m: out.get(m.group(1), m.group(0)), value
        )
    return out


def _extract_ts_object_string_values(src: str, const_name: str) -> dict[str, str]:
    """抽 ``NAME: Record<string, string> = { ... }`` 里**每个键的值**（含跨行拼接）。

    ⚠️ 2026-09-20：原先只用 ``re.findall(r"'key':\\s*'([^']*)'")`` ✗ —— 那只吃得到**多行值的
    **第一行**（实测 10 条里抽到 1 条，而且抽到的是**残缺值** ✓✗：TS 少了后半句却"看着像对的" ✓）。
    ⇒ 改成与 :func:`_extract_ts_string_constants` 同款的逐行累积：值以 ``+`` 续行或引号未闭合
    时继续收 ✓，最后把引号片段**按序拼接**（不做任何空白归一 ✓）。
    """
    block = re.search(
        rf"{const_name}:\s*Record<string, string>\s*=\s*\{{(.*?)\n\}}", src, re.S)
    if block is None:
        return {}
    lines = block.group(1).splitlines()
    out: dict[str, str] = {}
    index = 0
    while index < len(lines):
        match = re.match(r"\s*'?([a-z0-9-]+)'?:\s*(.*)$", lines[index])
        if not match:
            index += 1
            continue
        key, chunk = match.group(1), match.group(2)
        # ⚠️ 三种形态都要吃住（**初版漏了第一种** ✗ ⇒ 正表 0 条、负面表 10 条 ✓）：
        #   a) **值在下一行**（实测 TS 正表就是 `realistic:` ↓ `'...' +` ✓）
        #   b) 值以 `+` 续行；c) 引号未闭合
        while (not re.search(r"'[^']*'", chunk)
               or chunk.rstrip().endswith("+")
               or chunk.count("'") % 2 == 1):
            index += 1
            if index >= len(lines):
                break
            chunk += "\n" + lines[index]
        index += 1
        quoted = re.findall(r"'([^']*)'", chunk)
        if quoted:
            out[key] = "".join(quoted)
    return out


def _prompt_utils_drift() -> list[str]:
    """校验 ``services/prompt_utils.py`` 的字符串常量与 ``prompt-utils.ts`` 逐字一致。

    这是**最需要漂移守卫**的一处：那些词表就是生成结果本身，改一个词就改变出图/出片，
    而两边不一致时**不会报任何错**，只会「同一个剧在两套后端下风格不同」。

    覆盖范围（2026-09-20 扩过一轮 ✓）：**跨行拼接**的字符串常量 ✓ + 画风目录 key 集合 ✓
    + 两张映射表的 **key 与 value** ✓。

    ⚠️ 原文这里写着「**不覆盖**跨行拼接的常量（``VISUAL_STYLE_VIDEO``）」—— **实测是错的** ✗：
    探针跑下来 ``VISUAL_STYLE_VIDEO`` / ``VIDEO_MOTION_BASE`` / ``IMPERFECTION_ANCHORS`` /
    ``QUALITY_TAIL`` 四个都**抽得准、且与 Python 逐字相同** ✓ ⇒ 它们只是**没被列进 ``expected``** ✗
    （**"没被断言" ≠ "断不了"** ✓）。已补上 ✓ —— 等于白捡的覆盖 ✓。

    ⚠️ 映射表**值**当时确实是真盲区 ✗：`re.findall` 只吃到多行值的**第一行** ⇒ 10 条只抽到 1 条 ✓✗
    ⇒ 现在用与常量同款的「引号拼接 + 续行」逻辑逐条抽、**逐字比对全部值** ✓。
    """
    from app.services import prompt_utils as pu

    src = (_SRC_ROOT / "shared" / "prompt-utils.ts").read_text(encoding="utf-8")
    resolved = _extract_ts_string_constants(src)

    expected = {
        "VISUAL_STYLE_MASTER": pu.VISUAL_STYLE_MASTER,
        "VISUAL_STYLE_CHARACTER": pu.VISUAL_STYLE_CHARACTER,
        "VISUAL_STYLE_SCENE": pu.VISUAL_STYLE_SCENE,
        "UI_OVERLAY_RULE": pu.UI_OVERLAY_RULE,
        "UI_PLATE_CATEGORY": pu.UI_PLATE_CATEGORY,
        "NEGATIVE_BASE": pu.NEGATIVE_BASE,
        "CHARACTER_IMAGE_NEGATIVE": pu.CHARACTER_IMAGE_NEGATIVE,
        "SCENE_IMAGE_NEGATIVE": pu.SCENE_IMAGE_NEGATIVE,
        "STORYBOARD_IMAGE_NEGATIVE": pu.STORYBOARD_IMAGE_NEGATIVE,
        "VIDEO_NEGATIVE": pu.VIDEO_NEGATIVE,
        "PRESET_IMAGE_NEGATIVE": pu.PRESET_IMAGE_NEGATIVE,
        "VISUAL_STYLE_VIDEO": pu.VISUAL_STYLE_VIDEO,
        "VIDEO_MOTION_BASE": pu.VIDEO_MOTION_BASE,
        "IMPERFECTION_ANCHORS": pu.IMPERFECTION_ANCHORS,
        "QUALITY_TAIL": pu.QUALITY_TAIL,
        "IMPERFECTION_ANCHORS": pu.IMPERFECTION_ANCHORS,
        "QUALITY_TAIL": pu.QUALITY_TAIL,
        "THREE_VIEW_COMBINED_LAYOUT": pu.THREE_VIEW_COMBINED_LAYOUT,
        "THREE_VIEW_SIZE": pu.THREE_VIEW_SIZE,
        "ITEM_IMAGE_SIZE": pu.ITEM_IMAGE_SIZE,
    }

    problems: list[str] = []
    for name, py_value in expected.items():
        ts_value = resolved.get(name)
        if ts_value is None:
            problems.append(f"{name}: 未能从 TS 抽取（正则需要跟着改）")
        elif ts_value != py_value:
            problems.append(f"{name} 漂移：TS={ts_value!r} ／ Python={py_value!r}")

    catalog = re.search(r"ART_STYLE_CATALOG: ArtStyleOption\[\] = \[(.*?)\n\]", src, re.S)
    ts_keys = re.findall(r"key: '([^']+)'", catalog.group(1)) if catalog else []
    if not ts_keys:
        problems.append("ART_STYLE_CATALOG 未能从 TS 抽取（正则需要跟着改）")
    elif ts_keys != pu.ART_STYLE_KEYS:
        problems.append(f"画风目录漂移：TS={ts_keys} ／ Python={pu.ART_STYLE_KEYS}")

    style_map = re.search(r"DRAMA_ART_STYLE_MAP: Record<string, string> = \{(.*?)\n\}", src, re.S)
    ts_styles = re.findall(r"^  '?([a-z0-9-]+)'?:", style_map.group(1), re.M) if style_map else []
    if not ts_styles:
        problems.append("DRAMA_ART_STYLE_MAP 未能从 TS 抽取（正则需要跟着改）")
    elif set(ts_styles) != set(pu.DRAMA_ART_STYLE_MAP):
        problems.append(
            f"画风映射表 key 漂移：TS={sorted(set(ts_styles))} ／ Python={sorted(pu.DRAMA_ART_STYLE_MAP)}"
        )

    # ⭐ 2026-09-20 补：**映射表的值**也逐字比 ✓（这些英文词**就是生成结果** ✓，
    #    此前只比了 key ✗ ⇒ 值整句被改/被截断，守卫一声不吭 ✓✗）
    # ⚠️ 负面表在 Python 侧是**私有名** `_DRAMA_ART_NEGATIVE_MAP` ✓（TS 侧叫 DRAMA_ART_NEGATIVE_MAP）；
    #    此前**只**比了正表的 key ✗ ⇒ 负面表连 key 都没守 ✓✗（这一轮的值比对顺带把它的 key 也守上了 ✓）
    for const_name, py_map in (("DRAMA_ART_STYLE_MAP", pu.DRAMA_ART_STYLE_MAP),
                               ("DRAMA_ART_NEGATIVE_MAP", pu._DRAMA_ART_NEGATIVE_MAP)):
        ts_values = _extract_ts_object_string_values(src, const_name)
        if not ts_values:
            problems.append(f"{const_name} 未能从 TS 抽取值（抽取器要跟着改）")
            continue
        # ⭐ **检查器自检** ✓：抽到的条数必须与 Python 表**一样多** ——
        #    否则"只抽到一半"会被读成"另一半没漂移" ✗✗（本仓的老教训：**没读到 ≠ 通过** ✓）
        if len(ts_values) != len(py_map):
            problems.append(
                f"{const_name} 抽取条数 {len(ts_values)} ≠ Python 表 {len(py_map)}"
                f" ⇒ 抽取器漏了 ✗（**别把「没抽到」当成「没漂移」** ✓）")
            continue
        missing = sorted(k for k in ts_values if k not in py_map)
        if missing:
            problems.append(f"{const_name} 缺 key：{missing}")
        for key, ts_value in ts_values.items():
            py_value = py_map.get(key)
            if py_value is not None and py_value != ts_value:
                problems.append(
                    f"{const_name}[{key!r}] 值漂移：TS={ts_value!r} ／ Python={py_value!r}")

    # 表情预设：key 与前端共用，必须逐项一致（含英文表情词本身）
    presets = re.search(r"EXPRESSION_PRESETS: Array<[^>]*> = \[(.*?)\n\]", src, re.S)
    ts_pairs = (
        re.findall(r"key: '([^']+)', label: '[^']*', en: '([^']+)'", presets.group(1))
        if presets
        else []
    )
    py_pairs = [(p["key"], p["en"]) for p in pu.EXPRESSION_PRESETS]
    if not ts_pairs:
        problems.append("EXPRESSION_PRESETS 未能从 TS 抽取（正则需要跟着改）")
    elif ts_pairs != py_pairs:
        problems.append(f"表情预设漂移：TS={ts_pairs} ／ Python={py_pairs}")

    return problems


#: ── 适配器层镜像检查 ────────────────────────────────────────────────────
#: 适配器是逐字移植的，最危险的漂移是**静默的**：
#: ① 厂商注册表少一家（用户配了就报「未知 provider」）；
#: ② 默认模型名改了（行为变了但不报错）；
#: ③ 端点路径改了（请求打到 404）；
#: ④ 报错文案改了（用户看不懂，且前端可能按文案匹配）。
_ADAPTER_TS_DIR = _SRC_ROOT / "services" / "adapters"
_ADAPTER_PY_DIR = Path(__file__).resolve().parents[1] / "app" / "services" / "adapters"

#: TS 注册表名 → Python 字典名
_ADAPTER_REGISTRIES = {
    "imageAdapters": "image_adapters",
    "videoAdapters": "video_adapters",
    "ttsAdapters": "tts_adapters",
    "textAdapters": "text_adapters",
}

_REGISTRY_BLOCK = re.compile(r"export const (\w+)\s*:[^=]*=\s*\{(.*?)\n\}", re.S)
_REGISTRY_KEY = re.compile(r"^\s*(?:'([^']+)'|\"([^\"]+)\"|([A-Za-z_$][\w$]*))\s*:", re.M)

#: 用户可见的报错文案（两边都必须逐条找得到）
_ADAPTER_MESSAGES = [
    "No image URL or task_id in response",
    "No image URL in response",
    "No task_id or video_url in response",
    "No task_id in Vidu response",
    "No image data in Gemini response",
    "Gemini generation failed",
    "Gemini generation stopped: ",
    "Generation failed",
    "Video generation failed",
    "Vidu generation failed",
    "Unknown state: ",
    "TTS generation failed",
    "No audio data in response",
    "SD 未返回图片",
    "SD WebUI 同步模式，无需轮询",
    "CosyVoice 未返回音频数据",
    "Unexpected Ali image response: ",
    "Unexpected Ali video response: ",
    "Generate an image",
    "图片生成被内容安全拦截：Gemini 判定该提示词或参考图触发了安全策略。"
    "请修改关键帧/角色提示词或更换参考图后重试。",
]

#: 默认模型名（改了会静默改变行为）
_ADAPTER_DEFAULT_MODELS = [
    "gemini-2.5-flash-image",
    "doubao-seedream-5-0-lite",
    "doubao-seedance-1-5-pro-251215",
    "wan2.6-t2i",
    "wan2.6-i2v-flash",
    "viduq3-turbo",
    "speech-2.8-hd",
    "cosyvoice-v2",
    "dall-e-3",
    "DPM++ 2M Karras",
]

#: 端点路径片段。
#: ⚠️ 只放**真实存在于源码的字面量**：前缀与路径在两边都是 ``join(baseUrl, '/v1', '/x')``
#: 的**两个独立实参**，所以不能拿拼起来的 ``/v1/x`` 来比对（第一次写就踩到，误报 2 条）。
_ADAPTER_PATHS = [
    "/image_generation",
    "/image_generation/task/",
    "/images/generations",
    "/images/task/",
    "/api/v3/images/generations",
    "/images/generations/",
    "/services/aigc/image-generation/generation",
    "/services/aigc/video-generation/video-synthesis",
    "/v1beta",
    ":generateContent",
    "/sdapi/v1/txt2img",
    "/sdapi/v1/img2img",
    "/video_generation",
    "/video_generation/task/",
    "/api/v3/contents/generations/tasks",
    "/contents/generations/tasks/",
    "/ent/v2/img2video",
    "/t2a_v2",
    "/inference_zero_shot",
    "/compatible-mode/v1",
]


def _read_all(directory: Path, suffix: str) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(directory.glob(f"*{suffix}"))
    )


def _normalize_source(text: str) -> str:
    """去掉空白、引号与**转义换行**，让两边的字面量可比。

    两处坑（都是写这个守卫时踩到的）：

    1. TS 的**隐式拼接多行**字面量（``'a' + 'b'``）在 Python 里是相邻字面量（``"a" "b"``），
       去掉引号与空白后才是同一串；
    2. TS 源码里的 ``\\n`` 是**两个字符**（反斜杠 + n），而 Python 侧写 ``"…\\n"``
       在运行时是**真换行** —— 只去空白的话，前者会留下 ``\\n`` 造成假漂移。
       所以这里把 ``\\n`` 转义序列也一并抹掉（抹掉后两边都成了纯文本）。
    """
    return re.sub(r"[\s\"']", "", text.replace("\\n", ""))


def _adapters_drift() -> list[str]:
    from app.services.adapters import registry as adapter_registry

    ts_text = _read_all(_ADAPTER_TS_DIR, ".ts")
    py_text = _read_all(_ADAPTER_PY_DIR, ".py")
    ts_flat, py_flat = _normalize_source(ts_text), _normalize_source(py_text)
    problems: list[str] = []

    # ① 厂商注册表 key（含顺序：TS 用 Object.keys 列给用户看，顺序会出现在报错里）
    blocks = dict(_REGISTRY_BLOCK.findall(ts_text))
    for ts_name, py_name in _ADAPTER_REGISTRIES.items():
        block = blocks.get(ts_name)
        if block is None:
            problems.append(f"{ts_name} 未能从 registry.ts 抽取（正则需要跟着改）")
            continue
        ts_keys = [
            g1 or g2 or g3 for g1, g2, g3 in _REGISTRY_KEY.findall(block)
        ]
        py_keys = list(getattr(adapter_registry, py_name))
        if ts_keys != py_keys:
            problems.append(f"{ts_name} 注册表漂移：TS={ts_keys} ／ Python={py_keys}")

    # ②③④ 文案 / 默认模型 / 路径
    for label, literals in (
        ("文案", _ADAPTER_MESSAGES),
        ("默认模型", _ADAPTER_DEFAULT_MODELS),
        ("端点路径", _ADAPTER_PATHS),
    ):
        for literal in literals:
            flat = _normalize_source(literal)
            if flat not in ts_flat:
                problems.append(f"适配器{label}在 TS 侧找不到（可能已改）：{literal}")
            elif flat not in py_flat:
                problems.append(f"适配器{label}未同步到 Python：{literal}")

    return problems


#: ── text-generation 镜像检查 ────────────────────────────────────────────
#: 这里面的漂移最隐蔽：
#: * **提示词常量**是直接喂给模型的系统指令，改一个字模型行为就变；
#: * **词表**（38/32/29/7/56 条）里打错一个字，拆分结果就会悄悄不同。
_TEXT_GEN_TS = _SRC_ROOT / "services" / "text-generation.ts"

_TEXT_GEN_PROMPTS = [
    "ACTION_SYSTEM_PROMPT",
    "SPLIT_SYSTEM_PROMPT",
    "CONTINUE_RAW_SYSTEM_PROMPT",
    "CONTINUE_SCRIPT_SYSTEM_PROMPT",
    "OPTIMIZE_PROMPT_SYSTEM_PROMPT",
    "OPTIMIZE_TIMELINE_SYSTEM_PROMPT",
    "SPLIT_VISUALS_SYSTEM_PROMPT",
    "SPLIT_VISUALS_EXAMPLES",
    "VOICE_TAG_SYSTEM_PROMPT",
]

_TEXT_GEN_TABLES = [
    "SPLIT_CLOTHING_KEYWORDS",
    "SPLIT_WEAPON_KEYWORDS",
    "SPLIT_WEAPON_BANNED",
    "SPLIT_NOISE_KEYWORDS",
    "SPLIT_ACCESSORY_KEYWORDS",
    "SPLIT_PROP_NOISE_KEYWORDS",
    "SPLIT_REDUNDANT_PREFIXES",
    "SPLIT_REDUNDANT_SUFFIXES",
]


def _text_generation_drift() -> list[str]:
    from app.services import text_generation as tg

    src = _TEXT_GEN_TS.read_text(encoding="utf-8")
    # TS 里 prompt 是多段字面量用 `+` 拼的：先把 `' + '` 粘掉，再统一去引号与空白
    joined = re.sub(r"['\"]\s*\+\s*['\"]", "", src)
    flat = _normalize_source(joined)
    problems: list[str] = []

    for name in _TEXT_GEN_PROMPTS:
        value = _normalize_source(getattr(tg, name))
        if not value:
            problems.append(f"{name} 在 Python 侧为空")
        elif value not in flat:
            problems.append(f"{name} 与 TS 不一致（**提示词漂移**，会改变模型行为）")

    for name in _TEXT_GEN_TABLES:
        block = re.search(rf"\b{name}\b\s*=\s*\[(.*?)\]", src, re.S)
        if block is None:
            problems.append(f"{name} 未能从 TS 抽取（正则需要跟着改）")
            continue
        ts_items = re.findall(r"'([^']*)'", block.group(1))
        py_items = [str(item) for item in getattr(tg, name)]
        if ts_items != py_items:
            only_ts = [x for x in ts_items if x not in py_items]
            only_py = [x for x in py_items if x not in ts_items]
            problems.append(
                f"{name} 漂移（顺序也须一致）：TS 独有 {only_ts} ／ Python 独有 {only_py}"
            )

    return problems


#: ── json.dumps 紧凑性守卫 ──────────────────────────────────────────────
#: JS 的 ``JSON.stringify`` 永远是**紧凑**的（``{"a":1}``），而 Python 的
#: ``json.dumps`` 默认是 ``(', ', ': ')``（``{"a": 1}``）。这些字符串会：
#: 写进**与 Node 共用**的 DB 列、作为**请求体发给厂商**、或**直接作为响应体**返回 ——
#: 三种场景下多一个空格都是偏差。这个偏差已在 8 处真实存在过（image-generation、
#: asset-versions、dramas、presets、ai-configs、text-generation、resource-library、export）。
_JSON_DUMPS_ALLOW = {
    "color_grade.py",  # 深拷贝：json.loads(json.dumps(...))
    "task_logger.py",  # 深拷贝 + 日志格式化（indent=2 是有意的）
    "optimizer.py",  # 优化历史文件：镜像 `JSON.stringify(history, null, 2)`（indent=2 有意）
    "local_model_scan.py",  # configs/model-paths.json：镜像 `JSON.stringify(cfg, null, 2)`（indent=2 有意）
    "style_profiles.py",  # 提炼 prompt 里的测量事实：镜像 `JSON.stringify(measurements, null, 2)`（indent=2 有意）
    "jianying_draft.py",  # 剪映草稿 JSON：镜像 `JSON.stringify(content, null, 2)`（indent=2 有意）
    # 干跑报告：**本地调试产物**，给人看的（indent=1 有意 ✓）。与上面几条的区别是它
    # **没有 Node 对应物** ✗ ⇒ 根本不承担「镜像紧凑性」义务（这条守卫防的是
    # 「写进共用 DB 列 / 发给厂商 / 直接当响应体」的字符串 ✓ 三者它都不属于 ✓）。
    "dryrun.py",
    # 上下文摘要**提示词**（发给 LLM 的正文 ✓，indent=1 有意）：与 dryrun 同理 ——
    # **没有 Node 对应物** ✗、也不属于「写共用 DB 列 / 发给厂商的响应体 / 直接当 API 响应体」
    # 这三类 ✓。缩进是给模型读执行记录用的（1 字符缩进 ⇒ 可读性 vs token 的折中 ✓）。
    "context_budget.py",
}


def _json_dumps_drift() -> list[str]:
    """用 **AST** 找缺紧凑分隔符的 ``json.dumps``。

    为什么不用文本扫描：第一版就是文本扫的，结果把 **docstring 里讲解这个坑的示例**
    和 ``json.dumps(x, **_JSON)``（分隔符藏在别处的 kwargs 里）都误报了。
    AST 精确知道「这是不是一次 json.dumps 调用」「关键参数写在哪」，
    并且对 ``**kwargs`` 展开直接放行（假定调用方已处理）。
    """
    import ast  # noqa: PLC0415

    app_dir = Path(__file__).resolve().parents[1] / "app"
    problems: list[str] = []
    for path in sorted(app_dir.rglob("*.py")):
        # ⚠️ 2026-09-15 三目录并入 `app/` 后，`app/scripts/`（工具链）与 `app/skills/`（技能附带的
        #    脚本，如 `production-tools/beat-sync-editor/scripts/*.py`）也落在 `app/` 下 ——
        #    它们**不是** TS 镜像代码，`json.dumps` 用于 CLI/日志格式化，不该按「镜像紧凑性」判。
        #    实测：不加此排除会误报 **7 条** ✗。
        if {"scripts", "skills"} & set(path.relative_to(app_dir).parts):
            continue
        if path.name in _JSON_DUMPS_ALLOW:
            continue
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError:  # pragma: no cover
            continue
        relative = path.relative_to(app_dir).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == "dumps"
                and isinstance(func.value, ast.Name)
                and func.value.id == "json"
            ):
                continue
            if any(keyword.arg == "separators" for keyword in node.keywords):
                continue
            if any(keyword.arg is None for keyword in node.keywords):
                continue  # `**kwargs` 展开，无法静态判断
            problems.append(
                f"{relative}:{node.lineno} 的 json.dumps 缺紧凑分隔符"
                "（Node 的 JSON.stringify 无空格）—— 若确属深拷贝/日志格式化请加白名单"
            )
    return problems


#: ── 视觉图谱镜像检查 ────────────────────────────────────────────────────
#: `VISUAL_GRAPH` 是一张**数据契约表**：`zh` 别名要与 DB 里的实际取值对得上
#: （对不上就退化成「中文混进英文 prompt」），`en` 会直接进 prompt 文本。
#: 41 个节点、上百个别名，靠肉眼比对不现实 ⇒ 逐条机械比对。
_VISUAL_GRAPH_TS = _SRC_ROOT / "shared" / "visual-graph.ts"


def _visual_graph_drift() -> list[str]:
    from app.services import visual_graph as vgp

    src = _VISUAL_GRAPH_TS.read_text(encoding="utf-8")
    problems: list[str] = []
    for category in ("shot_size", "composition", "movement", "lighting"):
        block = re.search(rf"\b{category}:\s*\[(.*?)\n  \],", src, re.S)
        if block is None:
            problems.append(f"{category} 未能从 visual-graph.ts 抽取（正则需要跟着改）")
            continue
        ts_nodes = [
            (re.findall(r"'([^']*)'", zh_list), en)
            for zh_list, en in re.findall(
                r"\{\s*zh:\s*\[([^\]]*)\],\s*en:\s*'([^']+)'", block.group(1)
            )
        ]
        py_nodes = [(node["zh"], node["en"]) for node in vgp.VISUAL_GRAPH[category]]
        if ts_nodes == py_nodes:
            continue
        first_bad = next(
            (index for index, (ts, py) in enumerate(zip(ts_nodes, py_nodes)) if ts != py),
            min(len(ts_nodes), len(py_nodes)),
        )
        problems.append(
            f"{category} 图谱漂移：TS {len(ts_nodes)} 项 ／ Python {len(py_nodes)} 项；"
            f"首个不一致在第 {first_bad + 1} 项 "
            f"（TS={ts_nodes[first_bad] if first_bad < len(ts_nodes) else None} ／ "
            f"Python={py_nodes[first_bad] if first_bad < len(py_nodes) else None}）"
        )
    return problems


#: ── 运镜表 / 宫格角度表镜像检查 ──────────────────────────────────────────
#: 这两张表**直接进 prompt**（中文构图指导是用户可见的画面描述），
#: 且运镜表的**顺序就是匹配优先级**（错序会让「跟拍」被「斜线跟拍」捕获）⇒ 逐条比对。
_CMG_TS = _SRC_ROOT / "shared" / "camera-movement-guides.ts"
_PU_TS = _SRC_ROOT / "shared" / "prompt-utils.ts"


def _prompt_tables_drift() -> list[str]:
    from app.services import camera_movement_guides as cmg
    from app.services import prompt_utils as pu

    problems: list[str] = []

    # ① 运镜表：zh（含顺序）与全部 start/end 短语逐条比对
    ts_cmg = _CMG_TS.read_text(encoding="utf-8")
    ts_zh = re.findall(r"\{\s*zh:\s*'([^']+)'", ts_cmg)
    py_zh = [guide["zh"] for guide in cmg.CAMERA_MOVEMENT_GUIDES]
    if not ts_zh:
        problems.append("运镜表 zh 未能从 TS 抽取（正则需要跟着改）")
    elif ts_zh != py_zh:
        problems.append(f"运镜表顺序/内容漂移：TS={ts_zh} ／ Python={py_zh}")

    ts_start = re.findall(r"start:\s*'([^']+)'", ts_cmg)
    ts_end = re.findall(r"end:\s*'([^']+)'", ts_cmg)
    py_start = [guide["start"] for guide in cmg.CAMERA_MOVEMENT_GUIDES]
    py_end = [guide["end"] for guide in cmg.CAMERA_MOVEMENT_GUIDES]
    if ts_start != py_start or ts_end != py_end:
        problems.append(
            f"运镜首/尾帧短语漂移：start 差异={[a[:12] for a, b in zip(ts_start, py_start) if a != b]} "
            f"／ end 差异={[a[:12] for a, b in zip(ts_end, py_end) if a != b]}"
        )

    # ② 宫格角度表（25 条，顺序影响画面节奏）
    block = re.search(r"const GRID_ANGLES = \[(.*?)\]", _PU_TS.read_text(encoding="utf-8"), re.S)
    ts_angles = re.findall(r"'([^']+)'", block.group(1)) if block else []
    if not ts_angles:
        problems.append("GRID_ANGLES 未能从 TS 抽取（正则需要跟着改）")
    elif ts_angles != list(pu._GRID_ANGLES):
        problems.append(f"宫格角度表漂移：TS={ts_angles} ／ Python={list(pu._GRID_ANGLES)}")

    return problems


#: ── 提示词规范块镜像检查（S5 前置） ──────────────────────────────────────
#: 这四个块被 Agent 的 system prompt、工具 instruction、SKILL 文档**共同引用**，
#: 当初就是因为散成三份出现表述差异才收口到一处 ⇒ 必须逐字镜像（含示例英文与中文标点）。
_PROMPT_BLOCKS_TS = _SRC_ROOT / "shared" / "prompt-blocks.ts"

_PROMPT_BLOCK_NAMES = (
    "SCREENPLAY_FORMAT_RULES",
    "IMAGE_PROMPT_TEMPLATE_CHARACTER",
    "IMAGE_PROMPT_TEMPLATE_SCENE",
    "IMAGE_PROMPT_TEMPLATE_SHOT",
)


def _prompt_blocks_drift() -> list[str]:
    from app.services import prompt_blocks as pb

    ts_text = _PROMPT_BLOCKS_TS.read_text(encoding="utf-8")
    problems: list[str] = []

    for name in _PROMPT_BLOCK_NAMES:
        match = re.search(rf"export const {name} = `([^`]*)`", ts_text, re.S)
        if match is None:
            problems.append(f"提示词块 {name} 未能从 TS 抽取（正则需要跟着改）")
            continue
        ts_value = match.group(1)
        py_value = getattr(pb, name)
        if ts_value == py_value:
            continue
        # 报出首个差异位置，便于一眼定位（整段文本太长，只报位置与字符）
        for index, (ts_char, py_char) in enumerate(zip(ts_value, py_value)):
            if ts_char != py_char:
                problems.append(
                    f"提示词块漂移：{name} 第 {index} 字符 TS={ts_char!r} ／ Python={py_char!r}"
                )
                break
        else:
            problems.append(
                f"提示词块漂移：{name} 长度不同 TS={len(ts_value)} ／ Python={len(py_value)}"
            )

    return problems


#: ── 查询参数名镜像检查 ──────────────────────────────────────────────
#: Node 用 ``c.req.query('X')`` 读查询参数，**X 就是线上契约**（前端按 X 拼 URL）。
#: FastAPI 的形参名默认就是查询键 ⇒ 蛇形形参**收不到 camelCase 参数**，而且因为
#: 带默认值，会被**静默忽略**（不报错、只是功能悄悄失效）—— ``preset-framework`` 的
#: ``?excludeFamily=`` 就这样失效过：路由永远拿不到排除项，随机家族**偶尔**正好撞上
#: 被排除的那个，于是测试表现为「偶发失败」（单跑 6 次挂 2 次）。这类漂移文本比对
#: 抓不到（路径是对的），只能查**参数名**。
def _query_param_drift() -> list[str]:
    routes_dir = _SRC_ROOT / "routes"
    py_dir = REPO / "backend-py" / "app" / "routers"
    py_text = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(py_dir.glob("*.py"))
    )
    problems: list[str] = []
    for ts_name in MIGRATED:
        ts_file = routes_dir / ts_name
        if not ts_file.exists():
            continue
        names = set(re.findall(r"\.query\('([^']+)'\)", ts_file.read_text(encoding="utf-8")))
        for name in sorted(n for n in names if any(c.isupper() for c in n)):
            if f'"{name}"' not in py_text:
                problems.append(
                    f"{ts_name} 的查询参数 `{name}` 在 Python 路由里找不到 —— camelCase 必须显式 "
                    f"`Query(alias=...)`，否则被静默忽略"
                )
    return problems


#: ── Agent 类型清单镜像 ──────────────────────────────────────────────
#: ``validAgentTypes`` = ``Object.keys(DEFAULT_PROMPTS)`` ⇒ **顺序即声明序**。
#: 路由用它做合法性判定（顺序不影响判定，但「默认配置 / 类型列表」的展示顺序会漂）。
_AGENTS_TS = _SRC_ROOT / "agents" / "index.ts"


def _agent_types_drift() -> list[str]:
    from app.services.agent_registry import VALID_AGENT_TYPES

    text = _AGENTS_TS.read_text(encoding="utf-8")
    block = re.search(r"DEFAULT_PROMPTS[^=]*=\s*\{(.*?)\n\}", text, re.S)
    # 顶层键缩进恰好 2 空格（内层字段 ≥4 空格）⇒ 只取 6 个 Agent 类型
    names = re.findall(r"^\s{2}([a-z_]+):\s*\{", block.group(1), re.M) if block else []
    if not names:
        return ["DEFAULT_PROMPTS 类型未能从 TS 抽取（正则需要跟着改）"]
    if tuple(names) != VALID_AGENT_TYPES:
        return [f"Agent 类型清单漂移：TS={names} ／ Python={list(VALID_AGENT_TYPES)}"]
    return []


#: ── Agent 出厂提示词镜像（**逐字**）────────────────────────────────────
#: ``DEFAULT_PROMPTS[].instructions`` 从 TS 的**模板字面量**里抽出来，把 ``${...}``
#: 按 ``prompt_blocks`` 的同名常量做替换后，与 ``agent_prompts.DEFAULT_INSTRUCTIONS``
#: **逐字**比对 —— 提示词是行为本身，差一个标点都会让 Agent 的表现漂移。
def _agent_prompts_drift() -> list[str]:
    from app.services import agent_prompts, prompt_blocks

    text = _AGENTS_TS.read_text(encoding="utf-8")
    block = re.search(r"DEFAULT_PROMPTS[^=]*=\s*\{(.*?)\n\}", text, re.S)
    if not block:
        return ["DEFAULT_PROMPTS 未能从 TS 抽取（正则需要跟着改）"]

    placeholders = {
        "SCREENPLAY_FORMAT_RULES": prompt_blocks.SCREENPLAY_FORMAT_RULES,
        "IMAGE_PROMPT_TEMPLATE_CHARACTER": prompt_blocks.IMAGE_PROMPT_TEMPLATE_CHARACTER,
        "IMAGE_PROMPT_TEMPLATE_SCENE": prompt_blocks.IMAGE_PROMPT_TEMPLATE_SCENE,
        "IMAGE_PROMPT_TEMPLATE_SHOT": prompt_blocks.IMAGE_PROMPT_TEMPLATE_SHOT,
    }

    problems: list[str] = []
    for key, python_value in agent_prompts.DEFAULT_INSTRUCTIONS.items():
        entry = re.search(rf"^\s{{2}}{key}: \{{(.*?)\n\s{{2}}\}}", block.group(1), re.S | re.M)
        raw = re.search(r"instructions: `(.*?)`", entry.group(1), re.S) if entry else None
        if raw is None:
            problems.append(f"DEFAULT_PROMPTS.{key}.instructions 未能从 TS 抽取")
            continue
        ts_value = re.sub(
            r"\$\{(\w+)\}",
            lambda m: placeholders.get(m.group(1), m.group(0)),
            raw.group(1),
        )
        if ts_value != python_value:
            offset = next(
                (i for i, (a, b) in enumerate(zip(ts_value, python_value)) if a != b),
                min(len(ts_value), len(python_value)),
            )
            problems.append(
                f"DEFAULT_PROMPTS.{key} 提示词漂移 @{offset}："
                f"TS={ts_value[offset:offset + 24]!r} ／ Python={python_value[offset:offset + 24]!r}"
                f"（长度 TS={len(ts_value)} / Python={len(python_value)}）"
            )
    return problems


def main() -> int:
    routes_dir = _SRC_ROOT / "routes"

    node_routes: set[tuple[str, str]] = set()
    for filename, prefix in MIGRATED.items():
        text = (routes_dir / filename).read_text(encoding="utf-8")
        for method, ts_path in _TS_ROUTE.findall(text):
            node_routes.add((method.upper(), _node_path(prefix, ts_path)))

    # Python 已注册（走 OpenAPI schema 枚举 —— 唯一不受 _IncludedRouter 干扰的口径）
    registered: set[tuple[str, str]] = set()
    for path, ops in app.openapi()["paths"].items():
        for method in ops:
            registered.add((method.upper(), path))

    node_norm = {(m, _pattern(p)) for m, p in node_routes}
    registered_norm = {(m, _pattern(p)) for m, p in registered}
    # 只保留「Node 有、Python 完全没接管」的（参数名无关）
    unregistered = sorted(
        (m, p) for m, p in node_norm if (m, p) not in registered_norm
    )

    shadow: list[str] = []
    with TestClient(app, raise_server_exceptions=False) as client:
        for method, path in unregistered:
            response = client.request(method, _concrete(path))
            if response.status_code != 501:
                shadow.append(
                    f"{method} {path} → HTTP {response.status_code}"
                    f"  {response.text[:120]}"
                )

    extra = sorted(
        (m, p)
        for m, p in registered_norm
        if (m, p) not in node_norm
        and p.startswith("/api/v1/")
        and (m, p) not in _EXTRA_ALLOWLIST
    )

    migrated_count = len(node_norm & registered_norm)
    print(f"Node 路径 {len(node_norm)} 条 ｜ Python 已注册 {len(registered_norm)} 条 ｜ "
          f"未注册（应全部走兜底）{len(unregistered)} 条")
    for line in shadow:
        print(f"  SHADOW  {line}")
    for method, path in extra:
        print(f"  EXTRA   {method} {path}   ← Node 侧没有这条路径，确认是否写错")
    print(f"  （已迁移 {migrated_count} 条；"
          f"{len(unregistered) - len(shadow)} 条未迁移且正确走兜底）")

    print(f"\n{'FAIL' if shadow else 'OK'}: 被吞掉的未迁移端点 {len(shadow)} 条")

    drift = (
        _registry_drift()
        + _prompt_utils_drift()
        + _adapters_drift()
        + _text_generation_drift()
        + _visual_graph_drift()
        + _prompt_tables_drift()
        + _prompt_blocks_drift()
        + _json_dumps_drift()
        + _query_param_drift()
        + _agent_types_drift()
        + _agent_prompts_drift()
        + _numeric_thresholds_drift()
        + _subagent_registry_drift()
    )
    for line in drift:
        print(f"  DRIFT   {line}")
    print(f"{'FAIL' if drift else 'OK'}: 镜像常量漂移 {len(drift)} 条")
    return 1 if (shadow or drift) else 0


def _numeric_thresholds_drift() -> list[str]:
    """校验**数值型**镜像常量：``(TS 文件, 常量名, Python 模块)`` 三者逐值一致。

    守卫里其余镜像多为**字符串**常量；这一条专管**数值**阈值 —— 它们直接决定
    warning/info/缺陷的判定边界，两边不一致时**不会报任何错**，只会「同一份素材在两套后端下
    得到不同的严重度/分数」。新增此类常量时，把它加进下面的表即可。
    """
    import importlib  # noqa: PLC0415

    sources = (
        ("services/consistency-qc.ts", "CONSISTENCY_QC_THRESHOLDS", "app.services.consistency_qc"),
        ("services/technical-qc.ts", "TECH_QC_THRESHOLDS", "app.services.technical_qc"),
    )
    drift: list[str] = []
    for relative, const, module_path in sources:
        source = _SRC_ROOT / relative
        if not source.is_file():
            drift.append(f"{relative}: 文件不存在（守卫依赖的 TS 源码缺失）")
            continue
        src = source.read_text(encoding="utf-8")
        block = re.search(rf"{const}\s*=\s*\{{(.*?)\}}\s*as const", src, re.S)
        if block is None:
            drift.append(f"{relative}: 找不到 {const} 块（解析失败，需人工核对）")
            continue
        # 块内的 `name: 0.55` 形态（注释里没有「冒号 + 数字」，不会误匹配）
        ts_values = {name: float(value)
                     for name, value in re.findall(r"(\w+)\s*:\s*(-?[0-9.]+)", block.group(1))}
        py_values: dict[str, float] = getattr(importlib.import_module(module_path), const)
        for key, value in py_values.items():
            if key not in ts_values:
                drift.append(f"{const}.{key} 在 TS 侧不存在")
            elif abs(ts_values[key] - value) > 1e-12:
                drift.append(f"{const}.{key}: TS={ts_values[key]} / py={value}")
        for key in ts_values:
            if key not in py_values:
                drift.append(f"{const}.{key}: TS 有但 Python 侧缺")
    return drift


def _subagent_registry_drift() -> list[str]:
    """校验 ``agents/subagent.py`` 的 ``SUBAGENT_REGISTRY`` / ``MAX_SUBAGENT_DEPTH`` 与 TS 逐字一致。

    ⚠️ 这些 ``name``/``capability`` 文案是**发给模型看的能力清单**（orchestrator 据此决定把子任务
    派给谁）⇒ 改一边不改另一边会让两侧的 Agent 调度行为分叉，而且**不会报任何错**。
    """
    from app.agent import subagent as sa  # noqa: PLC0415

    src = (_SRC_ROOT / "agents" / "subagent.ts").read_text(encoding="utf-8")
    entries = re.findall(
        r"\{\s*type:\s*'([^']+)'\s*,\s*name:\s*'([^']+)'\s*,\s*capability:\s*'([^']+)'\s*\}",
        src)
    drift: list[str] = []
    if len(entries) != len(sa.SUBAGENT_REGISTRY):
        drift.append(f"SUBAGENT_REGISTRY 条数：TS={len(entries)} / py={len(sa.SUBAGENT_REGISTRY)}")
    for (ts_type, ts_name, ts_capability), entry in zip(entries, sa.SUBAGENT_REGISTRY):
        if (ts_type, ts_name, ts_capability) != (entry["type"], entry["name"], entry["capability"]):
            drift.append(f"SUBAGENT_REGISTRY[{ts_type}]：TS=({ts_name}/{ts_capability}) "
                         f"vs py=({entry['name']}/{entry['capability']})")
    depth = re.search(r"MAX_SUBAGENT_DEPTH\s*=\s*(\d+)", src)
    if depth is None or int(depth.group(1)) != sa.MAX_SUBAGENT_DEPTH:
        drift.append(f"MAX_SUBAGENT_DEPTH：TS={depth.group(1) if depth else '?'} "
                     f"/ py={sa.MAX_SUBAGENT_DEPTH}")
    return drift


if __name__ == "__main__":
    # ⚠️ Windows 中文控制台是 **GBK**（py3.14 及以前不会自动 UTF-8 ✗）⇒ 不加这行，裸跑第一条 `print` 就 `UnicodeEncodeError` 崩 ✗（守口径见 `check_cli_encoding.py` ✓）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
