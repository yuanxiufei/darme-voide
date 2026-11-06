"""画布插件契约守卫（源 `plugins/canvas/sdk/src/types.ts` + 6 个插件 ⇒ `services/canvas_plugins.py`）。

钉五件事：

1. **内置 5 个插件自洽** —— 走一遍 `validate_bundle`，全绿；`template` 不登记
   （它是源项目给插件作者的起步样板，注释原文「改我」，迁进来只会变成死样板 ✗）。
2. **⭐ 与源逐条对拍** —— 源仓在盘上时读 6 个 `index.tsx` 的 `id`/`version`，
   与注册表**逐项相等**；手抄错一个版本号就红（这类错没人会主动发现 ✗）。
   源仓不在盘上时打印 SKIP 而**不判失败** —— 测试不能依赖外部仓库。
3. **⭐ 与前端逐字段对拍** —— `RESERVED_METADATA_KEYS` 必须覆盖
   `useCanvas.ts::CanvasNode` 的每个平铺字段：漏一个，插件就能写坏节点状态。
4. **反证 11 种坏法** —— 每种坏法都对应一个真会出问题的场景（见各条注释），
   断言「必须报错」，防止校验退化成橡皮图章 ✓。
5. **端点与会话** —— 2 个插件接口挂上；`hosted` 如实为 `false`。

运行::

    ./backend-py/.venv/Scripts/python.exe backend-py/tests/canvas_plugins_test.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BP = REPO / "backend-py"


def _find_source() -> Path | None:
    """找源仓库（可能不在盘上，也可能挂在不同的父目录下）。"""
    tail = Path("ComfyUI-Local-Infinite-Canvas") / "infinite-canvas" / "plugins" / "canvas"
    for base in (REPO.parent / "anli", REPO.parent, REPO.parent.parent):
        candidate = base / tail
        if candidate.is_dir():
            return candidate
    return None


SOURCE_CANVAS = _find_source()

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _sample(**overrides) -> dict:
    """一个合法的单插件清单，再用 overrides 逐项弄坏 —— 每条反证只变一处 ✓。"""
    node = {
        "type": "demo:thing",
        "title": "演示",
        "icon": "🧪",
        "description": "演示节点",
        "defaultSize": {"w": 300, "h": 200},
        "defaultMetadata": {"content": ""},
        "contentComponent": "DemoNodeContent",
    }
    node.update(overrides.pop("node", {}))
    plugin = {
        "id": "demo",
        "name": "演示插件",
        "version": "1.0.0",
        "description": "演示用",
        "nodes": [node],
    }
    plugin.update(overrides)
    return plugin


def main() -> int:
    sys.path.insert(0, str(BP))
    from app.services import canvas_ops as ops  # noqa: PLC0415
    from app.services import canvas_plugins as plugins  # noqa: PLC0415

    # ---------------------------------------------------------------- ① 内置自洽
    registry = plugins.registry()
    report = registry["validation"]
    check("内置: 5 个插件全部通过校验", report["ok"], report)
    check("内置: 清单条数 = 5", report["count"] == 5, report["count"])
    check("内置: 无跨插件冲突", not report["conflicts"], report["conflicts"])
    check("内置: 5 个插件节点 type", len(registry["nodeTypes"]) == 5, registry["nodeTypes"])
    check("内置: template 不登记（起步样板不是能力）",
          "template" not in {item["id"] for item in registry["plugins"]},
          [item["id"] for item in registry["plugins"]])
    # 渲染槽已接（`CanvasNodeCard.vue` 走 `PluginNodeBody`）⇒ hosted 必须如实为 True。
    # 这条曾经断言 False（当年宿主还没接），改回 False 就说明谁把渲染槽摘了 ✗
    check("内置: hosted 如实为 True（渲染槽已接）", registry["hosted"] is True, registry["hosted"])
    check("内置: note 点明渲染槽与 values 落点",
          "渲染槽" in registry["note"] and "values" in registry["note"], registry["note"])

    # 插件节点 type 与内置节点类型必须完全不相交，否则卡片渲染时分不清
    overlap = set(registry["nodeTypes"]) & set(ops.NODE_TYPES)
    check("内置: 插件 type 与内置节点类型不相交", not overlap, sorted(overlap))
    check("内置: 每个 type 都带 <id>: 前缀",
          all(item.split(":")[0] == item.split(":")[0] for item in registry["nodeTypes"])
          and all(":" in item for item in registry["nodeTypes"]),
          registry["nodeTypes"])
    owners = {item["type"]: plugin["id"] for plugin in registry["plugins"] for item in plugin["nodes"]}
    check("内置: type 命名空间 = 所属插件 id",
          all(key.split(":")[0] == value for key, value in owners.items()),
          {key: value for key, value in owners.items() if key.split(":")[0] != value})

    # 内置节点尺寸要落在插件尺寸界内（同一套界，别一个宽一个紧）
    flat = [size for pair in ops.DEFAULT_NODE_SIZE.values() for size in pair]
    check("内置: 内置节点尺寸也在 SIZE_MIN–SIZE_MAX 内",
          all(plugins.SIZE_MIN <= size <= plugins.SIZE_MAX for size in flat), flat)

    # ---------------------------------------------------------------- ② 与源对拍
    if SOURCE_CANVAS is not None:
        source: dict[str, str] = {}
        for path in sorted(SOURCE_CANVAS.glob("*/src/index.tsx")):
            text = path.read_text(encoding="utf-8")
            found_id = re.search(r'definePlugin\(\{\s*id:\s*"([a-z0-9-]+)"', text)
            found_ver = re.search(r'version:\s*"(\d+\.\d+\.\d+)"', text)
            if found_id and found_ver:
                source[found_id.group(1)] = found_ver.group(1)
        check("源: 读到 6 个插件（含 template）", len(source) == 6, source)
        ours = {item["id"]: item["version"] for item in registry["plugins"]}
        check("源: 版本号逐项相等（手抄会错）",
              all(ours.get(key) == value for key, value in source.items() if key in ours),
              {"ours": ours, "source": source})
        check("源: 唯一不迁的正好是 template",
              set(source) - set(ours) == {"template"}, sorted(set(source) - set(ours)))
        source_types = set()
        for path in sorted(SOURCE_CANVAS.glob("*/src/index.tsx")):
            source_types |= set(re.findall(r'type:\s*"([a-z0-9-]+:[a-z0-9-]+)"',
                                           path.read_text(encoding="utf-8")))
        check("源: 节点 type 逐项一致",
              source_types - {"template:node"} == set(registry["nodeTypes"]),
              {"ours": sorted(registry["nodeTypes"]), "source": sorted(source_types)})
    else:
        _SKIPS.append(f"源仓不在盘上，跳过对拍：{SOURCE_CANVAS}")

    # ---------------------------------------------------------------- ③ 与前端对拍
    canvas_src = (REPO / "frontend" / "app" / "composables" / "useCanvas.ts").read_text(encoding="utf-8")
    interface = re.search(r"interface CanvasNode\b[^{]*\{(.*?)\n\}", canvas_src, re.S)
    check("前端: 解析到 CanvasNode 接口", interface is not None)
    fields = set(re.findall(r"^\s{2}(\w+)\??\s*:", interface.group(1), re.M)) if interface else set()
    check("前端: 解析到的字段不少于 10 个", len(fields) >= 10, sorted(fields))
    unguarded = sorted(fields - plugins.RESERVED_METADATA_KEYS)
    check("前端: 每个平铺字段都被 RESERVED 挡住", not unguarded, unguarded)

    # ------------------------------------------------- ③b 插件清单 ↔ 渲染组件（一字不差）
    spec_src = (REPO / "frontend" / "app" / "utils" / "canvasPlugins.ts").read_text(encoding="utf-8")
    # 一条清单 = 一个 `  { ... },` 块；块里的 `defaultSize` 还嵌一层花括号 ⇒ 按**缩进**切，不按花括号切
    body = re.search(r"export const PLUGIN_NODES:[^=]*=\s*\[(.*?)\n\]", spec_src, re.S)
    check("前端: 解析到 PLUGIN_NODES 清单", body is not None)
    entries: dict[str, dict] = {}
    for chunk in re.split(r"\n\s{2}\{\n", body.group(1) if body else ""):
        found = re.search(r"type: '([^']+)'", chunk)
        if not found:
            continue
        size = re.search(r"defaultSize: \{ w: (\d+), h: (\d+) \}", chunk)
        meta = re.search(r"defaultMetadata: \{(.*?)\}", chunk)
        entries[found.group(1)] = {
            "pluginId": (re.search(r"pluginId: '([^']+)'", chunk) or [None, ""])[1],
            "title": (re.search(r"title: '([^']*)'", chunk) or [None, ""])[1],
            "size": (int(size.group(1)), int(size.group(2))) if size else None,
            "component": (re.search(r"contentComponent: '([^']+)'", chunk) or [None, ""])[1],
            "keys": set(re.findall(r"(\w+):", meta.group(1))) if meta else set(),
        }
    backend_nodes = {
        node["type"]: (plugin["id"], node)
        for plugin in plugins.BUILTIN_PLUGINS
        for node in plugin["nodes"]
    }
    check("前端: 插件节点 type 与后端逐项相同",
          set(entries) == set(backend_nodes),
          {"onlyFront": sorted(set(entries) - set(backend_nodes)),
           "onlyBack": sorted(set(backend_nodes) - set(entries))})
    check("前端: 标题逐字相同（卡片头部直接显示它）",
          all(entries[t]["title"] == node["title"] for t, (_, node) in backend_nodes.items() if t in entries),
          {t: (entries[t]["title"], node["title"]) for t, (_, node) in backend_nodes.items()
           if t in entries and entries[t]["title"] != node["title"]})
    check("前端: 默认尺寸逐项相同（尺寸错 ⇒ 首屏就挤）",
          all(entries[t]["size"] == (node["defaultSize"]["w"], node["defaultSize"]["h"])
              for t, (_, node) in backend_nodes.items() if t in entries),
          {t: (entries[t]["size"], node["defaultSize"]) for t, (_, node) in backend_nodes.items()
           if t in entries and entries[t]["size"] != (node["defaultSize"]["w"], node["defaultSize"]["h"])})
    check("前端: pluginId 与所属插件 id 一致",
          all(entries[t]["pluginId"] == pid for t, (pid, _) in backend_nodes.items() if t in entries),
          {t: (entries[t]["pluginId"], pid) for t, (pid, _) in backend_nodes.items()
           if t in entries and entries[t]["pluginId"] != pid})
    check("前端: 默认键集合与 defaultMetadata 相同（少一个键 ⇒ 组件读到 undefined）",
          all(entries[t]["keys"] == set(node["defaultMetadata"]) for t, (_, node) in backend_nodes.items()
              if t in entries),
          {t: (sorted(entries[t]["keys"]), sorted(node["defaultMetadata"]))
           for t, (_, node) in backend_nodes.items()
           if t in entries and entries[t]["keys"] != set(node["defaultMetadata"])})

    # 渲染组件必须**真在盘上**且被分发器登记 —— 清单写了名字却没有 .vue ⇒ 卡片报「缺组件」
    component_dir = REPO / "frontend" / "app" / "components" / "canvas" / "plugins"
    missing = sorted(
        t for t, (_, node) in backend_nodes.items()
        if not (component_dir / f"{node['contentComponent']}.vue").is_file()
    )
    check("前端: 每个 contentComponent 都有 .vue 文件", not missing, missing)
    dispatcher = (component_dir / "PluginNodeBody.vue").read_text(encoding="utf-8")
    unregistered = sorted(
        node["contentComponent"] for _, node in backend_nodes.values()
        if node["contentComponent"] not in dispatcher
    )
    check("前端: 分发器登记了全部 contentComponent", not unregistered, unregistered)

    # 类型判据两边同规则（形态不一致 ⇒ 后端放行的 type 前端认不出，反之亦然）
    front_pattern = re.search(r"PLUGIN_TYPE_PATTERN = /(.+)/\n", spec_src)
    check("前端: 插件 type 正则与后端同规则",
          front_pattern is not None
          and front_pattern.group(1).replace("\\", "") == ops.PLUGIN_TYPE_PATTERN.pattern.replace("\\", ""),
          {"front": front_pattern.group(1) if front_pattern else None,
           "back": ops.PLUGIN_TYPE_PATTERN.pattern})

    # ---------------------------------------------------------------- ④ 反证
    check("反证 0: 合法样例通过", not plugins.validate_plugin(_sample()), plugins.validate_plugin(_sample()))
    check("反证 1: id 非 kebab-case 报错",
          plugins.validate_plugin(_sample(id="DemoPlugin")) != [])
    check("反证 2: id 撞内置节点类型报错",
          any("内置节点类型" in item for item in plugins.validate_plugin(_sample(id="text"))))
    check("反证 3: type 无命名空间报错",
          any("pluginId" in item for item in plugins.validate_plugin(_sample(node={"type": "thing"}))))
    check("反证 4: type 命名空间不是本插件报错",
          any("命名空间" in item for item in plugins.validate_plugin(_sample(node={"type": "other:thing"}))))
    check("反证 5: 尺寸越界报错",
          any("越界" in item for item in
              plugins.validate_plugin(_sample(node={"defaultSize": {"w": 40, "h": 200}}))))
    check("反证 5b: 源式 width/height 也认（只要求是数字）",
          any("越界" in item for item in
              plugins.validate_plugin(_sample(node={"defaultSize": {"width": 40, "height": 200}}))))
    check("反证 6: 占了节点真字段报错",
          any("真字段" in item for item in
              plugins.validate_plugin(_sample(node={"defaultMetadata": {"x": 10}}))))
    check("反证 7: hidePanel 与 builtinPanel 并存报错",
          any("矛盾" in item for item in plugins.validate_plugin(
              _sample(node={"hidePanel": True, "builtinPanel": {"mode": "image"}}))))
    #: 反证 8 锁的是「面板模式不许超出 GENERATION_MODES」✓。原先拿 `audio` 当反例 ✗ ——
    #: 2026-11-06 起 audio 是**合法档位** ✓，再拿它反证就是把一条纪律测成了一句过时话 ✗；
    #: 换成真不存在的 `voice`，纪律照旧锁住 ✓，另加一条正向：audio 面板**必须放行** ✓
    #: （音乐卡要能嵌宿主生成面板 ✓）。
    check("反证 8: 内置面板要 voice 报错（不在 GENERATION_MODES 里）",
          any("不支持" in item for item in plugins.validate_plugin(
              _sample(node={"builtinPanel": {"mode": "voice"}}))))
    check("正向 8b: 内置面板要 audio 放行（音乐卡要能嵌宿主面板 ✓）",
          not plugins.validate_plugin(_sample(node={"builtinPanel": {"mode": "audio"}})),
          plugins.validate_plugin(_sample(node={"builtinPanel": {"mode": "audio"}})))
    check("反证 9: icon 是对象报错（ReactNode 残留）",
          any("icon" in item for item in plugins.validate_plugin(_sample(node={"icon": {"$$typeof": 1}}))))
    check("反证 10: 缺 contentComponent 报错",
          any("contentComponent" in item for item in
              plugins.validate_plugin(_sample(node={"contentComponent": ""}))))
    check("反证 11: version 非 semver 报错",
          any("x.y.z" in item for item in plugins.validate_plugin(_sample(version="v1"))))

    # 批量：跨件唯一性
    duplicate_id = plugins.validate_bundle([_sample(), _sample()])
    check("批量: 同 id 两件报冲突", not duplicate_id["ok"] and bool(duplicate_id["conflicts"]), duplicate_id)
    clash = plugins.validate_bundle([
        _sample(id="alpha", node={"type": "alpha:thing"}),
        _sample(id="beta", node={"type": "alpha:thing"}),
    ])
    check("批量: 跨插件抢同一个 type 报冲突", bool(clash["conflicts"]), clash["conflicts"])
    check("批量: 空清单 ok（没插件不是错）", plugins.validate_bundle([])["ok"])

    # ---------------------------------------------------------------- ⑤ 端点
    from app.main import app  # noqa: PLC0415

    paths = app.openapi()["paths"]
    check("端点: GET /canvas/plugins 已挂", "/api/v1/canvas/plugins" in paths, sorted(
        item for item in paths if item.startswith("/api/v1/canvas")))
    check("端点: POST /canvas/plugins/validate 已挂",
          "/api/v1/canvas/plugins/validate" in paths)

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed_ok, detail in _RESULTS:
        print(("PASS  " if passed_ok else "FAIL  ") + name + ("" if passed_ok else f"   <<< {detail!r}"))
    for note in _SKIPS:
        print("SKIP  " + note)
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
