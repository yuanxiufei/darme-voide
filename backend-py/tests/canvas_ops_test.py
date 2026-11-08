"""画布工具层守卫（源 `canvas-agent/src/canvas/*.ts` ⇒ `services/canvas_ops.py` 重写）。

钉五件事：

1. **工具账对齐** —— 22 个已迁 + 12 个 `DROPPED_TOOLS` = 源的 34 个。少一个就是漏迁、
   多一个就是编的 ✗；每个 dropped 都必须带理由（空理由等于没交代）。
   （2026-11-06：`canvas_generate_audio` 从 dropped 挪进已迁 ✓ —— 画布补了 `audio` 节点类型后
   它才有宿主 ✓；总数仍是 34 ✓）
2. **⭐ 跨语言同源** —— `DEFAULT_NODE_SIZE` / `NODE_TYPES` / `MIN_MAX_SCALE` 必须与
   `frontend/app/utils/canvasNodeSize.ts`、`composables/useCanvas.ts` **逐项相等**。
   这一条是本次重写最要紧的守卫：两边各拍一套尺寸，症状是「同一张图在画布上被压扁」，
   而**后端测试与前端测试各自都绿** ✓✗ —— 只有交叉断言能抓到。
3. **op 编译正确** —— 生成链路要产出「文本节点 → 配置节点 → 连线 → 选中（→ 开跑）」，
   且配置卡的正文是 `@[node:<文本 id>]` **引用**而不是复制一份同样的字 ✗。
4. **整批校验** —— 未知 type / 未知生成模式 / 自连 / 指向不存在的节点，全部要报错而不是
   静默丢；有错时 `ops` 必须为空（不许半批落地）。⚠️ audio 自 2026-11-06 起是**合法**的
   节点类型与生成模式 ✓（原判据把它当反例 ✗ —— 那条随 Music3 真出歌一起作废 ✓）。
5. **前端执行器认得每个 op** —— `utils/canvasOps.ts` 里 8 种 type 都得有分支，
   否则后端编出来、前端当未知类型丢掉 ✗。
6. **节点字段白名单覆盖接口** —— `useCanvas.normalizeDoc()` 重建节点是显式白名单，
   `CanvasNode` 每加一个字段都必须搬进去；漏搬的症状是「刷新后档位/通道自己变了」
   （2026-11-06 实测：`engineStage` 漏搬 ⇒ 选过 Music3 刷新回落 Z-Image，且永久丢失 ✗）。

运行::

    ./backend-py/.venv/Scripts/python.exe backend-py/tests/canvas_ops_test.py
"""
from __future__ import annotations

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
    from app.services import canvas_ops as ops  # noqa: PLC0415

    # ---------------------------------------------------------------- ① 工具账
    #: 22/12（2026-11-06：``canvas_generate_audio`` 从「未迁」挪进「已迁」✓ ——
    #: Music3 真出 wav 之后画布有音频卡了，源的这条工具才有宿主 ✓；总数仍是源的 34 ✓）
    check("账: 已迁 22 个", len(ops.TOOL_NAMES) == 22, len(ops.TOOL_NAMES))
    check("账: 未迁 12 个", len(ops.DROPPED_TOOLS) == 12, len(ops.DROPPED_TOOLS))
    check("账: 22 + 12 = 源的 34", len(ops.TOOL_NAMES) + len(ops.DROPPED_TOOLS) == 34)
    empty_reason = [name for name, why in ops.DROPPED_TOOLS.items() if len(why.strip()) < 10]
    check("账: 每个未迁工具都写了理由", not empty_reason, empty_reason)
    overlap = set(ops.TOOL_NAMES) & set(ops.DROPPED_TOOLS)
    check("账: 已迁与未迁不重叠", not overlap, sorted(overlap))
    schemas = ops.tool_schemas()
    check("账: 每个工具都有 schema", len(schemas) == len(ops.TOOL_NAMES), len(schemas))
    check("账: schema 都带 description",
          all(item["description"].strip() for item in schemas),
          [item["name"] for item in schemas if not item["description"].strip()])

    # ---------------------------------------------------------------- ② 跨语言同源 ★
    node_size_src = (REPO / "frontend" / "app" / "utils" / "canvasNodeSize.ts").read_text(encoding="utf-8")
    parsed_sizes = {
        name: (float(width), float(height))
        for name, width, height in re.findall(r"(\w+):\s*\{\s*w:\s*(\d+),\s*h:\s*(\d+)\s*\}", node_size_src)
    }
    check("同源: 前端解析到 6 类尺寸", len(parsed_sizes) == 6, parsed_sizes)
    check("同源: DEFAULT_NODE_SIZE 与前端逐项相等",
          parsed_sizes == ops.DEFAULT_NODE_SIZE,
          {"python": ops.DEFAULT_NODE_SIZE, "ts": parsed_sizes})

    canvas_src = (REPO / "frontend" / "app" / "composables" / "useCanvas.ts").read_text(encoding="utf-8")
    match = re.search(r"CanvasNodeType\s*=\s*([^\n]+)", canvas_src)
    ts_types = tuple(re.findall(r"'([a-z]+)'", match.group(1))) if match else ()
    check("同源: NODE_TYPES 与前端 CanvasNodeType 相等",
          set(ts_types) == set(ops.NODE_TYPES),
          {"python": ops.NODE_TYPES, "ts": ts_types})
    #: audio 是 2026-11-06 补的 ✓（原判据是「本仓确实没有 audio 节点」✗ —— 那条随着
    #: Music3 真出歌失效了 ✓）。两边任一漏都会让 `set` 那条先红 ✓，这里只锁「补上了且没歪」✓
    check("同源: audio 已在册且两侧都有（原判据「本仓没有 audio」已作废 ✗）",
          "audio" in ops.NODE_TYPES and "audio" in ts_types, (ops.NODE_TYPES, ts_types))
    check("同源: GENERATION_MODES 与 NODE_TYPES 的生成子集一致（audio 两边都算生成档 ✓）",
          set(ops.GENERATION_MODES) == {"text", "image", "video", "audio"},
          ops.GENERATION_MODES)

    min_scale = re.search(r"MIN_SCALE\s*=\s*([\d.]+)", canvas_src)
    max_scale = re.search(r"MAX_SCALE\s*=\s*([\d.]+)", canvas_src)
    check("同源: MIN_SCALE 一致", min_scale and float(min_scale.group(1)) == ops.MIN_SCALE,
          min_scale.group(1) if min_scale else None)
    check("同源: MAX_SCALE 一致", max_scale and float(max_scale.group(1)) == ops.MAX_SCALE,
          max_scale.group(1) if max_scale else None)

    # ---------------------------------------------------------------- ②b 节点字段白名单 ★
    #: `useCanvas.normalizeDoc()` 重建节点时是**白名单**（`nodes.push({…})` 字面量）⇒
    #: `CanvasNode` 接口每加一个字段都必须搬进去 ✓✗。2026-11-06 真踩到：`engineStage` 漏在
    #: 白名单外 ⇒ 用户选了 Music3、刷新一次档位就回落成清单里第一个可用档（Z-Image），
    #: 点生成跑出来的是一张图；而 `persist()` 会把清洗后的对象写回盘 ⇒ **丢是永久的** ✗。
    #: 症状在浏览器里只是「档位怎么自己变了」，前后端测试各自都绿 ✓✗ —— 只有这种
    #: 「接口字段 vs 搬运字面量」的交叉断言抓得到 ✓（实测反向验证：删掉 `engineStage` 那行即红 ✓）。
    iface = re.search(r"export interface CanvasNode\s*\{(.*?)\n\}", canvas_src, re.S)
    iface_fields = set(re.findall(r"^\s{2}(\w+)\??:", iface.group(1), re.M)) if iface else set()
    lift = re.search(r"nodes\.push\(\{(.*?)\n    \}\)", canvas_src, re.S)
    #: 两种写法都要认 ✓：`key: value` 与**简写** `key,`（`id`/`type`/`w`/`h` 就是简写 ✓ ——
    #: 只认冒号那版会把四个字段误判成「没搬」✗，判据自己先假红一次）
    lifted = (set(re.findall(r"^\s{6}(\w+):", lift.group(1), re.M))
              | set(re.findall(r"^\s{6}(\w+),\s*$", lift.group(1), re.M))) if lift else set()
    check("同源: 两侧字段都解析得到（判据自身不许空转 ✓）",
          len(iface_fields) >= 10 and len(lifted) >= 10, (sorted(iface_fields), sorted(lifted)))
    check("同源: normalizeDoc 白名单覆盖 CanvasNode 全部字段（漏搬=刷新后静默丢档 ✗）",
          not (iface_fields - lifted), sorted(iface_fields - lifted))

    # ---------------------------------------------------------------- ③ op 编译
    flow = ops.build_request("canvas_generate_image", {"prompt": "一只猫在雨里"})
    kinds = [item["type"] for item in flow["ops"]]
    check("编译: 图片生成产出 5 条 op",
          kinds == ["add_node", "add_node", "connect_nodes", "select_nodes", "run_generation"], kinds)
    text_node, config_node = flow["ops"][0], flow["ops"][1]
    check("编译: 第一个是文本节点", text_node["nodeType"] == "text", text_node)
    check("编译: 第二个是配置节点", config_node["nodeType"] == "config", config_node)
    check("编译: 配置卡正文是引用而不是复制",
          config_node["text"] == f"@[node:{text_node['id']}]", config_node.get("text"))
    check("编译: 引用能反查到文本 id",
          ops.REFERENCE_PATTERN.findall(config_node["text"]) == [text_node["id"]], config_node["text"])
    check("编译: 文本节点带原文", text_node.get("text") == "一只猫在雨里", text_node.get("text"))
    check("编译: 配置卡在文本节点右侧", config_node["x"] > text_node["x"], (text_node["x"], config_node["x"]))
    check("编译: 连线是 text → config",
          flow["ops"][2]["connections"] == [{"from": text_node["id"], "to": config_node["id"]}],
          flow["ops"][2])
    check("编译: 选中配置卡", flow["ops"][3]["ids"] == [config_node["id"]], flow["ops"][3])
    check("编译: 开跑指向配置卡", flow["ops"][4]["nodeId"] == config_node["id"], flow["ops"][4])
    check("编译: 编出来的 op 自己就合法", ops.validate_ops(flow["ops"])[1] == [],
          ops.validate_ops(flow["ops"])[1])

    reuse = ops.build_request("canvas_generate_image", {
        "prompt": "@[node:img-1]",
        "referenceNodeIds": ["img-1"],
    })
    check("编译: 纯引用时不重复建文本节点",
          [item["type"] for item in reuse["ops"]] == ["add_node", "connect_nodes", "select_nodes", "run_generation"],
          [item["type"] for item in reuse["ops"]])

    determined = ops.build_request("canvas_generate_image", {"prompt": "同一句"})
    again = ops.build_request("canvas_generate_image", {"prompt": "同一句"})
    check("编译: 同输入产出同 id（重放不长两套）",
          determined["ops"][0]["id"] == again["ops"][0]["id"],
          (determined["ops"][0]["id"], again["ops"][0]["id"]))

    #: ⭐ 音频通路 ✓（2026-11-06 新建）：源的 `canvas_generate_audio` 从「未迁」挪进「已迁」✓。
    #: 三件一起核 —— 配置卡标题走 `_title_of` 的「音频生成」✓（回落成"图片生成"就是通道丢了 ✗）、
    #: 开跑那条 op 的 mode 是 audio ✓（`add_node` 不带 mode，模式只活在这一条 op 上 ✓）、
    #: 歌词与时长原样进 values ✓（一律字符串 ✓，数值类型漂开会让前端对不上 ✓）。
    audio_flow = ops.build_request("canvas_generate_audio", {
        "prompt": "温柔女声民谣，木吉他", "lyrics": "[Verse]月亮升起来", "seconds": 20,
    })
    audio_cfg = [item for item in audio_flow["ops"]
                 if item["type"] == "add_node" and item["nodeType"] == "config"][0]
    check("编译: 音频档标题是「音频生成」", audio_cfg.get("title") == "音频生成", audio_cfg.get("title"))
    check("编译: 歌词与时长进 values（一律字符串）",
          audio_cfg.get("values") == {"lyrics": "[Verse]月亮升起来", "seconds": "20"},
          audio_cfg.get("values"))
    audio_run = [item for item in audio_flow["ops"] if item["type"] == "run_generation"]
    check("编译: 音频档开跑是 mode=audio（不是静默跑成 image ✗）",
          bool(audio_run) and audio_run[0].get("mode") == "audio", audio_run)
    check("编译: 音频 op 自己就合法", ops.validate_ops(audio_flow["ops"])[1] == [],
          ops.validate_ops(audio_flow["ops"])[1])

    snapshot = {"nodes": [{"id": "a", "x": 100, "y": 0, "w": 340, "h": 240}], "edges": []}
    check("编译: nextCanvasX = 最右缘 + 80", ops.next_canvas_x(snapshot) == 520.0, ops.next_canvas_x(snapshot))
    check("编译: 空画布 nextCanvasX = 0", ops.next_canvas_x(None) == 0.0)
    placed = ops.build_request("canvas_create_text_node", {"text": "hi"}, snapshot)
    check("编译: 无 x 时落在画布右侧", placed["ops"][0]["x"] == 520.0, placed["ops"][0])

    column = ops.build_request("canvas_create_text_nodes", {
        "items": [{"text": "1"}, {"text": "2"}], "x": 0, "y": 0, "direction": "column", "gap": 10,
    })
    check("编译: 竖排 y 递增", column["ops"][1]["y"] > column["ops"][0]["y"],
          [item["y"] for item in column["ops"]])
    moved = ops.build_request("canvas_move_nodes",
                              {"items": [{"id": "a", "dx": 20, "dy": 5}]}, snapshot)
    check("编译: 相对位移按当前坐标累加", moved["ops"][0]["patch"] == {"x": 120.0, "y": 5.0}, moved["ops"])

    # ---------------------------------------------------------------- ④ 整批校验
    # ⭐ 老坑：带快照时，本批刚建的文本/配置节点在快照里还没有 ⇒ 自己的连线被判成幽灵。
    # 只在**接了真实画布**时才露头（不带快照一切正常），必须单独锁一条。
    wired = ops.build_request("canvas_generate_image", {"prompt": "一只猫在雨里"}, snapshot)
    check("编译: 带快照时本批自建的连线不算幽灵",
          not wired["errors"] and len(wired["ops"]) == 5, wired["errors"])
    _, errors = ops.validate_ops([{"type": "nope"}])
    check("校验: 未知 type 报错", bool(errors), errors)
    audio_ops, errors = ops.validate_ops([{"type": "add_node", "nodeType": "audio"}])
    check("校验: audio 节点放行（2026-11-06 起是内置类型 ✓）", audio_ops and not errors, errors)
    _, errors = ops.validate_ops([{"type": "connect_nodes", "connections": [{"from": "a", "to": "a"}]}])
    check("校验: 自连报错", bool(errors), errors)
    _, errors = ops.validate_ops(
        [{"type": "connect_nodes", "connections": [{"from": "a", "to": "ghost"}]}], snapshot)
    check("校验: 指向不存在节点报错", bool(errors), errors)
    # 判据的**本意**是「未知模式不许静默降级成 image」✓ —— audio 现在是合法档位 ✓，
    # 所以换一个真不存在的模式来锁同一条纪律 ✓（删掉这条＝把纪律一起删了 ✗）
    _, errors = ops.validate_ops([{"type": "run_generation", "nodeId": "c", "mode": "voice"}])
    check("校验: 未知模式报错（不静默降级成 image）", bool(errors), errors)
    audio_run, errors = ops.validate_ops([{"type": "run_generation", "nodeId": "a", "mode": "audio"}], snapshot)
    check("校验: audio 模式放行（音乐档要走这条路 ✓）", audio_run and not errors, errors)
    _, errors = ops.validate_ops([{"type": "add_node", "nodeType": "text", "x": "左边"}])
    check("校验: 坐标不是数字报错", bool(errors), errors)

    # 插件节点（`<pluginId>:<name>`）：**已登记才放行** —— 形态对但没登记必须当场拒，
    # 否则模型编一个幽灵类型、前端只能画成「未安装的插件」✗
    plugin_ops, errors = ops.validate_ops(
        [{"type": "add_node", "nodeType": "sticky-note:note", "values": {"content": "记一笔"}}])
    check("校验: 已登记插件节点放行", plugin_ops and not errors, errors)
    check("校验: 插件节点不编默认尺寸（前端清单说了算）",
          "w" not in plugin_ops[0] and "h" not in plugin_ops[0], plugin_ops[0])
    check("校验: 插件键原样留在 values", plugin_ops[0].get("values") == {"content": "记一笔"},
          plugin_ops[0].get("values"))
    _, errors = ops.validate_ops([{"type": "add_node", "nodeType": "nope:thing"}])
    check("校验: 未登记的插件 type 报错", bool(errors) and "未登记" in errors[0], errors)
    _, errors = ops.validate_ops([{"type": "add_node", "nodeType": "StickyNote"}])
    check("校验: 形态不合法的 type 报错", bool(errors), errors)
    made = ops.build_request("canvas_create_node", {"nodeType": "markdown:doc", "text": "hello"})
    check("编译: create_node 能建插件节点", not made["errors"] and bool(made["ops"]), made["errors"])
    check("编译: 插件节点不带 w/h（交给前端清单）",
          "w" not in made["ops"][0] and "h" not in made["ops"][0], made["ops"][0])
    bad = ops.build_request("canvas_create_node", {"nodeType": "nope:thing"})
    check("编译: create_node 拒未登记插件", bool(bad["errors"]), bad["errors"])

    clean, errors = ops.validate_ops(
        [{"type": "add_node", "nodeType": "text", "x": 1, "y": 2, "values": {"steps": 30}}], snapshot)
    check("校验: 合法 op 通过", clean and not errors, errors)
    check("校验: 参数一律转成字符串", clean[0]["values"] == {"steps": "30"}, clean[0].get("values"))

    scaled, _ = ops.validate_ops([{"type": "set_viewport", "x": "0", "y": 0, "scale": 99}])
    check("校验: 视口 scale 夹到上限", scaled[0]["viewport"]["scale"] == ops.MAX_SCALE, scaled)
    scaled, _ = ops.validate_ops([{"type": "set_viewport", "k": 0.01}])
    check("校验: 视口 scale 夹到下限", scaled[0]["viewport"]["scale"] == ops.MIN_SCALE, scaled)

    bad = ops.build_request("canvas_apply_ops", {"ops": [{"type": "nope"}]})
    check("校验: 编译期有错时 ops 为空（不半批落地）", bad["ops"] == [] and bad["errors"], bad)

    # ---------------------------------------------------------------- ⑤ 快照与执行器
    compact = ops.compact_snapshot({
        "nodes": [{"id": "n1", "type": "text", "text": "x" * 400, "values": {"a": "1"},
                   "x": 1, "y": 2, "w": 3, "h": 4, "private": "不该出现"}],
        "edges": [{"id": "e1", "from": "n1", "to": "n2"}],
        "viewport": {"x": 0, "y": 0, "scale": 1},
        "selection": ["n1"],
    })
    text = compact["nodes"][0]["text"]
    check("快照: 正文截断", len(text) <= ops.SNAPSHOT_TEXT_KEEP + 1 and text.endswith("…"), len(text))
    check("快照: 参数原样保留", compact["nodes"][0]["values"] == {"a": "1"}, compact["nodes"][0])
    check("快照: 未知字段不外泄", "private" not in compact["nodes"][0], compact["nodes"][0])
    check("快照: 连线与选中保留", compact["edges"] and compact["selection"] == ["n1"], compact)
    raised = False
    try:
        ops.compact_snapshot(None)
    except ValueError:
        raised = True
    check("快照: 没有画布时如实报错", raised)

    executor = (REPO / "frontend" / "app" / "utils" / "canvasOps.ts").read_text(encoding="utf-8")
    missing = [kind for kind in ops.OP_TYPES if f"'{kind}'" not in executor]
    check("执行器: 8 种 op 前端都有分支", not missing, missing)
    check("执行器: 检查了 connect 的返回值", "canvas.connect(" in executor and "false" in executor)
    check("执行器: 观测量与后端同名", "'run_generation'" in executor and "onGenerate" in executor)
    check("执行器: run_generation 的 mode 认得 audio（只写三种 ⇒ 音乐档被当成未知 ✗）",
          "'text' | 'image' | 'video' | 'audio'" in executor,
          [line.strip() for line in executor.splitlines() if "mode?" in line])

    # ---------------------------------------------------------------- ⑥ 音频通道（2026-11-06）
    #: 后端把 `audio` 放进 `NODE_TYPES` 只解决了「能建这一类」✓；**歌能不能真落到卡上**取决于
    #: 卡片这三处 ✓✗（后端自测与前端自测各自都绿，只有对着卡片源码断言才抓得到 ✓）：
    #: ① 产物后缀判成 audio ✓（否则 wav 被按图片卡建进来 ⇒ 一张裂图 ✗）
    #: ② 真有 `<audio controls>` ✓（只判类型不给播放器 ⇒ 卡上永远一个空框 ✗）
    #: ③ `lyrics` / `seconds` 真进了 payload ✓（只躺在 values 里不换算 ⇒ 歌照跑、只是没词 ✗）
    card_src = (REPO / "frontend" / "app" / "components" / "canvas"
                / "CanvasNodeCard.vue").read_text(encoding="utf-8")
    check("音频: 产物后缀判成 audio（mp3/wav/flac/m4a/ogg ✓）",
          "(mp3|wav|flac|m4a|ogg)" in card_src,
          [line.strip() for line in card_src.splitlines() if "mp3" in line])
    check("音频: 卡片真有 <audio controls>", "<audio" in card_src and 'class="cn-audio"' in card_src)
    check("音频: 歌词与时长进 payload",
          "payload.lyrics" in card_src and "payload.seconds" in card_src)
    check("音频: 档位表登记 music3 出音频（audio: true）",
          re.search(r"music3:\s*\{[^}]*audio:\s*true", card_src) is not None,
          [line.strip() for line in card_src.splitlines() if "music3:" in line])
    check("音频: 时长那一格按 needsSeconds 显示（只按 isVideoStage ⇒ 音乐档没有时长 ✗）",
          'v-if="needsSeconds"' in card_src)

    # 端点可见
    from app.main import app  # noqa: PLC0415

    paths = app.openapi()["paths"]
    # 插件端点归 canvas_plugins_test、agent 端点归 canvas_agent_test，这里只管 op 相关
    expected = {
        "/api/v1/canvas/tools",
        "/api/v1/canvas/tool",
        "/api/v1/canvas/ops/validate",
        "/api/v1/canvas/ops/snapshot",
        "/api/v1/canvas/next-x",
    }
    check("端点: 5 个画布 op 接口都已挂上", expected <= set(paths),
          sorted(item for item in paths if item.startswith("/api/v1/canvas")))

    # ---------------------------------------------------------------- ⑦ 对话代跑生成的接线（2026-11-08）
    #: 这一节全部是**跨层接线**判据：后端编出来、前端接不接得住。
    #: 单看任何一层都绿 ✓✗ —— 只有对着源码断言才抓得到「模型点了名、前端没理」。
    run_op, errors = ops.validate_ops(
        [{"type": "run_generation", "nodeId": "c", "mode": "video", "prompt": "@[node:t1]"}], snapshot)
    check("对话: 校验后 run_generation 仍带 mode/prompt（丢掉=模型这轮写的提示词白写 ✗）",
          run_op and run_op[0].get("mode") == "video" and run_op[0].get("prompt") == "@[node:t1]",
          run_op)

    refs_src = (REPO / "frontend" / "app" / "utils" / "canvasReferences.ts").read_text(encoding="utf-8")
    check("对话: 提示词有引用解析器（`@[node:…]` 原样送引擎=出图与用户写的字无关 ✗）",
          "export function resolvePromptText" in refs_src
          and "REFERENCE_TOKEN" in refs_src,
          [line.strip() for line in refs_src.splitlines() if "resolvePromptText" in line])
    check("对话: 引用解析结果进了 ReferenceInputs.prompt",
          re.search(r"prompt\?:\s*string", refs_src) is not None
          and "inputs.prompt" in refs_src,
          [line.strip() for line in refs_src.splitlines() if "inputs.prompt" in line])
    #: ⚠️ 必须**定位到函数体**再断言：这两个 build 里都写了 `promptOverride ?? promptText.value`，
    #: 只查「文件里存在这串」的话，把 engine 那处改回 `props.node.text` 仍然判绿 ✗
    #: （反向验证时真踩到了：整文件存在性断言对这种双处写法完全失效 ✗）
    engine_fn = re.search(r"function buildEnginePayload\([^)]*\)\s*\{(.*?)\n\}", card_src, re.S)
    comfy_fn = re.search(r"function buildPayload\([^)]*\)\s*\{(.*?)\n\}", card_src, re.S)
    check("对话: 引擎 payload 用解析后的正文（改回 props.node.text = 引用串直达引擎 ✗）",
          engine_fn is not None and "promptOverride ?? promptText.value" in engine_fn.group(1),
          [line.strip() for line in (engine_fn.group(1) if engine_fn else "").splitlines()
           if "prompt:" in line][:2])
    check("对话: 工作流 payload 同样用解析后的正文",
          comfy_fn is not None and "promptOverride ?? promptText.value" in comfy_fn.group(1),
          [line.strip() for line in (comfy_fn.group(1) if comfy_fn else "").splitlines()
           if "prompt:" in line][:2])
    check("对话: 卡片从 references.prompt 取解析结果",
          "references?.prompt" in card_src,
          [line.strip() for line in card_src.splitlines() if "references?.prompt" in line])
    check("对话: 卡片按 op.mode 对档位类别（类别不符要拒，不许静默按卡上档位跑 ✗）",
          "stageKind.value" in card_src and "STAGE_KIND_LABEL" in card_src,
          [line.strip() for line in card_src.splitlines() if "stageKind" in line])

    panel_src = (REPO / "frontend" / "app" / "components" / "canvas"
                 / "CanvasChatPanel.vue").read_text(encoding="utf-8")
    check("对话: 面板把 mode/prompt 一起交给画布（只传 nodeId = 丢掉新提示词 ✗）",
          "props.generate?.(nodeId, override)" in panel_src
          and "mode: run.mode, prompt: run.prompt" in panel_src,
          [line.strip() for line in panel_src.splitlines() if "override" in line])
    check("对话: 回执读引擎真状态（只写「已提交」= 用户永远不知道跑成没跑成 ✗）",
          "engine.stateOf(id).value" in panel_src and "function runBadge" in panel_src,
          [line.strip() for line in panel_src.splitlines() if "runBadge" in line][:3])

    page_src = (REPO / "frontend" / "app" / "pages" / "canvas.vue").read_text(encoding="utf-8")
    check("对话: 成品贴源卡右侧落位（固定落视口中心 ⇒ 多张叠在一起看不出哪张是哪次 ✗）",
          "function artifactSpot" in page_src and "payload.sourceId" in page_src,
          [line.strip() for line in page_src.splitlines() if "artifactSpot" in line][:3])
    check("对话: 成品与源卡连线（不连=下一轮 agent 眼里它是来路不明的图 ✗）",
          "connect(payload.sourceId, id)" in page_src)

    agent_src = (REPO / "frontend" / "app" / "utils" / "canvasAgent.ts").read_text(encoding="utf-8")
    check("对话: 落地前调整批自检（逐条 try 只会落半批 ✗）",
          "canvasAPI.validate(" in agent_src and "verified" in agent_src,
          [line.strip() for line in agent_src.splitlines() if "canvasAPI.validate" in line])
    check("对话: 自检自身失败如实提醒（不许静默吞掉 ✗）",
          "warnings.push" in agent_src)

    #: ── 谎报「已提交」的四处病灶（本轮浏览器实测抓到）─────────────────────────
    #: 场景：模型让一张 zimage 卡跑 `mode:"video"`，卡片如实拒绝（一条请求都没发），
    #: 面板却写下「已提交生成 1」。根因是两处「乐观返回 true」：同一批 op 刚建的卡
    #: 还没挂载、或 `/engine/stages` 还在路上，它们回答了「等」，而调用方读成「成」✗。
    #: 修法是让「等」显式变成 promise 一路传上去、由执行器 `await` 出真答案 ✓。
    ops_src = (REPO / "frontend" / "app" / "utils" / "canvasOps.ts").read_text(encoding="utf-8")
    check("对话: 执行器等代跑的真答复（不 await ⇒ 闸没过也记 applied ✗）",
          re.search(r"export async function applyCanvasOps\(", ops_src) is not None
          and "await handlers.onGenerate(op)" in ops_src,
          [line.strip() for line in ops_src.splitlines() if "onGenerate(op)" in line])
    board_src = (REPO / "frontend" / "app" / "components" / "canvas"
                 / "CanvasBoard.vue").read_text(encoding="utf-8")
    #: 重试到头必须如实给「没提交」，而且要说清是「卡没渲染出来」；改回 `return true`
    #: 顶账就是面板谎报，改回裸 `return false` 就把理由丢了（回执只能写「看卡片」✗）
    sched = re.search(r"function scheduleRun\([^)]*\)\s*:\s*Promise<RunAnswer>\s*\{(.*?)\n\}\n",
                      board_src, re.S)
    check("对话: 未挂载的代跑也如实答复（乐观 true 顶账 ⇒ 面板谎报已提交 ✗）",
          sched is not None and "ok: false" in sched.group(1)
          and "return true" not in sched.group(1)
          and "return false" not in sched.group(1),
          [line.strip() for line in (sched.group(1) if sched else "").splitlines()
           if line.strip().startswith("return")])
    #: `card.run(...) === true` 会把卡片交出来的 promise 判成 false ⇒ 该跑的静默没跑 ✗；
    #: 在一层把它折成布尔/丢掉 `reason` ⇒ 回执永远只有一句笼统话 ✗
    check("对话: 画布板原样转交答复（折成布尔 ⇒ 该跑的吞掉、理由也丢 ✗）",
          "card.run(resolved) ===" not in board_src
          and "card.run(resolved)" in board_src
          and "RunAnswer | Promise<RunAnswer>" in board_src)
    check("对话: 档位清单未到时把 promise 交出去（返回 true ⇒ 没档位也报已提交 ✗）",
          re.search(r"return engine\.loadStages\(\)\s*\n\s*\.then\(", card_src) is not None,
          [line.strip() for line in card_src.splitlines() if "loadStages()" in line])
    check("对话: 页面原样转交结果（`=== true` 会把 promise 判成 false ✗）",
          "RunAnswer | Promise<RunAnswer>" in page_src
          and "runNode(nodeId, override) === true" not in page_src,
          [line.strip() for line in page_src.splitlines() if "runNodeFromChat" in line])
    check("对话: 面板等真答复再记「已提交」（乐观 push ⇒ 回执说谎 ✗）",
          "const answer = props.generate?.(nodeId, override)" in panel_src
          and "isRunPending(answer)" in panel_src
          and "if (verdict.ok) startedRun.push(nodeId)" in panel_src,
          [line.strip() for line in panel_src.splitlines() if "verdict" in line][:3])

    #: ── 回执要把「为什么」说出来（本轮第二轮：理由明明算出来了却被丢掉）──────────
    #: 上一轮修完「不谎报」，回执只剩一句「生成没提交成：看卡片上的缺项提示」——
    #: 卡片其实算出了「zimage 是「图片」档，跑不了「视频」」，`emit` 给了 toast，
    #: 但 `return false` 让那句话到不了对话侧 ⇒ 用户看见的是一句指路牌 ✗。
    #: 参考实现也不给答案（`void generateNodeRef.current?.(…)` 是 fire-and-forget，
    #: 理由只挂节点 `metadata.errorDetails`），照抄只能学一半 ⇒ 本仓自己把理由回传 ✓。
    check("对话: 答复带理由（RunVerdict.reason 是回执的真源）",
          "export interface RunVerdict" in ops_src
          and re.search(r"export interface RunVerdict\s*\{[^}]*reason\?:", ops_src, re.S) is not None
          and "export function normalizeRunVerdict" in ops_src
          and "export function isRunPending" in ops_src,
          [line.strip() for line in ops_src.splitlines() if "RunVerdict" in line][:4])
    check("对话: 执行器把卡片理由写进「未生效」（写死一句笼统话 ⇒ 用户仍要去猜 ✗）",
          "normalizeRunVerdict(verdict)" in ops_src
          and "answer.reason ||" in ops_src,
          [line.strip() for line in ops_src.splitlines() if "answer.reason" in line])
    #: 闸口必须**每一处**都过 `fail()`：漏一处就是「这次没跑起来，但理由没了」✗。
    #: 判据盯函数体而不是整文件 —— 整个文件里别处还有 `emit('error')`，会蒙过去 ✓✗
    run_engine = re.search(r"function runEngine\([^)]*\)[^{]*\{(.*?)\n\}\n", card_src, re.S)
    body = run_engine.group(1) if run_engine else ""
    check("对话: 卡片闸口把理由一起交出去（`emit` 完就 return false ⇒ 理由到不了对话 ✗）",
          body.count("return fail(") >= 6 and "return false" not in body,
          {"fail": body.count("return fail("),
           "return_false": [line.strip() for line in body.splitlines()
                            if "return false" in line]})
    check("对话: 卡片承诺的「ok」与 emit 同源（两处各算一遍必然漂移 ✗）",
          "function fail(reason: string): RunVerdict" in card_src
          and "emit('error', reason)" in card_src)

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed_ok, detail in _RESULTS:
        print(("PASS  " if passed_ok else "FAIL  ") + name + ("" if passed_ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
