"""S? 自检：**盘上这批 H3 模板本身对不对**（子图摊平 / 画幅两套式子 / COMBO 白名单；纯 stdlib ✓）。

只读 + 纯计算 ✓ —— **不下载模型、不连 ComfyUI、不碰 GPU** ✓（所以能塞进 `run_all.py` 的常跑档 ✓）。

盘上 `app/local_services/h3/workflows/` 的模板分两来源（见该目录 `README.md` ✓）：

* **A 批** `MiniMax_H3_Fast_T2V.json` / `_I2V.json` —— 第三方实测模板（无许可声明 ✓）；
* **B 批** `video_minimax_h3_{t2v,i2v,i2v_continuation,r2v}.json` 与
  `api_minimax_h3_{t2v,flf2v,r2v}.json` —— `comfy-org/workflow_templates`（**MIT** ✓）原样副本 ✓。

覆盖三类「**看上去没事、实际会静默出错**」的点 ✓✗：

① **子图摊平**：B 批 4 个本地模板把主节点链包在 `definitions.subgraphs[*]` 里 ✗
   （顶层那个节点的 `type` 是子图 uuid ✗）⇒ 不摊平就是「类名不在 object_info 里」整图转换失败 ✗，
   摊平后还必须**连线两端与槽位都自洽** ✓（否则错的是别的节点的槽位 ✓✗，画面看得出来但查不出来 ✓）；
② **画幅两套式子**：模板的 `width`/`height` 是 `ResolutionSelector` **连线算出来**的 ✗
   ⇒ 权威式子（`comfy_extras/nodes_resolution.py` ✓）与本仓早先反推的
   `geometry.size_for_megapixels`（✓）**在一般档位会发散** ✓✗ ⇒ 两个都钉住 ✓（别拿一个的数当另一个的 ✗）；
③ **COMBO 白名单**：选项串抄了「界面展示标签」（如 `"21:9 (Cinemascope)"` ✗）⇒ 提交被 400 拒 ✓
   ⇒ `check_combo_choices()` 必须**点名** ✓、且**只报不改** ✓（静默换串 = 静默换产物 ✗）。

运行::

    ./.venv/Scripts/python.exe tests/h3_templates_test.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = BACKEND_PY / "app" / "local_services" / "h3" / "workflows"

sys.path.insert(0, str(BACKEND_PY))

from app.local_services.h3 import workflow as wf  # noqa: E402
from app.services.engine import geometry as geo  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, ok: bool, detail: object = "") -> bool:
    _RESULTS.append((name, bool(ok), detail))
    print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    return bool(ok)


def check_eq(name: str, got: object, want: object) -> bool:
    return check(name, got == want, f"got={got!r} want={want!r}")


def load(name: str) -> dict[str, Any]:
    return json.loads((WORKFLOW_DIR / name).read_text(encoding="utf-8"))


#: B 批本地模板（走本机权重 ✓）与「云端 API 节点」模板（**不是**本机推理路径 ✗）
LOCAL_BATCH_B = (
    "video_minimax_h3_t2v.json",
    "video_minimax_h3_i2v.json",
    "video_minimax_h3_i2v_continuation.json",
    "video_minimax_h3_r2v.json",
)
CLOUD_BATCH_B = (
    "api_minimax_h3_t2v.json",
    "api_minimax_h3_flf2v.json",
    "api_minimax_h3_r2v.json",
)
BATCH_A = ("MiniMax_H3_Fast_T2V.json", "MiniMax_H3_Fast_I2V.json")

#: 展平前后的顶层节点数（**实测值** ✓，见 README 的 B 批表 ✓）—— 只在结构被改坏时才动 ✓
EXPECTED_COUNTS = {
    "video_minimax_h3_t2v.json": (6, 26),
    "video_minimax_h3_i2v.json": (9, 29),
    "video_minimax_h3_i2v_continuation.json": (6, 26),
    "video_minimax_h3_r2v.json": (29, 29),
    "api_minimax_h3_t2v.json": (2, 2),
    "api_minimax_h3_flf2v.json": (4, 4),
    "api_minimax_h3_r2v.json": (3, 3),
    "MiniMax_H3_Fast_T2V.json": (23, 23),
    "MiniMax_H3_Fast_I2V.json": (30, 30),
}


# ══════════════════════════════════════════════════════════════════════════
# ① 子图摊平：结构
# ══════════════════════════════════════════════════════════════════════════
def _link_parts(link: Any) -> tuple[Any, int, Any, int] | None:
    if isinstance(link, (list, tuple)) and len(link) >= 5:
        return link[0], int(link[1]), link[3], int(link[4])
    if isinstance(link, dict):
        return link.get("id"), int(link.get("origin_id")), link.get("target_id"), int(link.get("target_slot"))
    return None


def link_integrity(graph: dict[str, Any]) -> list[str]:
    """连线自洽性：两端节点都在、源槽位在该节点的 ``outputs`` 里、目标槽位在 ``inputs`` 里 ✓。"""
    nodes = {str(node.get("id")): node for node in graph.get("nodes") or []}
    problems: list[str] = []
    for link in graph.get("links") or []:
        parts = _link_parts(link)
        if parts is None:
            problems.append(f"连线形态不认: {link!r}")
            continue
        link_id, origin_id, target_id, target_slot = parts
        origin, target = nodes.get(str(origin_id)), nodes.get(str(target_id))
        if origin is None or target is None:
            problems.append(f"连线 {link_id}: 端点在图上不存在（{origin_id}→{target_id}）")
            continue
        if origin_id == target_id and origin is target:
            problems.append(f"连线 {link_id}: 自环（{origin_id}）")
        n_out = len(origin.get("outputs") or [])
        # 源槽位：连线记在 link[2] ✓（这里只校目标，源槽位另算 ✓）
        if isinstance(link, (list, tuple)) and len(link) >= 3:
            if int(link[2]) >= n_out and n_out:
                problems.append(f"连线 {link_id}: 源槽位 {link[2]} 越界（{origin_id} 只有 {n_out} 个输出）")
        n_in = len(target.get("inputs") or [])
        if target_slot >= n_in and n_in:
            problems.append(f"连线 {link_id}: 目标槽位 {target_slot} 越界（{target_id} 只有 {n_in} 个输入）")
    return problems


def case_flatten_structure() -> None:
    for name in LOCAL_BATCH_B + CLOUD_BATCH_B + BATCH_A:
        raw = load(name)
        before = len(raw.get("nodes") or [])
        graph = wf.flatten_subgraphs(raw)
        after = len(graph.get("nodes") or [])
        want = EXPECTED_COUNTS[name]
        check_eq(f"① {name} 顶层节点数（摊平前）", before, want[0])
        check_eq(f"① {name} 节点数（摊平后）", after, want[1])

    # 摊平后顶层**不该**再留「type 是子图 uuid」的节点 ✗、也不该留虚拟节点 -10/-20 ✗
    for name in LOCAL_BATCH_B:
        raw = load(name)
        subs = {str(one.get("id")) for one in (raw.get("definitions") or {}).get("subgraphs") or []}
        graph = wf.flatten_subgraphs(raw)
        leftovers = [str(n.get("id")) for n in graph.get("nodes") or [] if str(n.get("type")) in subs]
        check(f"① {name} 摊平后没有残留子图节点", not leftovers, leftovers)
        phantom = [str(n.get("id")) for n in graph.get("nodes") or []
                   if str(n.get("type")) in (wf.SUBGRAPH_INPUT_ID, wf.SUBGRAPH_OUTPUT_ID)]
        check(f"① {name} 摊平后没有虚拟节点（-10/-20）", not phantom, phantom)
        classes = wf.node_classes(graph)
        check(f"① {name} 摊平后含 MiniMaxH3 核心节点",
              bool({cls for cls in classes if cls.startswith("MiniMaxH3")}), sorted(classes))

    # B 批 r2v 本来就没子图 ⇒ 「没子图就一个字都不动」✓（身份不变 ✓）
    graph = load("video_minimax_h3_r2v.json")
    check("① 没有子图的图：flatten 返回同一对象", wf.flatten_subgraphs(graph) is graph)

    # **就地**改（不是返回新对象 ✓）—— 调用方拿旧引用也有效 ✓
    graph = load("video_minimax_h3_t2v.json")
    same = wf.flatten_subgraphs(graph) is graph
    check("① 有子图的图：flatten 就地改并返回同一对象", same and len(graph["nodes"]) == 26, len(graph["nodes"]))

    # 幂等：再摊一次不该变 ✓（摊完没有子图了 ⇒ 第二次什么都不做 ✓）
    frozen = json.dumps(graph["nodes"], sort_keys=True) + json.dumps(graph["links"], sort_keys=True)
    wf.flatten_subgraphs(graph)
    check("① flatten 幂等（第二次摊平不该再动图）",
          frozen == json.dumps(graph["nodes"], sort_keys=True) + json.dumps(graph["links"], sort_keys=True))

    # 摊平后 `widget_values_override`（提升的 widget ✓）**不能**留在顶层图上当垃圾 ✗、也不该丢 ✓
    graph = load("video_minimax_h3_t2v.json")
    wf.flatten_subgraphs(graph)
    core = [n for n in graph["nodes"] if n.get("type") == "MiniMaxH3ImageToVideo"]
    check_eq("① 摊平后找到 1 个 MiniMaxH3ImageToVideo", len(core), 1)
    if core:
        check("① 摊平后核心节点带上提升的 widget 值（override）",
              bool(core[0].get("widget_values_override")), core[0].keys())


def case_flatten_links() -> None:
    for name in LOCAL_BATCH_B + CLOUD_BATCH_B + BATCH_A:
        graph = wf.flatten_subgraphs(load(name))
        problems = link_integrity(graph)
        check(f"① {name} 摊平后连线自洽（两端在、槽位不越界）", not problems, problems[:3])

    # 摊平后的连线数 = 「子图内部**两端都是真节点**的那些（换新 id ✓）」+ 「挂在实例节点上、**重建**了的外层线 ✓」
    # （内部那 16 条碰 ``-10``/``-20`` 的线是**虚拟端口**的表示 ✗ ⇒ 不直接搬 ✓：入参那 15 条要么变外层线 ✓、
    #  要么变提升的 widget 值 ✓；出参那 1 条变外层出线 ✓）
    raw = load("video_minimax_h3_t2v.json")
    subs = {str(one.get("id")): one for one in (raw.get("definitions") or {}).get("subgraphs") or []}
    inner = [link for one in subs.values() for link in one.get("links") or []]
    both_real = [link for link in inner
                 if -10 != link.get("origin_id") != -20 and -10 != link.get("target_id") != -20]
    parents = {str(n.get("id")) for n in raw.get("nodes") or [] if str(n.get("type")) in subs}
    attached = [link for link in raw.get("links") or []
                if str(link[1]) in parents or str(link[3]) in parents]
    before = len(raw.get("links") or [])
    check_eq("① 真实模板里确实有挂在子图实例上的外层线（G：摘线判据有对象 ✓）", len(attached), 3)
    graph = wf.flatten_subgraphs(raw)
    after = len(graph.get("links") or [])
    check_eq("① 摊平后的连线数 = 内部真节点线 + 重建的外层线", after, len(both_real) + len(attached))
    check_eq("① 摊平前的连线数（对照 ✓）", before, len(attached))


# ══════════════════════════════════════════════════════════════════════════
# ② 画幅：两套式子都钉住（权威 ✓ / 反推 ✓）
# ══════════════════════════════════════════════════════════════════════════
def _node_source_algo(megapixels: float, ratio: tuple[int, int], multiple: int) -> tuple[int, int]:
    """**逐字转写**自 `comfy_extras/nodes_resolution.py` 的 ``ResolutionSelector.execute`` ✓。

    独立转写的意义：拿它去核 ``workflow.resolution_selector_size`` ✓（照抄同一份式子不算 ✓）。

    .. code-block:: python

        w_ratio, h_ratio = ASPECT_RATIOS[aspect_ratio]
        total_pixels = megapixels * 1024 * 1024
        scale = math.sqrt(total_pixels / (w_ratio * h_ratio))
        width = round(w_ratio * scale / multiple) * multiple
        height = round(h_ratio * scale / multiple) * multiple
    """
    w_ratio, h_ratio = ratio
    total_pixels = megapixels * 1024 * 1024
    scale = math.sqrt(total_pixels / (w_ratio * h_ratio))
    return round(w_ratio * scale / multiple) * multiple, round(h_ratio * scale / multiple) * multiple


def _wrong_base_1e6(megapixels: float, ratio: tuple[int, int], multiple: int) -> tuple[int, int]:
    """**反面**：把预算基数写成 ``×1e6`` ✗（少 4.86% 像素 ✗）会算出什么 —— 只用来证明两套**可区分** ✓。"""
    w_ratio, h_ratio = ratio
    scale = math.sqrt(megapixels * 1e6 / (w_ratio * h_ratio))
    return round(w_ratio * scale / multiple) * multiple, round(h_ratio * scale / multiple) * multiple


def case_resolution_formula() -> None:
    auth, mult = wf.resolution_selector_size, 32
    check_eq("② 权威锚点 0.65MP @16:9 @32（A 批 T2V 真值）", auth(0.65, "16:9 (Widescreen)", multiple=mult), (1088, 608))
    check_eq("② 权威锚点 0.98MP @16:9 @32（A 批 I2V 真值）", auth(0.98, "16:9 (Widescreen)", multiple=mult), (1344, 768))
    check_eq("② 权威 1.0MP @16:9 @32（与反推**发散**的那档 ✓）", auth(1.0, "16:9 (Widescreen)", multiple=mult), (1376, 768))
    check_eq("② 权威 0.1MP @16:9 @32（下限档 ✓）", auth(0.1, "16:9 (Widescreen)", multiple=mult), (416, 256))
    check_eq("② 下限钳住：0.0MP ⇒ 按 0.1MP 算（schema min=0.1 ✓）", auth(0.0, "16:9", multiple=mult), (416, 256))

    # 独立转写的同一份式子：在网格上必须**逐一相等** ✓（证明本仓实现没走样 ✓）
    mismatches: list[str] = []
    for mp in (0.1, 0.2, 0.4, 0.5, 0.65, 0.8, 0.98, 1.0, 1.5, 2.0):
        for label, terms in wf.ASPECT_RATIO_OPTIONS.items():
            for step in (1, 8, 32, 64, 128):
                got = auth(mp, label, multiple=step)
                want = _node_source_algo(mp, terms, step)
                if got != want:
                    mismatches.append(f"{mp}MP {label} ×{step}: {got} != {want}")
    check("② 与节点源码逐字转写的式子网格一致（10 档 × 8 画幅 × 5 multiple）", not mismatches, mismatches[:3])

    # **反面**：预算基数若写成 ×1e6（或取整写成 ceil ✗）就会落到别的数上 ⇒ 证明我们钉的式子是**可区分**的 ✓
    check("② 反面：×1e6 基数在 1.0MP @16:9 上会算出别的数",
          _wrong_base_1e6(1.0, (16, 9), mult) != auth(1.0, "16:9 (Widescreen)", multiple=mult),
          _wrong_base_1e6(1.0, (16, 9), mult))

    # 本仓早先**反推**的口径（`geometry.size_for_megapixels` ✓）：实测锚点重合 ✓、一般档位发散 ✓
    check_eq("② 反推锚点重合 0.65MP @16:9", geo.size_for_megapixels(0.65, "16:9"), (1088, 608))
    check_eq("② 反推锚点重合 0.98MP @16:9", geo.size_for_megapixels(0.98, "16:9"), (1344, 768))
    check_eq("② 反推 1.0MP @16:9（= 1344×768 ✓ 与权威的 1376×768 不同 ✓）",
             geo.size_for_megapixels(1.0, "16:9"), (1344, 768))
    check_eq("② 反推 0.1MP @16:9（= 448×256 ✓ 与权威的 416×256 不同 ✓）",
             geo.size_for_megapixels(0.1, "16:9"), (448, 256))
    check("② 两套式子确实发散（1.0MP @16:9）",
          geo.size_for_megapixels(1.0, "16:9") != auth(1.0, "16:9 (Widescreen)", multiple=mult))

    # 画幅解析：选项串 / 全角冒号 / 元组 / 解不出的回落 ✓
    check_eq("② ratio_terms 选项串", wf.ratio_terms("16:9 (Widescreen)"), (16.0, 9.0))
    check_eq("② ratio_terms 全角冒号", wf.ratio_terms("16：9"), (16.0, 9.0))
    check_eq("② ratio_terms 元组", wf.ratio_terms((21, 9)), (21.0, 9.0))
    check_eq("② ratio_terms 解不出 ⇒ 回落 16:9", wf.ratio_terms("看不懂的画幅"), (16.0, 9.0))
    check_eq("② 选项表 8 项（与 AspectRatio 枚举一致 ✓）", len(wf.ASPECT_RATIO_OPTIONS), 8)
    check("② 选项表含 21:9 (Ultrawide) ✓（节点真枚举 ✓）", "21:9 (Ultrawide)" in wf.ASPECT_RATIO_OPTIONS)
    check("② 反面：展示标签 21:9 (Cinemascope) ✗ 不在表里",
          "21:9 (Cinemascope)" not in wf.ASPECT_RATIO_OPTIONS)


# ══════════════════════════════════════════════════════════════════════════
# ③ 「连线优先」（graphToPrompt 语义 ✓）：真模板的 width/height 真值
# ══════════════════════════════════════════════════════════════════════════
#: 只带「注入点 + Resolution selector」的最小字段表 ✓（真打 `ui_to_api` ✓）
SPECS: dict[str, Any] = {
    "MiniMaxH3ImageToVideo": {"input": {"required": {
        "prompt": ["STRING", {}], "width": ["INT", {"default": 1344}],
        "height": ["INT", {"default": 768}], "length": ["INT", {"default": 124}]}}},
    "ResolutionSelector": {"input": {"required": {
        "aspect_ratio": [["16:9 (Widescreen)", "21:9 (Ultrawide)", "1:1 (Square)"], {}],
        "megapixels": ["FLOAT", {"default": 0.65}], "multiple": ["INT", {"default": 8}]}}},
    "PrimitiveFloat": {"input": {"required": {"value": ["FLOAT", {"default": 1.0}]}}},
    #: 非 Primitive 的上游（用来钉「读不出就 None ✓ 不假装能算 ✗」）
    "ComfyMathExpression": {"input": {"required": {"expression": ["STRING", {}]}}},
}


def _graph_with_wired_width(ratio: str = "16:9 (Widescreen)", megapixels: Any = ["3", 0],
                            stale: tuple[int, int] = (1344, 768)) -> dict[str, Any]:
    """仿 `MiniMax_H3_Fast_T2V.json` 的真实接线 ✓：`width`/`height` 被 `ResolutionSelector` 拉着线 ✓，
    而 `widgets_values` 里**躺着旧值** ✗（这正是会静默出错的那个局 ✓）。"""
    return {
        "nodes": [
            {"id": 1, "type": "ResolutionSelector", "mode": 0, "title": "Resolution",
             "widgets_values": [ratio, 0.65, 32],
             "inputs": [{"name": "megapixels", "link": 12, "widget": {"name": "megapixels"}}],
             "outputs": [{"name": "width", "links": [10]}, {"name": "height", "links": [11]}]},
            {"id": 3, "type": "PrimitiveFloat", "mode": 0, "title": "QUALITY",
             "widgets_values": [0.65],
             "inputs": [], "outputs": [{"name": "value", "links": [12]}]},
            {"id": 2, "type": "MiniMaxH3ImageToVideo", "mode": 0, "title": "H3",
             "widgets_values": ["a shot", stale[0], stale[1], 124],
             "inputs": [{"name": "width", "link": 10, "widget": {"name": "width"}},
                        {"name": "height", "link": 11, "widget": {"name": "height"}}],
             "outputs": []},
        ],
        "links": [[10, 1, 0, 2, 0, "INT"], [11, 1, 1, 2, 1, "INT"], [12, 3, 0, 1, 0, "FLOAT"]],
    }


def case_connection_beats_widget() -> None:
    api = wf.ui_to_api(_graph_with_wired_width(), SPECS)
    core = api["2"]["inputs"]
    check_eq("③ 连线优先：width 取连线（不是 widgets_values 的旧值 1344）", core["width"], ["1", 0])
    check_eq("③ 连线优先：height 取连线", core["height"], ["1", 1])
    check("③ 旧值没有漏出去", 1344 not in (core["width"], core["height"]) or isinstance(core["width"], list))
    check_eq("③ 没拉线的 prompt / length 照旧吃 widgets_values", (core["prompt"], core["length"]), ("a shot", 124))

    # 负控：把连线摘掉 ⇒ 就该吃 widgets_values 的 1344×768 ✓（证明上面那条不是「碰巧相等」✓）
    graph = _graph_with_wired_width()
    for port in graph["nodes"][2]["inputs"]:
        port["link"] = None
    stripped = wf.ui_to_api(graph, SPECS)["2"]["inputs"]
    check_eq("③ 负控：没连线时才吃 widgets_values 的 1344×768", (stripped["width"], stripped["height"]), (1344, 768))

    # 提升 widget 的按名覆盖（`flatten_subgraphs` 的产物 ✓）：② 连线仍在 ⇒ **连线赢** ✓
    graph = _graph_with_wired_width()
    graph["nodes"][2]["widget_values_override"] = {"width": 999, "length": 100}
    overridden = wf.ui_to_api(graph, SPECS)["2"]["inputs"]
    check_eq("③ 覆盖值 + 连线：连线赢（覆盖在 ③ 步、连线在 ② 步之后 ✓）", overridden["width"], ["1", 0])
    check_eq("③ 覆盖值 + 无连线：覆盖值生效（length ✓）", overridden["length"], 100)

    # 真模板里就是「端口拉着线 + widgets_values 躺着旧值」✓（事实核对 ✓，两边都得对上 ✓）
    graph = wf.flatten_subgraphs(load("MiniMax_H3_Fast_T2V.json"))
    node = next((n for n in graph["nodes"] if n.get("type") == "MiniMaxH3ImageToVideo"), None)
    if check("③ 真模板里找到 MiniMaxH3ImageToVideo", node is not None):
        values = list(node.get("widgets_values") or [])
        wired = {port.get("name") for port in node.get("inputs") or [] if port.get("link") is not None}
        check("③ 真模板 G: width/height 端口确实拉着线", {"width", "height"} <= wired, sorted(wired))
        check("③ 真模板 G: widgets_values 里确实躺着旧值 1344/768", [1344, 768] == values[1:3], values[1:4])

    # `expected_resolution`：从 API 格式复算「模板会算出多大」✓（0.65MP @16:9 @32 ⇒ 1088×608 ✓）
    check_eq("③ expected_resolution 复算真值（1088×608 ✓）", wf.expected_resolution(api), (1088, 608))
    check_eq("③ expected_resolution 按标题收窄（找不到 ⇒ None ✓）",
             wf.expected_resolution(api, title_contains="不存在的标题"), None)
    check_eq("③ expected_resolution：图里没有 ResolutionSelector ⇒ None",
             wf.expected_resolution({"2": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {}}}), None)

    # 读不出就 None ✓（不猜 ✗）：画幅被牵线、或 megapixels 来自非 Primitive 节点 ✓
    wired_ratio = _graph_with_wired_width()
    wired_ratio["nodes"][0]["inputs"].append({"name": "aspect_ratio", "link": 13})
    wired_ratio["nodes"].append({"id": 4, "type": "ComfyMathExpression", "mode": 0,
                                 "widgets_values": ["16/9"], "inputs": [],
                                 "outputs": [{"name": "value", "links": [13]}]})
    wired_ratio["links"].append([13, 4, 0, 1, 1, "STRING"])
    check_eq("③ expected_resolution：画幅被牵线 ⇒ None（不猜 ✓）",
             wf.expected_resolution(wf.ui_to_api(wired_ratio, SPECS)), None)

    computed = _graph_with_wired_width()
    computed["nodes"].append({"id": 5, "type": "ComfyMathExpression", "mode": 0, "widgets_values": ["0.65"],
                              "inputs": [], "outputs": [{"name": "value", "links": [14]}]})
    computed["nodes"][0]["inputs"] = [{"name": "megapixels", "link": 14, "widget": {"name": "megapixels"}}]
    computed["links"].append([14, 5, 0, 1, 0, "FLOAT"])
    check_eq("③ expected_resolution：megapixels 来自非 Primitive ⇒ None（不假装能算 ✓）",
             wf.expected_resolution(wf.ui_to_api(computed, SPECS)), None)


# ══════════════════════════════════════════════════════════════════════════
# ④ COMBO 白名单（**只报不改** ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_combo_whitelist() -> None:
    check_eq("④ combo_options 新式声明", wf.combo_options(["COMBO", {"options": ["a", "b"]}]), ["a", "b"])
    check_eq("④ combo_options 老式声明（选项就在第一项 ✓）", wf.combo_options([["a", "b"], {}]), ["a", "b"])
    check_eq("④ combo_options 非 COMBO ⇒ None", wf.combo_options(["STRING", {}]), None)
    check_eq("④ combo_options 非 COMBO 列表型 ⇒ None", wf.combo_options(["INT", {"default": 1}]), None)
    check_eq("④ combo_options 认不出 ⇒ None（不猜 ✓）", wf.combo_options("MiniMaxH3ImageToVideo"), None)
    check_eq("④ combo_options：forceInput 的 COMBO 不算（值来自连线 ✓）",
             wf.combo_options(["COMBO", {"options": ["a"], "forceInput": True}]), None)

    graph = _graph_with_wired_width()
    api = wf.ui_to_api(graph, SPECS)
    object_info = {"ResolutionSelector": SPECS["ResolutionSelector"],
                   "MiniMaxH3ImageToVideo": SPECS["MiniMaxH3ImageToVideo"],
                   "PrimitiveFloat": SPECS["PrimitiveFloat"]}
    check_eq("④ 合法选项串 ⇒ 零问题", wf.check_combo_choices(api, object_info), [])
    check_eq("④ 连线值（list）不算 COMBO 问题 ✓",
             wf.check_combo_choices({"1": {"class_type": "ResolutionSelector",
                                           "inputs": {"aspect_ratio": ["9", 0]}}}, object_info), [])

    # **反面**：抄了展示标签 ✗ ⇒ 必须点名（节点 id + 字段 + 合法值举例 ✓）
    bad = json.loads(json.dumps(api))
    bad["1"]["inputs"]["aspect_ratio"] = "21:9 (Cinemascope)"
    problems = wf.check_combo_choices(bad, object_info)
    check_eq("④ 表外选项串 ⇒ 报 1 条", len(problems), 1)
    if problems:
        check("④ 报的文案点名了节点与字段", "1" in problems[0] and "aspect_ratio" in problems[0], problems[0])
        check("④ 报的文案给出了合法值举例", "21:9 (Ultrawide)" in problems[0], problems[0])
    check("④ **只报不改**：调用后 API 原样（没被静默换串 ✓）",
          bad["1"]["inputs"]["aspect_ratio"] == "21:9 (Cinemascope)")

    # 类不在 object_info 里 ⇒ 不在这里报 ✓（那条归 `missing_nodes` ✓）
    check_eq("④ 类不在 object_info ⇒ 不报（交给缺节点预检 ✓）",
             wf.check_combo_choices({"9": {"class_type": "某个没装的类", "inputs": {"x": "y"}}}, object_info), [])

    # 真模板自核：`ResolutionSelector` 的 `aspect_ratio` 必须在节点自己的选项表里 ✓
    for name in LOCAL_BATCH_B + CLOUD_BATCH_B + BATCH_A:
        graph = wf.flatten_subgraphs(load(name))
        bad_values = []
        for node in graph.get("nodes") or []:
            if node.get("type") != "ResolutionSelector":
                continue
            values = node.get("widgets_values")
            if isinstance(values, list) and values and isinstance(values[0], str):
                if values[0] not in wf.ASPECT_RATIO_OPTIONS:
                    bad_values.append((node.get("id"), values[0]))
        check(f"④ {name} 的 ResolutionSelector 画幅都在选项表里", not bad_values, bad_values)

    # 云端 API 节点与本地核心节点**别混** ✗（api_* 那三个不是本机推理路径 ✓）
    for name in CLOUD_BATCH_B:
        classes = sorted(wf.node_classes(load(name)))
        check(f"④ {name} 只有云端 API 节点（不含本地核心节点 ✗）",
              any(cls.startswith("MinimaxHailuo03") for cls in classes)
              and "MiniMaxH3ImageToVideo" not in classes, classes)
    for name in LOCAL_BATCH_B:
        classes = wf.node_classes(load(name))
        check(f"④ {name} 是本地路径（含 MiniMaxH3* 核心节点）",
              bool({c for c in classes if c.startswith("MiniMaxH3")}), sorted(classes))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(f"工作流目录: {WORKFLOW_DIR}")
    for name in LOCAL_BATCH_B + CLOUD_BATCH_B + BATCH_A:
        print(f"  模板: {name}  ({'在' if (WORKFLOW_DIR / name).exists() else '缺!!!'})")
    case_flatten_structure()
    case_flatten_links()
    case_resolution_formula()
    case_connection_beats_widget()
    case_combo_whitelist()
    failed = [name for name, ok, _ in _RESULTS if not ok]
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    for name in failed:
        print(f"  FAILED: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
