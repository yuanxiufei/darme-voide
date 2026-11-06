"""局部重绘 / 扩图两个变体的守卫（2026-11-03 立，随「画布补蒙版与扩图」一起落）。

为什么单独立这套：这两条是**本仓唯一会把「人画的蒙版」当输入**的工作流，而它们错起来
全是静默的 —— 图照样出得来，只是「改了不该改的地方」或「整张被拉伸」✓✗。所以判据只钉
四处会静默走样的地方：

1. **参考图顺序** ✗：`source` 是 `imageIndex 0`、`mask` 是 `imageIndex 1` ——
   顺序反了就是把蒙版当底图（出图会是一张黑白块），而画布侧的连线顺序是**用户可改**的 ✓；
2. **蒙版通道** ✗：`ImageToMask(channel="red")` —— 换成 `alpha` 会让
   「黑底 + 白笔」这张 PNG 的 alpha 全 1 ⇒ 蒙版退化成「全图重绘」（不报错 ✗）；
3. **扩图两段尺寸必须成对** ✗：`#2` 吃四边留白、`#18` 吃输出尺寸，
   只绑一半 ⇒ 撑大后的画布被硬缩回一个不相干的比例（整体拉伸变形）；
4. **数值类型** ✓：画布给的是**字符串**（`values` 只存 string），
   没有 `transform: "int"` 的话 `#2.left` 会变成 `"100"` 而不是 `100`（ComfyUI 会静默取默认值 ✗）。

运行::

    ./.venv/Scripts/python.exe tests/comfyui_inpaint_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(BACKEND_PY))

from app.services import comfyui as svc  # noqa: E402
from app.services import comfyui_catalog as catalog  # noqa: E402
from app.services.comfyui_bindings import qwen21_sigmas  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _variant(key: str) -> dict:
    entry = catalog.get_entry("image-qwen21")
    assert entry is not None, "注册表里没有 image-qwen21"
    return entry, (entry.get("variants") or {}).get(key) or {}


# ─────────────────── ⑨ 变体齐备：文件真的在盘上 ───────────────────

def case_variants_exist() -> None:
    entry, inpaint = _variant("inpaint")
    _, outpaint = _variant("outpaint")

    check("⑨ 注册表有 inpaint 变体", bool(inpaint), inpaint.get("label"))
    check("⑨ 注册表有 outpaint 变体", bool(outpaint), outpaint.get("label"))
    check("⑨ 两个变体都是人话名（前端下拉直接显示）",
          (inpaint.get("label"), outpaint.get("label")) == ("局部重绘", "扩图"),
          (inpaint.get("label"), outpaint.get("label")))

    root = catalog.workflows_root() if hasattr(catalog, "workflows_root") else None
    for key, overlay in (("inpaint", inpaint), ("outpaint", outpaint)):
        graph = catalog.load_graph(entry, key)  # 读不到会抛 ⇒ 这一行本身就是「文件在盘上」的判据 ✓
        check(f"⑨ {key} 工作流读得出且非空", isinstance(graph, dict) and len(graph) > 5, len(graph or {}))
        check(f"⑨ {key} 的 file 字段与磁盘一致", bool(overlay.get("file")), overlay.get("file"))
    check("⑨ workflows_root 可解析（拼路径不靠各模块自己拼 ✓）", root is None or Path(root).exists(), root)


# ─────────────────── ⑨ 局部重绘：参考图顺序与蒙版通道 ───────────────────

def case_inpaint_graph() -> None:
    graph = svc.build_registry_graph({
        "workflowId": "image-qwen21",
        "variant": "inpaint",
        "prompt": "same woman, now with short hair",
        "uploadedImages": [{"name": "source.png"}, {"name": "mask.png"}],
    })
    check("⑨ 重绘：源图落 #1（imageIndex 0）",
          graph["1"]["inputs"].get("image") == "source.png", graph["1"]["inputs"].get("image"))
    check("⑨ 重绘：蒙版落 #2（imageIndex 1）",
          graph["2"]["inputs"].get("image") == "mask.png", graph["2"]["inputs"].get("image"))
    check("⑨ 重绘：两张图不许同名（顺序反了会把蒙版当底图）",
          graph["1"]["inputs"].get("image") != graph["2"]["inputs"].get("image"),
          (graph["1"]["inputs"].get("image"), graph["2"]["inputs"].get("image")))
    check("⑨ 重绘：蒙版走 ImageToMask(red) —— 白=重绘，黑=保留",
          graph["3"]["class_type"] == "ImageToMask" and graph["3"]["inputs"].get("channel") == "red",
          (graph["3"]["class_type"], graph["3"]["inputs"].get("channel")))
    check("⑨ 重绘：提示词落到正面口",
          str(graph["7"]["inputs"].get("prompt") or "").startswith("same woman"),
          graph["7"]["inputs"].get("prompt"))
    # `#12 ManualSigmas` 的输入是**逗号串**（不是数组）：写成 JSON 数组会整段被拒 ✓
    check("⑨ 重绘：8 步走官方 sigma 串（逗号串、9 个值、末位必 0）",
          graph["12"]["inputs"].get("sigmas") == qwen21_sigmas(8),
          graph["12"]["inputs"].get("sigmas"))
    # 只钉「等于官方串」不够 —— 工作流自带的默认值**就是**官方串，那样等于没测注入。
    # 换成 4 步：必须真的按步数插值出 5 个值 ✓（信号是「steps 有没有进到这一格」）
    four = svc.build_registry_graph({
        "workflowId": "image-qwen21",
        "variant": "inpaint",
        "prompt": "x",
        "uploadedImages": [{"name": "source.png"}, {"name": "mask.png"}],
        "values": {"steps": "4"},
    })
    four_sigmas = str(four["12"]["inputs"].get("sigmas") or "")
    check("⑨ 重绘：steps 真进到 sigma 串（4 步 = 5 个值，不是照抄默认）",
          four_sigmas == qwen21_sigmas(4) and len(four_sigmas.split(",")) == 5,
          four_sigmas)

    # 缺蒙版 ⇒ 必须当场拒（否则会拿同一张图当蒙版，整图重绘 ✗）
    try:
        svc.build_registry_graph({
            "workflowId": "image-qwen21",
            "variant": "inpaint",
            "prompt": "x",
            "uploadedImages": [{"name": "source.png"}],
        })
        check("⑨ 重绘：只给一张图必须当场拒", False, "没抛")
    except Exception as exc:  # noqa: BLE001
        check("⑨ 重绘：只给一张图必须当场拒（并说清缺第 2 张）",
              "第 2 张" in str(exc) or "2" in str(exc), str(exc))


# ─────────────────── ⑨ 扩图：四边 + 两段尺寸成对 ───────────────────

def case_outpaint_graph() -> None:
    values = {
        "left": "160", "top": "0", "right": "160", "bottom": "0",
        "feathering": "40", "megapixels": "1.5",
        "outpaintWidth": "1216", "outpaintHeight": "1152",
    }
    graph = svc.build_registry_graph({
        "workflowId": "image-qwen21",
        "variant": "outpaint",
        "prompt": "continue the street and streetlights",
        "uploadedImages": [{"name": "source.png"}],
        "values": values,
    })

    edges = tuple(graph["2"]["inputs"].get(key) for key in ("left", "top", "right", "bottom"))
    check("⑨ 扩图：四边留白都是 **int**（字符串会被 ComfyUI 静默取默认值 ✗）",
          edges == (160, 0, 160, 0) and all(isinstance(item, int) for item in edges), edges)
    check("⑨ 扩图：羽化是 int", graph["2"]["inputs"].get("feathering") == 40,
          graph["2"]["inputs"].get("feathering"))
    check("⑨ 扩图：像素预算落在 #4", float(graph["4"]["inputs"].get("megapixels")) == 1.5,
          graph["4"]["inputs"].get("megapixels"))
    target = (graph["18"]["inputs"].get("width"), graph["18"]["inputs"].get("height"))
    check("⑨ 扩图：输出尺寸落在 #18", target == (1216, 1152), target)

    # 核心不变量：撑大后的尺寸 **必须**等于 #18 的目标尺寸（否则整体拉伸 ✗）
    entry, _ = _variant("outpaint")
    base = catalog.load_graph(entry, "outpaint")
    base_size = (int(base["18"]["inputs"].get("width")), int(base["18"]["inputs"].get("height")))
    check("⑨ 扩图：不传输出尺寸时走工作流自带目标（两段尺寸同源 ✓）",
          base_size == (int(base["18"]["inputs"].get("width")), int(base["18"]["inputs"].get("height"))),
          base_size)
    check("⑨ 扩图：四边与目标尺寸必须成对绑定（只绑一半 ⇒ 拉伸）",
          {"left", "top", "right", "bottom"} <= set((entry.get("variants") or {})["outpaint"]["bindings"])
          and {"outpaintWidth", "outpaintHeight"} <= set((entry.get("variants") or {})["outpaint"]["bindings"]),
          sorted(((entry.get("variants") or {})["outpaint"]["bindings"]).keys()))

    # 没给留白时不报错（走默认 0，等于「不改尺寸」✓ —— 但画布侧已经拦住这种提交）
    bare = svc.build_registry_graph({
        "workflowId": "image-qwen21", "variant": "outpaint",
        "prompt": "x", "uploadedImages": [{"name": "source.png"}],
    })
    check("⑨ 扩图：不给留白也能建图（默认 0，不编数 ✓）",
          bare["2"]["inputs"].get("left") == 0, bare["2"]["inputs"].get("left"))


# ─────────────────── ⑨ 前端契约：变体要带得出 id/label ───────────────────

def case_router_brief() -> None:
    from app.routers.comfyui import _workflow_brief

    entry, _ = _variant("inpaint")
    brief = _workflow_brief(entry)
    ids = [item["id"] for item in brief["variants"]]
    check("⑨ 前端拿到的是**数组**（注册表里是字典，转换在路由层 ✓）",
          isinstance(brief["variants"], list) and {"inpaint", "outpaint"} <= set(ids), ids)
    labels = {item["id"]: item["label"] for item in brief["variants"]}
    check("⑨ 变体 label 是人话（下拉框直接显示）",
          (labels.get("inpaint"), labels.get("outpaint")) == ("局部重绘", "扩图"), labels)
    check("⑨ 参数表里不掺八个扩图格（它们只在绑定里，画布由对话框代填 ✓）",
          not ({"left", "right", "outpaintWidth"} & {item.get("key") for item in brief["parameters"] if isinstance(item, dict)}),
          [item.get("key") for item in brief["parameters"] if isinstance(item, dict)])


def main() -> None:
    case_variants_exist()
    case_inpaint_graph()
    case_outpaint_graph()
    case_router_brief()

    failures = 0
    for name, ok, detail in _RESULTS:
        if ok:
            print(f"  ✓ {name}")
        else:
            failures += 1
            print(f"  ✗ {name}  -> {detail}")
    print(f"\nSUMMARY: {len(_RESULTS) - failures}/{len(_RESULTS)} passed")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
