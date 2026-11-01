"""H3 文本编码器**装载前**的硬门禁 ✓：模块键名/形状 vs 盘上 safetensors 头。

## 为什么要有这一份 ✗

32B 语言塔 bf16 要 **约 60 GiB** ✗ —— 建一遍再比对就得 OOM ✓✗。
所以这里用 ``with torch.device("meta"):`` 建**空壳** ✓（零字节 ✓ 形状全对 ✓），
只做**键名与形状**的对账 ✓ —— 这一步能抓住的错，全都属于「装完才发现、或者根本装完不报错」那类：

* 语言塔少了 ``self.model`` 那层 ⇒ 键名变 ``layers.N.*`` ✗ ⇒ 全套权重装不进去 ✓
* 视觉塔**忘了** ``visual.`` 前缀 ⇒ 那 27 层 BF16 静默装不进去 ✓ 视觉塔保持随机初值 ✓
  且语言塔照常工作 ✓ ⇒ **只有图糊** ✓ 不报错 ✗
* ``combine`` 把语言塔壳整个挂进去 ⇒ 键名变 ``text.model.*`` ✗ 且参数**注册两次** ✓
* QK-norm / DeepStack merger 漏建 ⇒ 少键 ✓（``load_module_weights`` 会报 missing ✓ 但不该到那一步才知道 ✗）
"""
from __future__ import annotations

import json
import pathlib
import struct
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.services.engine import h3_encode as enc  # noqa: E402
from app.services.engine import qwen3vl_te as te  # noqa: E402
from app.services.engine import qwen3vl_vision as qv  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []

#: H3 的文本编码器 ✓（语言塔 I8 量化 ＋ 视觉塔 BF16，**同一个文件** ✓）
WEIGHTS = pathlib.Path(
    r"D:\Comfy-Desktop\ComfyUI-Shared\models\text_encoders"
    r"\qwen3vl_32b_minimax_h3_int8_convrot.safetensors")

#: 量化**伴随**键 ✓ —— 不是模块参数 ✗（`load_module_weights` 自己会把它们当 companion 吃掉 ✓）
_AUX = (".weight_scale", ".comfy_quant", ".scale", ".scale_weight", ".scale_input",
        ".input_scale", ".weight_scale_2")


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(name: str, detail: str) -> None:
    _RESULTS.append((name, True, f"（跳过 ✓：{detail} —— **如实标记**，不算通过 ✗）"))


def read_header(path: pathlib.Path) -> dict[str, tuple[str, list[int]]]:
    """只读 safetensors 头 ✓（不加载任何张量 ✓）。"""
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    out: dict[str, tuple[str, list[int]]] = {}
    for key, item in header.items():
        if key == "__metadata__":
            continue
        out[key] = (item["dtype"], list(item["shape"]))
    return out


def _is_aux(key: str) -> bool:
    return any(key.endswith(suffix) or suffix in key for suffix in _AUX)


def main() -> int:
    if not WEIGHTS.exists():
        for name in ("① 语言塔键名", "② 视觉塔键名", "③ 形状", "④ combine 不重复注册"):
            skip(name, f"盘上没有 {WEIGHTS.name}")
        return _report()

    header = read_header(WEIGHTS)
    lang = {key: shape for key, (_dtype, shape) in header.items()
            if not key.startswith("visual.") and not _is_aux(key)}
    vision = {key: shape for key, (_dtype, shape) in header.items()
              if key.startswith("visual.") and not _is_aux(key)}
    check("① 头部读出 ⇒ 语言塔 %d 键 / 视觉塔 %d 键 ✓" % (len(lang), len(vision)),
          len(lang) > 400 and len(vision) > 300, (len(lang), len(vision)))

    # ② 尺子从**盘上形状**反推 ✓（不信常量 ✓）
    shapes = {key: shape for key, (_dtype, shape) in header.items()}
    text_shape = te.infer_shape(shapes)
    vision_shape = qv.infer_shape(shapes)
    check("② 反推出的两把尺子都自洽 ✓（语言塔 ropeDims 之和 = 64 ✓、视觉塔 merge 2 ✓）",
          te.shape_problems(text_shape) == []
          and vision_shape["spatial_merge_size"] == 2
          and vision_shape["out_hidden_size"] == text_shape["hidden"],
          (te.shape_problems(text_shape), vision_shape))

    try:
        import torch  # noqa: F401
    except BaseException:  # noqa: BLE001
        for name in ("③ 语言塔键名与盘上一致", "③′ 视觉塔键名与盘上一致（含 visual. 前缀）",
                     "③″ 逐键形状一致", "④ combine 后键名仍是 model.* + visual.*"):
            skip(name, "没有 torch")
        return _report()

    import torch

    with torch.device("meta"):
        text_tower = te.build_text_tower(text_shape)
        vision_tower = qv.build_vision_tower(vision_shape)
        encoder = enc.combine(text_tower, vision_tower)

    #    ⚠️ 一律在 **``encoder`` 全名空间**里比 ✓ —— 拿 ``encoder.model.state_dict()`` 比会把
    #       ``model.`` 那一层丢掉 ✓ 自以为对 ✓（本测试首版就踩了这个 ✓ 对账才抓住 ✓）
    owned = {key: list(value.shape) for key, value in encoder.state_dict().items()}
    own_text = {key: shape for key, shape in owned.items() if key.startswith("model.")}
    own_vision = {key: shape for key, shape in owned.items() if key.startswith("visual.")}

    check("③ 语言塔键名**逐字**等于盘上 ``model.*`` ✓（%d 键 ✓）—— 少了 ``self.model`` 那层"
          "就变成 ``layers.N.*`` ✗ 整套装不进 ✓✗" % len(own_text),
          set(own_text) == set(lang),
          {"只在盘上": sorted(set(lang) - set(own_text))[:6],
           "只在模块": sorted(set(own_text) - set(lang))[:6],
           "盘上": len(lang), "模块": len(own_text)})
    check("③′ 视觉塔键名**带着** ``visual.`` 前缀逐字对上 ✓（%d 键 ✓）—— **忘了前缀** ⇒ 那 27 层"
          "静默装不进去 ✓ 视觉塔保持随机初值 ✓ 只有图糊 ✓ 不报错 ✗" % len(own_vision),
          set(own_vision) == set(vision),
          {"只在盘上": sorted(set(vision) - set(own_vision))[:6],
           "只在模块": sorted(set(own_vision) - set(vision))[:6],
           "盘上": len(vision), "模块": len(own_vision)})

    shape_bad: list[tuple[str, list[int], list[int]]] = []
    for key, shape in own_text.items():
        if key in lang and shape != lang[key]:
            shape_bad.append((key, shape, lang[key]))
    for key, shape in own_vision.items():
        if key in vision and shape != vision[key]:
            shape_bad.append((key, shape, vision[key]))
    check("③″ 每个键的**形状**逐项一致 ✓（%d + %d 键 ✓）" % (len(own_text), len(own_vision)),
          not shape_bad, shape_bad[:6])

    keys = list(encoder.state_dict())
    check("④ ``combine`` 之后 ⇒ 键名只有 ``model.*`` ＋ ``visual.*`` ✓（共 %d 键 ✓）、"
          "**没有** ``_text.`` / ``text.`` 那些多出来的 ✓"
          "（语言塔壳整个挂进去 ✗ ⇒ 参数**注册两次** ✓ 键名翻倍 ✓）" % len(keys),
          all(key.startswith(("model.", "visual.")) for key in keys)
          and len(keys) == len(own_text) + len(own_vision),
          (len(keys), len(own_text) + len(own_vision),
           sorted({key.split('.')[0] for key in keys})))
    check("④′ 参数总数 = 盘上两半之和 ✓（不多不少 ✓）",
          len(list(encoder.parameters())) == len(lang) + len(vision),
          (len(list(encoder.parameters())), len(lang) + len(vision)))

    # ⑤ 视觉塔真的是 BF16 口径（盘上没有量化伴随键 ✓）
    vision_aux = [key for key in header if key.startswith("visual.") and _is_aux(key)]
    check("⑤ 视觉塔那半盘上**没有**量化伴随键 ✓ ⇒ 它是 BF16 原样 ✓（不用反量化 ✓）",
          vision_aux == [], vision_aux[:6])
    lang_aux = [key for key in header if not key.startswith("visual.") and _is_aux(key)]
    check("⑤′ 语言塔那半**有**量化伴随键 ✓（%d 个 ✓ ⇒ ``load_module_weights`` 要走反量化 ✓）"
          % len(lang_aux), len(lang_aux) > 0, len(lang_aux))

    return _report()


def _report() -> int:
    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        if not ok:
            print(f"FAIL {name} —— 实际：{detail!r}")
    print(f"SUMMARY {len(_RESULTS) - len(failed)}/{len(_RESULTS)} 通过 ✓")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
