"""S28 自检：**族盘点脚本**（`app/scripts/dit_families.py`，2026-10-30）。

这个脚本回答的是"盘上有几族、每族差什么才能跑" ✓。它的价值**全在口径上** ✗：
把"专用实现（H3/SDXL）"误报成"拓扑读不全"、把"没查过配套件"说成"缺"、
把"还没判族"说成"能跑" —— 三种错法都会让人去做错的事 ✓✗。

本套盯四件事：
1. **专用族不按通用口径判** ✓（H3/SDXL 的 depth/hidden 本来就"读不出来" ✓ 不是缺口 ✗）；
2. 配套件**缺就报缺**、在就报在（用临时目录造真文件 ✓，不碰共享池 ✓）；
3. `--strict` 只拦"没通路 / 配套件缺" ✓（"拓扑读不全"不在内 ✓）；
4. 不认识的权重进 `unrecognised` ✓（不是"能跑" ✓）。

运行::

    ./.venv/Scripts/python.exe tests/engine_dit_families_script_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.scripts import dit_families as script  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def fake_file(family: str, *, buildable: bool = True, label: str = "",
              unknown: tuple[str, ...] = (), name: str = "x.safetensors") -> dict[str, Any]:
    return {
        "path": f"<合成>/{name}", "name": name, "family": family,
        "spec": {
            "key": family, "label": label or family, "buildable": buildable, "depth": 28,
            "hidden": 6144, "unknown": list(unknown), "strippedPrefix": "",
            "norm": "rms_scale", "attn": "separate_qkv", "mlp": "swiglu_gate_up_down",
            "cond": "adaln_lin", "qkNorm": True, "attnGate": True, "dualStream": False,
            "evidence": [], "blockPrefix": "blocks.",
        },
    }


def _touch(root: Path, *parts: str) -> None:
    target = root.joinpath(*parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"\x00")


# ══════════════════════════════════════════════════════════════════════════
# ① 专用族：不许拿通用口径判
# ══════════════════════════════════════════════════════════════════════════
def case_dedicated() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _touch(root, "vae/minimax_h3_video_vae_fp16.safetensors")
        _touch(root, "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors")
        _touch(root, "vae/minimax_h3_audio_vae_fp32.safetensors")
        report = script.summarise(
            [fake_file("h3", buildable=False, label="MiniMax H3", unknown=("heads",))],
            root=root)
    bucket = report["families"]["h3"]
    check("① 专用族 buildable=False **不算**缺口（H3 走 `dit.py`，键名核对是 `h3_keys` ✓）",
          not any("拓扑读不全" in gap for gap in bucket["gaps"]), bucket["gaps"])
    check("① 专用族的 unknown 也不进缺点（那是它自己实现的范畴 ✓）",
          not any("读不到的参数" in gap for gap in bucket["gaps"]), bucket["gaps"])
    check("① 专用族该有的配套件齐 ⇒ 一句缺口都没有",
          bucket["gaps"] == [], bucket["gaps"])
    check("① 表里「可建」列写「专用」而不是 0/1 ✓",
          "专用" in script.render(report), script.render(report).splitlines()[4])


# ══════════════════════════════════════════════════════════════════════════
# ② 配套件：缺就报缺、在就报在（清单里的模式都实测过盘 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_companions() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bare = script._companions(root, "krea2")
        check("② 空目录 ⇒ TE/VAE 都报缺（**缺**不是「没查」✓）",
              bare["checked"] and len(bare["missing"]) == 2, bare["missing"])

        _touch(root, "clip/Qwen3_VL_4B/Krea2-Engineer-V1-fp8_scaled.safetensors")
        _touch(root, "vae/ae.safetensors")
        full = script._companions(root, "krea2")
        check("② 放上真文件名 ⇒ 一句不缺，且把命中的文件名报出来",
              full["missing"] == [] and full["text"] and full["vae"], full)

        empty = script._companions(root, "seedvr2")
        check("② 本来就不需要 TE 的族**不许**报「缺 TE」（SEEDVR2 只要 VAE ✓）",
              not any("缺 text" in item for item in empty["missing"]), empty["missing"])

        _touch(root, "vae/SEEDVR2/ema_vae_fp16.safetensors")
        check("② SEEDVR2 有 VAE ⇒ 齐（TE 那栏留空是**不需要**，不是缺 ✓）",
              script._companions(root, "seedvr2")["missing"] == [], None)

        check("② 清单里没登记的族 ⇒ 明确说「没查」，不许假装齐 ✓",
              script._companions(root, "某个还没登记的族")["checked"] is False, None)


# ══════════════════════════════════════════════════════════════════════════
# ③ --strict 的判据
# ══════════════════════════════════════════════════════════════════════════
def case_strict() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _touch(root, "clip/Qwen3_VL_4B/Krea2-Engineer-V1-bf16.safetensors")
        _touch(root, "vae/ae.safetensors")
        _touch(root, "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors")
        _touch(root, "vae/minimax_h3_video_vae_fp16.safetensors")
        _touch(root, "vae/minimax_h3_audio_vae_fp32.safetensors")
        # ⚠️ 判据**不许举某个具体族当"没通路"的例子** ✗✗：2026-10-31 实测 —— 原先拿 `krea2` 举例，
        #    后来 `dit_generic.py` 把 krea2 的骨架写出来了（`ENGINE_STATE` 多了一行 ✓），这条测试就
        #    变成"红得对、但说的是另一件事" ✓✗（它其实在测"ENGINE_STATE 里有没有 krea2" ✓）。
        # ⇒ 直接喂**纯函数** `blocked()` 造两个典型 ✓：没通路的族 + 有通路但配套件缺的族 ✓。
        report = script.summarise([fake_file("flux", label="Flux"),
                                   fake_file("h3", buildable=False, label="MiniMax H3")], root=root)
        check("③ 拦「没通路」的族（flux 骨架没写 ✓）—— H3 这种专用族有通路 ⇒ 不拦 ✓",
              script.blocked(report) == ["flux"], script.blocked(report))
        handmade = {"families": {
            "krea2": {"companions": {"missing": []}},                      # 有通路、配套齐 ⇒ 放 ✓
            "flux": {"companions": {"missing": []}},                       # 没通路 ⇒ 拦 ✓
            "sdxl": {"companions": {"missing": ["缺 vae：`x`"]}},          # 有通路但配套缺 ⇒ 也拦 ✓
        }}
        check("③ 拦**两类**：没通路 ✓ + 配套件缺 ✓（缺 TE/VAE 就是跑不起来，与骨架缺是两回事 ✓）",
              script.blocked(handmade) == ["flux", "sdxl"], script.blocked(handmade))


# ══════════════════════════════════════════════════════════════════════════
# ④ 渲染与"认不出"
# ══════════════════════════════════════════════════════════════════════════
def case_render() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # 同 ③：别拿"现在恰好有通路"的族当反例 ⇒ 用 flux（骨架确实还没写 ✓）
        report = script.summarise([fake_file("flux", label="Flux", unknown=("heads",))], root=root,
                                  unrecognised=[{"path": "<某>.safetensors", "why": "键名不在指纹里"}])
    text = script.render(report)
    check("④ 认不出的单列一节，且**不写**成能跑 ✓",
          "认不出的" in text and "键名不在指纹里" in text, text[-300:])
    check("④ 没通路的族在表里露出来（`**无**`）",
          "**无**" in text, text.splitlines()[4])
    check("④ 未登记配套件的族在 TE/VAE 栏写 `?`（**没查**≠通过 ✓）",
          script.render(script.summarise([fake_file("mystery")], root=root)).count("?") >= 1, None)
    check("④ 空盘不炸（0 份也要出一张像样的表 ✓）",
          "0 份" in script.render(script.summarise([], root=root)), None)


def main() -> int:
    case_dedicated()
    case_companions()
    case_strict()
    case_render()
    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"\n      ↳ {detail}"))
    print(f"\nSUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed"
          + ("  ❌" if failed else "  ✅"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
