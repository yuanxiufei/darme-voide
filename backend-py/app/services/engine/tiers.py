"""**档位表**：步数 / 分辨率 / 加速件 / 显存建议 / **必备模型** ✓（2026-09-24 补 ✓ 零依赖 ✓）。

## 口径来源（**外证** ✓，不是拍的 ✗）
三档与其中的倍率、显存建议、实测耗时都来自**参考实现的档位表** ✓（本仓 2026-09-24 逆向记档 ✓，
见 `.codebuddy/memory/TOPICS.md` §最值得参考：三档引擎表 ✓）—— ⚠️ 它们是**同一个模型家族（MiniMax H3）**
的实测口径 ✓ ⇒ 可以当**起点**用 ✓，但**换机器/换分辨率都要重新量** ✗。

| 档 | steps | 分辨率 | 加速件 | 实测倍率 | 显存建议 |
|---|---|---|---|---|---|
| ``official`` 官方稳妥流（入门 ✓） | 25 | 832×480 | **无**（零第三方依赖 ✓） | 1×（5 秒段 **10~15 分钟** ✓） | 8G 可跑 |
| ``std8`` 标准加速流（中级 ✓） | 8 | 1024×576 | 8 步蒸馏 LoRA（LightX2V 系 ✓） | ≈2.1× | 16G+ |
| ``fast4`` 极速流（旗舰 ✓） | 4 | 1024×576 | 4 步蒸馏（**官方 Comfy-Org R2V 模板同款** ✓） | ≈3.44× | 8G 甜点 ✓ |

## 本模块能回答什么（判据都在代码里 ✓）
* `resolve(key)` → 档位 ✓（**不认识就报错并列出合法的** ✗ 不硬套 ✓）；
* `for_steps(steps)` → 猜得出来是哪档 ✓；**猜不出来 ⇒ ``None``** ✓（= 自定义档 ✓ 不硬塞 ✗）；
* `estimate_minutes(...)` → 耗时区间 ✓；⚠️ **跨分辨率一律拒** ✗✗（倍率是在**各自分辨率**下量的 ✓✗）；
* 每档的**必备模型** ✓（含「零第三方」这一档的**没有 LoRA** ✓ —— 这条最容易在部署时漏 ✗）。

## 不猜
* ⚠️ 显存建议是**建议**不是硬门槛 ✓（``advised`` / ``vram_min_gib`` ✓ —— 别拿它当准入检查 ✗）；
* 8 步那档的 LoRA **具体文件名未核** ✗ ⇒ 只记**品类**（LightX2V 系 ✓），4 步那档才有实锤文件名 ✓；
* 倍率是「**相对 official**」的口径 ✓，所以本模块的耗时估算**只在同分辨率下成立** ✓✗。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

__all__ = ["TIER_PRESETS", "TierPreset", "advised", "estimate_minutes", "for_steps", "presets",
           "resolve", "summary"]

#: 基础必备模型（**每档都要** ✓；口径：H3 官方那套 ✓）—— ⚠️ 文件名里那串量化的后缀会变 ✗
#: （int8 / fp8 / convrot ✓），所以这里只钉**品类 + 关键前缀** ✓。
BASE_MODELS: tuple[str, ...] = (
    "UNET：`minimax_h3_*ref2va*` / `*fl2va*`（参考素材那半边只能用 Ref2VA ✓）",
    "CLIP：`qwen3vl_*minimax_h3*`（type=minimax ✓）",
    "VAE：`minimax_h3_video_vae_*`",
    "音频 VAE：`minimax_h3_audio_vae_*`",
)


@dataclass(frozen=True)
class TierPreset:
    """一个档位 ✓（``third_party_free`` 与 ``models`` 是一对 ✗：零第三方那档**不许**带加速件 ✓）。"""

    key: str
    label: str
    steps: int
    size: tuple[int, int]
    accelerator: str
    speedup: float
    vram_min_gib: float
    vram_hint: str
    models: tuple[str, ...]
    third_party_free: bool = False
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "label": self.label, "steps": self.steps, "size": list(self.size),
                "accelerator": self.accelerator, "speedup": self.speedup,
                "vramMinGiB": self.vram_min_gib, "vramHint": self.vram_hint,
                "models": list(self.models), "thirdPartyFree": self.third_party_free,
                "notes": list(self.notes)}


def _presets() -> dict[str, TierPreset]:
    official = TierPreset(
        key="official", label="官方稳妥流（入门 ✓ 零第三方依赖）", steps=25, size=(832, 480),
        accelerator="none", speedup=1.0, vram_min_gib=8.0, vram_hint="8G 可跑",
        models=BASE_MODELS, third_party_free=True,
        notes=("**零第三方依赖** ✓（任何加速插件都不装也能跑 ✓ —— 排查问题时的基准档 ✓）",
               "实测：**5 秒段约 10~15 分钟** ✓（倍率基准 = 1× ✓）"))
    std8 = TierPreset(
        key="std8", label="标准加速流（中级 ✓）", steps=8, size=(1024, 576),
        accelerator="8 步蒸馏 LoRA（LightX2V 系 ✓）", speedup=2.1, vram_min_gib=16.0,
        vram_hint="16G+",
        models=BASE_MODELS + ("LoRA：8 步蒸馏（LightX2V 系 ✓ —— ⚠️ **具体文件名未核** ✗）",),
        notes=("分辨率比 official 高一档（1024×576 ✓）⇒ ⚠️ **不能**拿它的倍率去套 832×480 ✗✗",))
    fast4 = TierPreset(
        key="fast4", label="极速流（旗舰 ✓）", steps=4, size=(1024, 576),
        accelerator="4 步蒸馏（**官方 Comfy-Org R2V 模板同款** ✓）", speedup=3.44, vram_min_gib=8.0,
        vram_hint="8G 甜点 ✓",
        models=BASE_MODELS + (
            "LoRA：`minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16` ✓（实锤文件名 ✓）",),
        notes=("实锤：4 步档在 8G 上就是甜点 ✓", "高端卡可再叠加速（+Sol-Attn +NVFP4 ⇒ 1344×768 ✓）"))
    return {preset.key: preset for preset in (official, std8, fast4)}


#: 三档表 ✓（⚠️ 想加档位就加在这里 ✓ —— 别在别处再抄一份 ✗）。
TIER_PRESETS: dict[str, TierPreset] = _presets()


def presets() -> tuple[TierPreset, ...]:
    """全部档位 ✓（按 **steps 从多到少** ✓ = 从稳到快 ✓）。"""
    return tuple(sorted(TIER_PRESETS.values(), key=lambda preset: -preset.steps))


def resolve(key: str) -> TierPreset:
    """按 key 取档位 ✓；⚠️ 不认识 ⇒ **报错并列出合法的** ✗（不静默回落默认档 ✗✗）。"""
    wanted = str(key or "").strip()
    if wanted not in TIER_PRESETS:
        raise ValueError(
            f"档位 {key!r} 不认识 ✗ ⇒ 合法的是：" + " / ".join(sorted(TIER_PRESETS))
            + " ✓（⚠️ **不静默回落到默认档** ✗ —— 那会让「选了极速」变成「跑了稳妥」且不报错 ✓✗）")
    return TIER_PRESETS[wanted]


def for_steps(steps: int) -> TierPreset | None:
    """按步数反查档位 ✓；**查不出来就 ``None``** ✓（= 自定义档 ✓ —— 不硬塞进某一档 ✗）。"""
    count = int(steps)
    for preset in presets():
        if preset.steps == count:
            return preset
    return None


def advised(preset: TierPreset, *, gpu_gib: float | None) -> dict[str, Any]:
    """显存**建议**比对 ✓ ⇒ ``{enough, ...}`` ✓。

    ⚠️ 这是**建议**不是硬门槛 ✗：不够也可能跑得动（看分辨率/量化/卸载策略 ✓），
    够了也可能 OOM（看别的进程 ✓）⇒ 只用来**提示** ✓ 别拿它当准入检查 ✗✗。
    """
    if gpu_gib is None:
        return {"enough": None, "advisedGiB": preset.vram_min_gib, "hint": preset.vram_hint,
                "note": "没给显存 ⇒ **没比** ✗（不是通过 ✓）"}
    enough = float(gpu_gib) >= float(preset.vram_min_gib)
    return {"enough": enough, "advisedGiB": preset.vram_min_gib, "hint": preset.vram_hint,
            "note": ("建议（不是硬门槛 ✗）：够不代表一定不 OOM ✓、不够也不代表一定跑不动 ✓")}


def summary(*, gpu_gib: float | None = None) -> dict[str, Any]:
    """给**就绪报告 / 面板**用的摘要 ✓（``inventory.readiness()`` 就接在这里 ✓）。

    ⚠️ 每档都带**必备模型** ✓（部署时最容易漏的就是「零第三方那档以为也要装 LoRA」✗）；
    ⚠️ ``gpu_gib`` 没给 ⇒ 各档的 ``advised.enough`` 是 ``None`` ✓（**没比** ✗ 不是通过 ✓）。
    """
    return {
        "presets": [{**preset.to_dict(), "advised": advised(preset, gpu_gib=gpu_gib)}
                    for preset in presets()],
        "gpuGiB": None if gpu_gib is None else float(gpu_gib),
        "note": ("⚠️ 显存是**建议**；倍率是「**相对 official、且各自分辨率下**」量的 ✗ "
                 "⇒ 换机器/换分辨率都要重新量 ✓（见 `estimate_minutes` ✓）"),
    }


def estimate_minutes(preset: TierPreset, seconds: float, *,
                     size: Sequence[int] | None = None) -> dict[str, Any]:
    """估耗时 ✓ ⇒ ``{low, high, basis}``（分钟 ✓）。

    * ``official`` 是**基准档** ✓：口径是实测「5 秒段 10~15 分钟」⇒ 按比例给**区间** ✓；
    * 其余档按**实测倍率**缩 ✓；
    * ⚠️⚠️ 给了 ``size`` 且**与档位自己的分辨率不一致** ⇒ **拒** ✗✗（倍率是在各自分辨率下量的 ✓✗
      —— 跨分辨率套数字等于编 ✓）。
    """
    length = float(seconds)
    if length <= 0:
        raise ValueError(f"时长必须为正（收到 {seconds!r} ✗）")
    if size is not None:
        given = (int(size[0]), int(size[1]))
        if given != preset.size:
            raise ValueError(
                f"跨分辨率**不估** ✗✗：档位 {preset.key} 的倍率是在 {preset.size[0]}×{preset.size[1]} "
                f"上量的 ✓，而这里给的是 {given[0]}×{given[1]} ✓ ⇒ **本仓不套** ✗"
                f"（要估就先把那个分辨率**量一遍** ✓）")
    base_low = 10.0 / 5.0 * length
    base_high = 15.0 / 5.0 * length
    factor = float(preset.speedup) or 1.0
    return {"low": round(base_low / factor, 2), "high": round(base_high / factor, 2),
            "basis": (f"基准：official 实测「5 秒段 10~15 分钟」✓ × {length:g}s"
                      + (f" ÷ {factor:g}×（{preset.key} 的实测倍率 ✓）" if factor != 1.0 else ""))}
