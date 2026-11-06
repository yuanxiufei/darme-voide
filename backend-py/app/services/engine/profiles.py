"""**用途档位表** ✓ —— 「哪个用途用多大的画布 / 几步 / 多少 cfg / 什么采样器」的**唯一数据源**。

## 为什么要有这一层（本仓踩过的形状 ✓）

同一族的画布与采样档在**工作流 JSON** 与引擎 ``_fill_*_plan`` 的**常量**里各写一份 ✓✗：
生产线 2512 的定妆板是 ``2048×1536``、场景卡是 ``1344×768``，而引擎默认给 ``1.76 MP``
（``qwen_image`` 一族 ⇒ ``1792×1024`` 或 ``1328×1328`` 那档 ✓）—— **两边都不是同一个数** ✓✗。
症状是「参数全对、图就是不像」✓，且**不报错** ✗。

⇒ 档位**外部化**成这张表 ✓：引擎不再"猜一个默认"，而是「**没有点名用途就按请求跑**（旧行为逐位不变 ✓）、
**点名了就照表执行并写明取自哪一档**」✓。

## 纪律（与本仓既有口径一致 ✓）

* **不猜** ✗：表里 ``width/height`` 为 ``None`` 的档（例如 H3 由导演台节点内部生成画布）⇒
  :func:`app.services.engine.canvas.plan_canvas` 会**明确报错**，而不是填一个"看起来合理"的数 ✗；
* **可追溯** ✓：每一档的 ``source`` 写**工作流文件名** ✓ —— 数值对不上时能直接回去核 ✗；
* **只描述、不替创作者裁定** ✗：这是"用途声明"✓ 不是"引擎默认值"✗ ——
  默认档之争（定妆板还是场景卡）不属于代码 ✓。

## 数值出处

全部来自 ``comfy-studio/ComfyUI/user/default/workflows/AIGC中国风漫剧/`` 下各工作流的
``EmptySD3LatentImage`` 与 ``KSampler`` 的 ``widgets_values`` ✓（``[seed, steps, cfg, sampler, scheduler,
denoise]`` ✓）。**不是**从模板文档抄的 ✗。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "CanvasProfile",
    "PROFILES",
    "STAGES",
    "as_capabilities",
    "for_stage",
    "profile_ids",
    "resolve",
]

#: 引擎认识的 stage ✓（与 :mod:`app.services.engine.pipeline` 的 ``*_STAGE`` 常量**同值** ✓ ——
#: 由 ``tests/engine_profiles_test.py`` 交叉钉住 ✗ 不靠人记 ✓）。
STAGES = ("h3", "sdxl", "zimage", "qwen_image21", "qwen_image", "krea2")


@dataclass(frozen=True)
class CanvasProfile:
    """一个**用途档** ✓ = 画布 + 采样档 + 出处。

    ⚠️ ``width/height`` 存的是**实测画布**（像素）✓ 不是"MP + 比例"✗ ——
    ``1344×768`` 与 ``16:9`` 的 ``1.03 MP`` **不闭合**（比例 1.75 vs 1.778 ✓），
    拿 MP 重算会得到 ``1360×768`` ✓✗ ⇒ 真值只能是像素 ✓。
    """

    id: str
    stage: str
    width: int | None
    height: int | None
    steps: int | None = None
    cfg: float | None = None
    sampler: str | None = None
    schedule: str | None = None
    denoise: float | None = None
    label: str = ""
    source: str = ""

    @property
    def target(self) -> str:
        """``"qwen_image/character-sheet"`` ⇒ ``"character-sheet"`` ✓（前端填的口径 ✓）。"""
        return self.id.split("/", 1)[1] if "/" in self.id else self.id

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "stage": self.stage, "target": self.target,
            "label": self.label, "width": self.width, "height": self.height,
            "steps": self.steps, "cfg": self.cfg, "sampler": self.sampler,
            "schedule": self.schedule, "denoise": self.denoise, "source": self.source,
            # ⚠️ **派生量、不落字段** ✗：档位里 ``denoise<1``（``20_分镜尾帧`` ✓ 0.55）就
            #    **必须**连 ``initImage`` 一起传 ✓✗（管线缺图当场 400 ✓）—— 回执先替前端把
            #    这条说清 ✓，否则"照抄档位参数"必撞一个 400 而看不出为什么 ✓。
            "requiresInitImage": self.requires_init_image,
        }

    @property
    def requires_init_image(self) -> bool:
        """这档要不要**起点图** ✓（``denoise<1`` ⇒ 是 ✓✗ —— 只砍 σ 不换起点 = 灰图 ✓）。"""
        return self.denoise is not None and float(self.denoise) < 1.0


def _p(profile_id: str, *, width: int | None, height: int | None, label: str,
       steps: int | None = None, cfg: float | None = None, sampler: str | None = None,
       schedule: str | None = None, denoise: float | None = None,
       source: str = "") -> CanvasProfile:
    stage = profile_id.split("/", 1)[0]
    return CanvasProfile(id=profile_id, stage=stage, width=width, height=height, steps=steps,
                         cfg=cfg, sampler=sampler, schedule=schedule, denoise=denoise,
                         label=label, source=source)


#: 全部用途档 ✓（key = ``"<stage>/<target>"`` ✓）。新增档位**只改这里** ✓ 不改管线代码 ✗。
PROFILES: dict[str, CanvasProfile] = {p.id: p for p in (
    # ── Qwen-Image 2512 / Edit 2511 ✓ ────────────────────────────────────────
    _p("qwen_image/character-sheet", width=2048, height=1536, label="角色定妆板",
       steps=30, cfg=5.5, sampler="euler", schedule="simple", denoise=1.0,
       source="01_角色定妆板_Qwen2512.json"),
    _p("qwen_image/character-sheet-lightning", width=2048, height=1536, label="角色定妆板（Lightning 4 步）",
       steps=4, cfg=1.0, sampler="euler", schedule="simple", denoise=1.0,
       source="01_角色定妆板_Qwen2512_Lightning.json"),
    _p("qwen_image/scene-card", width=1344, height=768, label="场景设定卡",
       steps=30, cfg=4.0, sampler="euler", schedule="simple", denoise=1.0,
       source="02_场景设定卡_Qwen2512.json"),
    _p("qwen_image/storyboard-first", width=1344, height=768, label="分镜首帧",
       steps=30, cfg=4.0, sampler="euler", schedule="simple", denoise=1.0,
       source="03_分镜首帧_Qwen2512.json"),
    _p("qwen_image/crowd", width=1344, height=768, label="背景层群像",
       steps=30, cfg=4.0, sampler="euler", schedule="simple", denoise=1.0,
       source="13_背景层群像_Qwen2512.json"),
    _p("qwen_image/storyboard-last", width=1344, height=768, label="分镜尾帧（参考图精修 0.55）",
       steps=30, cfg=4.0, sampler="euler", schedule="simple", denoise=0.55,
       source="20_分镜尾帧_Qwen2512参考图.json"),
    _p("qwen_image/reference-sheet", width=2048, height=1152, label="角色参考定妆（Edit 2511）",
       steps=30, cfg=4.0, sampler="euler", schedule="normal", denoise=1.0,
       source="10_角色参考定妆_QwenEdit2511.json"),
    # ── Qwen-Image 2.1 ✓ ────────────────────────────────────────────────────
    _p("qwen_image21/reference-edit", width=None, height=None, label="参考图编辑（画布继承参考图）",
       steps=30, cfg=4.0, sampler="euler", schedule="simple", denoise=1.0,
       source="11_参考图编辑_Qwen2.1.json"),
    # ── Z-Image ✓ ───────────────────────────────────────────────────────────
    _p("zimage/character-sheet", width=2048, height=1536, label="角色定妆板（Z-Image）",
       steps=30, cfg=4.0, sampler="res_multistep", schedule="simple", denoise=1.0,
       source="01_角色定妆板_ZImage.json"),
    _p("zimage/scene-card", width=1344, height=768, label="场景设定卡（Z-Image）",
       steps=30, cfg=4.0, sampler="res_multistep", schedule="simple", denoise=1.0,
       source="02_场景设定卡_ZImage.json"),
    # ── Krea2 ✓ ─────────────────────────────────────────────────────────────
    _p("krea2/character-sheet", width=2048, height=1536, label="角色定妆板（Krea2 参考图）",
       steps=8, cfg=2.0, sampler="euler", schedule="simple", denoise=1.0,
       source="16_角色定妆板_Krea2参考图.json"),
    _p("krea2/scene-card", width=1344, height=768, label="场景卡（Krea2）",
       steps=8, cfg=2.0, sampler="euler", schedule="simple", denoise=1.0,
       source="18_场景卡_Krea2.json"),
    _p("krea2/storyboard-first", width=720, height=1280, label="分镜首帧（Krea2 竖屏）",
       steps=30, cfg=4.0, sampler="euler", schedule="simple", denoise=1.0,
       source="17_分镜首帧_Krea2.json"),
    _p("krea2/storyboard-last", width=720, height=1280, label="分镜尾帧（Krea2 参考图）",
       steps=30, cfg=4.0, sampler="euler", schedule="simple", denoise=1.0,
       source="19_分镜尾帧_Krea2参考图.json"),
    _p("krea2/storyboard-last-dual", width=720, height=1280, label="分镜尾帧（Krea2 双参考）",
       steps=30, cfg=4.0, sampler="euler", schedule="simple", denoise=1.0,
       source="21_分镜尾帧_Krea2双参考.json"),
    # ── H3 视频 ✓（画布由导演台节点内部生成 ⇒ 表里**不给**数 ✓，明确说"没定"✗ 不猜 ✗）─
    _p("h3/storyboard-768p", width=None, height=None, label="768p 试片（导演台出画布）",
       steps=None, cfg=None, sampler=None, schedule=None,
       source="04_视频_768p试片_fl2v.json"),
    _p("h3/final-1080p", width=None, height=None, label="1080p 正片（导演台出画布）",
       steps=None, cfg=None, sampler=None, schedule=None,
       source="05_视频_1080p正片_fl2v.json"),
)}


def profile_ids(stage: str | None = None) -> tuple[str, ...]:
    """列出档位 id ✓（给 ``stage`` 就只列该 stage ✓）。"""
    if stage is None:
        return tuple(sorted(PROFILES))
    wanted = str(stage).strip().lower()
    return tuple(sorted(pid for pid, item in PROFILES.items() if item.stage == wanted))


def for_stage(stage: str) -> tuple[CanvasProfile, ...]:
    """某个 stage 的全部档位 ✓（按 id 排序，输出稳定 ✓）。"""
    return tuple(PROFILES[pid] for pid in profile_ids(stage))


def resolve(stage: str, target: str | None) -> CanvasProfile:
    """``stage`` + 用途名 → **档位** ✓；不认识 ⇒ **报错并列出合法的** ✗✗（不回落 ✗）。

    ``target`` 可写 ``"character-sheet"`` ✓ 也可写全 id ``"qwen_image/character-sheet"`` ✓。
    """
    if target in (None, ""):
        raise ValueError(f"没给用途名（target）✗ —— {stage} 的合法档位：{list(profile_ids(stage))} ✓")
    key = str(target).strip()
    if "/" not in key:
        key = f"{str(stage).strip().lower()}/{key}"
    profile = PROFILES.get(key)
    if profile is None:
        raise ValueError(
            f"未知用途档 {target!r} ✗；{stage} 可用："
            f"{[pid.split('/', 1)[1] for pid in profile_ids(stage)]} ✓"
            "（⚠️ 别随手编一个名字 ✓ —— 它会决定画布/步数/cfg ✓✗）")
    return profile


def as_capabilities() -> list[dict[str, Any]]:
    """给 ``/capabilities`` 用的**纯数据** ✓（前端据此列出可选档位 ✓）。"""
    return [PROFILES[pid].to_dict() for pid in profile_ids()]
