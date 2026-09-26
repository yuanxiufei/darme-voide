"""**提示词质感层**（移植 ``short-drama-agent`` 的 **Mx-Shell 五段式** ✓ 零依赖 ✓）。

## 它解决什么问题

短剧视频提示词最容易写成**空话堆砌** ✗：``电影感`` / ``大片`` / ``4K`` / ``high quality`` ——
这些话对扩散模型**几乎不产生像素级约束** ✗（参考项目原话：*不要依赖 cinematic/epic/stunning 这类词，
除非它们与**物理**的相机、光线、材质、运动细节成对出现* ✓）。

所以这里做的不是"润色文采" ✗，而是**逼出可执行的物理锚点** ✓，按参考项目的**五段**组织 ✓：

| 段 | 内容 | 本模块的**硬约束** ✓ |
|---|---|---|
① ``core_theme_tags`` | 3-6 个**具体**标签（格式 → 类型 → 美学 ✓） | 少于 3 个 ⇒ 报问题 ✓ |
② ``locked_character_scene`` | 脸/发/服装/状态 + 材质 + 场景地理 + 道具当前状态 | 空 ⇒ 报问题 ✓ |
③ ``atmosphere_quality`` | **真实**相机+镜头型号、调色、光线、胶片颗粒/物理质感 | 镜头锚点空 ⇒ 报问题 ✓ |
④ ``camera_rules`` | 景别/角度/运动/**画面方向**；手持时必须有**呼吸般轻微浮动** | 手持 ⇒ **自动补**浮动句 ✓ |
⑤ ``storyboard_slice`` | **只做一件事** ✓（动作 / 相机 / 声音 / 可选 VFX ✓） | 空 ⇒ 报问题 ✓ |

外加参考项目强调的三条 ✓：**声音政策**（无配乐、只用同期声 ✓）、
**瑕疵锚点 ≥2**（真实感的关键 ✓）、**克制结尾** ✓。

## 两条纪律（写进代码，不只是写在文档里 ✓）

1. **本层不新增剧情动作** ✗ —— 它只做"表达强化" ✓。所以它**不接收**也不要生成新的动作描述 ✓；
2. **空话必须成对** ✓ —— 出现 ``cinematic``/``4K`` 这类词但没有物理细节 ⇒ 进 ``issues`` ✓
   （自检直接钉住这条 ✓）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["PolishInputs", "PolishedPrompt", "VAGUE_TERMS", "build_prompt_layer",
           "find_vague_terms"]

#: 空话词表 ✓（参考项目点名的那批 + 常见中文对应 ✓）—— 它们**本身不是错** ✗，
#: 只是**必须**与物理细节成对出现 ✓（否则就是无效约束 ✓）。
VAGUE_TERMS: tuple[str, ...] = (
    "cinematic", "epic", "premium", "stunning", "high quality", "4k", "8k",
    "perfect", "advanced", "masterpiece", "best quality", "ultra detailed",
    "电影感", "大片感", "高级感", "极致", "完美", "高清",
)

#: 手持/主观镜头的**轻微浮动** ✓（参考项目原句 ✓ —— 刻意区别于"剧烈晃动" ✓）
HANDHELD_FLOAT = "手持拍摄，全程保持极其轻微的、如呼吸般的镜头浮动，增强临场感；不要变成剧烈晃动。"

#: 声音政策 ✓（参考项目原句 ✓）
SOUND_POLICY = "Sound: No score. Production audio only."

#: 内心独白必须声明**不张嘴** ✓（否则模型会让角色说话 ✗ —— 参考项目专门点了这条 ✓）
NO_MOUTH_OPEN = "此处为内心独白：角色**不张嘴**，声音走后期配音。"


@dataclass(frozen=True)
class PolishInputs:
    """输入 ✓（**全部由上游的连续性镜头卡给出** ✓ —— 本层不发明内容 ✗）。"""

    theme_tags: tuple[str, ...] = ()
    character_scene: str = ""
    lens: str = ""
    palette: str = ""
    texture: str = ""
    shot_type: str = ""
    movement: str = ""
    screen_direction: str = ""
    slice_text: str = ""
    sounds: tuple[str, ...] = ()
    imperfections: tuple[str, ...] = ()
    handheld: bool = False
    inner_monologue: bool = False
    restrained_ending: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "themeTags": list(self.theme_tags), "characterScene": self.character_scene,
            "lens": self.lens, "palette": self.palette, "texture": self.texture,
            "shotType": self.shot_type, "movement": self.movement,
            "screenDirection": self.screen_direction, "slice": self.slice_text,
            "sounds": list(self.sounds), "imperfections": list(self.imperfections),
            "handheld": self.handheld, "innerMonologue": self.inner_monologue,
            "restrainedEnding": self.restrained_ending,
        }


@dataclass
class PolishedPrompt:
    """结果 ✓（``sections`` 五段可分别展示 ✓；``text`` 是拷走即用的整段 ✓）。"""

    sections: dict[str, str] = field(default_factory=dict)
    text: str = ""
    issues: list[str] = field(default_factory=list)
    vagueTerms: list[str] = field(default_factory=list)
    inserted: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """**够硬** ⇔ 没有 issues ✓（issues 全是"约束不够具体"类 ✓ 不是运行时错误 ✓）。"""
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {"sections": dict(self.sections), "text": self.text, "ok": self.ok,
                "issues": self.issues, "vagueTerms": self.vagueTerms,
                "inserted": self.inserted}


def find_vague_terms(text: str) -> list[str]:
    """找出空话词 ✓（大小写不敏感 ✓）。"""
    lowered = str(text or "").lower()
    return sorted({term for term in VAGUE_TERMS if term.lower() in lowered})


def _clean(parts: list[str]) -> str:
    return "，".join(part.strip() for part in parts if part and part.strip())


def build_prompt_layer(inputs: PolishInputs) -> PolishedPrompt:
    """按**五段式**组装 ✓，并把"约束不够具体"的地方**逐条报出** ✓（不静默润色 ✗）。"""
    result = PolishedPrompt()

    # ── ① 主题标签 ✓（3-6 个 ✓ 从格式到类型到美学 ✓）──────────────────────
    tags = [str(tag).strip() for tag in inputs.theme_tags if str(tag).strip()]
    if len(tags) < 3:
        result.issues.append(
            f"core_theme_tags 只有 {len(tags)} 个 ✗ ⇒ 至少要 3 个具体标签"
            f"（格式 → 类型 → 美学 ✓），**不要**用「电影感」这类空话凑数 ✓")
    if len(tags) > 6:
        result.issues.append(f"core_theme_tags 有 {len(tags)} 个 ✗ ⇒ 超过 6 个会互相稀释 ✓")
    result.sections["core_theme_tags"] = " / ".join(tags)

    # ── ② 锁定角色/场景（含道具当前状态 ✓）────────────────────────────────
    if not inputs.character_scene.strip():
        result.issues.append("locked_character_scene 为空 ✗ ⇒ 脸/发/服装/状态与场景几何必须写死 ✓"
                             "（否则逐镜之间角色会漂 ✗）")
    result.sections["locked_character_scene"] = inputs.character_scene.strip()

    # ── ③ 气氛与画质（**真实**相机/镜头 + 调色 + 物理质感 ✓）──────────────
    if not inputs.lens.strip():
        result.issues.append(
            "camera_lens_profile 为空 ✗ ⇒ 要写**真实或可信的**镜头锚点"
            "（例如「Sony Venice + Canon K-35」「35mm 胶片 + 漂白旁路」✓），"
            "光写 cinematic 不给像素级约束 ✗")
    atmosphere = _clean([inputs.lens, inputs.palette, inputs.texture])
    if not atmosphere:
        result.issues.append("atmosphere_quality 三段（镜头/调色/质感）全空 ✗")
    result.sections["atmosphere_quality"] = atmosphere

    # ── ④ 相机规则（含手持浮动 ✓ 与画面方向 ✓）────────────────────────────
    camera_parts = [inputs.shot_type, inputs.movement, inputs.screen_direction]
    if inputs.handheld:
        camera_parts.append(HANDHELD_FLOAT)          # **自动补** ✓ 不用调用方记 ✓
        result.inserted.append("handheldFloat")
    camera = _clean(camera_parts)
    if not inputs.screen_direction.strip():
        result.issues.append("screen_direction 为空 ✗ ⇒ 相邻镜头会左右跳（假连续性 ✗）")
    if not camera:
        result.issues.append("camera_rules 为空 ✗ ⇒ 景别/角度/运动必须给 ✓")
    result.sections["camera_rules"] = camera

    # ── ⑤ 分镜切片（**只做一件事** ✓）────────────────────────────────────
    slice_parts = [inputs.slice_text.strip()]
    if not inputs.slice_text.strip():
        result.issues.append("storyboard_slice 为空 ✗ ⇒ 每个镜头只做**一件事** ✓（动作/相机/声音 ✓）")
    slice_parts.append(SOUND_POLICY)                 # 声音政策必带 ✓
    if inputs.sounds:
        slice_parts.append("场景同期声：" + "、".join(str(item) for item in inputs.sounds))
    if inputs.inner_monologue:
        slice_parts.append(NO_MOUTH_OPEN)            # 内心独白不张嘴 ✓
        result.inserted.append("innerMonologueRule")
    if inputs.restrained_ending:
        slice_parts.append("结尾克制：不做总结式抒情、不留空镜抒情长镜 ✓")
    result.sections["storyboard_slice"] = _clean(slice_parts)

    # ── 瑕疵锚点（≥2 ✓ 真实感的关键 ✓）──────────────────────────────────
    imperfections = [str(item).strip() for item in inputs.imperfections if str(item).strip()]
    if len(imperfections) < 2:
        result.issues.append(
            f"imperfection_anchors 只有 {len(imperfections)} 个 ✗ ⇒ 现实感场景**至少 2 个** ✓"
            f"（湿水泥/磨损金属/布料灰尘/皮肤瑕疵/玻璃裂纹/胶片颗粒 ✓）")
    result.sections["imperfection_anchors"] = "、".join(imperfections)

    # ── 空话检查 ✓（**成对**才允许 ✓）────────────────────────────────────
    assembled = _clean(list(result.sections.values()))
    result.vagueTerms = find_vague_terms(assembled)
    physical = bool(inputs.lens.strip() or inputs.palette.strip() or inputs.texture.strip()
                    or imperfections)
    if result.vagueTerms and not physical:
        result.issues.append(
            f"出现了空话词 {result.vagueTerms[:3]} ✗ 但**没有任何物理锚点** ✓ ⇒ "
            f"这些词对模型不构成像素级约束 ✗（补镜头/光线/材质/运动细节 ✓）")
    if result.vagueTerms and physical:
        result.issues.append(
            f"空话词 {result.vagueTerms[:3]} ✓ 已有物理锚点陪衬 ✓ —— 建议直接删掉它们 ✓"
            f"（占 token 且稀释注意力 ✓）")

    result.text = assembled + "。"
    return result
