"""ComfyUI 口径的「模型目录类别」表 + **任意目录**模型根声明解析（自研重写 ✓）。

⚠️ 口径来源（2026-09-26 逐条核对 ✓）：``ComfyUI/folder_paths.py``（类别表 ✓）与
``ComfyUI/utils/extra_config.py``（``extra_model_paths.yaml`` 语义 ✓）——
⚠️⚠️ 那份参考副本是 **GPL-3.0** ⇒ 这里只搬**语义**（类别名 / 目录名 / 扩展名 / 声明格式 ✓），
代码**自己写** ✗（不拷源码 ✗）。

为什么要有它（2026-09-26 用户口径 ✓：模型可装**任意目录** ✓、也要能扫**任意目录** ✓、不许写死路径 ✗）：
1. :func:`category_of_path`：按**父目录名**（ComfyUI 口径 ✓）再按**文件名**认出「这是哪一类」✓
   —— 比只看文件名准得多 ✓（``…/models/vae/xxx.safetensors`` 一定是 VAE ✓，同名文件在别处就未必 ✓）；
2. :func:`declared_extra_dirs`：读用户**已经声明**的目录 ✓（ComfyUI 的 ``extra_model_paths.yaml`` ✓、
   本仓 ``model-paths.json`` 的 ``extra_paths`` ✓、环境变量 ``EXTRA_MODEL_PATHS`` ✓）
   ⇒ 用户把模型放哪个盘哪个目录，**声明一下就能用** ✓（不用改代码 ✓）；
3. :func:`default_subdir` / :func:`category_extensions`：下载/安装时挑**类别子目录** ✓ 与校验扩展名 ✓
   （``model_manager.py --kind`` 的默认值就取这儿 ✓）。
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable, NamedTuple

__all__ = [
    "LEGACY_CATEGORY_ALIASES",
    "PT_EXTENSIONS",
    "FOLDER_SPECS",
    "category_dirs",
    "category_extensions",
    "category_of_path",
    "config_declared_dirs",
    "default_subdir",
    "declared_extra_dirs",
    "env_declared_dirs",
    "load_extra_paths_file",
    "normalize_category",
    "resolve_extra_paths",
]

#: ComfyUI 的 ``supported_pt_extensions``（口径一致 ✓ —— 想改要两边一起想 ✓）
PT_EXTENSIONS = frozenset({".ckpt", ".pt", ".pt2", ".bin", ".pth", ".safetensors", ".pkl", ".sft"})

#: 旧名 → 现名（ComfyUI ``folder_paths.map_legacy`` 的口径 ✓）：老工作流/老配置还这么叫 ✓
LEGACY_CATEGORY_ALIASES = {
    "unet": "diffusion_models",
    "clip": "text_encoders",
    "text_encoder": "text_encoders",
    "text_encoders": "text_encoders",
    "t2i_adapter": "controlnet",
    "lycoris": "loras",
    "lora": "loras",
    "checkpoint": "checkpoints",
    "embedding": "embeddings",
    "upscale": "upscale_models",
    "clip_vision_model": "clip_vision",
}


class FolderSpec(NamedTuple):
    """一个模型类别的口径：目录名 ✓ / 认的扩展名 ✓ / 我们扫描器的 ``kind`` 与 ``role`` ✓。"""

    dirs: tuple[str, ...]
    extensions: frozenset[str]
    kind: str
    role: str


def _spec(*dirs: str, exts: frozenset[str] = PT_EXTENSIONS, kind: str = "image",
          role: str = "standalone") -> FolderSpec:
    return FolderSpec(tuple(dirs), exts, kind, role)


#: 类别表（键 = ComfyUI 的 ``folder_names_and_paths`` 键 ✓，2026-09-26 逐条对齐 ✓）
#: ⚠️ ``kind`` / ``role`` 是**我们**的字段 ✓（供扫描结果与注册建议用 ✓），ComfyUI 那边没有 ✓。
FOLDER_SPECS: dict[str, FolderSpec] = {
    "checkpoints": _spec("checkpoints"),
    "loras": _spec("loras", "lycoris"),
    "vae": _spec("vae", role="component"),
    "text_encoders": _spec("text_encoders", "clip", role="component"),
    # ⚠️ 顺序 = ``default_subdir`` 的落点 ✓：新装的放**新名** ``diffusion_models`` ✓（``unet`` 只是老名字 ✓，
    #    两种目录都认 ✓ 但别把新模型塞进老目录 ✗）
    "diffusion_models": _spec("diffusion_models", "unet"),
    "clip_vision": _spec("clip_vision", role="component"),
    "style_models": _spec("style_models", role="component"),
    "embeddings": _spec("embeddings", role="component"),
    "diffusers": _spec("diffusers", exts=frozenset({"folder"})),
    "vae_approx": _spec("vae_approx", role="component"),
    "controlnet": _spec("controlnet", "t2i_adapter", role="component"),
    "gligen": _spec("gligen", role="component"),
    "upscale_models": _spec("upscale_models", role="component"),
    "latent_upscale_models": _spec("latent_upscale_models", role="component"),
    "hypernetworks": _spec("hypernetworks", role="component"),
    "photomaker": _spec("photomaker", role="component"),
    "model_patches": _spec("model_patches", role="component"),
    "audio_encoders": _spec("audio_encoders", kind="audio", role="component"),
    "background_removal": _spec("background_removal", role="component"),
    "frame_interpolation": _spec("frame_interpolation", kind="video", role="component"),
    "geometry_estimation": _spec("geometry_estimation", role="component"),
    "optical_flow": _spec("optical_flow", kind="video", role="component"),
    "detection": _spec("detection", role="component"),
    "classifiers": _spec("classifiers", exts=frozenset({""}), role="component"),
    "custom_nodes": _spec("custom_nodes", exts=frozenset()),   # 不是模型 ✓（只为口径完整 ✓）
    "datasets": _spec("datasets", exts=frozenset()),
    "configs": _spec("configs", exts=frozenset({".yaml"})),
}

#: 目录名 → 类别（**目录名优先** ✓；含别名 ✓）。按「长名优先」匹配 ✓（``text_encoders`` 先于 ``clip`` ✓）
_DIR_TO_CATEGORY: dict[str, str] = {}
for _category, _folder in FOLDER_SPECS.items():
    _DIR_TO_CATEGORY.setdefault(_category, _category)
    for _dir in _folder.dirs:
        _DIR_TO_CATEGORY.setdefault(_dir, _category)

#: 文件名兜底口径（父目录认不出时才用 ✓；⚠️ **只认很确定的** ✗ —— 瞎猜会把类别搅乱 ✗）
_NAME_HINTS: tuple[tuple[str, str], ...] = (
    (r"controlnet|t2i[-_]?adapter|(^|[^a-z])canny([^a-z]|$)|depth_anything|openpose|dwpose", "controlnet"),
    (r"esrgan|swinir|hat[-_]?gan|realesr|upscaler|4x[-_]|2x[-_]", "upscale_models"),
    (r"(^|[^a-z])vae([^a-z]|$)|vae[-_]?ft|sdxl[-_]?vae|taesd|kl[-_]?f8", "vae"),
    (r"t5xxl|t5[-_]?encoder|clip[-_]?l|clip[-_]?g|text[-_]?encoder|umt5|byt5", "text_encoders"),
    (r"clip[-_]?vision|siglip|vision[-_]?encoder", "clip_vision"),
    (r"lora|lycoris|locon|loha", "loras"),   # ⚠️ 用子串不用边界 ✓：`MyLora_v2.safetensors` 也要认 ✓
    (r"embedding|textual[-_]?inversion", "embeddings"),
    (r"ip[-_]?adapter|ipadapter", "model_patches"),
    (r"insightface|buffalo|faceid", "detection"),
    (r"sdxl|sd[-_]?1\.?5|sd15|flux|z[-_]?image|dreamshaper|pony|illustrious", "checkpoints"),
)


def normalize_category(name: str) -> str:
    """类别名归一（去空白/小写/查别名 ✓）—— 认不出来**原样返回小写** ✓（不瞎改 ✓）。"""
    text = str(name or "").strip().lower().replace("-", "_")
    return LEGACY_CATEGORY_ALIASES.get(text, text)


def category_dirs(category: str) -> tuple[str, ...]:
    """该类别的目录名（ComfyUI 口径 ✓；未知类别 ⇒ 空元组 ✓）。"""
    spec = FOLDER_SPECS.get(normalize_category(category))
    return spec.dirs if spec else ()


def category_extensions(category: str) -> frozenset[str]:
    """该类别认的扩展名（未知类别 ⇒ ComfyUI 的 ``supported_pt_extensions`` ✓ —— 别把人挡在门外 ✗）。"""
    spec = FOLDER_SPECS.get(normalize_category(category))
    return spec.extensions if spec else PT_EXTENSIONS


def default_subdir(category: str) -> str:
    """该类别在 ``models/`` 下的**默认子目录**（下载/安装落点 ✓；未知类别 ⇒ 原样名字 ✓）。"""
    normalize = normalize_category(category)
    dirs = category_dirs(normalize)
    return dirs[0] if dirs else normalize


def category_of_path(path: str) -> str | None:
    """猜「这个文件/目录属于哪个 ComfyUI 类别」：**父目录名优先** ✓，认不出再按**文件名**兜底 ✓。

    ⚠️ 返回 ``None`` = **认不出** ✓（不硬塞一个类别 ✗ —— 认错比认不出更糟 ✗）。
    """
    text = str(path or "").replace("\\", "/").rstrip("/")
    if not text:
        return None
    parts = [segment for segment in text.split("/") if segment]
    if len(parts) >= 2:
        parent = normalize_category(parts[-2])
        if parent in _DIR_TO_CATEGORY:
            return _DIR_TO_CATEGORY[parent]
    leaf = normalize_category(parts[-1] if parts else "")
    if leaf in _DIR_TO_CATEGORY:
        return _DIR_TO_CATEGORY[leaf]
    from re import search as _search  # 局部 import：模块头保持轻 ✓

    name = (parts[-1] if parts else "").lower()
    for pattern, category in _NAME_HINTS:
        if _search(pattern, name):
            return category
    return None


def _split_values(value: Any) -> list[str]:
    """声明里的值 ⇒ 路径串列表 ✓（**多行字符串**按行拆 ✓，也收列表/单串 ✓ —— 与 ComfyUI 一致 ✓）。"""
    if isinstance(value, (list, tuple)):
        items: list[str] = []
        for item in value:
            items.extend(_split_values(item))
        return items
    if not isinstance(value, str):
        return []
    out: list[str] = []
    for line in value.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        text = line.strip()
        if text:
            out.append(text)
    return out


def _resolve_one(raw: str, base_dir: str) -> str:
    """一个声明路径 ⇒ 绝对路径 ✓（展开 ``$VAR`` / ``~`` ✓；相对 ⇒ 相对 ``base_dir`` ✓，同 ComfyUI ✓）。"""
    text = os.path.expandvars(os.path.expanduser(str(raw or "").strip()))
    if not text:
        return ""
    if not os.path.isabs(text):
        text = os.path.join(base_dir or os.getcwd(), text)
    return os.path.normpath(os.path.abspath(text))


def resolve_extra_paths(config: dict[str, Any], base_dir: str = "") -> dict[str, list[str]]:
    """解析「额外模型路径」声明的**语义** ✓ ⇒ ``{类别: [绝对目录…]}``（2026-09-26 对齐 ComfyUI ✓）。

    格式与 ComfyUI 的 ``extra_model_paths.yaml`` **一致** ✓（用户已有的配置**直接能用** ✓）::

        comfyui:                    # 段的键 = 名字（我这边不用 ✓，只为兼容 ✓）
            base_path: D:/ai/models # 该段的相对路径基准 ✓（可省 ✓）
            is_default: true        # 只作标记 ✓（扫描**不分**默认与否 ✓ —— 都要能看见 ✓）
            checkpoints: |
                checkpoints
                D:/另一个盘/ckpt
            loras: loras

    ⚠️ ``base_path`` 展开 ``$VAR`` / ``~`` ✓、相对 ⇒ 相对**配置文件所在目录** ✓（和 ComfyUI 一样 ✓）。
    """
    resolved: dict[str, list[str]] = {}
    if not isinstance(config, dict):
        return resolved
    for _, section in config.items():
        if not isinstance(section, dict):
            continue
        raw_base = section.get("base_path")
        base_path = ""
        if isinstance(raw_base, str) and raw_base.strip():
            base_path = _resolve_one(raw_base, base_dir)
        for key, value in section.items():
            if key in ("base_path", "is_default"):
                continue
            category = normalize_category(key)
            for raw in _split_values(value):
                full = _resolve_one(raw, base_path or base_dir) if base_path else _resolve_one(raw, base_dir)
                if full:
                    resolved.setdefault(category, [])
                    if full not in resolved[category]:
                        resolved[category].append(full)
    return resolved


def load_extra_paths_file(path: str) -> dict[str, list[str]]:
    """读声明文件（``*.yaml`` / ``*.yml`` / ``*.json`` ✓）⇒ :func:`resolve_extra_paths` 的结果 ✓。

    ⚠️ 读不开/格式不对 ⇒ **抛 ValueError** ✓（不静默跳过 ✗ —— 「声明了却没生效」必须能被看见 ✓）。
    ⚠️ yaml 需要 ``PyYAML``（本仓 ``requirements.txt`` 已有 ✓）；实在没有 ⇒ 文案里说清「改用 JSON」✓。
    """
    target = str(path or "").strip()
    if not target or not os.path.isfile(target):
        raise ValueError(f"声明文件不存在：{target or '(未指定)'}")
    base_dir = os.path.dirname(os.path.abspath(target))
    raw = open(target, encoding="utf-8").read()
    if target.lower().endswith(".json"):
        try:
            parsed = json.loads(raw)
        except ValueError as error:
            raise ValueError(f"JSON 解析失败：{error}") from error
    else:
        try:
            import yaml  # noqa: PLC0415
        except ImportError as error:  # pragma: no cover - 环境缺 PyYAML 才会走到
            raise ValueError(
                "读 yaml 声明需要 PyYAML（`pip install PyYAML`）；也可以改用等价的 .json 声明"
            ) from error
        try:
            parsed = yaml.safe_load(raw)
        except Exception as error:  # noqa: BLE001 - yaml 的异常类型随版本变 ✓ 一律换成 ValueError ✓
            raise ValueError(f"YAML 解析失败：{error}") from error
    return resolve_extra_paths(parsed if isinstance(parsed, dict) else {}, base_dir)


def env_declared_dirs(environ: dict[str, str] | None = None) -> list[str]:
    """环境变量声明的目录/声明文件 ✓：``EXTRA_MODEL_PATHS`` / ``COMFYUI_EXTRA_MODEL_PATHS`` ✓。

    值用 ``;``（Windows 的 ``os.pathsep`` ✓）或 ``,`` 分隔 ✓；每一项可以是**目录** ✓ 或**声明文件** ✓
    （``.yaml/.yml/.json`` ⇒ 走 :func:`load_extra_paths_file` ✓）。⚠️ 文件读不开 ⇒ 如实抛 ✓ 不静默 ✗。
    """
    env = os.environ if environ is None else environ
    out: list[str] = []
    for key in ("EXTRA_MODEL_PATHS", "COMFYUI_EXTRA_MODEL_PATHS"):
        raw = str(env.get(key) or "")
        for item in raw.replace(",", os.pathsep).split(os.pathsep):
            text = item.strip().strip('"')
            if not text:
                continue
            text = _resolve_one(text, "")
            if os.path.isfile(text):
                for dirs in load_extra_paths_file(text).values():
                    out.extend(dirs)
            else:
                out.append(text)
    return out


def config_declared_dirs(cfg: dict[str, Any], base_dir: str = "") -> list[str]:
    """本仓 ``model-paths.json`` 里的 ``extra_paths`` 段 ✓（**与 ComfyUI 同一种格式** ✓）。

    两种写法都收 ✓：① ``extra_paths`` 直接就是 ``{类别: 路径…}`` 的**一段** ✓
    （等价于 yaml 里只有一段 ✓）；② ``extra_paths`` 是 ``{段名: 段}`` ✓（完全对齐 yaml ✓）。
    """
    if not isinstance(cfg, dict):
        return []
    section = cfg.get("extra_paths")
    if not isinstance(section, dict):
        return []
    looks_like_sections = any(isinstance(value, dict) for value in section.values())
    resolved = resolve_extra_paths({"_": section} if not looks_like_sections else section, base_dir)
    out: list[str] = []
    for dirs in resolved.values():
        out.extend(dirs)
    return out


def comfyui_declared_dirs(comfyui_roots: Iterable[str]) -> list[str]:
    """各 ComfyUI 根下的 ``extra_model_paths.yaml`` ✓ —— **用户已有的声明直接接管** ✓。

    ⚠️ 解析失败**不抛** ✓（那是别人的配置文件 ✗ 不该让我们的扫描整体挂掉 ✗）：如实跳过 ✓，
    把原因记进返回值之外的日志由调用方决定（这里只保证「能读就读 ✓」）。
    """
    out: list[str] = []
    for root in comfyui_roots or []:
        for name in ("extra_model_paths.yaml", "extra_model_paths.yml"):
            candidate = os.path.join(str(root or ""), name)
            if not os.path.isfile(candidate):
                continue
            try:
                for dirs in load_extra_paths_file(candidate).values():
                    out.extend(dirs)
            except ValueError:
                continue
    return out


def declared_extra_dirs(cfg: dict[str, Any] | None = None, base_dir: str = "",
                        comfyui_roots: Iterable[str] = ()) -> list[str]:
    """**所有**「用户声明过的任意目录」的合集 ✓（去重、保序 ✓）—— 接进扫描根里就能直接用 ✓。

    顺序 = 环境变量 ✓ → ``model-paths.json`` ✓ → 各 ComfyUI 的 ``extra_model_paths.yaml`` ✓
    （⚠️ **显式声明优先于探测** ✓ —— 用户说的算 ✓）。
    """
    out: list[str] = []
    seen: set[str] = set()
    sources = list(env_declared_dirs())
    sources.extend(config_declared_dirs(cfg or {}, base_dir))
    sources.extend(comfyui_declared_dirs(comfyui_roots))
    for item in sources:
        text = str(item or "").strip()
        if not text:
            continue
        key = os.path.normcase(os.path.abspath(text))
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out
