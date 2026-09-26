"""本地模型扫描服务 —— 与 ``services/local-model-scan.ts``（709 行）对齐。

让 Drama Studio 能「**发现**」电脑上任意位置的模型，而不再依赖 ``configs/models.json``
静态清单。识别结果供注册到 ``ai_service_configs`` 后由现有 adapter（provider + baseUrl
切换）调用。

**识别策略**：纯启发式（目录名 + 文件名 + 扩展名），零依赖，**按优先级命中**。规则顺序
与 ``configs/models.json`` 的 category/runtime/kind 语义对齐。

⚠️ 七处保真点（都会让「扫不到 / 识错类」而**不报错**）：

1. **规则顺序就是一切**（先命中先返回）：H3 的 text_encoders（qwen3vl）属**视频**、必须排在
   文本 LLM 之前；Wan/Hunyuan 的 GGUF 必须排在通用 GGUF LLM 之前；组件目录
   （vae/text_encoders/clip…）必须排在「目录=LLM」与「扩散主权重」之前；
2. **不跳过任何模型**：VAE/CLIP/ControlNet/Upscale/Embeddings 都是「组件」，
   照样扫出来，只是 ``role='component'``；``SKIP_DIRS`` 只跳过**非模型**目录；
3. **系统目录要跳过**（``SYSTEM_DIRS``）：磁盘/全盘扫描时不进 Windows/Program Files/AppData，
   否则大量 EACCES + 杀软拦截 + OneDrive 占位文件；
4. **ComfyUI 只作「只读扫描来源」**：``baseUrl`` 为空、``callable=false``、不注册为可调用后端；
5. ``comfyui_root`` 的判定顺序是 **环境变量 > model-paths.json > 动态探测** ✓（探测 = 家目录 + **各盘符**
   往下找「名字含关键词且带 ``models/``」的目录 ✓ ⇒ 多装几个 ComfyUI **全部可见** ✓；⚠️ 旧的写死候选表
   且**只取第一个** ⇒ 后面的**被遮住** ✗✗，2026-09-26 修的正是这条 ✓）；
6. **同步版与异步版上限不同**：同步 ``maxDepth=5 / maxFiles=8000``，异步 ``8 / 50000``，
   且异步版**每 1000 个文件**报一次进度、随时可取消；
7. **排序**：``video > image > text > audio > unknown``，同类按**体积降序**（体积更能代表重要性）；
8. ⭐ **要覆盖「世界各大模型」**（2026-09-25 用户口径 ✓）：默认根目录含 HuggingFace / ModelScope /
   LM Studio / GPT4All / Jan / llama.cpp / text-generation-webui / SD WebUI / Stability Matrix /
   InvokeAI / torch hub 的本地落点 ✓（表在 ``services/model_ecosystems.py`` ✓），**全程只读、零外部依赖** ✓
   —— **不许**要求 ``ollama serve`` / LM Studio 等**服务在跑**才能扫到 ✗✗（那正是「盘上有模型却
   看不见」的老 bug ✓）；Ollama 的权重是无扩展名 blob ⇒ 文件遍历看不见 ✗，改由 **manifests 清单**读出 ✓
   （见 ``_merge_ollama`` ✓）。
9. ⭐ **不许写死机器路径** ✗（2026-09-26 用户口径 ✓）：模型机可以装到**用户指定目录** ✓（``models_dir`` ✓）、
   也要能扫**电脑任意目录** ✓（``extra_roots`` ✓ + 盘符枚举 ``list_drives`` ✓，整盘扫也行 ✓）——
   权威来源只有**配置 / 环境变量 / 盘符枚举** ✓，源码里**不出现** ``D:/…`` 这类字面量 ✗
   （静态守卫在同一套自检里 ✓：``local_models_test`` ✓）。
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..core import config
from ..core.config import PROJECT_ROOT
from . import model_ecosystems, model_folders, ollama_store

#: 后端包根（``backend-py/``）。**「本地服务根」的默认值落在这里**：
#: 2026-09-15 从仓库根 ``local_services/`` 迁入 —— 它是「``model_manager.py`` 会 ``git clone``
#: 的独立服务 + 项目自带的 ``h3`` 薄封装」的落脚点，属后端资产，与后端代码同包更归拢。
#: ⚠️ **唯一权威在 ``app/core/config.py``**（2026-09-15 收口：本文件原先自己算一遍 ``parents[2]`` ✗）；
#: ``scripts/model_manager.py`` 那份是同名独立实现（scripts 不 import ``app.*``），靠注释同步。
#: ⚠️ 与 ``config.BACKEND_PY_ROOT`` 区分：**资产根是 `app/`**（三类资产住在这里）⇒ 用 `APP_ROOT`。
LOCAL_SERVICES_ROOT = config.APP_ROOT / "local_services"

__all__ = [
    "COMFYUI_BLANKET_DEPTH",
    "COMFYUI_DIR_HINTS",
    "COMFYUI_PROBE_DEPTH",
    "ECOSYSTEM_SKIP_PREFIXES",
    "MODEL_EXTS",
    "SKIP_DIRS",
    "SYSTEM_DIRS",
    "ScanCancelledError",
    "build_suggestion",
    "classify",
    "default_models_dir",
    "detect_comfyui",
    "detect_comfyui_roots",
    "get_default_roots",
    "get_extra_roots",
    "get_model_paths",
    "human_size",
    "list_drives",
    "save_extra_roots",
    "save_model_paths",
    "scan_local_models",
    "scan_local_models_async",
]

#: 模型权重扩展名（大小写不敏感）
MODEL_EXTS = frozenset([
    ".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx", ".sft", ".engine",
])

#: 递归时跳过的目录名 —— **仅跳过「非模型」目录**。
#: ⚠️ clip / vae / controlnet / upscale_models / embeddings / photomaker / style_models /
#: ipadapter 等目录下**都是模型权重**，不能跳过，需被扫描识别。
SKIP_DIRS = frozenset([
    "node_modules", ".git", "venv", ".venv", "__pycache__", ".cache", ".huggingface",
    "output", "temp", "input", "assets", "metadata", "samples", "custom_nodes",
    ".codebuddy", "python", "lib", "site-packages",
])

#: 系统级目录（磁盘/全盘扫描时跳过）。这里用**目录名小写**匹配，模型几乎不可能放在这些目录下。
SYSTEM_DIRS = frozenset([
    "windows", "program files", "program files (x86)", "programdata",
    "$recycle.bin", "system volume information", "recovery", "perflogs",
    "msocache", "documents and settings", "windows.old", "appdata",
    "onedrive", "onedrivetemp", "intel", "amd", "nvidia", "drivers",
])

#: ⭐ 2026-09-26 用户口径 ✓：**不许写死路径** ✗ —— 模型既可以装在**用户指定的任意目录** ✓，
#: 也要能扫**电脑上任意目录**里已有的模型 ✓。所以这里**没有** ``D:/Comfy-Desktop/…`` 这类机器字面量 ✗，
#: 只有「**按名字动态探测 + 全盘枚举**」的口径 ✓：
#:
#: * **权威来源**（优先于探测）：环境变量 ``COMFYUI_PATH`` / ``model-paths.json`` 的 ``comfyui_root`` ✓、
#:   用户的「额外扫描目录」``extra_roots`` ✓（**任意目录**都行 ✓，见 ``get_default_roots`` ✓）；
#:   下载/存放的**主目录** = ``MODELS_DIR`` / ``model-paths.json`` 的 ``models_dir`` ✓（默认 ``<data_root>/models`` ✓）；
#: * **探测回退** = 家目录 + **各现存盘符**（``list_drives`` ✓）往下找「名字含关键词」的目录 ✓，
#:   判定「像 ComfyUI 根」只看**目录下有没有 ``models/``** ✓（与旧口径一致 ✓ ⇒ 桌面端的**共享模型目录**
#:   ``…/ComfyUI-Shared`` ✓ 也算一个根 ✓，本机那份 SDXL 正在它下面 ✓）；
#: * ⚠️ 关键词是**产品名**不是机器路径 ✓：用户把 ComfyUI 装在 ``D:\AI\MyComfy`` 这类名字里照样命中 ✓
#:   （名字毫不相干的目录 = 不瞎猜 ✗，靠用户显式配 ``extra_roots`` / ``comfyui_root`` ✓）。
COMFYUI_DIR_HINTS = ("comfyui", "comfy-desktop", "comfyui-desktop", "comfy")
#: 探测深度（层）：**关键词目录**最深看到这里 ✓。
#: ⚠️ 桌面端把安装藏在 ``<盘>/Comfy-Desktop/ComfyUI-Installs/<名字>/ComfyUI`` ✓ ⇒ **5 层**才够 ✓。
COMFYUI_PROBE_DEPTH = 5
#: 「顺便看一眼」的深度（层）：名字**不沾边**的目录只在最上面这几层进一下 ✓ ——
#: 因为开发/自建目录常挂在 ``<盘>/code/ComfyUI`` 这类**父目录名不含关键词**的地方 ✗
#: （只认关键词会漏掉它 ✗）；再深就**不乱翻** ✗（省时的同时避开无关目录树 ✓）。
COMFYUI_BLANKET_DEPTH = 2

#: 运行时 → 默认 baseUrl。
#: ⚠️ ``comfyui`` **有意为空**：它只作「只读的可选扫描来源」，不生成可调用注册建议。
RUNTIME_BASE_URLS: dict[str, str] = {
    "comfyui": "",
    "ollama": "http://localhost:11434",
    "local-sd": "http://localhost:7860",
    "h3": "http://localhost:8765",
    "cosyvoice": "http://localhost:9880",
    "unknown": "",
}

#: 目录名**前缀**跳过（生态专用，见 ``model_ecosystems``）✓：
#: HuggingFace 的 ``datasets--*`` 是**数据集**不是模型 ✗、``.no_exist`` 是「查过、不存在」的
#: 负缓存标记 ✗（里面是 0 字节的同名占位文件 ⇒ 不跳会扫出**幽灵模型** ✗✗）。
ECOSYSTEM_SKIP_PREFIXES = model_ecosystems.skip_prefixes()

#: 类别排序权重（``video`` 最先 —— 体积大、最能代表机器上的模型资产）
_KIND_ORDER = {"video": 0, "image": 1, "text": 2, "audio": 3, "unknown": 4}


class ScanCancelledError(Exception):
    """扫描被取消。⚠️ 文案固定 ``scan cancelled``（路由按类型判定，不按文案）。"""

    def __init__(self) -> None:
        super().__init__("scan cancelled")
        self.name = "ScanCancelledError"


# ============================================================
# 识别规则引擎
# ============================================================
#: 每条规则：``(id, kind, runtime, confidence, role, test)``。
#: ``test(p, dir, name, ext)`` 的四个入参都已**小写**（p = 全路径）。
#: ⚠️ **顺序即优先级**（先命中先返回），改动顺序前先读模块头的第 1 条。
_RULES: list[tuple[str, str, str, str, str, Callable[[str, str, str, str], bool]]] = [
    # ===== 视频 =====
    # 1. H3 视频 DiT（minimax_h3 系列，可独立驱动生成）
    #    注意排除 qwen 开头的文本编码器（``qwen3vl_32b_minimax_h3`` 是 TE 而非 DiT）
    ("h3-dit", "video", "h3", "high", "standalone",
     lambda _p, _d, name, _e: bool(re.search(r"minimax.?h3", name)) and not re.search(r"vae", name)
     and not re.search(r"qwen", name) and not re.search(r"te\d*$", name)),
    # 2. H3 视频 VAE（组件）
    ("h3-vae", "video", "h3", "high", "component",
     lambda _p, _d, name, _e: bool(re.search(r"minimax.?h3", name)) and bool(re.search(r"vae", name))),
    # 3. H3 文本编码器（qwen3vl 32b / qwen_3_4b TE，组件）
    ("h3-text-encoder", "video", "h3", "high", "component",
     lambda _p, d, name, _e: (d == "text_encoders" and bool(re.search(r"qwen|minimax", name)))
     or bool(re.search(r"qwen3vl", name))),
    # 4. Wan / Hunyuan 视频 GGUF（可灵类，可独立驱动）
    ("wan-video-gguf", "video", "comfyui", "high", "standalone",
     lambda _p, _d, name, ext: ext == ".gguf" and bool(re.search(r"wan|hunyuan|kling", name))),
    # 5. Wan / Hunyuan 视频 safetensors 主权重
    ("wan-video-safetensors", "video", "comfyui", "high", "standalone",
     lambda _p, d, name, _e: bool(re.search(r"wan|hunyuan|kling", name))
     and bool(re.search(r"diffusion_models|unet", d))),

    # ===== 音频 =====
    # 6. 语音合成（CosyVoice / GPT-SoVITS / VITS 等）
    # ⚠️ 整条路径**只认「以关键词开头的路径段」** ✗ —— 不再无条件拼进正则 ✗✗：
    #    `…\ecosystems_tmp\…\flux1-dev.safetensors` 的 "**ecosys**" 含 "cosy" ⇒
    #    原来会把**图像**权重判成 TTS 模型 ✗，并给它配一个 CosyVoice:9880 的**错建议** ✗✗
    #    （用户目录里叫 `ecosystems` / `my-tts-*` 都很常见 ⇒ 这不是理论风险 ✓）。
    ("tts-service", "audio", "cosyvoice", "high", "standalone",
     lambda p, d, name, _e: bool(re.search(
         r"cosyvoice|cosy|tts|speech|vits|gpt.?sovits|bert.?vits|fish.?speech|chat.?tts|xtts",
         f"{d} {name}"))
     or bool(re.search(r"(?:^|[\\/])(?:cosy|tts|speech|vits|xtts|sovits)", p))),

    # ===== 文本 =====
    # 7. 独立 LLM GGUF（量化文本模型，可被 Ollama 导入）
    ("gguf-llm", "text", "ollama", "high", "standalone",
     lambda _p, _d, name, ext: ext == ".gguf"
     and not re.search(r"wan|hunyuan|kling|flux", name)),
    # 8. 目录「LLM」或知名 LLM 名（ComfyUI LLM_party 挂载）
    ("llm-dir", "text", "comfyui", "medium", "standalone",
     lambda p, d, name, _e: bool(re.search(r"/llm/", p)) or d == "llm"
     or bool(re.search(
         r"llm|qwen3-?4b|qwen\d+\.?\d*|llama|mistral|phi-?|deepseek|gemma|glm|baichuan|"
         r"chatglm|yi-?|internlm", f"{d} {name}"))),

    # ===== 图像：主权重 =====
    # 9. diffusion 主权重（checkpoints / diffusion_models / unet / loras）
    ("diffusion-weights", "image", "comfyui", "high", "standalone",
     lambda _p, d, name, _e: bool(re.search(r"checkpoint|diffusion_models|unet|loras|dora", d))
     and not re.search(r"minimax.?h3", name)),

    # ===== 图像：组件 =====
    # 10. ComfyUI 图像生成组件（VAE/CLIP/ControlNet/Upscale/Embeddings…）
    ("image-component", "image", "comfyui", "high", "component",
     lambda _p, d, _n, _e: bool(re.search(
         r"vae|clip|clip_vision|text_encoders|controlnet|upscale_models|embeddings|"
         r"photomaker|insightface|ipadapter|style_models|blip|onnx|flux|t5|hypernetworks", d))),

    # 11. 知名图像模型文件名
    ("image-by-name", "image", "comfyui", "medium", "standalone",
     lambda _p, d, name, _e: bool(re.search(
         r"flux|sd.?xl|sdxl|sd15|sd_?1\.?5|z-?image|dreamshaper|realistic.?vision|juggernaut|"
         r"animagine|pony|illustrious|noobai|majicmix|chilloutmix", f"{d} {name}"))),

    # ===== 兜底 =====
    # 12. 其余 safetensors/ckpt/pt/pth/bin 默认判为图像权重
    ("default-weights", "image", "comfyui", "low", "standalone",
     lambda _p, _d, _n, ext: ext in (".safetensors", ".ckpt", ".pt", ".pth", ".bin",
                                     ".onnx", ".sft", ".engine")),
]


def classify(path: str) -> dict[str, Any]:
    """按规则命中（返回 ``kind/runtime/confidence/role/matchedBy``）。**先命中先返回**。"""
    directory = os.path.basename(os.path.dirname(path)).lower()
    name = os.path.basename(path).lower()
    ext = os.path.splitext(name)[1]
    full = path.lower()
    for rule_id, kind, runtime, confidence, role, test in _RULES:
        if test(full, directory, name, ext):
            return {"kind": kind, "runtime": runtime, "confidence": confidence,
                    "role": role, "matchedBy": [rule_id]}
    return {"kind": "unknown", "runtime": "unknown", "confidence": "low",
            "role": "standalone", "matchedBy": []}


def build_suggestion(hit: dict[str, Any], name: str, ext: str) -> dict[str, Any] | None:
    """由命中结果生成「注册建议」（供阶段 2 写进 ``ai_service_configs``）。"""
    model = re.sub(r"\.[^.]+$", "", name)
    role = hit["role"]
    kind = hit["kind"]
    runtime = hit["runtime"]

    # ⚠️ ComfyUI 至多作为「只读的可选扫描来源」：只探测/展示模型列表，
    #    不生成可调用注册建议（不写入 ai_service_configs，也不作为推理后端）
    if runtime == "comfyui":
        return {
            "serviceType": "image" if kind == "unknown" else kind,
            "provider": "", "baseUrl": "", "model": model,
            "runtime": "comfyui", "role": role, "callable": False,
            "note": "ComfyUI 仅作只读扫描来源，不直接调用；"
                    "如需使用请部署为项目自有服务（H3/local-sd/Ollama/CosyVoice）",
        }

    if kind == "text":
        # 此处只可能是 ollama（comfyui 已在上方短路）
        return {
            "serviceType": "text", "provider": "openai",
            "baseUrl": RUNTIME_BASE_URLS["ollama"],
            "model": model if ext == ".gguf" else name,
            "runtime": "ollama", "role": role, "callable": True,
            "note": "需先 `ollama create <name> -f Modelfile` 导入该 GGUF",
        }
    if kind == "image":
        # 当前 image 规则均为 comfyui（已短路）；将来命中 local-sd 规则时走到这里
        return {
            "serviceType": "image", "provider": "local-sd",
            "baseUrl": RUNTIME_BASE_URLS["local-sd"], "model": model,
            "runtime": "local-sd", "role": role, "callable": True,
            "note": "本地 SD WebUI 服务，可直接调用",
        }
    if kind == "video":
        return {
            "serviceType": "video", "provider": "minimax",
            "baseUrl": RUNTIME_BASE_URLS["h3"], "model": "hailuo-02",
            "runtime": "h3", "role": role, "callable": True,
            "note": ("H3 组件（VAE/文本编码器），由 H3 服务 checkpoint_map 引用，不单独注册"
                     if role == "component" else "本地 H3 服务，通过 checkpoint_map 路由到该权重"),
        }
    if kind == "audio":
        return {
            "serviceType": "audio", "provider": "cosyvoice",
            "baseUrl": RUNTIME_BASE_URLS["cosyvoice"], "model": model,
            "runtime": "cosyvoice", "role": role, "callable": True,
            "note": "CosyVoice 本地服务",
        }
    return None


# ============================================================
# 路径解析（环境变量 > configs/model-paths.json > 探测）
# ============================================================

def _paths_config_file() -> str:
    return str(PROJECT_ROOT / "configs" / "model-paths.json")


def _read_paths_config() -> dict[str, Any]:
    try:
        path = _paths_config_file()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                parsed = json.load(handle)
            return parsed if isinstance(parsed, dict) else {}
    except (OSError, ValueError):
        pass
    return {}


def _write_paths_config(cfg: dict[str, Any]) -> None:
    """⚠️ 缩进 2 空格 + 末尾换行（与原 TS 的 ``JSON.stringify(cfg, null, 2) + '\\n'`` 一致）。"""
    os.makedirs(os.path.dirname(_paths_config_file()), exist_ok=True)
    with open(_paths_config_file(), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")


def _probe_bases() -> list[str]:
    """探测起点：家目录 + 各现存盘根（``list_drives`` ✓ —— POSIX 上取不到盘符 ⇒ 只剩家目录 ✓）。"""
    bases: list[str] = []
    home = os.path.expanduser("~")
    if home and os.path.isdir(home):
        bases.append(home)
    for drive in list_drives():
        root = str(drive.get("root") or "")
        if root and os.path.isdir(root):
            bases.append(root)
    return bases


def detect_comfyui_roots(bases: list[str] | None = None) -> list[str]:
    """**动态**探测本机全部「ComfyUI 类」目录（保序：起点顺序 + 浅→深 ✓）。

    ⚠️ 与旧 ``detect_comfyui()`` 的关键差别：这里返回**全部**命中 ✓ —— 旧实现只取**第一个** ⇒
    机器上装了两个以上 ComfyUI 时，**后面的全被遮住** ✗✗（本机就是这么漏掉那份 SDXL 的 ✓：
    候选表把 ``ComfyUI-Installs/…`` 排在前 ⇒ ``ComfyUI-Shared/models`` 永远扫不到 ✗）。

    做法：从起点往下走，**名字含关键词**（``COMFYUI_DIR_HINTS``）的目录一路下潜
    （最多 ``COMFYUI_PROBE_DEPTH`` 层 ✓），名字不沾边的只在最上面 ``COMFYUI_BLANKET_DEPTH`` 层
    「顺便看一眼」✓ —— 因为自建/开发目录常挂在 ``<盘>/code/ComfyUI`` 这种**父目录名不含关键词**的地方 ✗；
    命中判定 = **名字含关键词 + 目录下有 ``models/``** ✓（与旧口径一致 ✓）。``SKIP_DIRS`` / ``SYSTEM_DIRS``
    照旧跳过 ✓（⇒ 再怎么「顺便看」也不会进 ``Windows`` / ``AppData`` ✓）。

    ``bases=None`` ⇒ 家目录 + 各现存盘根 ✓；测试**注入临时目录** ✓（不碰真实磁盘 ✓）。
    """
    found: list[str] = []
    seen: set[str] = set()
    queue: list[tuple[str, int]] = [
        (base, 0) for base in (bases if bases is not None else _probe_bases())
    ]
    while queue:
        current, depth = queue.pop(0)
        try:
            with os.scandir(current) as scanned:
                children: list[tuple[str, str, bool]] = []
                for entry in sorted(scanned, key=lambda item: item.name.lower()):
                    try:
                        children.append((entry.name.lower(), os.path.abspath(entry.path),
                                         entry.is_dir(follow_symlinks=False)))
                    except OSError:
                        continue
        except OSError:
            continue
        for name, path, is_dir in children:
            if not is_dir or name.startswith(".") or name in SKIP_DIRS or name in SYSTEM_DIRS:
                continue
            hinted = any(hint in name for hint in COMFYUI_DIR_HINTS)
            # ⚠️ 名字不沾边的**只在最上面几层**「顺便看一眼」✓（``<盘>/code/ComfyUI`` 这种 ✗）；
            #    再往下只跟着关键词走 ✗ —— 既不乱翻无关目录树 ✓、对普通名字也**不瞎猜**是根 ✓
            #    （用户的任意目录另有显式入口：``extra_roots`` / ``comfyui_root`` ✓）
            if not hinted and depth + 1 >= COMFYUI_BLANKET_DEPTH:
                continue
            key = os.path.normcase(path)
            if key in seen:
                continue
            seen.add(key)
            if hinted and os.path.isdir(os.path.join(path, "models")):
                found.append(path)
            if depth + 1 < COMFYUI_PROBE_DEPTH:
                queue.append((path, depth + 1))
    return found


def detect_comfyui() -> str:
    """探测**第一个** ComfyUI 安装位置（保留给「单值要用」的调用方 ✓）。

    ⚠️ **扫描一律走** :func:`detect_comfyui_roots` ✓（只取第一个 = 漏 ✗，2026-09-26 修的正是这条 ✓）。
    """
    roots = detect_comfyui_roots()
    return roots[0] if roots else ""


def default_models_dir() -> str:
    """本仓**自己的**模型存储目录 ✓：``<data_root>/models`` ✓。

    2026-09-25 自主化 ✓：模型不再依赖 ComfyUI 目录 ✗ —— 下载/扫描/就绪/加载统一落到这里 ✓
    （``MODELS_DIR`` 环境变量或 ``model-paths.json`` 的 ``models_dir`` 仍可覆盖 ✓）。
    """
    return str(Path(config.get_data_root()) / "models")


def get_extra_roots() -> list[str]:
    """用户自定义的额外扫描目录（``configs/model-paths.json`` 的 ``extra_roots``）。"""
    extra = _read_paths_config().get("extra_roots")
    if isinstance(extra, list):
        return [x for x in extra if isinstance(x, str) and x.strip()]
    return []


def get_default_roots() -> list[str]:
    """解析「默认扫描根目录」集合：环境变量 > model-paths.json > 动态探测 > 默认。

    ⭐ 2026-09-25 语义修正 ✓：**下载/存储**默认到本仓 <data_root>/models（自主 ✓），
    但**扫描发现**要尽量覆盖电脑里已有的模型 ✓ —— 含 ComfyUI 目录 ✗（**发现不排斥第三方** ✓：
    自主的是「从哪下载、存到哪」，不是「不许看见别人已装的模型」✓）。

    ⭐ 2026-09-26 用户口径 ✓：**不许写死路径** ✗、**任意目录**都要能装能用 ✓：
    * ``models_dir``（``MODELS_DIR`` / ``models_dir`` ✓）= 用户指定的**存放/下载主目录** ✓；
    * ``extra_roots``（用户显式加的**任意目录** ✓）**排在约定落点之前** ✓（显式意图优先 ✓）；
    * ComfyUI 类目录 = **显式配的** ``comfyui_root`` ✓ + :func:`detect_comfyui_roots` 的**全部**命中 ✓
      —— ⚠️ 修的就是「只取第一个 ⇒ 后面几个被遮住」✗✗（本机因此扫不到 ``ComfyUI-Shared/models`` ✓）。
    """
    cfg = _read_paths_config()
    roots: list[str] = []

    models_dir = (os.environ.get("MODELS_DIR") or cfg.get("models_dir")
                  or default_models_dir())
    services_dir = (os.environ.get("LOCAL_SERVICES_DIR") or cfg.get("local_services_dir")
                    or str(LOCAL_SERVICES_ROOT))
    # ⚠️ 显式配置（env / json）**优先且靠前** ✓；探测结果只是**补充** ✓（探测错了也不影响显式配置 ✓）
    comfyui_root = os.environ.get("COMFYUI_PATH") or cfg.get("comfyui_root") or ""
    comfyui_roots: list[str] = []
    for candidate in [comfyui_root, *detect_comfyui_roots()]:
        text = os.path.abspath(str(candidate)) if str(candidate or "").strip() else ""
        if text and os.path.normcase(text) not in {os.path.normcase(r) for r in comfyui_roots}:
            comfyui_roots.append(text)

    def add(candidate: str) -> None:
        if candidate and os.path.exists(candidate):
            resolved = os.path.abspath(candidate)
            if os.path.normcase(resolved) not in {os.path.normcase(r) for r in roots}:
                roots.append(resolved)

    add(models_dir)
    add(services_dir)
    # ⚠️ 用户**显式**加的目录排在约定落点之前 ✓：显式意图优先于约定 ✓
    for candidate in get_extra_roots():
        add(candidate)
    # ⭐ 2026-09-26：**用户声明过的任意目录** ✓（ComfyUI 的 ``extra_model_paths.yaml`` ✓、
    #    ``model-paths.json`` 的 ``extra_paths`` ✓、环境变量 ``EXTRA_MODEL_PATHS`` ✓）——
    #    「模型放任意目录，**声明一下就能用**」✓，口径见 ``model_folders`` ✓。
    #    ⚠️ 与 ``extra_roots`` 同属**显式意图** ⇒ 也排在生态落点之前 ✓（显式优先于约定 ✓）。
    #    ⚠️ **环境变量**指向的声明文件读不开 ⇒ 如实抛 ✓（用户自己指的文件 ✗ 不许静默跳过 ✗）；
    #       而 ComfyUI 那边别人的 ``extra_model_paths.yaml`` 坏掉 ⇒ 只跳过 ✓（不连坐 ✗）。
    for candidate in model_folders.declared_extra_dirs(
            cfg, os.path.dirname(_paths_config_file()), comfyui_roots):
        add(candidate)
    # ⭐ 2026-09-25：**世界各大模型生态**的落点 ✓（HuggingFace / ModelScope / LM Studio / GPT4All /
    # Jan / llama.cpp / text-generation-webui / SD WebUI / Stability Matrix / InvokeAI / torch hub ✓，
    # 表在 ``model_ecosystems`` ✓；只读 ✓、不要求任何生态的服务在跑 ✓）。
    # ⚠️ 排在 ComfyUI **之前** ✗：ComfyUI 的 ``models/`` 动辄上万文件 ⇒ 后置会被 ``maxFiles``
    # 上限吃光 ✗，生态落点就「**扫不到也不报错**」了 ✗✗ —— 那正是本轮要修的那类 bug ✓。
    for _, path in model_ecosystems.roots():
        add(path)
    # ⭐ 2026-09-26 修「只取第一个」✗：**每个** ComfyUI 类目录的 ``models/`` 都加 ✓
    #    （本机三个：``ComfyUI-Installs/…/ComfyUI`` / ``ComfyUI-Shared`` / ``D:/code/ComfyUI/ComfyUI`` ✓
    #     —— 旧实现只加第一个 ⇒ 共享目录里那份 SDXL 扫不到 ✗✗）
    for root in comfyui_roots:
        add(os.path.join(root, "models"))
    # ⚠️ **只有显式配的那个根**整体也加上 ✓（``COMFYUI_PATH`` / ``comfyui_root`` ✓）：
    #    探测到多个安装时，各自的整包（web / tests … 动辄上万文件）会把 ``maxFiles`` 吃光 ✗
    #    ⇒ 生态落点又变成「扫不到也不报错」✗✗；而根下除 ``models/`` 外基本是代码/输出 ✓（不是模型 ✓）。
    if comfyui_root:
        add(comfyui_root)

    # 若完全没有可扫描目录，回退到本仓默认模型目录
    if not roots:
        default = os.path.abspath(default_models_dir())
        if os.path.exists(default):
            roots.append(default)
    return roots


def save_extra_roots(roots: list[str]) -> list[str]:
    """保存额外扫描目录，返回去重后的**规范绝对路径**（保序）。"""
    cleaned: list[str] = []
    for raw in roots or []:
        text = str(raw).strip()
        if not text:
            continue
        resolved = os.path.abspath(text)
        if resolved not in cleaned:
            cleaned.append(resolved)
    cfg = _read_paths_config()
    cfg["extra_roots"] = cleaned
    _write_paths_config(cfg)
    return cleaned


def get_model_paths() -> dict[str, Any]:
    """完整模型路径配置（comfyui_root / models_dir / nodes_dir / local_services_dir / extra_roots）。

    ⚠️ ``comfyui_root`` 是**单值**字段 ✓（给「要写一份配置」的地方用 ✓）：显式配置优先 ✓、
    否则取**第一个**探测命中 ✓；**扫描不靠它** ✗ —— 扫描走 :func:`get_default_roots` ✓
    （那里把**所有** ComfyUI 类目录的 ``models/`` 都收进来 ✓）。
    """
    cfg = _read_paths_config()
    return {
        "comfyui_root": os.environ.get("COMFYUI_PATH") or cfg.get("comfyui_root")
        or detect_comfyui() or "",
        "models_dir": os.environ.get("MODELS_DIR") or cfg.get("models_dir")
        or default_models_dir(),
        "nodes_dir": cfg.get("nodes_dir") or "",
        "local_services_dir": cfg.get("local_services_dir") or "",
        "extra_roots": get_extra_roots(),
    }


def save_model_paths(patch: dict[str, Any]) -> dict[str, Any]:
    """合并写 ``model-paths.json``：**仅更新传入的字符串字段，空串表示清空**。"""
    cfg = _read_paths_config()
    for key in ("comfyui_root", "models_dir", "nodes_dir", "local_services_dir"):
        value = patch.get(key)
        if not isinstance(value, str):
            continue
        text = value.strip()
        if text:
            cfg[key] = text
        else:
            cfg.pop(key, None)
    _write_paths_config(cfg)
    return cfg


# ============================================================
# 扫描实现
# ============================================================

def _js_to_fixed(value: float, digits: int) -> str:
    """``value.toFixed(digits)``。

    ⚠️ 不能直接用 Python 的 ``format`` —— Python 是**银行家舍入**，JS 是**四舍五入**
    （``(2.5).toFixed(0)`` 是 ``"3"``，Python ``format(2.5, '.0f')`` 是 ``"2"``）。
    """
    quantized = Decimal(repr(value)).quantize(
        Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    return f"{quantized:.{digits}f}"


def human_size(num_bytes: float) -> str:
    """人类可读体积（``KB/MB/GB/TB``；≥100 不留小数位）。"""
    if num_bytes < 1024:
        return f"{int(num_bytes)} B"
    units = ["KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    index = -1
    while True:
        value /= 1024
        index += 1
        if not (value >= 1024 and index < len(units) - 1):
            break
    return f"{_js_to_fixed(value, 0 if value >= 100 else 1)} {units[index]}"


def _resolve_roots(roots: list[str] | None) -> list[str]:
    base = roots if roots else get_default_roots()
    return [os.path.abspath(r) for r in base if os.path.exists(r)]


def _sort_models(models: list[dict[str, Any]]) -> None:
    """⚠️ ``Array.sort`` 在 V8 里是**稳定**排序，Python 的 ``sorted`` 同样稳定 ⇒ 等价。"""
    models.sort(key=lambda item: (_KIND_ORDER[item["kind"]], -item["sizeBytes"]))


def _make_model(full_path: str, filename: str, dir_name: str, ext: str,
                size_bytes: int, kinds_filter: set[str] | None) -> dict[str, Any] | None:
    hit = classify(full_path)
    # ⭐ 生态标注 ✓：文件**在哪**（HuggingFace 缓存 / LM Studio / SD WebUI…）——
    # ⚠️ 只改「弱命中」（comfyui/unknown/ollama）✗，强命中（h3/local-sd/cosyvoice）保留事实判断 ✓
    # ⚠️ 路径不在任何已知生态里 ⇒ ``""`` ✓（不硬猜 ✗）
    ecosystem = model_ecosystems.detect(full_path)
    if ecosystem:
        hit = model_ecosystems.apply_ecosystem(hit, ecosystem)
    if kinds_filter and hit["kind"] not in kinds_filter:
        return None
    suggested = (model_ecosystems.build_suggestion(ecosystem, hit, filename, ext)
                 if ecosystem else None)
    return {
        "path": full_path,
        "filename": filename,
        "dirname": dir_name,
        "sizeBytes": size_bytes,
        "sizeHuman": human_size(size_bytes),
        "ext": ext,
        "kind": hit["kind"],
        "runtime": hit["runtime"],
        "confidence": hit["confidence"],
        "role": hit["role"],
        "matchedBy": hit["matchedBy"],
        "ecosystem": ecosystem,
        "suggested": suggested if suggested is not None else build_suggestion(hit, filename, ext),
    }


def _is_under(path: str, roots: list[str]) -> bool:
    """``path`` 是否落在 ``roots`` 之下（含相等 ✓；大小写/分隔符按平台归一 ✓）。"""
    if not path:
        return False
    target = os.path.normcase(os.path.abspath(path))
    for root in roots:
        base = os.path.normcase(os.path.abspath(root)).rstrip("\\/")
        if base and (target == base or target.startswith(base + os.sep)):
            return True
    return False


def _merge_ollama(models: list[dict[str, Any]], roots: list[str],
                  explicit_roots: bool, kinds_filter: set[str] | None) -> None:
    """把 Ollama **清单**里的模型并进结果 ✓（它的权重是无扩展名的 blob ⇒ 文件遍历看不见 ✗）。

    ⚠️ 触发条件**故意保守**（不无条件并 ✗，否则「只扫某个盘」会凭空冒出别的盘的模型 ✗）：
    1. 调用方**没指定** roots ⇒ 默认扫描 = 「把这台机器的模型都找出来」✓；
    2. 或指定的 roots **覆盖了** Ollama 模型库 ✓（扫 ``C:`` 时它本来就在下面 ✓）。

    ⚠️ ``path`` 填**清单文件**的真实路径 ✓（blob 路径对调用方没用 ✗）；``ext`` 留空 ✓
    （硬写成 ``.gguf`` 是编 ✗：清单不保证底层 blob 的类型 ✓）。
    """
    if explicit_roots and not _is_under(ollama_store.models_root(), roots):
        return
    if kinds_filter and "text" not in kinds_filter:
        return
    seen = {str(m.get("path", "")).lower() for m in models}
    for item in ollama_store.list_models():
        name = str(item.get("name") or "")
        path = str(item.get("path") or "")
        if not name or (path and path.lower() in seen):
            continue
        seen.add(path.lower())
        size = int(item.get("size") or 0)
        missing = int(item.get("missingBlobs") or 0)
        models.append({
            "path": path,
            "filename": name,
            "dirname": os.path.basename(os.path.dirname(path)) or "ollama",
            "sizeBytes": size,
            "sizeHuman": human_size(size),
            "ext": "",
            "kind": "text",
            "runtime": "ollama",
            "confidence": "high",
            "role": "standalone",
            "matchedBy": ["ollama-store"],
            "ecosystem": "ollama",
            "suggested": model_ecosystems.build_ollama_manifest_suggestion(name, size, missing),
        })


def scan_local_models(opts: dict[str, Any] | None = None) -> dict[str, Any]:
    """**同步**扫描本地模型（``roots`` 为空时用默认候选目录）。

    单个根目录递归扫描，带**深度/数量上限**，跳过无关目录。默认 ``maxDepth=5``、
    ``maxFiles=8000``（⚠️ 与异步版不同）。
    """
    opts = opts or {}
    start = _now_ms()
    max_depth = opts.get("maxDepth") if opts.get("maxDepth") is not None else 5
    max_files = opts.get("maxFiles") if opts.get("maxFiles") is not None else 8000
    kinds = opts.get("kinds") or []
    kinds_filter = set(kinds) if kinds else None

    explicit_roots = bool(opts.get("roots"))
    roots = _resolve_roots(opts.get("roots"))
    models: list[dict[str, Any]] = []
    file_count = 0
    state = {"truncated": False, "cancelled": False}

    seen: set[str] = set()
    # 已访问目录（realpath 去重，防 junction/符号链接死循环与重复扫描）
    visited: set[str] = set()
    for root in roots:
        try:
            visited.add(os.path.realpath(root))
        except OSError:
            pass

    def walk(directory: str, depth: int) -> None:
        nonlocal file_count
        if state["truncated"] or depth > max_depth:
            return
        try:
            with os.scandir(directory) as handle:
                entries = list(handle)
        except OSError:
            return

        for entry in entries:
            if state["truncated"] or file_count >= max_files:
                state["truncated"] = True
                return
            full = os.path.join(directory, entry.name)

            # 目录条目：⚠️ `follow_symlinks=False` 与 Node 的 DirEntry 语义一致
            # （软链接**不**被当作目录 ⇒ 与 TS 的 `ent.isSymbolicLink()` 跳过一致）
            if entry.is_dir(follow_symlinks=False):
                low = entry.name.lower()
                if low in SKIP_DIRS or low in SYSTEM_DIRS:
                    continue
                if low.startswith(ECOSYSTEM_SKIP_PREFIXES):
                    continue  # 生态的非模型目录（HF 数据集 / 负缓存标记…）
                if entry.is_symlink():
                    continue  # 不跟随符号链接/junction
                try:
                    real = os.path.realpath(full)
                except OSError:
                    continue
                if real in visited:
                    continue
                visited.add(real)
                walk(full, depth + 1)
                continue

            file_count += 1
            ext = os.path.splitext(entry.name)[1].lower()
            if ext not in MODEL_EXTS:
                continue

            # 去重（同一文件通过不同根目录重复出现）
            key = full.lower()
            if key in seen:
                continue
            seen.add(key)

            try:
                size_bytes = os.stat(full).st_size
            except OSError:
                size_bytes = 0

            model = _make_model(full, entry.name, os.path.basename(directory), ext,
                                size_bytes, kinds_filter)
            if model is not None:
                models.append(model)

    for root in roots:
        walk(root, 0)

    # ⭐ Ollama 清单里的模型也并进来 ✓（权重是无扩展名 blob ⇒ 文件遍历看不见 ✗，见 _merge_ollama）
    _merge_ollama(models, roots, explicit_roots, kinds_filter)

    by_kind = {"text": 0, "image": 0, "video": 0, "audio": 0, "unknown": 0}
    for model in models:
        by_kind[model["kind"]] += 1
    _sort_models(models)

    return {
        "roots": roots,
        "models": models,
        "total": len(models),
        "truncated": state["truncated"],
        "elapsedMs": _now_ms() - start,
        "byKind": by_kind,
        "byEcosystem": model_ecosystems.by_ecosystem(models),
    }


async def scan_local_models_async(opts: dict[str, Any] | None = None) -> dict[str, Any]:
    """**异步**扫描：``readdir/stat`` 走线程池，扫描期间不阻塞事件循环。

    支持进度回调与取消（``shouldCancel`` 返回真即中止）。默认 ``maxDepth=8``、
    ``maxFiles=50000``（⚠️ 比同步版宽松得多，用于磁盘/全盘级扫描）。
    """
    opts = opts or {}
    start = _now_ms()
    max_depth = opts.get("maxDepth") if opts.get("maxDepth") is not None else 8
    max_files = opts.get("maxFiles") if opts.get("maxFiles") is not None else 50000
    kinds = opts.get("kinds") or []
    kinds_filter = set(kinds) if kinds else None
    on_progress: Callable[[dict[str, Any]], None] | None = opts.get("onProgress")
    should_cancel: Callable[[], bool] | None = opts.get("shouldCancel")

    explicit_roots = bool(opts.get("roots"))
    roots = _resolve_roots(opts.get("roots"))
    models: list[dict[str, Any]] = []
    counters = {"files": 0, "truncated": False, "cancelled": False}

    seen: set[str] = set()
    visited: set[str] = set()
    for root in roots:
        try:
            visited.add(await asyncio.to_thread(os.path.realpath, root))
        except OSError:
            pass

    def emit(current_dir: str) -> None:
        if on_progress:
            on_progress({
                "scannedFiles": counters["files"],
                "foundModels": len(models),
                "currentDir": current_dir,
                "done": False,
                "cancelled": counters["cancelled"],
            })

    async def walk(directory: str, depth: int) -> None:
        if counters["truncated"] or counters["cancelled"] or depth > max_depth:
            return
        if should_cancel and should_cancel():
            counters["cancelled"] = True
            return
        entries = await asyncio.to_thread(_read_entries, directory)
        if entries is None:
            return

        for name, is_dir, is_symlink in entries:
            if counters["truncated"] or counters["cancelled"] or counters["files"] >= max_files:
                counters["truncated"] = True
                return
            if should_cancel and should_cancel():
                counters["cancelled"] = True
                return
            full = os.path.join(directory, name)

            if is_dir:
                low = name.lower()
                if low in SKIP_DIRS or low in SYSTEM_DIRS:
                    continue
                if low.startswith(ECOSYSTEM_SKIP_PREFIXES):
                    continue  # 生态的非模型目录（HF 数据集 / 负缓存标记…）
                if is_symlink:
                    continue
                try:
                    real = await asyncio.to_thread(os.path.realpath, full)
                except OSError:
                    continue
                if real in visited:
                    continue
                visited.add(real)
                await walk(full, depth + 1)
                continue

            counters["files"] += 1
            if counters["files"] % 1000 == 0:
                emit(directory)
            ext = os.path.splitext(name)[1].lower()
            if ext not in MODEL_EXTS:
                continue

            key = full.lower()
            if key in seen:
                continue
            seen.add(key)

            try:
                size_bytes = (await asyncio.to_thread(os.stat, full)).st_size
            except OSError:
                size_bytes = 0

            model = _make_model(full, name, os.path.basename(directory), ext,
                                size_bytes, kinds_filter)
            if model is not None:
                models.append(model)

    for root in roots:
        if counters["cancelled"] or counters["truncated"]:
            break
        await walk(root, 0)

    if counters["cancelled"]:
        raise ScanCancelledError()

    # ⭐ Ollama 清单里的模型也并进来 ✓（与同步版**同一份**规则 ✓，见 _merge_ollama）
    _merge_ollama(models, roots, explicit_roots, kinds_filter)

    by_kind = {"text": 0, "image": 0, "video": 0, "audio": 0, "unknown": 0}
    for model in models:
        by_kind[model["kind"]] += 1
    _sort_models(models)

    return {
        "roots": roots,
        "models": models,
        "total": len(models),
        "truncated": counters["truncated"],
        "elapsedMs": _now_ms() - start,
        "byKind": by_kind,
        "byEcosystem": model_ecosystems.by_ecosystem(models),
    }


def _read_entries(directory: str) -> list[tuple[str, bool, bool]] | None:
    """一次性读出目录条目（**在线程里跑**）：``(名字, 是否目录, 是否软链接)``。

    ⚠️ 用 ``follow_symlinks=False`` —— 与 Node ``DirEntry`` 一致（软链接不会被当作目录）。
    这里**故意不 stat 文件大小**：与原 TS 一样，只有模型文件才 stat（大目录下省掉 N 次系统调用）。
    """
    try:
        with os.scandir(directory) as handle:
            return [(entry.name, entry.is_dir(follow_symlinks=False),
                     entry.is_symlink()) for entry in handle]
    except OSError:
        return None


def list_drives() -> list[dict[str, str]]:
    """枚举可用盘符（Windows 探测 A-Z；POSIX 上结果为空 —— 与原 TS 行为一致）。"""
    drives: list[dict[str, str]] = []
    for code in range(65, 91):
        letter = chr(code)
        root = f"{letter}:\\"
        try:
            if os.path.exists(root):
                drives.append({"letter": letter, "root": root})
        except OSError:
            pass
    return drives


def _now_ms() -> float:
    """``Date.now()`` 的等价物（只用于耗时统计）。"""
    import time

    return time.time() * 1000
