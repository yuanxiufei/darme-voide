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
   InvokeAI / torch hub 的本地落点 ✓（表在 ``app/core/model_ecosystems.py`` ✓，本层再导出 ✓），**全程只读、零外部依赖** ✓
   —— **不许**要求 ``ollama serve`` / LM Studio 等**服务在跑**才能扫到 ✗✗（那正是「盘上有模型却
   看不见」的老 bug ✓）；Ollama 的权重是无扩展名 blob ⇒ 文件遍历看不见 ✗，改由 **manifests 清单**读出 ✓
   （见 ``_merge_ollama`` ✓）。
9. ⭐ **不许写死机器路径** ✗（2026-09-26 用户口径 ✓）：模型机可以装到**用户指定目录** ✓（``models_dir`` ✓）、
   也要能扫**电脑任意目录** ✓（``extra_roots`` ✓ + 盘符枚举 ``list_drives`` ✓，整盘扫也行 ✓）——
   权威来源只有**配置 / 环境变量 / 盘符枚举** ✓，源码与行文里都**不出现**任何**盘符字面量** ✗
   （静态守卫在同一套自检里 ✓：``local_models_test`` ✓）。
"""
from __future__ import annotations

import asyncio
import os
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..core import comfyui_roots as core_comfyui_roots
from ..core import model_paths
from ..core import model_roots as core_model_roots
from . import model_ecosystems, model_fingerprint, ollama_store

#: 后端包根（``backend-py/``）。**「本地服务根」的默认值**：
#: 2026-09-15 从仓库根 ``local_services/`` 迁入 —— 它是「``model_manager.py`` 会 ``git clone``
#: 的独立服务 + 项目自带的 ``h3`` 薄封装」的落脚点，属后端资产，与后端代码同包更归拢。
#: ⚠️ **唯一权威在 ``app/core/config.py``**（2026-09-15 收口：本文件原先自己算一遍 ``parents[2]`` ✗）；
#: ``scripts/model_manager.py`` 那份是同名独立实现（scripts 不 import ``app.*``），靠注释同步。
#: ⚠️ 与 ``config.BACKEND_PY_ROOT`` 区分：**资产根是 `app/`**（三类资产住在这里）⇒ 用 `APP_ROOT`。
#: ⚠️ 2026-11-08 起**再导出** :data:`app.core.model_roots.LOCAL_SERVICES_ROOT` ✓（同一个 ``Path`` ✓）：
#: 「默认扫描根」的拼装整体搬进 ``app.core.model_roots`` 了（引擎 ``inventory`` 也要那份根表 ✓），
#: 这个常量是根表的一行 ⇒ 跟着走 ✓，本层不再自己算一遍 ✗。
LOCAL_SERVICES_ROOT = core_model_roots.LOCAL_SERVICES_ROOT

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
    "get_comfyui_roots",
    "get_default_roots",
    "get_extra_roots",
    "get_model_paths",
    "human_size",
    "list_drives",
    "save_extra_roots",
    "save_model_paths",
    "scan_local_models",
    "scan_local_models_async",
    "weight_fingerprint",
]

#: 模型权重扩展名（大小写不敏感）
MODEL_EXTS = frozenset([
    ".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx", ".sft", ".engine",
])

#: ⚠️ 跳过表与「ComfyUI 安装根探测」的**实现**都在 :mod:`app.core.comfyui_roots` ✓✗
#: （2026-11-08 搬 ✓）：那一套读的**只有配置与盘** ✓，不碰业务数据 ⇒ 与 ``app.core.model_paths``
#: 同一类 ✓；而引擎读权重时**也要**答「安装根在哪」（``bridge.resolve_tokenizer_path`` 找
#: ``comfy/sd1_tokenizer`` ✓ —— SDXL 的 CLIP-BPE 词表不是权重，扫 ``.safetensors`` 永远找不着 ✓✗），
#: 留在服务层就等于逼引擎 ``import app.services.local_model_scan`` ✓✗。
#: 这里做的是**再导出同一批对象** ✓（``SKIP_DIRS is app.core.comfyui_roots.SKIP_DIRS`` ✓），不是副本 ✗。
#: ⚠️ 分层只搬到「安装根探测」为止 ✗✗：``get_default_roots`` 的另外几类根（``extra_roots`` /
#: ``model_ecosystems`` 的世界生态落点 / ``model_folders`` 的声明表）要读生态约定与用户声明，
#: 属「模型库功能」⇒ 仍留在这里 ✓（``_SEAMS`` 里 ``inventory.py`` 那条登记的就是剩下的这半 ✓）。
SKIP_DIRS = core_comfyui_roots.SKIP_DIRS
SYSTEM_DIRS = core_comfyui_roots.SYSTEM_DIRS
COMFYUI_DIR_HINTS = core_comfyui_roots.COMFYUI_DIR_HINTS
COMFYUI_PROBE_DEPTH = core_comfyui_roots.COMFYUI_PROBE_DEPTH
COMFYUI_BLANKET_DEPTH = core_comfyui_roots.COMFYUI_BLANKET_DEPTH
list_drives = core_comfyui_roots.list_drives
detect_comfyui_roots = core_comfyui_roots.detect_comfyui_roots
detect_comfyui = core_comfyui_roots.detect_comfyui

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

#: ⚠️ 下面四个是**薄委托** ✓✗：配置面（文件位置 / JSON 读写 / ``models_dir`` 优先级）的唯一真源
#: 在 :mod:`app.core.model_paths` ✓ —— 引擎要在不 import 服务层的前提下答「模型根在哪」✓，
#: 那份解析搬去 ``app.core`` 后本层只留**发现**逻辑 ✓（ComfyUI 探测 / 生态落点 / 声明表 ✓）。
#: 谁在这里再抄一遍优先级，就会「引擎说这里、扫描说那里」✓✗（判据：``tests/local_models_test.py``）。
def _paths_config_file() -> str:
    return str(model_paths.paths_config_file())


def _read_paths_config() -> dict[str, Any]:
    #: ⚠️ 读与写**两个**薄委托都必须经 ``_paths_config_file()`` ✓✗（它是本层唯一可 patch 的口）：
    #: 只把读走它、写仍直连 ``model_paths`` 时，自检里换掉的文件只有读生效 ⇒ 一次 ``PUT /roots``
    #: 就把真配置的 ``models_dir`` 冲掉、塞进一条 ``%TEMP%`` 的 ``extra_roots`` ✓✗（2026-11-08 实测；
    #: 判据 ``tests/local_models_test.py``「真配置文件一字未动」）。
    return model_paths.read_paths_config(path=Path(_paths_config_file()))


def _write_paths_config(cfg: dict[str, Any]) -> None:
    model_paths.write_paths_config(cfg, path=Path(_paths_config_file()))


def default_models_dir() -> str:
    """本仓**自己的**模型存储目录 ✓：``<data_root>/models`` ✓（实现见 ``app.core.model_paths`` ✓）。

    2026-09-25 自主化 ✓：模型不再依赖 ComfyUI 目录 ✗ —— 下载/扫描/就绪/加载统一落到这里 ✓
    （``MODELS_DIR`` 环境变量或 ``model-paths.json`` 的 ``models_dir`` 仍可覆盖 ✓）。
    """
    return model_paths.default_models_dir()


def get_extra_roots() -> list[str]:
    """用户自定义的额外扫描目录（``configs/model-paths.json`` 的 ``extra_roots``）。"""
    extra = _read_paths_config().get("extra_roots")
    if isinstance(extra, list):
        return [x for x in extra if isinstance(x, str) and x.strip()]
    return []


def get_comfyui_roots() -> list[tuple[str, str]]:
    """ComfyUI 类**安装根** + 来源说明 ✓ ⇒ ``[(根, 来源)]`` ✓（**显式配的排在前** ✓）。

    三处来源（顺序固定 ✓）：``COMFYUI_PATH`` ✓ → ``configs/model-paths.json`` 的 ``comfyui_root``
    ✓ → :func:`detect_comfyui_roots` 的**全部**命中 ✓（⚠️ 是**全部** ✗ 不是第一个 ✓：只取第一个
    会把后面几个安装整份遮住 ✗✗，见 :func:`detect_comfyui_roots` 的说明 ✓）。

    ⚠️ 与 :func:`get_default_roots` 的分工 ✗：那个要的是「**扫模型**的根」（``<根>/models`` ✓），
    这个要的是「**安装根**」（``<根>/comfy/…`` 这类**安装内部**的落点 ✓）—— 两者**共用**这一份
    根表 ✓（哪一块都不许各写一份 ✗：一处改了另一处不跟着改，就会出现「扫描能看见、加载看不见」✓✗）。

    ⚠️ 实现已搬 :mod:`app.core.comfyui_roots` ✓（引擎 ``inventory`` / ``bridge`` 也要这一份 ✓，
    而引擎不许 import 服务层 ✓）。这里只剩**一层薄委托**，目的是把本模块**两个可 patch 的名字**
    显式传下去 ✓✗（``tests/local_models_test.py`` 就是 patch 它们来注入临时配置与假根 ✓）：
    ``_paths_config_file()`` ✓ 与 ``detect_comfyui_roots`` ✓。
    ⚠️ 两个都**必须**传 ✗✗：只传配置路径而不传探测函数 ⇒ core 会走它自己的全局 ⇒ 本模块这次
    patch 静默失效 ✓✗（2026-11-08 实测：``get_default_roots`` 那条「每个 ComfyUI 根都要收进来」
    当场红 ✓ —— 跨模块 monkeypatch 不生效，只能靠显式注入 ✓）。
    """
    return core_comfyui_roots.comfyui_roots(
        path=Path(_paths_config_file()), detect=detect_comfyui_roots
    )


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

    ⚠️ **实现已搬 :mod:`app.core.model_roots`** ✓（2026-11-08 ✓）：引擎 ``inventory.candidate_roots``
    要的是**同一份**根表 ✓，而引擎不许 import 服务层 ✓ ⇒ 拼装整体下沉，本函数只剩**一层薄委托** ✓。
    这里显式下传两个隔离口 ✓✗（``tests/local_models_test.py`` 就是 patch 它们注入临时配置与假根）：
    ``path=``（:func:`_paths_config_file` ✓）与 ``comfyui=``（:func:`get_comfyui_roots` ✓ ——
    它自身已把 ``detect_comfyui_roots`` 传下去了 ✓）。**两个都少不得** ✗✗：
    只传一个 ⇒ core 对另一半走自己的全局 ⇒ 本模块这次 patch 静默失效 ✓✗（2026-11-08 实测）。
    """
    return core_model_roots.default_roots(
        path=Path(_paths_config_file()),
        comfyui=get_comfyui_roots,
        extra_roots=get_extra_roots(),
    )


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
        #: ⚠️ ``models_dir`` 走 core 那唯一一份优先级 ✓（引擎 ``inventory.models_dir()`` 用的是
        #: **同一个函数** ✓ ⇒ 「体检说这里、加载说那里」这类漂移从根上没了 ✓）。
        #: ⚠️ 隔离口同 :func:`get_default_roots` —— 不传就是读真配置文件 ✓✗（端点自检因此假绿过 ✓）。
        "models_dir": model_paths.models_dir(path=Path(_paths_config_file())),
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


# ============================================================
# 权重指纹（读文件头，不信文件名）
# ============================================================
#: 这些 kind 才能写进扫描结果的 ``kind`` 字段（``_KIND_ORDER`` 只认这五种 ⇒
#: 指纹判出的 ``3d``/``detect``/``depth`` 只进 ``weight`` 明细，**不改分类** ✗✗）。
_SCAN_KINDS = frozenset({"video", "image", "text", "audio"})
#: 只有这两档置信度够硬到能改写文件名规则的结论（GC 都实测核过 ✓）。
#: ⚠️ ``declared``（GGUF 自报 architecture）**不够硬** ✗ —— 实测有量化脚本写错标签的实例
_WEIGHT_TRUSTED = frozenset({"exact", "single-key"})
#: 指纹缓存：``(normcase(path), size, mtime_ns) -> 结果``。文件一动键就变 ⇒ 不会吃旧结论 ✓
_FP_CACHE: dict[tuple[str, int, int], dict[str, Any]] = {}
_FP_CACHE_MAX = 4096


def weight_fingerprint(path: str, ext: str = "") -> dict[str, Any] | None:
    """给扫描结果补「权重真身」（架构 / 模态 / 变体 / 证据）。

    * 只读**文件头**（safetensors header / GGUF kv+tensor info）⇒ 实测 ~2ms/个 ✓；
    * 只对 ``.safetensors`` / ``.gguf`` 做（其余格式没有可读的结构头 ✗）；
    * **不抛异常**：读不动就回 ``{"error": …}``，扫描因此不会因一个坏文件整体失败 ✓；
    * 命中缓存则零读盘（同一文件重复扫、或每次 API 请求都扫时不重复付代价 ✓）。
    """
    ext = (ext or os.path.splitext(path)[1]).lower()
    if ext not in (".safetensors", ".gguf"):
        return None
    try:
        info = os.stat(path)
        key = (os.path.normcase(path), info.st_size, info.st_mtime_ns)
    except OSError:
        key = (os.path.normcase(path), 0, 0)
    cached = _FP_CACHE.get(key)
    if cached is not None:
        return cached
    result = model_fingerprint.fingerprint_file(path)
    if len(_FP_CACHE) >= _FP_CACHE_MAX:
        _FP_CACHE.clear()  # 简单粗暴：清掉重来（比 LRU 记账便宜，且这是缓存不是状态）
    _FP_CACHE[key] = result
    return result


def _reconcile_kind(hit: dict[str, Any],
                    weight: dict[str, Any] | None) -> tuple[dict[str, Any], bool]:
    """**名字规则 vs 权重指纹**不一致时，以权重为准（仅在指纹够硬时）。

    真机上的实例（改前全是错的）：
    * ``Krea2/krea2_turbo-Q4_K_M.gguf`` —— 命中 ``gguf-llm`` ⇒ 被当成**文本模型** ✗，
      实际张量里有 ``txtfusion.projector.weight`` ⇒ 图像模型 ✓；
    * ``SEEDVR2/seedvr2_ema_7b-Q4_K_M.gguf`` —— 同样落到 ``gguf-llm`` ⇒ 文本 ✗，实际是**视频修复** ✓。

    改写时把 ``runtime`` 固定成 ``comfyui``：这些权重是 ComfyUI 格式的 DiT/修复网，
    **不是** H3 服务也不是 Ollama ⇒ 建议只能是「只读来源」，**不能**据此注册一个假服务 ✗。
    """
    if not weight or weight.get("error"):
        return hit, False
    kind = str(weight.get("kind") or "")
    if kind not in _SCAN_KINDS:
        return hit, False
    if str(weight.get("confidence") or "") not in _WEIGHT_TRUSTED:
        return hit, False
    if kind == hit["kind"]:
        return hit, False
    fixed = dict(hit)
    fixed["kind"] = kind
    fixed["runtime"] = "comfyui"
    fixed["role"] = "standalone"
    fixed["confidence"] = "high"
    fixed["matchedBy"] = [*hit["matchedBy"], "weight-fingerprint"]
    return fixed, True


def _is_broken(weight: dict[str, Any] | None) -> bool:
    """权重文件是否**结构上坏了**（头里对账对不上），与「架构认不出」严格分开 ✓。

    ``status`` 只有 ``truncated`` / ``trailing`` 算坏（``padded`` 是对齐填充、``unknown``
    是读不全 tensor info —— 后者已由 ``weight.error`` 或 ``confidence`` 表达，不重复计 ✗）。
    """
    if not weight or weight.get("error"):
        return False
    return str((weight.get("integrity") or {}).get("status") or "") in ("truncated", "trailing")


def _make_model(full_path: str, filename: str, dir_name: str, ext: str,
                size_bytes: int, kinds_filter: set[str] | None,
                fingerprint: bool = True) -> dict[str, Any] | None:
    hit = classify(full_path)
    # ⭐ 生态标注 ✓：文件**在哪**（HuggingFace 缓存 / LM Studio / SD WebUI…）——
    # ⚠️ 只改「弱命中」（comfyui/unknown/ollama）✗，强命中（h3/local-sd/cosyvoice）保留事实判断 ✓
    # ⚠️ 路径不在任何已知生态里 ⇒ ``""`` ✓（不硬猜 ✗）
    ecosystem = model_ecosystems.detect(full_path)
    if ecosystem:
        hit = model_ecosystems.apply_ecosystem(hit, ecosystem)
    # ⭐ 权重指纹 ✓：读文件头拿**真架构** —— 名字可以是错的（改名/量化工具贴错标签），张量名不会 ✗
    weight = weight_fingerprint(full_path, ext) if fingerprint else None
    hit, overridden = _reconcile_kind(hit, weight)
    if kinds_filter and hit["kind"] not in kinds_filter:
        return None
    # 被指纹改写过的，生态建议（按名字猜的落地方式）就不作数了 ⇒ 退回通用建议 ✓
    suggested = (model_ecosystems.build_suggestion(ecosystem, hit, filename, ext)
                 if ecosystem and not overridden else None)
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
        # ``None`` = 这个格式没有可读结构头（.ckpt/.pt/.bin/.onnx…）⇒ 只能按名字判 ✓
        "weight": weight,
        "weightVerified": bool(weight and not weight.get("error")
                               and weight.get("architecture") not in (None, "", "unknown")),
        # ⭐ 头里的数据区坐标与文件大小对不上 ⇒ 这份权重**是坏文件**（截断/拼接），
        # 与「架构认不出」是两件事：跑起来只会报 shape 错或半截加载，早点点名比事后查快 ✓
        "weightBroken": _is_broken(weight),
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
    # 默认开：读文件头判真架构（~2ms/个，命中缓存后 0）；``fingerprint=False`` 时才纯按名字判
    fingerprint = opts.get("fingerprint") is not False

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
                                size_bytes, kinds_filter, fingerprint)
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
        "weightSummary": model_fingerprint.summarize(
            [m["weight"] for m in models if m.get("weight")]),
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
    fingerprint = opts.get("fingerprint") is not False

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
                                size_bytes, kinds_filter, fingerprint)
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
        "weightSummary": model_fingerprint.summarize(
            [m["weight"] for m in models if m.get("weight")]),
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
