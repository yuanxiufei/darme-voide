"""ComfyUI 提示词模板（增强 / 反推）的**编译层** —— 2026-11-03 自 ``comfyui-gateway/src/prompt-optimizer.js`` 迁入 ✓。

模板本体是四张**工作流图**（``configs/comfyui/prompts/*.json`` ✓），它们的节点来自上游插件
（``TE_*`` / ``llama_cpp`` / ``Florence2``），**不是** :mod:`app.services.comfyui_catalog` 那 21 条
出图工作流。迁移时只搬「**参数落到哪个节点哪一格**」这层语义 ✓，不搬 JS 的 HTTP 服务与云端 chat
通路 ✗（本仓没有云端 chat 客户端 ⇒ 搬过来就是没有调用口的死代码 ✗）。

## 为什么要单独一层（不让调用方自己改 JSON）

四个模板的落点**全是中文输入名**（``输入提示词`` / ``任务类型`` / ``图片2`` …），且有两处
「不是常量、要按上下文算」的格：

* H3 的 ``#3.输入模式``（有参考图＝``图片``、无图＝``文本``）；
* 图的 ``#1.任务模式``（有参考图＝``图生图``、无图＝``文生图``）。

让前端自己拼 ⇒ 这两格必错，而**错了不报错**：``输入模式`` 停在 ``文本`` 时参考图白传（模型不看图 ✗）；
``任务模式`` 停在 ``文生图`` 时参考图只当"风格参考"（不参与编辑 ✗）—— 都是「跑通了、结果不对」那类 ✗✗。

## 口径（逐条对齐上游 ✓）

1. **落点缺格必须报** ✓：模板是数据、可被编辑 ⇒ 填之前先核 ``node`` / ``input`` 真的在，
   不在就抛 :class:`PromptTemplateError`（不静默新建一格 ✗：往 ComfyUI 塞它不认的输入只会得到
   ``required input is missing`` 这种指不到源头的错 ✗）。
   唯一例外是**参考图连线格**（``图片`` / ``图片2`` …）：它们是上游节点的**可选输入**，
   模板里只声明了其中一部分（H3 模板连第一格都没有 ✓）⇒ :func:`compile_template` 允许新建 ✓。
2. **枚举必须校验** ✓：``任务类型`` 只认 :data:`H3_MODES` 五个（与上游 ``MODES`` 同表 ✓），
   写错**当场拒** ✓；``文生视频(T2VA)`` 这种带括号的写法原样保留 ✓（改一个字就对不上模板 ✗）。
3. **未知参数必须报** ✓（上游是 JS 解构，拼错的键被**静默忽略** ✗）：本仓宁可直接 400 ✓ ——
   ``{"secods": 10}`` 这种拼错，静默忽略的结果是「按默认 5 秒跑」，而人以为改成了 10 ✗✗。
4. **时长 1..15 夹紧** ✓（上游 ``Math.min(15, Math.max(1, seconds))`` ✓，缺省 5 ✓）：
   夹紧**同时给一条 warning** ✓（静默改写数值同样是骗人 ✗）。
5. **不静默截断提示词** ✗：``prompt`` 上限 8000 字、``referenceNote`` 上限 4000 字（与上游同数 ✓），
   超了**报错** ✓（上游也是 ``prompt_too_long`` ✓）—— 截掉的那段会让人以为整段都送进去了 ✗。
6. **参考图最多 8 张** ✓（上游 ``slice(0, 8)`` ✓），超出**报错** ✓（静默丢图＝出图少一张参考 ✗）。
7. **模板每次都从盘上重读** ✓：H3 那条要往 ``系统提示词`` **追加**语言指令与输出锁，
   若把模板缓存起来再追加，第二次就叠成两段、第三次三段 ✗（上游用 ``deepClone`` 绕开了这件事 ✓，
   这里用"不缓存"更直白 ✓；四份文件都在 2 KB 上下，重读的代价可以忽略 ✓）。
8. **不提交、不连上游** ✓：本层只产出**可提交的 API 图**（用
   :func:`comfyui_catalog.is_api_format` 自检 ✓），真要跑把结果塞进
   ``POST /api/v1/comfyui/runs`` 的 ``workflow`` ✓（那条路才管 DB / 记账 / 产物 / 归因 ✓）。
   也因此它**离线可用** ✓（不需要 ComfyUI 在线，更不需要显卡 ✓）。

⚠️ 参考图在**本层只是名字** ✓：``image`` / ``images`` 收的是 ComfyUI ``input`` 目录里已有的图名
（或 ``{"name","subfolder"}`` ✓）。编译层**不上传不下载** ✗；要上传走
:func:`app.services.comfyui.stage_reference_uploads` 那条路 ✓。

运行::

    from app.services.comfyui_prompts import catalog, compile_template

    catalog()                                   # 目录：四份模板 + 各自参数 + 所需节点类
    compile_template("h3-enhance", {"prompt": "雨夜的便利店", "seconds": 6})
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..core.config import PROJECT_ROOT
from .comfyui_catalog import is_api_format

__all__ = [
    "BEHAVIOR_KEYS",
    "H3_MODES",
    "MAX_IMAGES",
    "MAX_NOTE",
    "MAX_PROMPT",
    "PROMPT_DIR",
    "PromptTemplateError",
    "SKILL_DIR",
    "TEMPLATES",
    "Template",
    "catalog",
    "compile_template",
    "get_template",
    "h3_skill_assets",
    "image_value",
    "prompt_path",
    "required_nodes",
]

#: 模板目录（四份 JSON + ``skills/`` 文本资产 ✓）
PROMPT_DIR = PROJECT_ROOT / "configs" / "comfyui" / "prompts"

#: H3 提示词写作 skill（自 ``TE`` 插件作者的 MIT 仓库逐字拷入 ✓，随模板一起分发 ✓）
SKILL_DIR = PROMPT_DIR / "skills" / "h3-prompt-writing"

#: 提示词上限（与上游同数 ✓）
MAX_PROMPT = 8000
#: 参考素材说明上限（与上游同数 ✓）
MAX_NOTE = 4000
#: 参考图张数上限（与上游 ``slice(0, 8)`` 同数 ✓）
MAX_IMAGES = 8

#: H3 五个任务类型（**逐字**，少一个字模板里的格就填不进去 ✓）
H3_MODES = (
    "文生视频(T2VA)",
    "首帧图生视频(I2VA)",
    "首尾帧视频(FL2VA)",
    "尾帧图生视频(L2VA)",
    "全参考模式(Reference to Video)",
)

_ZH_LANGUAGE_WORDS = frozenset({"zh", "cn", "chinese", "中文", "简体中文"})
_EN_LANGUAGE_WORDS = frozenset({"en", "english", "英文", "英语"})

_TRUE_WORDS = frozenset({"1", "true", "yes", "on", "是", "开"})
_FALSE_WORDS = frozenset({"0", "false", "no", "off", "否", "关"})

#: 追加在 H3 ``系统提示词`` 末尾的**输出锁**（上游原句 ✓：不锁住的话模型会把"思考过程"当地一段交付 ✗）
_H3_OUTPUT_LOCK = "只输出最终提示词正文；禁止输出思考过程、分析、推理、解释、Markdown 代码块、前后缀或自我说明。"

_H3_LANGUAGE = {
    "zh": ("字段正文全部使用中文（覆盖 skill 默认英文）；台词、旁白、歌词保留原文语言（中文仍为中文）；"
           "官方字段名保留英文。"),
    "en": ("Write every field body in English. Keep dialogue, lyrics, and on-screen visible text in their "
           "original language; Chinese dialogue stays Chinese inside <d>[Chinese]...</d>. "
           "Official field names stay English."),
}

_CAPTION_SYSTEM = {
    "zh": ("你是一位专业的AI图像生成提示词工程师。请详细描述这张图像主体、前景、中景、背景、构图、视觉引导、"
           "光影氛围等细节并创作出具有深度、氛围和艺术感或日常业余设备拍摄的图像提示词。\n\n"
           "要求：中文提示词，不用出现对图像水印的描述，不要出现无关的文字和符号，不需要总结，限制在800字以内"),
    "en": ("You are a professional prompt engineer for AI image generation. Describe the subject, foreground, "
           "middle ground, background, composition, visual guidance, lighting and atmosphere, then write a "
           "detailed prompt with depth and mood. Requirements: English prompt only; do not describe watermarks; "
           "no irrelevant text or symbols; no summary; at most 800 words."),
}

_CAPTION_USER = {
    "zh": "请把这张图片反推成可直接用于 AI 生图的中文提示词。",
    "en": "Write a detailed English prompt that could recreate this image.",
}


class PromptTemplateError(ValueError):
    """模板用不了（**当场抛** ✓ 绝不静默跳过 ✗）。``code`` 供接口层区分 404 / 400 ✓。"""

    def __init__(self, message: str, *, code: str = "prompt_template_failed") -> None:
        super().__init__(message)
        self.code = code


# ══════════════════════════════════════════════════════════════════════════
# 目录：参数怎么落的**唯一声明**（一族一份，填法只有一条公共流程 ✓）
# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Param:
    """一个**落点明确**的参数 ✓（``node`` / ``input`` 都是模板里的原字 ✓）。"""

    key: str            #: 对外键名（英文 ✓ 调用方稳定 ✓）
    label: str          #: 人话（前端表单标题 ✓）
    node: str           #: 落到哪个节点 id ✓
    input: str          #: 落到哪个输入格（中文原名 ✓ 逐字 ✓）
    kind: str = "text"  #: ``text`` | ``int`` | ``enum`` | ``bool``
    required: bool = False
    default: Any = None
    choices: tuple[str, ...] = ()
    max_length: int | None = None
    min_value: int | None = None
    max_value: int | None = None


@dataclass(frozen=True)
class ImagesWiring:
    """参考图怎么挂 ✓：第 i 张 ⇒ ``LoadImage`` 节点 ``9101+i`` ⇒ 连线塞进 ``node`` 的某一格 ✓。"""

    node: str        #: 哪个节点收连线 ✓
    first: str       #: 第 1 张的格名 ✓（H3 / 生图增强都是 ``图片`` ✓）
    nth: str         #: 第 2 张起的格名模板 ✓（``"图片{i}"``，``i`` 从 2 起 ✓ 与上游同口径 ✓）
    label: str       #: 节点标题前缀 ✓（排障时在 ComfyUI 里一眼能认出这几张是参考图 ✓）
    max: int = MAX_IMAGES


@dataclass(frozen=True)
class Template:
    id: str
    title: str
    purpose: str
    file: str
    params: tuple[Param, ...]
    extras: tuple[str, ...] = ()       #: 吃哪些**行为键**（:data:`BEHAVIOR_KEYS` ✓）
    images: ImagesWiring | None = None  #: 不吃参考图 ⇒ ``None`` ✓（传了会拒 ✗）
    post: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], list[str]] | None = None
    notice: str = ""


#: 行为键（不落在某一格上，而是**按上下文算**的几件事 ✓）
BEHAVIOR_KEYS: dict[str, dict[str, Any]] = {
    "language": {
        "label": "输出语言",
        "kind": "enum",
        "choices": ["zh", "en"],
        "default": "zh",
        "hint": "zh ⇒ 中文正文；en ⇒ 英文正文（台词/歌词保留原语言 ✓）",
    },
    "images": {
        "label": "参考图",
        "kind": "images",
        "hint": "ComfyUI input 目录里的图名，或 {name, subfolder}；最多 8 张 ✓（本层不上传不下载 ✓）",
    },
    "instruction": {
        "label": "反推指令",
        "kind": "text",
        "hint": "给了就整段顶掉默认 system prompt ✓（留空＝按 language 用默认那段 ✓）",
    },
}


def _h3_post(graph: dict[str, Any], values: dict[str, Any], ctx: dict[str, Any]) -> list[str]:
    """H3：``输入模式`` 按有没有图算 ✓；``系统提示词`` 追加语言指令 + 输出锁 ✓。"""
    inputs = _slot(graph, "3", "h3-enhance.输入模式")
    inputs["输入模式"] = "图片" if ctx["images"] else "文本"
    base = str(inputs.get("系统提示词") or "").strip()
    tail = f"{_H3_LANGUAGE[ctx['language']]}\n{_H3_OUTPUT_LOCK}"
    inputs["系统提示词"] = f"{base}\n{tail}" if base else tail
    return []


def _image_post(graph: dict[str, Any], values: dict[str, Any], ctx: dict[str, Any]) -> list[str]:
    """生图增强：``任务模式`` / ``输出语言`` 按上下文算 ✓；``增强方式`` 每次钉回本地档 ✓。"""
    inputs = _slot(graph, "1", "image-enhance.任务模式")
    inputs["任务模式"] = "图生图" if ctx["images"] else "文生图"
    inputs["输出语言"] = "英文" if ctx["language"] == "en" else "中文"
    # ⚠️ 模板里的 ``增强方式`` 可被人改成「云端」⇒ 那份格在本地跑必然失败 ✗（它要 api_base_url/api_key ✓）。
    #    上游也是**每次都覆盖**成 ``本地官方PE`` ✓，这里照做 ✓。
    inputs["增强方式"] = "本地官方PE"
    inputs["生成后自动卸载模型"] = True
    return []


def _caption_post(graph: dict[str, Any], values: dict[str, Any], ctx: dict[str, Any]) -> list[str]:
    """Qwen3.5-VL 反推：``system_prompt``（可被 instruction 顶掉）+ ``custom_prompt`` 按语言选 ✓。"""
    language = ctx["language"]
    instruction = str(values.get("instruction") or "").strip()
    inputs = _slot(graph, "3", "image-caption.反推指令")
    inputs["system_prompt"] = instruction or _CAPTION_SYSTEM[language]
    inputs["custom_prompt"] = _CAPTION_USER[language]
    return []


TEMPLATES: tuple[Template, ...] = (
    Template(
        id="h3-enhance",
        title="H3 提示词增强",
        purpose=("把一句话扩写成 MiniMax H3 直接可用的多段式提示词"
                 "（integrated_multimodal_description / overall_soundscape / non_diegetic_music 三段口径 ✓）"),
        file="h3-prompt-enhance.json",
        params=(
            Param("prompt", "原始提示词", "2", "输入提示词", required=True, max_length=MAX_PROMPT),
            Param("mode", "任务类型", "2", "任务类型", kind="enum", choices=H3_MODES, default=H3_MODES[0]),
            Param("seconds", "时长（秒）", "2", "视频时长", kind="int", default=5, min_value=1, max_value=15),
            Param("referenceNote", "参考素材说明", "2", "参考素材说明", max_length=MAX_NOTE, default=""),
        ),
        extras=("language", "images"),
        images=ImagesWiring(node="3", first="图片", nth="图片{i}", label="H3 Ref"),
        post=_h3_post,
        notice="增强方式固定为「本地模板」✓；要「云端增强」需另配 base_url / api_key（本层不代填 ✗）",
    ),
    Template(
        id="image-enhance",
        title="生图提示词增强",
        purpose="把一句话改写成可直接用的文生图 / 图生图提示词（本地官方 PE ✓）",
        file="image-prompt-enhance.json",
        params=(
            Param("prompt", "原始提示词", "1", "输入提示词", required=True, max_length=MAX_PROMPT),
        ),
        extras=("language", "images"),
        images=ImagesWiring(node="1", first="图片", nth="图片{i}", label="Image Ref"),
        post=_image_post,
        notice="有参考图就自动切「图生图」✓；「保留什么、改什么」由提示词正文负责（本层不改写正文 ✗）",
    ),
    Template(
        id="image-caption",
        title="图片反推提示词（Qwen3.5-VL）",
        purpose="把一张图反推成可直接用于生图的中文 / 英文提示词（llama_cpp + Qwen3.5-VL ✓）",
        file="image-caption.json",
        params=(
            Param("image", "图片", "1", "image", kind="image", required=True, max_length=512),
        ),
        extras=("language", "instruction"),
        post=_caption_post,
        notice="``image`` 是 ComfyUI input 目录里的图名 ✓（本层不上传 ✗）；模板 #3 的 images 已连在 #1 上 ✓",
    ),
    Template(
        id="image-caption-florence",
        title="图片反推提示词（Florence2）",
        purpose="用 Florence-2 PromptGen 反推混合描述（更快、更标签化 ✓）",
        file="image-caption-florence.json",
        params=(
            Param("image", "图片", "1", "image", kind="image", required=True, max_length=512),
        ),
        extras=(),
        post=None,
        notice="Florence2 的 task 是固定值「prompt_gen_mixed_caption_plus」✓ ⇒ 不吃 language / instruction（传了会被拒 ✗，不静默忽略 ✓）",
    ),
)


# ══════════════════════════════════════════════════════════════════════════
# 取模板 / 读模板
# ══════════════════════════════════════════════════════════════════════════
def get_template(template_id: str) -> Template:
    """按 id 取模板 ✓（找不到**抛** ✓ —— 静默回落到第一份就是「点了 A 跑了 B」✗）。"""
    wanted = str(template_id or "").strip()
    for tpl in TEMPLATES:
        if tpl.id == wanted:
            return tpl
    raise PromptTemplateError(
        f"没有提示词模板 {wanted!r}（可选：{[tpl.id for tpl in TEMPLATES]}）",
        code="unknown_template",
    )


def prompt_path(template: Template | str) -> Path:
    """模板文件路径 ✓（``PROMPT_DIR`` 走模块属性 ✓ ⇒ 测试可整目录替换 ✓）。"""
    tpl = get_template(template) if isinstance(template, str) else template
    return PROMPT_DIR / tpl.file


def _load(tpl: Template) -> dict[str, Any]:
    """读一份模板并核它是**API 格式** ✓（UI 格式提交不了 ✗）。"""
    path = prompt_path(tpl)
    if not path.is_file():
        raise PromptTemplateError(
            f"模板文件不在盘上：{path}（四份模板应随 configs/comfyui/prompts 一起分发 ✓）",
            code="missing_template",
        )
    try:
        graph = json.loads(path.read_text(encoding="utf-8"))
    except Exception as err:  # noqa: BLE001 - 坏 JSON 与读不到都要说清楚 ✓
        raise PromptTemplateError(f"模板 {tpl.id} 不是合法 JSON：{err}", code="bad_template") from err
    if not is_api_format(graph):
        raise PromptTemplateError(
            f'模板 {tpl.id} 不是 API 格式工作流（要 {{"<节点 id>": {{"class_type": …}}}} ✓）',
            code="bad_template",
        )
    return graph


def _slot(graph: dict[str, Any], node: str, where: str) -> dict[str, Any]:
    """取某节点的 ``inputs`` ✓（节点不在**抛** ✓ —— 模板被改过时不能假装填成功 ✗）。"""
    item = graph.get(node)
    inputs = item.get("inputs") if isinstance(item, dict) else None
    if not isinstance(inputs, dict):
        raise PromptTemplateError(
            f"{where}：模板里没有节点 {node}（模板被改过？现有节点：{sorted(graph)}）",
            code="missing_node",
        )
    return inputs


def _write(graph: dict[str, Any], node: str, key: str, value: Any, where: str) -> None:
    """写一格 ✓（**格不在就抛** ✗ —— 新键对 ComfyUI 等于"没这个输入"，只会报 required input is missing ✗）。"""
    inputs = _slot(graph, node, where)
    if key not in inputs:
        raise PromptTemplateError(
            f"{where}：节点 {node} 没有输入格 {key!r}（模板被改过？现有：{sorted(inputs)}）",
            code="missing_input",
        )
    inputs[key] = value


# ══════════════════════════════════════════════════════════════════════════
# 取值归一（**唯一**判据都在这几个函数里 ✓）
# ══════════════════════════════════════════════════════════════════════════
def image_value(ref: Any) -> str:
    """参考图的「图名」✓：``{"name","subfolder"}`` ⇒ ``subfolder/name``（与上游 ``imageValue`` 同口径 ✓）。"""
    if isinstance(ref, dict):
        name = str(ref.get("name") or "").strip()
        subfolder = str(ref.get("subfolder") or "").strip()
        if not name:
            return ""
        return f"{subfolder}/{name}" if subfolder else name
    return str(ref or "").strip()


def _as_bool(raw: Any) -> bool | None:
    """宽容布尔 ✓（前端表单只会送字符串 ✓）；认不出返回 ``None`` ✓（由调用方报错 ✗ 不猜 ✗）。"""
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    text = str(raw).strip().lower()
    if text in _TRUE_WORDS:
        return True
    if text in _FALSE_WORDS:
        return False
    return None


def _language(values: dict[str, Any]) -> str:
    """``language`` 归一成 ``zh`` / ``en`` ✓（缺省 zh ✓ 与上游 ``normalizePromptLanguage`` 同口径 ✓）。"""
    raw = values.get("language")
    if raw is None or str(raw).strip() == "":
        return "zh"
    text = str(raw).strip().lower()
    if text in _EN_LANGUAGE_WORDS:
        return "en"
    if text in _ZH_LANGUAGE_WORDS:
        return "zh"
    raise PromptTemplateError(f"language 只支持 zh / en（也认 中文 / 英文），收到 {raw!r}", code="bad_value")


def _coerce(param: Param, raw: Any, where: str) -> tuple[Any, str | None]:
    """按 ``kind`` 归一 ✓；返回 ``(值, warning)`` ✓（夹紧要留痕 ✓，越界不静默改 ✗）。"""
    if param.kind == "int":
        try:
            value = int(float(raw))
        except (TypeError, ValueError) as err:
            raise PromptTemplateError(f"{where} 要整数，收到 {raw!r}", code="bad_value") from err
        if param.min_value is not None and value < param.min_value:
            return param.min_value, f"{where} {value} 低于下限 {param.min_value} ⇒ 已夹到 {param.min_value}"
        if param.max_value is not None and value > param.max_value:
            return param.max_value, f"{where} {value} 高于上限 {param.max_value} ⇒ 已夹到 {param.max_value}"
        return value, None
    if param.kind == "bool":
        value = _as_bool(raw)
        if value is None:
            raise PromptTemplateError(f"{where} 要布尔值，收到 {raw!r}", code="bad_value")
        return value, None
    if param.kind == "enum":
        text = str(raw).strip()
        if text not in param.choices:
            raise PromptTemplateError(
                f"{where} 只支持 {list(param.choices)}，收到 {text!r}"
                "（别名不算 ✓：模板里的中文取值要逐字对上 ✓）",
                code="bad_value",
            )
        return text, None
    if param.kind == "image":
        # 单张图也要走 ``image_value`` ✓：``{"name","subfolder"}`` 直接 ``str()`` 会变成
        # ``{'name': 'a.png', …}`` 塞进 LoadImage ✗（ComfyUI 会报「图不存在」，而人以为自己给对了 ✗）
        text = image_value(raw)
        if not text:
            raise PromptTemplateError(f"{where} 是空的（要 ComfyUI input 目录里的图名，或 {{name, subfolder}} ✓）",
                                      code="bad_value")
        if param.max_length is not None and len(text) > param.max_length:
            raise PromptTemplateError(f"{where} {len(text)} 字，超过上限 {param.max_length}", code="too_long")
        return text, None
    text = str(raw if raw is not None else "").strip()
    if param.max_length is not None and len(text) > param.max_length:
        raise PromptTemplateError(
            f"{where} {len(text)} 字，超过上限 {param.max_length}"
            "（不静默截断 ✗：截掉的那段会让人以为整段都送进去了 ✗）",
            code="too_long",
        )
    return text, None


def _images(tpl: Template, values: dict[str, Any]) -> list[str]:
    """收参考图 ✓（张数超限**抛** ✓ 不静默丢 ✗）。"""
    wiring = tpl.images
    if wiring is None:
        if values.get("images"):
            raise PromptTemplateError(f"模板 {tpl.id} 不吃参考图（只接单张 image 或不接图 ✓）", code="bad_value")
        return []
    raw = values.get("images")
    if raw is None:
        return []
    items = list(raw) if isinstance(raw, (list, tuple)) else [raw]
    if len(items) > wiring.max:
        raise PromptTemplateError(
            f"参考图最多 {wiring.max} 张，收到 {len(items)} 张（静默丢图＝出图少一张参考 ✗）",
            code="too_many_images",
        )
    names: list[str] = []
    for index, item in enumerate(items):
        name = image_value(item)
        if not name:
            raise PromptTemplateError(
                f"images[{index}] 是空的（要 ComfyUI input 目录里的图名，或 {{name, subfolder}} ✓）",
                code="bad_value",
            )
        names.append(name)
    return names


def _wire_images(tpl: Template, graph: dict[str, Any], names: list[str]) -> None:
    """把参考图接成 ``LoadImage(9101+i)`` 并连到对应格 ✓（这两格**允许新建** ✓：上游节点把它们声明为可选输入 ✓）。"""
    wiring = tpl.images
    if wiring is None or not names:
        return
    inputs = _slot(graph, wiring.node, f"{tpl.id} 参考图")
    for index, name in enumerate(names):
        node_id = str(9101 + index)
        graph[node_id] = {
            "class_type": "LoadImage",
            "inputs": {"image": name},
            "_meta": {"title": f"{wiring.label} {index + 1}"},
        }
        slot = wiring.first if index == 0 else wiring.nth.format(i=index + 1)
        inputs[slot] = [node_id, 0]


def required_nodes(graph: dict[str, Any]) -> list[str]:
    """这张图要 ComfyUI 有哪些节点类 ✓（拿去与 ``GET /catalog/nodes`` 比，缺哪个当场能看出来 ✓）。"""
    names = {
        str(node.get("class_type"))
        for node in graph.values()
        if isinstance(node, dict) and node.get("class_type")
    }
    return sorted(names)


# ══════════════════════════════════════════════════════════════════════════
# 编译 / 目录
# ══════════════════════════════════════════════════════════════════════════
def compile_template(template_id: str, values: dict[str, Any] | None = None) -> dict[str, Any]:
    """把参数填进模板 ⇒ **一张可直接提交的 API 图** ✓（只编译，不提交 ✓）。

    返回 ``{"template","title","workflow","requiredNodes","images","language","warnings"}`` ✓；
    ``workflow`` 原样塞进 ``POST /api/v1/comfyui/runs`` 的 ``workflow`` 就能跑 ✓。
    """
    tpl = get_template(template_id)
    given = dict(values or {})
    allowed = {param.key for param in tpl.params} | set(tpl.extras)
    unknown = sorted(key for key in given if key not in allowed)
    if unknown:
        raise PromptTemplateError(
            f"模板 {tpl.id} 不认识这些键：{unknown}（可用：{sorted(allowed)}）"
            "—— 拼错的键若被静默忽略，跑出来的还是模板默认值 ✗",
            code="unknown_key",
        )

    images = _images(tpl, given)
    graph = _load(tpl)
    warnings: list[str] = []

    for param in tpl.params:
        where = f"{tpl.id}.{param.key}"
        if param.key in given:
            value, note = _coerce(param, given[param.key], where)
            if note:
                warnings.append(note)
        elif param.required:
            raise PromptTemplateError(f"{where} 必填（{param.label} ✓）", code="missing_value")
        elif param.default is not None:
            value = param.default
        else:
            continue
        _write(graph, param.node, param.input, value, where)

    _wire_images(tpl, graph, images)
    ctx = {"images": images, "language": _language(given)}
    if tpl.post is not None:
        warnings.extend(tpl.post(graph, given, ctx))

    if not is_api_format(graph):
        raise PromptTemplateError(f"模板 {tpl.id} 编译后不是 API 格式（实现有洞 ✗）", code="bad_template")

    return {
        "template": tpl.id,
        "title": tpl.title,
        "workflow": graph,
        "requiredNodes": required_nodes(graph),
        "images": images,
        "language": ctx["language"],
        "warnings": warnings,
    }


def _drift_problems(tpl: Template, graph: dict[str, Any]) -> list[str]:
    """目录阶段的**漂移体检** ✓：声明的每个落点在模板里都还在吗？

    为什么目录也要查（编译已经会查）：前端拿目录去画表单，若某个落点已经不在，
    表单就是**填了也不生效**的假表单 ✗ —— 人填完点「编译」才在最后一步吃 400，
    而这期间他已经在按"能改"的错觉调参数了 ✗。目录阶段点名，能立刻看出模板被谁改坏了 ✓。
    （参考图连线格 ``图片/图片2…`` 不查 ✓：那是可选输入，允许编译时新建 ✓。）
    """
    problems: list[str] = []
    for param in tpl.params:
        item = graph.get(param.node)
        inputs = item.get("inputs") if isinstance(item, dict) else None
        if not isinstance(inputs, dict):
            problems.append(f"参数 {param.key} 指向的节点 {param.node} 不在模板里")
        elif param.input not in inputs:
            problems.append(f"参数 {param.key} 指向的输入格 {param.node}.{param.input} 不在模板里")
    return problems


def _param_brief(param: Param) -> dict[str, Any]:
    return {
        "key": param.key,
        "label": param.label,
        "node": param.node,
        "input": param.input,
        "kind": param.kind,
        "required": param.required,
        "default": param.default,
        "choices": list(param.choices),
        "maxLength": param.max_length,
        "min": param.min_value,
        "max": param.max_value,
    }


def h3_skill_assets() -> dict[str, Any]:
    """H3 提示词 skill 的文本资产在不在 ✓（字节数由盘上实测 ✓ 不写死 ✗）。"""
    files: dict[str, int | None] = {}
    for name in ("SKILL.md", "references/base-en.txt", "references/ref-en.txt"):
        path = SKILL_DIR / name
        files[name] = path.stat().st_size if path.is_file() else None
    return {
        "dir": str(SKILL_DIR),
        "available": all(size is not None for size in files.values()),
        "files": files,
        "consumedBy": (
            "``h3-enhance`` 的云端分支（本仓尚未接云端 chat 通路 ⇒ 目前只作文本资产留存 ✓；"
            "它随 MIT LICENSE 一并拷入 ✓）"
        ),
    }


def catalog() -> dict[str, Any]:
    """四份模板的目录 ✓（**离线可用** ✓：不连上游、不碰显卡 ✓）。

    单份坏掉**不拖垮整个目录** ✓：它在自己那行报 ``error``，其余三份照常可用 ✓
    （一份缺文件就让整个提示词面板 500，是把「局部坏」升级成「全坏」✗）。
    """
    rows: list[dict[str, Any]] = []
    for tpl in TEMPLATES:
        path = prompt_path(tpl)
        row: dict[str, Any] = {
            "id": tpl.id,
            "title": tpl.title,
            "purpose": tpl.purpose,
            "file": tpl.file,
            "exists": path.is_file(),
            "notice": tpl.notice,
            "params": [_param_brief(param) for param in tpl.params],
            "extras": [{"key": key, **BEHAVIOR_KEYS[key]} for key in tpl.extras],
            "images": (
                {"node": tpl.images.node, "first": tpl.images.first,
                 "nth": tpl.images.nth, "max": tpl.images.max}
                if tpl.images is not None else None
            ),
        }
        if row["exists"]:
            try:
                graph = _load(tpl)
            except PromptTemplateError as err:
                row["error"] = str(err)
            else:
                row["nodeCount"] = len(graph)
                row["requiredNodes"] = required_nodes(graph)
                drift = _drift_problems(tpl, graph)
                if drift:
                    row["error"] = "；".join(drift)
        rows.append(row)
    return {"templates": rows, "h3Skill": h3_skill_assets(), "promptDir": str(PROMPT_DIR)}
