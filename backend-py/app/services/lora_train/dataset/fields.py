"""**素材工具的参数体系** ✓ —— 改名 / 转格式 / 自动打标 三个工具的取值口径 ✓。

出处（参考实现 ✓）：``reference/lora/LoRAMaster/dataset_manager/`` 下的
``image_rename_settings.toml`` ✓、``image_convert_settings.toml`` ✓、``caption_settings.toml`` ✓，
以及 ``image_rename.py`` / ``image_convert.py`` / ``AutoCaptioning.py`` 里读这些键的代码 ✓。
形状与 :mod:`app.services.lora_train.options` **完全一致** ✓（同样是
「参数名 / 类型 / 默认值 / 中文标签 / 取值域」提成数据 ✓ ⇒ 后端据此校验 ✓、前端据此渲染 ✓）。

## ⚠️ **四处刻意不照抄** ✓（每一处都写清理由 ✓，不是漏了 ✗）

1. **不搬作者机器上的路径与触发词** ✗：参考 toml 里 ``dataset_path = "E:\\train\\..."`` ✓✗
   与 ``trigger_word = "This is a frxxz_style picture, "`` ✓✗ 都是**那一台机器/那一个项目**的东西 ✓；
   抄进来只会变成"默认值指向不存在的目录、默认拼一句别人的咒语" ✓✗ ⇒ 本仓一律留空 ✓
   （与 ``options.py`` 同一口径 ✓）。
2. **``remove_original`` 默认 ``False``** ✓（**参考实现写死删原图** ✗✗）：
   ``image_convert.py`` 第 126 行 ``remove_original = True`` ✓ —— 用完就把你的素材删了 ✓✗。
   本仓把它**提成开关** ✓ 且**默认不删** ✓：破坏性动作必须显式选择 ✓。
3. **``overwrite`` / ``quality`` / ``backup`` 提成参数** ✓：参考实现三个都写死在代码里
   （``overwrite = True`` / ``quality = 95`` / 备份目录名写死 ✓）⇒ 用户改不了 ✓✗。
4. **量化档提成参数** ✓：参考实现 ``JoyCaptionPredictor(quantization_mode="nf4")`` 是**写死**的 ✓✗；
   本仓提成 ``quantization`` ✓（``nf4`` / ``int8`` / ``bf16`` ✓）—— 显存够就用 bf16 更准 ✓，
   不够就 nf4 ✓，这个差别对出标质量影响很大 ✓，不该由代码替用户定 ✓。

⚠️ ``fp8`` 这个键照抄保留 ✓ 但标 ``used=False`` ✓：参考的 ``caption_settings.toml`` 第 37 行有它 ✓，
界面上也有个开关 ✓，但 ``QUANTIZATION_CONFIGS`` 里**根本没有 fp8** ✓、
而且生成时用的是写死的 ``"nf4"`` ✓ ⇒ 它**当前不生效** ✓（与 ``options`` 里 flux 的
``discrete_flow_shift`` 同一处理 ✓：留着为了读回老配置 ✓，但标签里写明 ✓）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..errors import LoraTrainConfigError
from ..options import (
    KIND_BOOL,
    KIND_FLOAT,
    KIND_INT,
    KIND_PATH_DIR,
    KIND_SELECT,
    KIND_TEXT,
    KIND_TEXTAREA,
)
from .prompts import CAPTION_LENGTHS, CAPTION_TYPE_MAP, EXTRA_OPTION_MAP, QUANTIZATION_MODES

#: 三个素材工具 ✓
TOOL_RENAME = "rename"
TOOL_CONVERT = "convert"
TOOL_CAPTION = "caption"

#: 打标模型能吃的图 ✓（参考实现 ``AutoCaptioning.py`` 第 546 行 ✓；转格式那边多一个 tiff ✓）
CAPTION_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
#: 转格式那边认的输入格式 ✓（参考实现 ``image_convert.py`` 第 119 行 ✓）
CONVERT_INPUT_EXTENSIONS = (".png", ".jpeg", ".jpg", ".bmp", ".tiff", ".webp")


@dataclass(frozen=True)
class Field:
    """一个素材工具参数 ✓（字段名/形状与 :class:`app.services.lora_train.options.Option` 一致 ✓）。"""

    key: str
    kind: str
    default: Any
    label: str
    choices: tuple[str, ...] = ()
    help: str = ""
    #: 是否**真的被用到** ✓ —— ``False`` = 参考实现读了但没用 ✓（本仓保留以便读回老配置 ✓）
    used: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "kind": self.kind, "default": self.default,
            "label": self.label, "choices": list(self.choices), "help": self.help,
            "used": self.used,
        }


@dataclass(frozen=True)
class DatasetToolSpec:
    """一个素材工具的**全部静态口径** ✓。"""

    key: str
    label: str
    description: str
    fields: tuple[Field, ...]
    #: 这个工具会不会**改动磁盘** ✓（只读的扫描不算 ✓）—— 前端据此提醒 ✓
    mutates: bool = True

    def field(self, key: str) -> Field | None:
        return next((f for f in self.fields if f.key == key), None)

    def defaults(self) -> dict[str, Any]:
        return {f.key: f.default for f in self.fields}

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "label": self.label, "description": self.description,
            "mutates": self.mutates, "fields": [f.as_dict() for f in self.fields],
        }


def _toggles() -> tuple[Field, ...]:
    """27 个打标补充开关 ✓ —— 标签直接用 :data:`EXTRA_OPTION_MAP` 的中文要求句 ✓（照抄 ✓）。"""
    return tuple(
        Field(key=key, kind=KIND_BOOL, default=False, label=text, help="勾上就加进打标提示词")
        for key, text in EXTRA_OPTION_MAP.items()
    )


_RENAME = DatasetToolSpec(
    key=TOOL_RENAME,
    label="素材重命名",
    description="把素材按「前缀 + 指定位数序号」批量改名，如 AIJBS0001.jpg",
    fields=(
        Field("dataset_path", KIND_PATH_DIR, "", "素材文件夹", help="要改名的图片所在目录"),
        Field("target_suffix", KIND_TEXT, "jpg", "素材格式", help="只改名这个后缀的文件，如 jpg"),
        Field("target_prefix", KIND_TEXT, "", "命名前缀", help="如 AIJBS ⇒ 命名成 AIJBS0001.jpg"),
        Field("target_num", KIND_TEXT, "4", "命名位数", help="如 4 ⇒ 0001.jpg、0002.jpg"),
        Field("rename_caption", KIND_BOOL, True, "顺带改名同名 txt",
              help="参考实现只改图片、不管 .txt ⇒ 改完名标签就全对不上了；本仓默认一起改"),
    ),
)

_CONVERT = DatasetToolSpec(
    key=TOOL_CONVERT,
    label="图片素材转格式",
    description=f"把 {'/'.join(CONVERT_INPUT_EXTENSIONS)} 统一转成目标格式（可选先备份）",
    fields=(
        Field("dataset_path", KIND_PATH_DIR, "", "素材文件夹", help="要转格式的图片所在目录（会递归子目录）"),
        Field("target_suffix", KIND_TEXT, "jpg", "目标格式", help="如 jpg / png / webp"),
        Field("backup", KIND_BOOL, True, "先备份再转",
              help="备份到**素材目录之外**的同级目录 —— 参考实现的备份建在素材目录里面，"
                   "会把备份自己再抄一遍（无终止递归），本仓挪到同级（见 files.py 说明）"),
        Field("remove_original", KIND_BOOL, False,
              "转完删掉原文件", help="⚠️ 破坏性：参考实现是**默认删**的，本仓默认不删"),
        Field("overwrite", KIND_BOOL, True, "覆盖同名结果", help="关掉则目标已存在时跳过"),
        Field("quality", KIND_INT, 95, "质量", help="JPEG/WebP 这类有损格式的质量（1~100）"),
    ),
)

_CAPTION = DatasetToolSpec(
    key=TOOL_CAPTION,
    label="自动打标 (JoyCaption)",
    description="用 JoyCaption 给每张图生成一句/一段标签，写成与图片同名的 .txt",
    fields=(
        Field("dataset_path", KIND_PATH_DIR, "", "素材文件夹", help="要打标的图片所在目录"),
        Field("caption_type", KIND_SELECT, "Descriptive", "打标类型",
              choices=tuple(CAPTION_TYPE_MAP),
              help="决定标签的风格（描述 / SD 提示词 / Danbooru 标签 …）"),
        Field("caption_length", KIND_SELECT, "long", "打标长度", choices=tuple(CAPTION_LENGTHS),
              help="any = 不限；数字 = 限字数；其余 = 长度短语"),
        Field("user_prompt", KIND_TEXTAREA, "", "自定义提示词",
              help="填了就完全用它（参考实现有这条路但界面上没接出来）", used=True),
        Field("trigger_word", KIND_TEXT, "", "触发词", help="拼到标签最前面，如 aijbs,"),
        Field("filter_word", KIND_TEXT, "", "过滤词", help="生成结果里把这个词删掉"),
        Field("character_name", KIND_TEXT, "Huluwa", "角色称呼",
              help="勾了「必须用角色名称呼」时用它填进提示词"),
        Field("temperature", KIND_FLOAT, 0.6, "温度", help="0 =  greedy（不可复现）；越大越发散"),
        Field("top_p", KIND_FLOAT, 0.9, "top_p", help="核采样"),
        Field("top_k", KIND_INT, 0, "top_k", help="0 = 关"),
        Field("max_new_tokens", KIND_INT, 741, "最大新 token 数", help="标签最长能有多长"),
        Field("quantization", KIND_SELECT, "nf4", "量化档", choices=QUANTIZATION_MODES,
              help="显存够用 bf16 更准，不够用 nf4；int8 居中"),
        Field("device", KIND_TEXT, "cuda", "设备", help="打标模型跑在哪，如 cuda / cuda:1 / cpu"),
        Field("free_vram", KIND_BOOL, True, "开跑前卸掉推理引擎",
              help="推理引擎与本打标模型**抢同一块显存**；默认开跑前把它卸掉（引擎正忙则报错，"
                   "不静默 OOM）。关掉则你自己保证显存够"),
        *_toggles(),
        Field("fp8", KIND_BOOL, False, "fp8（当前不生效）",
              help="参考的 toml 有这个键，但它的量化表里没有 fp8、生成时又写死 nf4 ⇒ 当前不生效",
              used=False),
    ),
)

#: 三个工具 ✓（键即 API 里的 ``tool`` ✓）
TOOLS: dict[str, DatasetToolSpec] = {
    TOOL_RENAME: _RENAME,
    TOOL_CONVERT: _CONVERT,
    TOOL_CAPTION: _CAPTION,
}

DEFAULT_TOOL = TOOL_CAPTION


def tool_keys() -> tuple[str, ...]:
    return tuple(TOOLS)


def spec(tool: str) -> DatasetToolSpec:
    """取一个工具的规格 ✓；不认识 ⇒ **点名报错** ✓（不悄悄退到默认 ✗）。"""
    found = TOOLS.get(str(tool))
    if found is None:
        raise LoraTrainConfigError(f"不认识的素材工具 ✗：{tool!r}；合法值：{sorted(TOOLS)}")
    return found


def merge(tool: str, payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """默认值 + 覆盖 ✓ ⇒ 完整取值 ✓。

    ⚠️ **不认识的键当场报错** ✗（唯一例外：``_`` 开头的注释键 ✓）——
    与 :func:`app.services.lora_train.options.merge` 同一纪律 ✓：
    「悄悄丢掉一个拼错的开关、然后按默认值把素材删了/转了」是本类工具最贵的失败方式 ✓。
    """
    resolved = spec(tool)
    values = resolved.defaults()
    if not payload:
        return values
    unknown = [key for key in payload if key not in values and not str(key).startswith("_")]
    if unknown:
        raise LoraTrainConfigError(
            f"素材工具「{resolved.label}」不认识的参数 ✗：{sorted(unknown)}；"
            f"它认这些：{sorted(values)}"
        )
    for key, raw in payload.items():
        if str(key).startswith("_"):
            continue
        values[key] = _coerce(resolved.field(key), raw)
    return values


def _coerce(field: Field | None, raw: Any) -> Any:
    """单键取值规范化 ✓（TOML 里数字常被写成字符串 ✓ ⇒ 这里统一回来 ✓）。"""
    if field is None:  # pragma: no cover - merge 已经拦掉了 ✓
        return raw
    if field.kind == KIND_BOOL:
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            return raw.strip().lower() in ("1", "true", "yes", "on", "是")
        return bool(raw)
    if field.kind == KIND_INT:
        if isinstance(raw, bool):
            raise LoraTrainConfigError(f"「{field.label}」要整数 ✗，给的是布尔：{raw!r}")
        try:
            return int(str(raw).strip())
        except (TypeError, ValueError) as err:
            raise LoraTrainConfigError(f"「{field.label}」要整数 ✗，给的是 {raw!r}") from err
    if field.kind == KIND_FLOAT:
        if isinstance(raw, bool):
            raise LoraTrainConfigError(f"「{field.label}」要小数 ✗，给的是布尔：{raw!r}")
        try:
            return float(str(raw).strip())
        except (TypeError, ValueError) as err:
            raise LoraTrainConfigError(f"「{field.label}」要小数 ✗，给的是 {raw!r}") from err
    if raw is None:
        return ""
    if isinstance(raw, bool):
        raise LoraTrainConfigError(f"「{field.label}」要文本 ✗，给的是布尔：{raw!r}")
    return str(raw)


def catalog() -> dict[str, Any]:
    """给前端的清单 ✓（三个工具的字段表 ✓ + 打标类型/补充开关的取值域 ✓）。"""
    return {
        "tools": [TOOLS[key].as_dict() for key in TOOLS],
        "defaultTool": DEFAULT_TOOL,
        "captionTypes": list(CAPTION_TYPE_MAP),
        "captionLengths": list(CAPTION_LENGTHS),
        "quantizationModes": list(QUANTIZATION_MODES),
        "extraOptions": list(EXTRA_OPTION_MAP),
    }
