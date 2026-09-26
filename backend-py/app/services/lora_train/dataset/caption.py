"""**JoyCaption 自动打标** ✓ —— 模型封装 + 批量打标（LoRAMaster 移植 ✓）。

出处（参考实现 ✓）：``reference/lora/LoRAMaster/dataset_manager/AutoCaptioning.py``

* 量化表 :data:`QUANTIZATION_CONFIGS` —— 第 100~111 行 ✓（``nf4`` 带
  ``llm_int8_skip_modules=["vision_tower", "multi_modal_projector"]`` ✓，见第 265 行 ✓）；
* 模型封装 ``JoyCaptionPredictor`` —— 第 247~298 行 ✓（``device_map="auto"`` ✓、
  ``pixel_values`` 转 ``bfloat16`` ✓、``temperature > 0`` 才采样 ✓）；
* 批量循环 ``run_caption()`` —— 第 377~452 行 ✓（拼提示词 ✓、``trigger_word`` 前置 ✓、
  ``filter_word`` 替换掉 ✓、逐张写同名 ``.txt`` ✓）；
* 只用触发词打标 ``run_trigger_caption()`` —— 第 453~466 行 ✓；
* 事后过滤标签 ``filter_caption_word()`` —— 第 468~486 行 ✓。

## ⚠️ **改了参考实现三处** ✓

1. **打标失败不再"原地重试一次"** ✗（第 421~441 行 ✓）：参考把**同一个调用**
   原封不动再跑一遍 ✓✗ —— 失败原因（显存不够 ✓、图坏了 ✓）一个都没变 ✓，
   所以第二次必然还失败 ✓，只是白等一次 + 日志里多一段重复堆栈 ✓。
   本仓改成**记下来、跳过、继续** ✓，最后统一汇报失败清单 ✓（一次失败不该中断整批 ✓，
   但**也不能当作没发生** ✗）。
2. **加载时接上 CPU 线程预算** ✓（:func:`app.core.cpu_budget.apply_torch` ✓）：
   参考没有这一步 ✗ ⇒ ``transformers`` + ``torch`` 会按逻辑核数开满 ✓✗。见 ``runner.py`` 模块头第 3 条 ✓。
3. **``device`` 真能指定** ✓：参考把 ``device="cuda"`` 写成默认参数 ✓、
   同时用 ``device_map="auto"`` ✓✗ —— 两者在多卡机器上可能**不一致** ✓
   （auto 放到 cuda:0 ✓，输入却往 cuda:1 上搬 ✓✗ ⇒ 报设备不匹配 ✓）。
   本仓：给了明确的 ``cuda:N`` ⇒ ``device_map={"": "cuda:N"}`` ✓；只给 ``"cuda"`` / ``"cpu"`` ⇒ 用 ``"auto"`` ✓。

⚠️ **量化与设备的硬约束**（不是本仓加的 ✗，是 ``bitsandbytes`` 的 ✓）：
``nf4`` / ``int8`` 都**必须在 CUDA 上** ✓ ⇒ ``device="cpu"`` 配量化 ⇒ 本仓**当场报错点名** ✓
（而不是让它去抛一句看不出所以然的 CUDA 错 ✓✗）。
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from app.core import cpu_budget

from ..errors import LoraTrainConfigError
from . import files
from .prompts import (
    DEFAULT_CAPTION_MODEL,
    EXTRA_OPTION_MAP,
    QUANTIZATION_MODES,
    SYSTEM_PROMPT,
    build_prompt,
)

#: 收到一条进展就调一次 ✓（不许抛 ✗）
Emit = Callable[[str], None]
#: 该不该停 ✓（运行器每张图问一次 ✓）
ShouldStop = Callable[[], bool]


def _noop(_message: str) -> None:
    """不发进展时的默认实现 ✓。"""
    return None


@dataclass(frozen=True)
class CaptionRequest:
    """一次打标的**全部取值** ✓（从 ``fields.merge`` 的结果里抽出来 ✓）。"""

    dataset: Path
    caption_type: str
    caption_length: str
    user_prompt: str
    trigger_word: str
    filter_word: str
    character_name: str
    temperature: float
    top_p: float
    top_k: int
    max_new_tokens: int
    quantization: str
    device: str
    extra_options: tuple[str, ...]

    @classmethod
    def from_values(cls, values: Mapping[str, Any]) -> "CaptionRequest":
        """从参数表抽 ✓（``dataset_path`` 换成验过的 :class:`Path` ✓）。"""
        dataset = files.require_directory(values.get("dataset_path", ""))
        extra = tuple(key for key in EXTRA_OPTION_MAP if bool(values.get(key)))
        quantization = str(values.get("quantization", "nf4")).strip().lower()
        if quantization not in QUANTIZATION_MODES:
            raise LoraTrainConfigError(
                f"不认识的量化档 ✗：{quantization!r}；合法值：{list(QUANTIZATION_MODES)}"
            )
        device = str(values.get("device", "cuda")).strip() or "cuda"
        if quantization != "bf16" and not device.startswith("cuda"):
            raise LoraTrainConfigError(
                f"量化档「{quantization}」只能在 CUDA 上用 ✗，但设备写的是 {device!r} ✓；"
                "⇒ 要么把设备改成 cuda ✓，要么把量化档改成 bf16 ✓"
            )
        return cls(
            dataset=dataset,
            caption_type=str(values.get("caption_type", "Descriptive")),
            caption_length=str(values.get("caption_length", "long")),
            user_prompt=str(values.get("user_prompt", "") or ""),
            trigger_word=str(values.get("trigger_word", "") or ""),
            filter_word=str(values.get("filter_word", "") or ""),
            character_name=str(values.get("character_name", "Huluwa") or "Huluwa"),
            temperature=float(values.get("temperature", 0.6)),
            top_p=float(values.get("top_p", 0.9)),
            top_k=int(values.get("top_k", 0)),
            max_new_tokens=int(values.get("max_new_tokens", 741)),
            quantization=quantization,
            device=device,
            extra_options=extra,
        )

    def prompt(self) -> str:
        """拼好的**用户提示词** ✓（打标全程只用这一条 ✓）。"""
        return build_prompt(
            self.caption_type,
            self.caption_length,
            list(self.extra_options),
            user_prompt=self.user_prompt,
            character_name=self.character_name,
        )


def quantization_config(quantization: str) -> dict[str, Any]:
    """量化表 ✓（照抄参考第 100~111 行 ✓，但把 ``torch.bfloat16`` **惰性**取 ✓）。"""
    if quantization not in QUANTIZATION_MODES:  # pragma: no cover - 上层已拦 ✓
        raise LoraTrainConfigError(f"不认识的量化档 ✗：{quantization!r}")
    if quantization == "bf16":
        return {}
    torch = _torch()
    if quantization == "nf4":
        return {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
            "bnb_4bit_use_double_quant": True,
        }
    return {"load_in_8bit": True}


def _torch():  # pragma: no cover - 只在真跑的时候用 ✓
    try:
        import torch  # noqa: PLC0415
        return torch
    except ImportError as err:
        raise LoraTrainConfigError(
            f"打标需要 torch ✗ —— 当前解释器里 import torch 失败 ✓：{err}\n"
            "⇒ 装到**跑后端的那个**解释器里 ✓"
        ) from err


def _transformers():  # pragma: no cover - 只在真跑的时候用 ✓
    try:
        from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration
        return AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration
    except ImportError as err:
        raise LoraTrainConfigError(
            f"打标需要 transformers ✗ —— 当前解释器里 import 失败 ✓：{err}\n"
            "⇒ 装到**跑后端的那个**解释器里 ✓（JoyCaption 是 Llava 架构 ✓）"
        ) from err


class CaptionEngine:
    """**进程内单例**的 JoyCaption 封装 ✓（懒加载 ✓、可卸 ✓、加载过就在 ✓）。

    ⚠️ 与 :mod:`app.services.engine` 的**同一份显存** ✓✗：
    推理引擎占着 VRAM 时再加载 8B 打标模型，很可能 OOM ✓。
    本仓**不替调用方决定** ✗（那属于编排策略 ✓）—— 但把事实摆在
    :meth:`loaded` / :meth:`describe` 里 ✓，路由层据此先卸引擎 ✓（见 ``routers/lora_train.py`` ✓）。
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._model: Any = None
        self._processor: Any = None
        self._device: str | None = None
        self._quantization: str | None = None
        self._model_path: str | None = None

    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._model is not None

    def describe(self) -> dict[str, Any]:
        with self._lock:
            return {
                "loaded": self._model is not None,
                "model": self._model_path,
                "device": self._device,
                "quantization": self._quantization,
                "defaultModel": DEFAULT_CAPTION_MODEL,
                "modelEnvVar": "VOIDE_CAPTION_MODEL",
                "cudaVisibleDevices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            }

    def unload(self) -> bool:
        """把打标模型卸掉并清显存 ✓ ⇒ 之前是否真加载着 ✓。"""
        with self._lock:
            if self._model is None:
                return False
            self._model = None
            self._processor = None
            self._device = None
            self._quantization = None
        try:  # pragma: no cover - 真跑才走 ✓
            torch = _torch()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - 清理失败不该把卸载判成失败 ✓（模型引用已经放了 ✓）
            pass
        return True

    def ensure(self, *, quantization: str, device: str) -> None:
        """按需加载 ✓（**已经加载了就复用** ✓ —— 换量化档/设备才重载 ✓）。"""
        with self._lock:
            if (self._model is not None and self._quantization == quantization
                    and self._device == device):
                return
            if self._model is not None:
                # 参数变了 ⇒ 先卸再装 ✓（否则会把两份权重都扛在显存里 ✓✗）
                self._release_locked()
            model_path = os.environ.get("VOIDE_CAPTION_MODEL", "").strip() or DEFAULT_CAPTION_MODEL
            self._model_path = model_path
            try:
                cpu_budget.apply_torch()
            except cpu_budget.CpuBudgetError as err:
                raise LoraTrainConfigError(f"CPU 线程预算不可用 ✗：{err}") from err
            AutoProcessor, BitsAndBytesConfig, LlavaModel = _transformers()
            torch = _torch()
            processor = AutoProcessor.from_pretrained(model_path)
            if quantization == "bf16":
                model = LlavaModel.from_pretrained(
                    model_path, torch_dtype=torch.bfloat16, device_map=_device_map(device),
                )
            else:
                config = BitsAndBytesConfig(
                    **quantization_config(quantization),
                    llm_int8_skip_modules=["vision_tower", "multi_modal_projector"],
                )
                model = LlavaModel.from_pretrained(
                    model_path, torch_dtype="auto", device_map=_device_map(device),
                    quantization_config=config,
                )
            model.eval()
            self._processor = processor
            self._model = model
            self._device = device
            self._quantization = quantization

    def _release_locked(self) -> None:
        self._model = None
        self._processor = None
        self._device = None
        self._quantization = None

    def caption(self, image_path: Path, prompt: str, *, temperature: float, top_p: float,
                top_k: int, max_new_tokens: int) -> str:
        """给一张图出一句标签 ✓（**不写盘** ✗ —— 写盘由调用方决定 ✓）。"""
        with self._lock:
            if self._model is None or self._processor is None:
                raise LoraTrainConfigError("打标模型还没加载 ✗ ⇒ 先调 ensure() ✓")
            model, processor, device = self._model, self._processor, self._device or "cuda"
        Image = files.require_pillow()
        torch = _torch()
        with Image.open(image_path) as raw:
            image = raw.convert("RGB")
        convo = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt.strip()},
        ]
        convo_string = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[convo_string], images=[image], return_tensors="pt").to(device)
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        with torch.inference_mode():
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=bool(float(temperature) > 0),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=None if int(top_k) == 0 else int(top_k),
                use_cache=True,
            )[0]
        generate_ids = generate_ids[inputs["input_ids"].shape[1]:]
        caption = processor.tokenizer.decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return caption.strip()


def _device_map(device: str) -> Any:
    """设备 → ``device_map`` ✓（见模块头第 3 条 ✓）。"""
    text = device.strip()
    if text.startswith("cuda:") and text[5:].isdigit():
        return {"": text}
    return "auto"


#: 进程内单例 ✓（和 ``runtime.lora_train_runtime`` 同一形状 ✓）
caption_engine = CaptionEngine()


# ---------------------------------------------------------------------------
# 批量动作 ✓
# ---------------------------------------------------------------------------


def compose_caption(caption: str, *, trigger_word: str, filter_word: str) -> str:
    """把模型输出加工成**落盘的那一句** ✓（照抄参考第 414~417 行 ✓）。"""
    text = trigger_word + caption.strip()
    if filter_word:
        text = text.replace(filter_word, "")
    return text


def run_caption(request: CaptionRequest, *, emit: Emit | None = None,
                should_stop: ShouldStop | None = None) -> dict[str, Any]:
    """**批量打标** ✓ ⇒ 事实 ✓（``images/labeled/failed/failures`` ✓）。

    ⚠️ 三件事的**顺序**是刻意的 ✓：先把整批图扫出来 ✓ ⇒ 再加载模型 ✓ ⇒ 再逐张跑 ✓。
    扫描在前面 ⇒ "目录里一张图都没有"会**在下载 8B 权重之前**就报出来 ✓✗。
    """
    log = emit or _noop
    stop = should_stop or (lambda: False)

    images = files.list_images(request.dataset)
    if not images:
        raise LoraTrainConfigError(
            f"素材目录里没有可打标的图 ✗：{request.dataset}\n"
            f"⇒ 认这些后缀：{list(files.CAPTION_IMAGE_EXTENSIONS)} ✓"
        )
    prompt = request.prompt()
    log(f"素材 {len(images)} 张 ✓；打标提示词：{prompt}")

    log(f"加载打标模型（{request.quantization} / {request.device} ✓），"
        "第一次要下载权重、可能要很久 ✓……")
    caption_engine.ensure(quantization=request.quantization, device=request.device)
    log(f"模型就绪 ✓：{caption_engine.describe().get('model')}")

    labeled = 0
    failures: list[dict[str, str]] = []
    for index, image in enumerate(images, start=1):
        if stop():
            log(f"已在第 {index}/{len(images)} 张停下 ✓")
            break
        try:
            raw = caption_engine.caption(
                image, prompt,
                temperature=request.temperature, top_p=request.top_p,
                top_k=request.top_k, max_new_tokens=request.max_new_tokens,
            )
            text = compose_caption(raw, trigger_word=request.trigger_word,
                                   filter_word=request.filter_word)
            files.write_caption(image, text)
            labeled += 1
            log(f"[{index}/{len(images)}] {image.name} → {text}")
        except Exception as err:  # noqa: BLE001 - 一张失败不该中断整批 ✓（但要记下来 ✓）
            failures.append({"path": str(image), "reason": f"{type(err).__name__}: {err}"})
            log(f"[{index}/{len(images)}] ✗ 失败：{image.name} —— {type(err).__name__}: {err}")
    log(f"----- 打标完成 ----- 成功 {labeled} ✓、失败 {len(failures)} ✓、共 {len(images)} ✓")
    return {"images": len(images), "labeled": labeled, "failed": len(failures),
            "failures": failures, "prompt": prompt}


def run_trigger_caption(dataset: Path, trigger_word: str, *, emit: Emit | None = None,
                        should_stop: ShouldStop | None = None) -> dict[str, Any]:
    """**只用触发词打标** ✓（照抄参考 ``run_trigger_caption`` 第 453~466 行 ✓）。"""
    log = emit or _noop
    stop = should_stop or (lambda: False)
    images = files.list_images(dataset)
    if not images:
        raise LoraTrainConfigError(f"素材目录里没有可打标的图 ✗：{dataset}")
    if not trigger_word.strip():
        raise LoraTrainConfigError("触发词是空的 ✗ —— 这个动作就是「把触发词写进去」 ✓")
    written = 0
    for index, image in enumerate(images, start=1):
        if stop():
            log(f"已在第 {index}/{len(images)} 张停下 ✓")
            break
        files.write_caption(image, trigger_word)
        written += 1
        log(f"[{index}/{len(images)}] {image.name} -> {trigger_word}")
    log(f"----- 完成 ----- 写了 {written} 个标签 ✓")
    return {"images": len(images), "written": written, "triggerWord": trigger_word}


def run_filter_caption(dataset: Path, filter_word: str, *, emit: Emit | None = None,
                       should_stop: ShouldStop | None = None) -> dict[str, Any]:
    """**事后过滤标签** ✓（照抄参考 ``filter_caption_word`` 第 468~486 行 ✓）。

    ⚠️ 与参考的差别 ✓：参考走的是**内存里那份已加载的** ``dataset.caption`` ✓✗ ⇒
    它必须先在界面上点过「加载训练素材」才有效 ✓，而且内存与磁盘不同步时改的是旧值 ✓✗。
    本仓**直接读盘上那个 ``.txt``** ✓ ⇒ 与谁先点谁后点无关 ✓（幂等 ✓）。
    """
    log = emit or _noop
    stop = should_stop or (lambda: False)
    if not filter_word:
        raise LoraTrainConfigError("过滤词是空的 ✗ —— 这个动作就是「把过滤词删掉」 ✓")
    images = files.list_images(dataset)
    if not images:
        raise LoraTrainConfigError(f"素材目录里没有可处理的图 ✗：{dataset}")
    changed = scanned = 0
    for index, image in enumerate(images, start=1):
        if stop():
            log(f"已在第 {index}/{len(images)} 张停下 ✓")
            break
        scanned += 1
        current = files.read_caption(image)
        if not current:
            continue
        updated = current.replace(filter_word, "")
        if updated != current:
            files.write_caption(image, updated)
            changed += 1
        log(f"[{index}/{len(images)}] {image.name} -> {updated}")
    log(f"----- 完成 ----- 看了 {scanned} 个标签 ✓、改掉 {changed} 个 ✓")
    return {"images": len(images), "scanned": scanned, "changed": changed, "filterWord": filter_word}
