"""进程内 transformers 后端：只用 torch + transformers 跑 PaddleOCR-VL。

为什么不走 vLLM / Docker / 独立服务
===================================

``transformers 5.21`` 已把 ``paddleocr_vl`` 收进 ``CONFIG_MAPPING_NAMES``（三个
model_type：``paddleocr_vl`` / ``paddleocr_vl_text`` / ``paddleocr_vl_vision``），
本仓现成的 ``torch 2.14+cu130`` 直接能跑 ``PaddleOCRVLForConditionalGeneration``。
所以既不需要 paddlepaddle，也不需要 vLLM，更不需要容器。

**双保险的形态**：本地 rapidocr 先读（CPU、快、便宜），只有它自己判定 suspect /
reject 的页才进这里重读，两遍结果按 ``ocr_vlm.decide`` 取优。这比"换掉本地引擎"
安全 —— 本地引擎在 97% 的页上是够用的，VLM 只在它不确定的地方花钱。

⚠️ 三条只有真跑才会暴露的约束，都写在这儿，别"优化"掉
-----------------------------------------------------

1. **懒加载**：``ocr_index --run`` 会 spawn 多个 worker。若在 import 期就加载
   0.9B 权重，8 个 worker 会各占一份显存直接 OOM（本机 A5000 22.49 GiB）。
   所以所有权重加载推迟到"第一次真遇到可疑页"，没遇到可疑页的 worker 一分显存
   都不花。
2. **显存门槛**：OCR 是 CPU 推理、worker 数量按 CPU 核数配，GPU 只归 VLM 用。
   加载前先查剩余显存，不够就**当场放弃本次升级**（``Escalator`` 会把它记进审计
   并继续用本地结果），而不是硬装进去把别人的任务挤崩。
3. **输出必须洗标记**：初版的原始输出带 ``<|LOC_123|>`` 这类布局定位 token。
   ``decide`` 按字数比对，标记会把它污染成"两遍差异巨大" ⇒ 每页都被判分歧、
   全部标 doubt。洗掉标记是正确性问题，不是美观问题。1.6 实测已不再输出标记
   （``raw == clean``），清洗**保留**：对初版必需，且对无标记输出是幂等的。

配置全部来自环境变量（进程池是 spawn 的，只认环境变量）::

    OCR_VLM_BACKEND=local            # http（默认）| local
    OCR_VLM_MODEL_PATH=<目录>        # 默认 data/models/PaddleOCR-VL-1.6
    OCR_VLM_MIN_FREE_GIB=3.0         # 加载前要求的显存余量
    OCR_VLM_MAX_PIXELS=             # 可选，限制视觉 token 上限（大页可降速）
"""
from __future__ import annotations

import os
import re
import threading
from pathlib import Path

__all__ = ["LocalEngine", "DEFAULT_PROMPT", "SPECIAL_TOKEN", "strip_layout_tokens",
           "free_gib", "default_model_dir", "MIN_FREE_GIB"]

#: PaddleOCR-VL 训练时的官方指令就是这一条短词；自然语言长指令它也能听，但短指令更稳。
#: 客户端仍可用 ``OCR_VLM_PROMPT`` 覆盖（比对信息时以客户端的为准）。
DEFAULT_PROMPT = "OCR:"

#: 布局/结构标记：``<|LOC_123|>``、``<|line_1|>``、``<|end_of_sentence|>`` 等
SPECIAL_TOKEN = re.compile(r"<\|[^|<>]{1,64}\|>")

#: 加载门槛（GiB）。0.9B bf16 权重约 1.8 GiB，加视觉塔激活与 KV 通常落到 2.5~3.5 GiB；
#: 卡上连 3 GiB 都腾不出来时加载只会半路 OOM，不如当场放弃。
MIN_FREE_GIB = 3.0


def default_model_dir() -> Path:
    """默认权重落点，与 ``ocr_vlm_fetch.default_dir()`` 同一坐标（**1.6**）。"""
    backend_py = Path(__file__).resolve().parents[3]
    return backend_py.parent / "data" / "models" / "PaddleOCR-VL-1.6"


def free_gib() -> float | None:
    """当前卡剩余显存（GiB）。无 CUDA / 查询失败 ⇒ ``None``（表示"不知道"，不是 0）。"""
    try:
        import torch  # noqa: PLC0415 —— 只在真要判显存时才 import
    except Exception:  # noqa: BLE001
        return None
    try:
        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return round(free / 2 ** 30, 2)
    except Exception:  # noqa: BLE001
        return None


def strip_layout_tokens(text: str) -> str:
    """去掉 ``<|...|>`` 标记并规整空白（**不改动任何正文用字**）。"""
    if not text:
        return ""
    cleaned = SPECIAL_TOKEN.sub("", text)
    #: 标记删掉后常留下孤立的空行；收成单个空行，保住段落边界
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


class LocalEngine:
    """一个 worker 私有的 VLM 实例。**构造不加载权重**，``recognize`` 时才加载。"""

    def __init__(self, model_path: str | Path | None = None, *, device: str = "",
                 max_new_tokens: int = 8192, min_free_gib: float | None = None) -> None:
        self.model_path = Path(model_path) if model_path else default_model_dir()
        env_device = (os.environ.get("OCR_VLM_DEVICE") or "").strip()
        self.device = (device or env_device or "cuda").strip()
        self.max_new_tokens = int(max_new_tokens)
        self.min_free_gib = MIN_FREE_GIB if min_free_gib is None else float(min_free_gib)
        self._model = None
        self._processor = None
        self._torch = None
        self._lock = threading.Lock()
        self._error: str | None = None

    # ---- 状态 -------------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def error(self) -> str | None:
        return self._error

    def describe(self) -> str:
        if self._error:
            return f"VLM（进程内）：不可用 —— {self._error}"
        state = "已加载" if self.loaded else "待首次可疑页加载"
        target = self.device if self.device == "cpu" else f"{self.device}（门槛 {self.min_free_gib} GiB）"
        return f"VLM（进程内）：{self.model_path.as_posix()} · {state} · {target}"

    # ---- 加载 -------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        """按需加载。失败一次就记下来，后续调用直接抛同一句人话（不反复撞 OOM）。"""
        if self._model is not None:
            return
        if self._error:
            raise RuntimeError(self._error)

        if not (self.model_path / "config.json").is_file():
            self._error = (f"权重不在盘上（{self.model_path.as_posix()}）：先跑 "
                           f"python -X utf8 app/scripts/corpus/ocr_vlm_fetch.py --fetch")
            raise RuntimeError(self._error)

        if self.device != "cpu":
            free = free_gib()
            if free is not None and free < self.min_free_gib:
                #: 不算永久错误：下一个 worker 可能腾得出显存，但**本次**不装
                raise RuntimeError(
                    f"显存不足（剩 {free} GiB < 门槛 {self.min_free_gib} GiB），本次跳过 VLM 升级")

        try:
            import torch  # noqa: PLC0415
            from transformers import AutoProcessor, PaddleOCRVLForConditionalGeneration  # noqa: PLC0415

            dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
            model = PaddleOCRVLForConditionalGeneration.from_pretrained(
                str(self.model_path), dtype=dtype, low_cpu_mem_usage=True)
            model.eval()
            model.to(self.device)
            processor_kwargs: dict = {}
            max_pixels = (os.environ.get("OCR_VLM_MAX_PIXELS") or "").strip()
            if max_pixels.isdigit():
                #: 视觉 token 上限直接决定显存与延迟；大页（整幅 2048 宽）可以压一档
                processor_kwargs["max_pixels"] = int(max_pixels)
            processor = AutoProcessor.from_pretrained(str(self.model_path), **processor_kwargs)
        except Exception as exc:  # noqa: BLE001 —— 加载失败要留成可读状态
            self._error = f"加载失败：{type(exc).__name__}: {exc}"
            raise RuntimeError(self._error) from exc

        self._torch, self._model, self._processor = torch, model, processor

    # ---- 推理 -------------------------------------------------------------
    def recognize(self, image, prompt: str = "") -> str:
        """一张页图（numpy RGB 或 PIL）→ 转录文本。失败抛 ``RuntimeError``。"""
        with self._lock:
            self._ensure_loaded()
            try:
                from PIL import Image  # noqa: PLC0415

                if not isinstance(image, Image.Image):
                    image = Image.fromarray(image)
                image = image.convert("RGB")
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt or DEFAULT_PROMPT},
                ]}]
                text = self._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                inputs = self._processor(text=[text], images=[image], return_tensors="pt")
                if "input_ids" not in inputs:
                    raise RuntimeError("processor 没给出 input_ids（chat template 可能不认 image 段）")
                inputs = inputs.to(self.device)
                with self._torch.inference_mode():
                    output = self._model.generate(
                        **inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
                new_tokens = output[0][inputs["input_ids"].shape[1]:]
                answer = self._processor.batch_decode([new_tokens], skip_special_tokens=True)[0]
            except RuntimeError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc
        return strip_layout_tokens(answer)
