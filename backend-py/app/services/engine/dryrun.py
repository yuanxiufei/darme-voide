"""**干跑后端**（零依赖 ✓）—— 不装 torch、不要权重，也能把**整条管线**真跑一遍。

## 它是什么 / 不是什么（先把边界说清 ✓）

* **是**：:class:`app.services.engine.pipeline.GenerationBackend` 的一个真实现 ✓ ——
  它让 `plan → encode → init → sample → decode → write` **真的按序执行** ✓，
  于是「阶段顺序 / 进度事件 / 取消 / 错误归因 / 算法层集成」这些**编排正确性**都能在
  **没有 GPU、没有权重、没有 torch** 的机器上钉死 ✓。
* **不是**：生成器 ✗ —— 它**不产生任何真图/真视频** ✗，输出一律带 ``synthetic: true`` ✓，
  落盘的是**干跑报告 JSON**（不是 mp4 ✗）。拿它冒充生成结果是不允许的 ✓。

## 张量表示（这是"零依赖"的关键 ✓）

管线只要求张量支持 ``+ − * float``（见 :mod:`app.services.engine.sampler` 的鸭子类型约定 ✓）
⇒ 这里用 :class:`TinyTensor`（纯 Python ``list[float]`` 包装 ✓）当"张量" ✓。
**同一段采样代码**在真机上收 ``torch.Tensor`` ✓、在这里收 ``TinyTensor`` ✓ ⇒ 算法层不需要两套 ✓。

## 确定性（可重放 ✓）

* 文本条件：``sha256(prompt \\x00 negative)`` ⇒ **stable** ✓
  （⚠️ 刻意**不用** Python 内置 ``hash()`` ✗ —— 它对字符串**每进程随机加盐**，
  会让"同 prompt 同种子可复现"这条承诺失效 ✗）。
* 初始噪声：``random.Random(seed)`` ✓（同种子同结果 ✓）。
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import tempfile
from pathlib import Path
from typing import Any

from .pipeline import GenerationPlan, GenerationRequest

__all__ = ["DryRunBackend", "TinyTensor"]


class TinyTensor:
    """够用的"张量"：``list[float]`` + ``+ − * float`` ✓（只为零依赖自检而存在 ✓）。"""

    __slots__ = ("data",)

    def __init__(self, data: list[float]) -> None:
        self.data = [float(value) for value in data]

    def __len__(self) -> int:
        return len(self.data)

    def __add__(self, other: Any) -> "TinyTensor":
        if isinstance(other, TinyTensor):
            return TinyTensor([a + b for a, b in zip(self.data, other.data)])
        return NotImplemented

    def __sub__(self, other: Any) -> "TinyTensor":
        if isinstance(other, TinyTensor):
            return TinyTensor([a - b for a, b in zip(self.data, other.data)])
        return NotImplemented

    def __mul__(self, other: Any) -> "TinyTensor":
        if isinstance(other, TinyTensor):
            return TinyTensor([a * b for a, b in zip(self.data, other.data)])
        return TinyTensor([value * float(other) for value in self.data])

    __rmul__ = __mul__

    def norm(self) -> float:
        return math.sqrt(sum(value * value for value in self.data))

    def distance(self, other: "TinyTensor") -> float:
        """L2 距离 ✓（自检用它验"采样真的收敛了" ✓）。"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(self.data, other.data)))

    def head(self, count: int = 8) -> list[float]:
        return [round(value, 4) for value in self.data[:count]]

    def __repr__(self) -> str:  # pragma: no cover - 只为人看
        return f"TinyTensor(n={len(self.data)}, norm={self.norm():.4f})"


class DryRunBackend:
    """零依赖后端 ✓（``synthetic=True`` ✓ ⇒ 输出会被显式标注为合成 ✓）。"""

    name = "dryrun"
    synthetic = True

    #: "张量"长度（够看出轨迹又不拖慢自检 ✓）
    width = 64

    def _condition(self, request: GenerationRequest) -> TinyTensor:
        """文本 → 条件向量 ✓（sha256 ⇒ 跨进程稳定 ✓）。"""
        digest = hashlib.sha256(
            f"{request.prompt}\x00{request.negative}".encode("utf-8")).digest()
        values = []
        for index in range(self.width):
            byte = digest[index % len(digest)]
            values.append((byte / 255.0) * 2.0 - 1.0)   # → [-1, 1] ✓
        return TinyTensor(values)

    # ── GenerationBackend 协议 ────────────────────────────────────────────
    def encode_text(self, request: GenerationRequest) -> dict[str, TinyTensor]:
        """回 ``{"positive", "negative"}`` ✓ ⇒ 管线才会做 CFG ✓。

        两个条件的取值都来自 ``sha256(prompt \\x00 negative)`` ✓ ⇒ 负提示词不同 ⇒ 条件向量不同 ✓
        （空负提示词也有自己的向量 ✓）⇒ 引导的**数学效果**在干跑里就能验 ✓。
        """
        return {"positive": self._condition(request), "negative": self._negative(request)}

    def _negative(self, request: GenerationRequest) -> TinyTensor:
        """负条件：把 prompt 与 negative **对调**算哈希 ⇒ 与正条件天然不同 ✓（且仍是确定性的 ✓）。"""
        digest = hashlib.sha256(
            f"{request.negative}\x00{request.prompt}".encode("utf-8")).digest()
        return TinyTensor([(digest[index % len(digest)] / 255.0) * 2.0 - 1.0
                           for index in range(self.width)])

    def init_latents(self, plan: GenerationPlan, request: GenerationRequest) -> TinyTensor:
        rng = random.Random(int(request.seed))
        return TinyTensor([rng.uniform(-1.0, 1.0) for _ in range(self.width)])

    def denoise(self, latents: TinyTensor, sigma: float,
                condition: TinyTensor, request: GenerationRequest) -> TinyTensor:
        """**故意做成一阶收敛的模型** ✓：``denoised = (1−g)·cond + g·x``，``g → 0`` 当 ``σ → 0`` ✓。

        这样：① 采样**真的收敛**（末态≈条件 ✓，自检可断言 ✓）；② ``g`` 依赖 σ ⇒
        Euler / Heun / 多阶的轨迹**会不同** ✓ ⇒ 自检能证明「换采样器真的换了算法」✓
        （如果模型与 x 无关，三种采样器结果会一模一样 ✗，那种自检就是假绿 ✗）。
        """
        g = min(0.5, float(sigma) / (float(sigma) + 1.0)) if sigma > 0 else 0.0
        return condition * (1.0 - g) + latents * g

    def condition_first_frame(self, latents: TinyTensor, image_path: str, mask: list[float],
                              plan: GenerationPlan, request: GenerationRequest) -> TinyTensor:
        """首帧条件 ✓（干跑版：把图片路径**确定性地**哈希成一个"图片潜变量"✓，再按掩码混合 ✓）。

        ⚠️ 这里**不读真图** ✗（干跑不碰磁盘/不装解码器 ✓）—— 但**掩码语义是真的** ✓：
        权重大小、单调性、边界都能验 ✓（真后端换成"真编码 + 套到自己的布局"✓ 即可 ✓）。
        """
        digest = hashlib.sha256(str(image_path).encode("utf-8")).digest()
        image_latents = TinyTensor([(digest[index % len(digest)] / 255.0) * 2.0 - 1.0
                                    for index in range(len(latents.data))])
        weights = list(mask) + [0.0] * max(0, len(latents.data) - len(mask))
        blended = [value * (1.0 - weight) + image * weight
                   for value, image, weight in zip(latents.data, image_latents.data, weights)]
        return TinyTensor(blended)

    def decode(self, latents: TinyTensor, plan: GenerationPlan,
               request: GenerationRequest) -> dict[str, Any]:
        """**不产生真帧** ✗ —— 只回报可核对的事实（帧数/音频样本数/数值预览 ✓）。"""
        return {
            "synthetic": True,
            "frames": int(plan.frames),
            "frameWidth": int(plan.width),
            "frameHeight": int(plan.height),
            "audioSamples": int(plan.frames / max(1, plan.fps) * 32000),
            "preview": latents.head(),
            "latentNorm": round(latents.norm(), 6),
        }

    def write(self, outputs: dict[str, Any], plan: GenerationPlan,
              request: GenerationRequest) -> dict[str, Any]:
        """落盘**干跑报告 JSON** ✓（不是视频 ✗ —— 名字与 ``kind`` 都把这点写明 ✓）。"""
        where = Path(request.outputs_dir) if request.outputs_dir else Path(
            tempfile.mkdtemp(prefix="engine_dryrun_"))
        where.mkdir(parents=True, exist_ok=True)
        target = where / f"dryrun_seed{int(request.seed)}.json"
        payload = {
            "synthetic": True,
            "note": "干跑报告：只验证引擎编排，**不是生成的画面** ✗",
            "request": request.to_dict(),
            "plan": plan.to_dict(),
            "decode": outputs,
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        return {
            "synthetic": True,
            "artifacts": [{"kind": "dryrun-report", "path": str(target),
                           "bytes": target.stat().st_size}],
            "videoPath": None,       # ⚠️ 明确为空 ✓：干跑没有视频 ✓
            "primaryPath": str(target),
        }
