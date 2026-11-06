"""ComfyUI **模型名相似匹配**（``model-match`` 语义 ✓，自研重写 ✓）。

要解决的问题：工作流里写的是**作者当初用的那个文件名**
（``z_image_turbo_bf16.safetensors``），而本机盘上装的常常是**同一个模型的另一档**
（``unet/zimage/z_image_turbo_fp8.safetensors``、``…_int8_convrot.gguf``…）。
精确比对 ⇒ **误报「缺模型」** ✗；乱模糊 ⇒ **把 VAE 当底模、把 Qwen 当 Z-Image** ✗✗。
本模块给出「**红线拦掉 + 相似度排序**」的口径。

⚠️ 口径来源（2026-11-02 逐条核对 ✓）：上游 ``ComfyUI-Local-Infinite-Canvas``（MIT ✓）
的 ``comfyui-gateway/src/model-match.js`` ⇒ 只搬**语义与阈值**，代码自己写 ✗。

## 七条红线（**任一命中即 0 分**，不是「扣分」✗）

1. ⭐ **格式是红线** ✓：两边都得有模型扩展名，且**必须相同** ——
   ``.safetensors`` ↔ ``.gguf`` 一律不算同一件（加载器不同，顶替会直接炸 ✗）。
   注意 ``.engine``（TensorRT）**不在**可模糊集合里 ✓（它认不出来源，不许猜）。
2. ⭐ **角色必须同集** ✓：``vae``/``clip``/``te``/``lora``/``controlnet``/``upscale``…
   一边有一边没有就不匹配 ⇒ 防止「拿 VAE 顶底模」✗（这是最容易出事的一条）。
3. ⭐ **语义限定词必须同集** ✓：``audio``/``video``/``vision``/``image`` ——
   H3 的**音频 VAE** 与**视频 VAE** 不能互顶 ✗。
4. ⭐ **代次** ✓：Qwen Image **2.1** 的 VAE 不许拿旧 Qwen/Wan 视频 VAE 顶替
   （判据是 ``qwen + image + vae + 2 + 1`` 五个 token 同现 ✓）；同代量化差异**仍允许** ✓。
5. ⭐ **规模不跨** ✓：``4B`` / ``8B``、``9B`` / ``27B`` 不是同一件。
   只有**两边都写了**参数量时才硬拦 ✓（一边没写就放行 —— 兼容不写规模的文件名 ✓）。
6. ⭐ **训练规格不互顶** ✓：``turbo`` ⇄ 非 turbo、``raw`` ⇄ 非 raw 都拦；
   而 ``fp8``/``bf16``/``int8`` 这类**量化**差异**允许**匹配 ✓（这正是要认出来的情形 ✓）。
7. ⭐ **家族不互顶** ✓：Qwen / Z-Image / Krea2 / MOSS / Wan / Flux… 各自为界，
   去尾部数字归一后**无交集即拦** ✓。

## 两处「宁可不选，也不乱选」

* 相似度 = ``0.65 × Dice(压缩串二元文法) + 0.35 × Jaccard(token 集合)`` ✓，
  家族 token 同现再 ``+0.08`` ✓；低于 **0.5** 不选 ✓。
* ⭐ **歧义时不自动选** ✓：最高分与第二名差 < **0.03** ⇒ 返回 ``ambiguous`` 与候选，
  **交给用户确认** —— 自动选错会把整条生成引到错模型上，代价远大于让用户点一下 ✗。

运行::

    from app.services.comfyui_match import find_similar_model
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, NamedTuple

__all__ = [
    "FUZZY_MODEL_EXTENSIONS",
    "MODEL_EXTENSIONS",
    "ModelNameParts",
    "find_similar_model",
    "model_name_similarity",
    "parse_model_name",
]

#: 认它是模型文件的扩展名（**判定用**）
MODEL_EXTENSIONS = (".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx", ".engine")

#: 参与**模糊匹配**的扩展名（``.engine`` 除外 ✓ —— TensorRT 引擎认不出来源，不许猜 ✗）
FUZZY_MODEL_EXTENSIONS = frozenset({".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".bin", ".gguf"})

_EXT_RE = re.compile(r"\.(safetensors|sft|ckpt|pt|pth|bin|gguf|onnx|engine)$", re.IGNORECASE)
_SPLIT_RE = re.compile(r"[^a-z0-9\u4e00-\u9fa5]+")
_SIZE_RE = re.compile(r"^\d+(?:\.\d+)?b$", re.IGNORECASE)

#: 模型家族（去尾部数字后归一 ✓，如 ``qwen2``/``qwen3`` ⇒ ``qwen`` ✓）
FAMILY_TOKENS = frozenset(
    {
        "moss", "krea2", "krea", "qwen", "qwen2", "qwen3", "h3", "minimax", "seedvr", "seedvr2",
        "flux", "wan", "zimage", "ltx", "ltxv", "ltx2", "hidream", "chroma", "sd3", "sdxl",
        "pony", "illustrious", "noob", "pixart", "kolors", "cascade", "hunyuan", "mochi",
    }
)

#: 角色词（「这是干什么用的」）—— 一边有一边没有 ⇒ 不匹配 ✗
ROLE_TOKENS = frozenset(
    {"vae", "clip", "text", "encoder", "te", "lora", "lycoris", "controlnet", "ipadapter", "upscale", "esrgan", "embedding"}
)

#: 语义限定词（模态）—— 音频 VAE 不能顶视频 VAE ✗
QUALIFIER_TOKENS = frozenset({"audio", "video", "image", "vision"})

#: 低于这个分不认 ✓
SCORE_THRESHOLD = 0.5
#: 最高分与第二名差不到这个值 ⇒ 判歧义，不自动选 ✓
AMBIGUOUS_MARGIN = 0.03


class ModelNameParts(NamedTuple):
    """一个模型文件名的拆解结果 ✓。

    ``tokens`` 是**小写切分**后的词（数字会独立成词，所以 ``2.1`` 会拆成 ``2`` 与 ``1`` ✓，
    这正是判「Qwen Image 2.1 VAE」的依据 ✓）；``compact`` 是 tokens 直接拼接（供 Dice 用 ✓）。
    """

    base: str
    ext: str
    stem: str
    tokens: tuple[str, ...]
    compact: str


class SimilarityResult(NamedTuple):
    """匹配结果：``ok`` 为真才可用；``reason`` 是被红线拦下的**原因码** ✓。"""

    ok: bool
    score: float
    reason: str = ""


def parse_model_name(value: Any) -> ModelNameParts:
    """拆文件名 ✓（统一正斜杠、取最后一段、切扩展名与 tokens ✓）。"""
    normalized = str(value or "").replace("\\", "/")
    base = normalized.rsplit("/", 1)[-1]
    match = _EXT_RE.search(base)
    ext = match.group(0).lower() if match else ""
    stem = base[: -len(ext)] if ext else base
    tokens = tuple(token for token in _SPLIT_RE.split(stem.lower()) if token)
    return ModelNameParts(base=base, ext=ext, stem=stem, tokens=tokens, compact="".join(tokens))


def _bigrams(text: str) -> Counter:
    if len(text) < 2:
        return Counter(text)
    return Counter(text[index : index + 2] for index in range(len(text) - 1))


def _dice(a: str, b: str) -> float:
    left, right = _bigrams(a), _bigrams(b)
    if not left or not right:
        return 0.0
    overlap = sum(min(count, right[gram]) for gram, count in left.items())
    return (2 * overlap) / (sum(left.values()) + sum(right.values()))


def _jaccard(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    left, right = set(a), set(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _same_set(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    return set(a) == set(b)


def _pick(tokens: tuple[str, ...], table: frozenset[str]) -> tuple[str, ...]:
    return tuple(token for token in tokens if token in table)


def _is_qwen_image_21_vae(tokens: tuple[str, ...]) -> bool:
    return all(name in tokens for name in ("qwen", "image", "vae", "2", "1"))


def _family_roots(tokens: tuple[str, ...], compact: str) -> tuple[str, ...]:
    roots = {
        family.rstrip("0123456789") or family
        for family in FAMILY_TOKENS
        if any(token == family or token.startswith(family) for token in tokens) or family in compact
    }
    return tuple(sorted(roots))


def model_name_similarity(required: Any, candidate: Any) -> SimilarityResult:
    """两件模型名能不能互顶 ✓（返回 ``(ok, score, reason)``，``reason`` 是被哪条红线拦的 ✓）。"""
    left = parse_model_name(required)
    right = parse_model_name(candidate)
    # ① 格式红线
    if not left.ext or not right.ext:
        return SimilarityResult(False, 0.0, "format")
    if left.ext != right.ext:
        return SimilarityResult(False, 0.0, "format")
    # ② 角色 ③ 限定词
    if not _same_set(_pick(left.tokens, ROLE_TOKENS), _pick(right.tokens, ROLE_TOKENS)):
        return SimilarityResult(False, 0.0, "role")
    if not _same_set(_pick(left.tokens, QUALIFIER_TOKENS), _pick(right.tokens, QUALIFIER_TOKENS)):
        return SimilarityResult(False, 0.0, "qualifier")
    # ④ 代次（Qwen Image 2.1 VAE 不许拿旧视频 VAE 顶）
    if _is_qwen_image_21_vae(left.tokens) != _is_qwen_image_21_vae(right.tokens):
        return SimilarityResult(False, 0.0, "generation")
    # ⑤ 规模（两边都写了才拦）
    left_size = tuple(token for token in left.tokens if _SIZE_RE.match(token))
    right_size = tuple(token for token in right.tokens if _SIZE_RE.match(token))
    if left_size and right_size and not _same_set(left_size, right_size):
        return SimilarityResult(False, 0.0, "size")
    # ⑥ 训练规格（量化差异**允许**，turbo/raw 不许互顶）
    if ("turbo" in left.tokens) != ("turbo" in right.tokens):
        return SimilarityResult(False, 0.0, "variant")
    if ("raw" in left.tokens) != ("raw" in right.tokens):
        return SimilarityResult(False, 0.0, "variant")
    # ⑦ 家族
    left_roots = _family_roots(left.tokens, left.compact)
    right_roots = _family_roots(right.tokens, right.compact)
    if left_roots and right_roots and not (set(left_roots) & set(right_roots)):
        return SimilarityResult(False, 0.0, "family")

    score = 0.65 * _dice(left.compact, right.compact) + 0.35 * _jaccard(left.tokens, right.tokens)
    left_family = [token for token in left.tokens if token in FAMILY_TOKENS]
    right_family = [token for token in right.tokens if token in FAMILY_TOKENS]
    if set(left_family) & set(right_family):
        score += 0.08
    score = max(0.0, min(1.0, score))
    return SimilarityResult(score >= SCORE_THRESHOLD, score)


def find_similar_model(
    required: Any,
    candidates: Any,
    *,
    threshold: float = SCORE_THRESHOLD,
    margin: float = AMBIGUOUS_MARGIN,
    limit: int = 200,
) -> dict[str, Any]:
    """在一堆盘上文件名里找**唯一**能顶替 ``required`` 的那件 ✓。

    返回三种形状（调用方要分开处理 ✗ 只看 ``file`` 会把歧义当没找到）：
    * 没找到 ⇒ ``{"file": "", "score": 最高分}``；
    * ⭐ **歧义** ⇒ ``{"file": "", "ambiguous": True, "candidates": [前 5]}``（**不替用户选** ✓）；
    * 找到 ⇒ ``{"file": 名字, "score": 分}``。

    同名（忽略大小写）的候选会被跳过 ✓ —— 那是精确命中，不该走模糊这一趟 ✗。
    """
    wanted = parse_model_name(required)
    scored: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in candidates if isinstance(candidates, (list, tuple, set)) else []:
        text = str(raw or "").strip()
        if not text:
            continue
        parsed = parse_model_name(text)
        if parsed.ext not in FUZZY_MODEL_EXTENSIONS:
            continue
        if parsed.base.lower() == wanted.base.lower():
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result = model_name_similarity(required, text)
        if result.ok:
            scored.append({"file": text, "score": result.score})
        if len(scored) >= limit:
            break

    scored.sort(key=lambda item: item["score"], reverse=True)
    if not scored:
        return {"file": "", "score": 0.0}
    top = scored[0]
    if top["score"] < threshold:
        return {"file": "", "score": top["score"]}
    if len(scored) > 1 and top["score"] - scored[1]["score"] < margin:
        return {"file": "", "score": top["score"], "ambiguous": True, "candidates": scored[:5]}
    return {"file": top["file"], "score": top["score"]}
