r"""**分词器总入口（优化改造层）** —— 自研 BPE 优先 ✓ 参考实现回退 ✓ 批量 + 缓存 ✓ 可互校 ✓。

## 它把「分词」这件事升级成什么（2026-09-20）

上一版只有两块：**自研 BPE**（`engine/tokenizer_bpe.py` ✓ 零依赖 ✓ 离线 ✓ **但只覆盖
ByteLevel-BPE** ✗）与 `HFTokenizer`（依赖 `transformers` ✓ 只覆盖「能装进 `AutoTokenizer`」✗）。
两块**各管一段** ✓✗ ⇒ 调用方得先知道词表是什么形态才能挑 ✗。本模块是**统一入口** ✓：

* **形态嗅探** ✓：读 ``tokenizer.json`` 的 ``model.type`` + ``pre_tokenizer.type``
  （``Sequence`` 递归下去 ✓；只有 ``vocab.json``+``merges.txt`` ⇒ ByteLevel-BPE ✓）
  ⇒ **自己选后端** ✓，调用方只给路径 ✓；
* ⭐⭐ **自研优先（三种血统 ✓ 2026-09-20 第二次扩充）** ✓：
  * ``BPE`` + ``ByteLevel`` ⇒ `tokenizer_bpe` ✓（零依赖 ✓ 离线 ✓ 往返恒等 ✓ 与参考逐例同 id ✓）；
  * ``Unigram`` + ``Metaspace``（Llama/Qwen/T5 系 ✓）⇒ `tokenizer_own.UnigramTokenizer` ✓
    （**Viterbi** 走最优路径 ✓ + `fuse_unk` ✓）；
  * ``WordPiece`` + ``BertPreTokenizer``（BERT 系 ✓）⇒ `tokenizer_own.WordPieceTokenizer` ✓
    （贪心最长匹配 ✓ + **整词失败 ⇒ 整词 `[UNK]`** ✓）；
  ⇒ 上面三条**都不需要** `transformers` ✗（且各自与参考实现**逐例同 id** ✓，见两套自检 ✓）；
* ⭐ **`normalizer` 也自研了** ✓（2026-09-21 补 ✓）：`BertNormalizer`（clean_text / CJK 两侧加空格 /
  小写 / 去重音 ✓，**四步顺序照参考 ✓**）/ `NFC`·`NFD`·`NFKC`·`NFKD` ✓ / `Lowercase` /
  `StripAccents`（⚠️ **不先 NFD** ✓ 与参考一致 ✓）/ `Strip` / `Prepend` / `Replace` / `Sequence` ✓
  —— 统一用一层 :class:`~app.services.engine.tokenizer_own.NormalizedTokenizer` 套住 ✓
  （三种模型共用一个包装 ✓）。⚠️ 自研实测出**两处容易搞反的语义** ✓：`BertNormalizer` 的
  `cf. clean_text` **不 trim / 不合并空格** ✓、`StripAccents` **不分解** ✓✗（而 BertNormalizer 的去重音
  **会**分解 ✓ —— 两处语义确实不同 ✓）；`Precompiled` **拒绝** ✓（要 SentencePiece charsmap 表 ✗）；
* ⭐ **预分词器已扩到十种** ✓（2026-09-21 补 ✓）：`ByteLevel` / `Metaspace` / `BertPreTokenizer` ✓
  + `Whitespace`（按**字母数字↔非字母数字边界**切 ✓ 连续非字母数字**留一起** ✓）/
  `WhitespaceSplit` / `Punctuation`（**5 种 behavior** ✓ 含 `removed`/`contiguous` ✓）/
  `Digits`（两种口径 ✓）/ `CharDelimiterSplit` / `Split`（**5 种 behavior** ✓）/ `FixedLength` ✓
  ⇒ 覆盖 **3 种模型 × 10 种预分词器** ✓；⭐ **`byte_fallback` 的 Unigram 也自研了** ✓（**段级判据** ✓）；
* ⭐ **参考实现回退** ✓：只剩自研**明确拒绝**的形态（`Precompiled` 等未实现 normalizer ✗、
  `UnicodeScripts` ✗（要 script 表 ✓ 标准库没有 ✓）、别的模型的 `byte_fallback` ✗、
  `Split` 的空匹配正则 / `\p{…}` ✗ —— 理由由 `tokenizer_own.own_support` 给出 ✓）才走 `transformers` ✓
  （`tokenizers` Rust 后端 ✓）；两者都没有 ⇒ **明确报错**并给出安装命令 ✓
  （不静默换一个词表把文本编成一串"看着正常"的错 id ✗✗）；
* ⭐ **性能优化** ✓：同文本 ``encode`` 走 **LRU 缓存** ✓（管线里同一提示词会被反复编码 ✓）、
  批量走 ``encode_batch`` ✓（Rust 侧真并行 ✓；自研侧逐条 ✓ 但**语义与逐条一致** ✓ —— 这是钉住的不变量 ✓）；
* ⭐ **用法优化（针对 `transformers`）** ✓：**已收口到** :mod:`.tokenizers_tuning` ✓（默认**离线** ✓
  ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE`` ✓ 不外呼探测 ⇒ 不会在没网时卡住 ✓、
  **关遥测** ✓、**缓存目录指到本仓数据根** ✓ 不污染用户主目录 ✓、还有**受控补丁**兜住
  `from_pretrained` ✓ 含 **Auto 工厂**那条漏网点 ✓）；
* ⭐ **可互校** ✓：:meth:`TokenizerHub.verify_against_reference` 用另一条实现逐例比对 ✓
  （自研 ↔ 参考 ✓）—— 自检里做过的事，**运行期也能做** ✓（换词表后当场复验 ✓）。

## 事实来源与边界

* 形态字段语义：HF ``tokenizer.json`` 的公开格式 ✓（``model.type`` / ``pre_tokenizer.type`` ✓）；
* ⚠️ 本模块**不 fork / 不改 `transformers` 源码** ✗ —— 只做**用法层**的优化改造 ✓
  （离线 ✓ 缓存 ✓ 遥测 ✓ 批量 ✓ 懒加载 ✓）。要 fork 得先解决 License/维护成本 ✓（其许可为
  Apache-2.0 ✓ 允许 ✓，但本仓目前**不需要**它 ✓：自研 BPE 已覆盖主路径 ✓）。
* ⚠️ 版本：本机 `transformers` **5.17.0 已是 PyPI 最新** ✓（2026-09-20 实测 `pip index versions` ✓）
  ⇒ 没有「升版本」这件事可做 ✓；要升级只会在**有新版本时**再谈 ✓。
"""
from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from . import tokenizer_own
from . import tokenizers_tuning

__all__ = ["HubConfig", "TokenizerHub", "TokenizerHubError", "detect_form", "load",
           "reference_available"]

#: LRU 缓存条目数 ✓（提示词级重复多 ✓ 词表规模无关 ✓ ⇒ 条目数够用即可 ✓）
DEFAULT_CACHE_ENTRIES = 512

#: 自研 BPE **能覆盖**的形态 ✓（其余形态 ⇒ 走参考实现 ✓ 或明确报错 ✓）
OWN_BPE_FORM = ("BPE", "ByteLevel")

_FORM_HINT: dict[str, str] = {
    "Unigram": "SentencePiece 血统的 Unigram（如 T5 / Llama 系 ✓）",
    "WordPiece": "BERT 血统的 WordPiece ✓",
    "Metaspace": "SentencePiece 风格的空格替换（如 Llama / Qwen 系 ✓）",
    "BertPreTokenizer": "BERT 的预分词 ✓",
    "Whitespace": "纯空格切分 ✓",
}


class TokenizerHubError(ValueError):
    """分词器装不上 / 形态不支持 ✓（**明确报错**，不静默换实现 ✗）。"""


@dataclass(frozen=True)
class HubConfig:
    """装载参数 ✓（默认**离线** ✓ —— 生产机上没网也不该卡在探测上 ✗）。"""

    #: 本地目录 / 文件 ✓（生产机上的**主路径** ✓ 词表随权重来 ✓）
    path: str | None = None
    #: HF 仓库名 ✓（**需要** `transformers` ✓ 且**需要**网络或本地缓存 ✓）
    repo: str | None = None
    #: HF 缓存目录 ✓（缺省 ⇒ `<数据根>/hf-cache` ✓ 不写用户主目录 ✓）
    cache_dir: str | None = None
    #: 离线 ✓（``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE`` ✓）
    offline: bool = True
    #: 自研优先 ✓（形态可自研覆盖时**不加载** `transformers` ✗）
    prefer_own_bpe: bool = True
    #: `encode` 的 LRU 条目数 ✓（0 ⇒ 关缓存 ✓）
    cache_entries: int = DEFAULT_CACHE_ENTRIES


def reference_available() -> bool:
    """参考实现（`transformers` ✓）装了没 ✓（**不导入**它 ✓ ⇒ 毫秒级 ✓）。"""
    import importlib.util  # noqa: PLC0415

    try:
        return importlib.util.find_spec("transformers") is not None
    except (ImportError, ValueError):  # pragma: no cover - 极少数损坏安装
        return False


def _default_cache_dir() -> str | None:
    """HF 缓存目录缺省值 ✓ —— **转给** :func:`tokenizers_tuning.default_cache_dir` ✓（单一来源 ✗）。

    ⚠️ 别在本模块再抄一份 ✗：两处实现会在某次改动里漂移 ✓（缓存目录写哪儿是**会咬人**的 ✓）。
    """
    return tokenizers_tuning.default_cache_dir()


def detect_form(path: str | Path) -> dict[str, Any]:
    """嗅探词表**形态** ✓（``modelType`` / ``preTokenizer`` / **``normalizerType``** /
    ``byteFallback`` / ``source`` / ``ownBpeOk`` ✓ 纯读文件 ✓）。

    ⚠️ 判据是 ``tokenizer.json`` 里的**结构字段** ✓，不是文件名 / 大小 ✗（那正是
    「名字对、形态错」的来源 ✓✗）。
    ⚠️ ``normalizerType`` **必须**一起报 ✗：它是「自研该不该接手」的**关键判据** ✓
    （`BertNormalizer` 会小写/去重音 ✓ 硬套 = 静默算错 id ✓✗）—— 第一版漏了它 ✓✗，
    结果拒绝理由说不清 ✓（自检 ⑫ 当场红 ✓）。
    """
    target = Path(path)
    json_path = target if target.is_file() and target.name.endswith(".json") else target / "tokenizer.json"
    if json_path.exists():
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as err:
            raise TokenizerHubError(f"{json_path} 不是合法 JSON ✗（{err} ✓）") from err
        model = payload.get("model") or {}
        model_type = str(model.get("type") or "")
        pre_spec = payload.get("pre_tokenizer")
        pre = _pre_tokenizer_type(pre_spec)
        normalizer_spec = payload.get("normalizer")
        return {"modelType": model_type, "preTokenizer": pre, "source": str(json_path),
                # ⚠️ 预分词器的**原始规格**也要带上 ✗：`Split` 的**正则** / `FixedLength` 的
                #    **length** 都会决定「自研能不能接」✓（只看类型名会误判 ✓✗）。
                "preTokenizerSpec": pre_spec,
                "normalizerType": _normalizer_type(normalizer_spec),
                # ⚠️ **展平后的每一项**都要报 ✗：`Sequence` 里只要有一项没实现 ⇒ 自研就得让位 ✓
                #    （只看顶层类型会把 `Sequence` 读成"能接" ✓✗）。
                "normalizerTypes": tokenizer_own.normalizer_spec_types(normalizer_spec),
                "byteFallback": bool(model.get("byte_fallback")),
                "ownBpeOk": (model_type, pre) == OWN_BPE_FORM}
    vocab_json, merges_txt = target / "vocab.json", target / "merges.txt"
    if vocab_json.exists() and merges_txt.exists():
        # 经典 GPT-2 格式 ✓ —— 它**就是** ByteLevel-BPE ✓（没有 pretokenizer 字段可读 ✓）
        return {"modelType": "BPE", "preTokenizer": "ByteLevel", "normalizerType": "",
                "normalizerTypes": [], "byteFallback": False, "preTokenizerSpec": None,
                "source": f"{vocab_json.name}+{merges_txt.name}", "ownBpeOk": True}
    raise TokenizerHubError(
        f"{target} 里没有可识别的词表 ✗（找过 `tokenizer.json` ✓ 与 `vocab.json`+`merges.txt` ✓）")


def added_tokens_of(path: str | Path) -> dict[str, int]:
    """词表目录里**登记的特殊 token 字面量** → id ✓（``{"<|im_start|>": 151644, …}`` ✓）。

    ⚠️ 为什么必须有这一步 ✗（2026-10-31 实测踩到 ✓）：``vocab.json`` + ``merges.txt`` **只含
    BPE 词表本体** ✓，``<|im_start|>`` 这类**后加的特殊符**登记在 ``tokenizer_config.json`` 的
    ``added_tokens_decoder`` 里 ✗。自研 BPE 若不做这层切分 ⇒ 它会被**当成普通文本**逐字节切 ✓✗：
    实测 ``"<|im_start|>user\\nhi"`` 编成 ``[27, 91, 318, 62, …]`` ✓（27 个 id ✓），
    而参考实现是 ``[151644, 872, …]`` ✓（4 个 ✓）—— **不报错、不越界、只是 id 全错** ✓✗。
    Z-Image 的提示词模板正是靠这两个特殊符定界 ✓ ⇒ 错在这里 = 文本塔收到的不是「对话起点」✓✗。

    两个来源都认 ✓（谁在就读谁 ✓）：
    * ``tokenizer_config.json`` → ``added_tokens_decoder``：``{id: {content: …}}`` ✓；
    * ``tokenizer.json`` → ``added_tokens``：``[{id, content}]`` ✓。

    ⚠️ 前缀写法要**整串匹配** ✗（不做「前缀最长匹配 + 剩余交给 BPE」✗ 之外的花样 ✓）——
    本函数只负责**把事实读出来** ✓，切分策略在 :class:`TokenizerHub` 里 ✓（单一职责 ✓）。
    """
    target = Path(path)
    folder = target if target.is_dir() else target.parent
    table: dict[str, int] = {}
    cfg_path = folder / "tokenizer_config.json"
    if cfg_path.exists():
        try:
            payload = json.loads(cfg_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        decoder = payload.get("added_tokens_decoder") or {}
        if isinstance(decoder, dict):
            for key, entry in decoder.items():
                content = entry.get("content") if isinstance(entry, dict) else None
                if content:
                    table[str(content)] = int(key)
    json_path = folder / "tokenizer.json"
    if json_path.exists() and not table:
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        for entry in payload.get("added_tokens") or []:
            if isinstance(entry, dict) and entry.get("content") is not None:
                table[str(entry["content"])] = int(entry.get("id") or 0)
    return table


def _normalizer_type(spec: Any) -> str:
    """取 ``normalizer.type`` ✓（`Sequence` 里多项 ⇒ 直接报 ``Sequence`` ✓ 反正自研不接 ✓）。"""
    if not spec or not isinstance(spec, dict):
        return ""
    return str(spec.get("type") or "")


def _pre_tokenizer_type(spec: Any) -> str:
    """取 ``pre_tokenizer.type`` ✓；``Sequence`` 里只有一项时**递归**进去 ✓（多段 ⇒ 报字段本身 ✓）。"""
    node = spec
    if isinstance(node, dict) and str(node.get("type")) == "Sequence":
        inner = node.get("pretokenizers") or []
        if len(inner) == 1:
            node = inner[0]
    if node is None:
        return "ByteLevel"          # BPE 字节模型的默认切分 ✓（见 tokenizer_bpe 的同一口径 ✓）
    if not isinstance(node, dict):
        return ""
    return str(node.get("type") or "")


class TokenizerHub:
    """统一分词器 ✓（`Tokenizer` 协议 ✓：``vocab_size`` + ``encode`` ✓ ⇒ 可直接注入 TE ✓）。

    ``backend`` 字段会如实报出这条是怎么编的 ✓：``"own-bpe"``（自研 ✓ 零依赖 ✓）/
    ``"transformers"``（参考实现回退 ✓）—— 「用的哪条」必须一眼可见 ✗（否则"自研优先"就成了口号 ✓✗）。
    """

    #: ⚠️ 类级默认 ``None`` ✓ —— 直接构造（不经 :func:`load` ✓）时 `encode` **不该炸** ✗（走无缓存路径 ✓）
    _cached: Callable[[str, bool], tuple[int, ...]] | None = None

    def __init__(self, backend: str, impl: Any, *, config: HubConfig,
                 vocab_size: int, required_vocab_size: int, detail: dict[str, Any] | None = None,
                 notes: Sequence[str] = (),
                 added_tokens: dict[str, int] | None = None) -> None:
        self.backend = backend
        self._impl = impl
        self.config = config
        self._vocab_size = int(vocab_size)
        self._required = int(required_vocab_size)
        self.detail = dict(detail or {})
        self.notes = list(notes)
        #: 特殊 token 字面量 → id ✓（见 :func:`added_tokens_of` ✓）。**长串优先** ✓：
        #: ``<|im_start|>`` 不能被更短的 ``<|im_…`` 先抢走 ✓（这里按长度降序试配 ✓）。
        self._added = dict(added_tokens or {})
        self._added_order = sorted(self._added, key=lambda item: (-len(item), item))

    # ── 协议面 ✓
    @property
    def name(self) -> str:
        """后端名 ✓（自研报实现自己的名字：`bpe` / `own-unigram` / `own-wordpiece` ✓；
        回退报 `transformers` ✓ ⇒ `describe().tokenizer` 一眼可辨用的是**哪条** ✓）。"""
        impl_name = getattr(self._impl, "name", None)
        return str(impl_name or ("transformers" if self.backend == "transformers"
                                 else self.backend))

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def required_vocab_size(self) -> int:
        """嵌入表**至少**要多大 ✓（含特殊符 ✓ —— 拿 `vocab_size` 建表会**越界** ✓✗）。"""
        return self._required

    # ── 编码 ✓（带 LRU ✓）
    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        if self.config.cache_entries <= 0 or self._cached is None:
            return self._encode_uncached(str(text or ""), add_special_tokens)
        return list(self._cached(str(text or ""), add_special_tokens))

    def _encode_uncached(self, text: str, add_special_tokens: bool) -> list[int]:
        if self._added and self.backend.startswith("own-"):
            return self._encode_with_added(text, add_special_tokens)
        if self.backend == "own-bpe":
            return list(self._impl.encode(text, add_special_tokens=add_special_tokens))
        return [int(item) for item in self._impl.encode(text, add_special_tokens=add_special_tokens)]

    def _encode_with_added(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """按**登记的特殊 token** 切开 ✓：命中的段直接给 id ✓，其余段走词表本体 ✓。

        ⚠️ 只在自研侧用 ✗（参考实现自己就会处理 added tokens ✓ 再切一遍等于重复 ✓✗）。
        ⚠️ 词表段**一律**传 ``add_special_tokens=False`` ✗：那会让某些词表在**每段**前面补 BOS ✓✗
        （Qwen2 没有 BOS ✓，但这条路径将来要接别的词表 ✓ ⇒ 语义定死：**片段不加特殊符** ✓）。

        ⚠️ **2026-10-31 修**：原实现直接**忽略** ``add_special_tokens`` ✗✗ ⇒ 同一个方法在
        「词表里有 added tokens」（走这里，不加 ✗）与「没有」（走 impl，**加 BOS＋EOS** ✗）两种
        输入上行为相反 ✓✗，而调用方看不出区别 ✓ —— 静默不一致比任何一种都糟 ✓。
        现在：``add_special_tokens=True`` ⇒ 在**首尾各加一次**（BOS/EOS 取自 impl ✓，缺就不加 ✗）。
        """
        if not text:
            return []
        ids: list[int] = []
        index = 0
        total = len(text)
        while index < total:
            hit = next((item for item in self._added_order if text.startswith(item, index)), None)
            if hit is not None:
                ids.append(int(self._added[hit]))
                index += len(hit)
                continue
            start = index
            index += 1
            while index < total and not any(
                    text.startswith(item, index) for item in self._added_order):
                index += 1
            ids.extend(int(value) for value in
                       self._impl.encode(text[start:index], add_special_tokens=False))
        if add_special_tokens:
            head = [int(item) for item in getattr(self._impl, "bos_ids", ()) or ()]
            tail = [int(item) for item in getattr(self._impl, "eos_ids", ()) or ()]
            ids = head + ids + tail
        return ids

    def encode_batch(self, texts: Iterable[str], *, add_special_tokens: bool = True
                     ) -> list[list[int]]:
        """批量编码 ✓（⚠️ **必须与逐条 `encode` 完全一致** ✓ —— 这是钉住的不变量 ✓）。

        参考实现走 ``encode_batch`` ✓（Rust 侧真并行 ✓）；自研侧没有 Rust 那层 ✓ ⇒ 逐条走
        **同一个缓存** ✓（语义一致 = 唯一要紧的事 ✓，并行只是快慢 ✓）。
        """
        items = [str(text or "") for text in texts]
        if self.backend.startswith("own-"):
            # 自研侧：逐条走**同一个缓存** ✓（语义一致是要紧的 ✓ 并行只是快慢 ✓）
            return [self.encode(text, add_special_tokens=add_special_tokens) for text in items]
        raw = self._impl.encode_batch(items, add_special_tokens=add_special_tokens)
        return [[int(item) for item in row] for row in raw]

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        return str(self._impl.decode(list(ids), skip_special_tokens=skip_special_tokens))

    @property
    def special_tokens(self) -> dict[str, str]:
        """``{bos_token: "<s>", eos_token: "</s>", …}`` ✓ —— 供 chat 骨架的上下文用 ✓。

        ⚠️ 为什么要有 ✗：参考实现（HF ``apply_chat_template`` ✓）会把这些**自动注入**模板，
        而本项目原先渲染骨架时没给 ✓ ⇒ 模板里的 ``{{ bos_token }}`` 直接渲染成**空串** ✗✗
        （Llama-3 系骨架**首行**就是它 ✓；``{% if not bos_token %}`` 一类分支还会走错 ✓），
        **而且不报错** ✓✗。空 = 这个词表确实没有 ✓（不猜 ``<s>`` ✗）。
        """
        out: dict[str, str] = {}
        for name in ("bos_token", "eos_token", "unk_token", "pad_token"):
            ids = getattr(self._impl, f"{name[:-6]}_ids", None)
            if ids:
                text = self.decode([int(ids[0])], skip_special_tokens=False)
                if text:
                    out[name] = text
                continue
            text = getattr(self._impl, name, "")       # transformers 侧直接是字符串 ✓
            if isinstance(text, str) and text:
                out[name] = text
        return out

    # ── 缓存 ✓（可观测：命中数要能读出来 ✗ 否则"加了缓存"无法自证 ✓）
    def _install_cache(self) -> None:
        @functools.lru_cache(maxsize=int(self.config.cache_entries))
        def _cached(text: str, add_special_tokens: bool) -> tuple[int, ...]:
            return tuple(self._encode_uncached(text, add_special_tokens))

        self._cached: Callable[[str, bool], tuple[int, ...]] = _cached  # type: ignore[assignment]

    def cache_stats(self) -> dict[str, int]:
        info = getattr(self._cached, "cache_info", None)
        if info is None:                                  # pragma: no cover - 关缓存时
            return {"maxsize": 0, "hits": 0, "misses": 0, "currsize": 0}
        stats = info()
        return {"maxsize": int(stats.maxsize), "hits": int(stats.hits),
                "misses": int(stats.misses), "currsize": int(stats.currsize)}

    def clear_cache(self) -> None:
        clear = getattr(self._cached, "cache_clear", None)
        if callable(clear):
            clear()

    # ── 自述 / 互校 ✓
    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.backend, "name": self.name,
            "vocabSize": self.vocab_size, "requiredVocabSize": self.required_vocab_size,
            # ⭐ 特殊 token 条数 ✓（自研侧会按字面量整串切分 ✓；0 说明这份词表没登记特殊符 ✓）
            "addedTokens": len(self._added),
            "cacheEntries": int(self.config.cache_entries), "offline": bool(self.config.offline),
            "detail": dict(self.detail), "notes": list(self.notes),
        }

    def fingerprint(self) -> dict[str, Any]:
        """词表指纹 ✓（换词表 ⇒ 指纹变 ✓ —— 「换没换过词表」要能机械回答 ✓）。"""
        impl_fingerprint = getattr(self._impl, "fingerprint", None)
        if callable(impl_fingerprint):
            return {"backend": self.backend, **impl_fingerprint()}
        if self.backend == "transformers":
            vocab = getattr(self._impl, "get_vocab", None)
            if callable(vocab):
                return {"backend": self.backend, "vocabSize": len(vocab()),
                        "name": self.name}
        return {"backend": self.backend, "vocabSize": self.vocab_size, "name": self.name}

    def verify_against_reference(self, samples: Sequence[str],
                                 *, add_special_tokens: bool = False) -> dict[str, Any]:
        """用**另一条实现**逐例互校 ✓（自研 ↔ 参考 ✓）—— 换词表后当场复验 ✓。

        ⚠️ 两侧必须能被**同一个词表**驱动 ✓：自研侧覆盖不了该形态时只跑参考侧 ✓，
        并**说明为什么**（``compared=0`` ✓ 不假装比过 ✗）。
        """
        texts = [str(text) for text in samples]
        result: dict[str, Any] = {"backend": self.backend, "compared": 0, "mismatched": {},
                                 "notes": []}
        if self.backend.startswith("own-"):
            reference, reason = _build_reference_encoder_from_path(self._source_path)
            if reference is None:
                result["notes"].append(f"{reason} ⇒ 只验自研侧**往返恒等** ✓")
                result["roundTrip"] = all(
                    self.decode(self.encode(text, add_special_tokens=add_special_tokens),
                                skip_special_tokens=not add_special_tokens) == text
                    for text in texts)
                return result
            for text in texts:
                mine = self.encode(text, add_special_tokens=add_special_tokens)
                theirs = reference(text)
                if mine != theirs:
                    result["mismatched"][text] = {"own": mine, "reference": theirs}
                else:
                    result["compared"] += 1
            result["ok"] = not result["mismatched"]
            return result
        # 参考侧：拿**自研实现**当对照 ✓（覆盖不了该形态 ⇒ 如实说明 ✓）
        own_impl = tokenizer_own.load_own_tokenizer(self._source_path or "")
        if own_impl is None:
            support = tokenizer_own.own_support(detect_form(self._source_path or ""))
            result["notes"].append(f"自研覆盖不了这个形态 ✓（{support.reason}）⇒ 只跑参考侧 ✓")
            result["ok"] = None
            return result
        for text in texts:
            mine = self.encode(text, add_special_tokens=add_special_tokens)
            theirs = own_impl.encode(text, add_special_tokens=add_special_tokens)
            if mine != theirs:
                result["mismatched"][text] = {"reference": mine, "own": list(theirs)}
            else:
                result["compared"] += 1
        result["ok"] = not result["mismatched"]
        return result


def load(config: HubConfig | None = None, **kwargs: Any) -> TokenizerHub:
    """**统一装载** ✓（自研优先 ✓ 参考实现回退 ✓）—— 调用方只给路径/仓库 ✓。

    ⚠️ 形态不支持且参考实现没装 ⇒ **报错并给出可执行命令** ✗（不静默换一个词表 ✓✗）。
    """
    cfg = config or HubConfig(**kwargs)
    # ⚠️ 「离线 + 关遥测 + 缓存目录」统一由 `tokenizers_tuning` 落环境变量 ✓（**单一来源** ✗）——
    #    而且它还会**顺手给 `transformers` 打上受控补丁** ✓（离线兜底挂在 Auto 工厂上 ✓
    #    见那里的第 1 条 ✓：只包基类会漏 ✓✗）。
    tokenizers_tuning.apply_tuning(offline=cfg.offline, cache_dir=cfg.cache_dir)
    if cfg.path:
        return _load_from_path(cfg)
    if cfg.repo:
        return _load_from_reference(cfg, cfg.repo)
    raise TokenizerHubError("既没给 `path` 也没给 `repo` ✗ ⇒ 不知道装哪份词表 ✓")


def _load_from_path(cfg: HubConfig) -> TokenizerHub:
    form = detect_form(cfg.path or "")
    hint = _FORM_HINT.get(form["preTokenizer"], "")
    support = tokenizer_own.own_support(form)
    if support.ok and cfg.prefer_own_bpe:
        impl = tokenizer_own.load_own_tokenizer(cfg.path or "")
        if impl is not None:
            kind = str(getattr(impl, "name", "own"))
            backend = {"bpe": "own-bpe"}.get(kind, kind)
            # ⚠️ 特殊 token 的 id **不在** BPE 词表里 ✗（``<|im_start|>``=151644 ✓ 而 vocab.json 只到
            #    151642 ✓）⇒ 嵌入表至少要比 `vocab_size` 大 ✓；漏了这条，下一个真挂 TE 的人会在
            #    `embed_tokens` 上撞 **越界** ✓✗（而且报的是 aten 的 `index out of range` ✗ 指不到这里 ✓）。
            added = added_tokens_of(cfg.path or "")
            required = max(int(impl.required_vocab_size), int(impl.vocab_size),
                           max(added.values(), default=-1) + 1)
            hub = TokenizerHub(
                backend, impl, config=cfg, vocab_size=impl.vocab_size,
                required_vocab_size=required,
                detail={"form": form, **impl.describe(),
                        "addedTokens": len(added),
                        **({"addedTokenMaxId": max(added.values())} if added else {})},
                added_tokens=added,
                notes=[f"{support.reason} ⇒ **不需要 `transformers`** ✗（零依赖 ✓ 离线 ✓）"]
                      + ([f"特殊 token {len(added)} 条按**字面量整串**切分 ✓"
                          f"（最大 id {max(added.values())} ✓）"] if added else []))
            hub._source_path = str(cfg.path)                                      # noqa: SLF001
            hub._install_cache()                                                   # noqa: SLF001
            return hub
    if not reference_available():
        raise TokenizerHubError(
            f"这份词表是 `{form['modelType']}` + `{form['preTokenizer']}` ✗"
            f"{('（' + hint + '）') if hint else ''} —— 自研实现覆盖不了 ✓（理由：{support.reason}）"
            f" ⇒ 需要参考实现 ✓，但本机没装 ✗："
            f"`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers` ✓"
            f"（可选依赖 ✓ 缺它不影响其它路径 ✓）")
    return _load_from_reference(cfg, str(cfg.path), form=form, hint=hint)


def _build_reference_encoder_from_path(path: str | None
                                       ) -> tuple[Callable[[str], list[int]] | None, str]:
    """按**词表文件**造参考侧 ``text -> idx`` ✓（`tokenizers` 直连 ✓ 任意形态都行 ✓）。

    ⚠️⚠️ 返回 ``(encoder, reason)`` ✗ —— **不能只回 ``None``** ✓：
    「没装参考实现」与「装了但这份文件它**装不了**」是**两回事** ✓✗，
    合成一个 ``None`` 会让文案说成前者 ✓✗（本仓纪律：**别让判据说谎** ✓）。
    实测踩过 ✓：自检手写的极简 ``tokenizer.json`` 缺字段 ⇒ 参考侧报
    ``missing field `single_word` `` ✓ —— 那是"文件不完整"，不是"没装" ✓。
    """
    if not reference_available():
        return None, "参考实现（`transformers`）**没装** ✓"
    if not path:
        return None, "没有词表路径 ✓（无从造参考侧 ✓）"
    target = Path(path)
    if not target.exists():
        return None, f"词表路径**不存在** ✓：{target} ✓"
    # ① 有 `tokenizer.json`（fast 定义）⇒ 直接喂 `tokenizers` ✓
    json_path = target if target.is_file() else target / "tokenizer.json"
    if json_path.exists():
        try:
            from tokenizers import Tokenizer as ReferenceTokenizer  # noqa: PLC0415

            tokenizer = ReferenceTokenizer.from_file(str(json_path))
            return (lambda text: [int(item) for item in
                                  tokenizer.encode(text, add_special_tokens=False).ids]), ""
        except Exception as err:  # noqa: BLE001 —— 如实回报原因 ✓（不是"没装"✗）
            return None, (f"参考实现**装不了这份词表** ✓（`tokenizer.json` 本身坏了 ✓）："
                          f"{type(err).__name__}: {err} ✓")
    # ② 只有 `vocab.json` + `merges.txt`（GPT-2 风格 BPE，常带 `tokenizer_config.json`）
    #    ⇒ 交给 `AutoTokenizer` **按词表自带的 config 选类** ✓（**不猜** ✗）。
    #    ⚠️ 这条通路是 2026-10-30 补的：原来只认 `tokenizer.json` ⇒ Qwen 系发布包（如
    #    `qwen25_tokenizer`，H3 的 TE = Qwen3-VL-32B 就该用它）**根本进不来** ✗，
    #    于是「互校」长期停在「没比成」✓✗。实测 `AutoTokenizer.from_pretrained(该目录)`
    #    按 config 判成 `Qwen2Tokenizer`（与 ComfyUI `qwen3vl.py::Qwen3VLSDTokenizer` 同一实现 ✓）。
    if not (target / "vocab.json").exists():
        return None, (f"这份词表目录里既没有 `tokenizer.json` 也没有 `vocab.json` ✓"
                      f"（{target} ✓ ⇒ 参考侧无从下手 ✓）")
    try:
        from . import tokenizers_tuning  # noqa: PLC0415

        tokenizers_tuning.apply_tuning(offline=True, cache_dir=None)
        from transformers import AutoTokenizer  # noqa: PLC0415

        tokenizer = AutoTokenizer.from_pretrained(str(target))
        return (lambda text: [int(item) for item in
                              tokenizer.encode(text, add_special_tokens=False)]), ""
    except Exception as err:  # noqa: BLE001 —— 如实回报原因 ✓
        return None, f"参考实现**装不了这份词表** ✓：{type(err).__name__}: {err} ✓"


#: H3 的 TE 是 **Qwen3-VL-32B** ⇒ 词表**只认 Qwen 系** ✓（判据 = 词表自带的 ``tokenizer_class`` ✓）
H3_TOKENIZER_CLASSES: tuple[str, ...] = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

#: 词表在**模型根**下的子目录 ✓ —— 与权重同一层 ✓（部署时把词表丢进这里最省事 ✓）
H3_TOKENIZER_SUBDIR = "tokenizers/qwen3vl_32b"

#: 自动探测的落点（按顺序试 ✓）—— **相对仓根** ✓（开发机把词表放这儿就够 ✓）
H3_TOKENIZER_CANDIDATES: tuple[str, ...] = ("models/tokenizers/qwen3vl_32b",)


def _tokenizer_dirs() -> list[tuple[Path, str]]:
    """候选目录 + 它的**来历** ✓（仓内 / 模型根 ✓）—— 去重后按顺序返回 ✓。"""
    repo = Path(__file__).resolve().parents[4]      # …/darme-voide（backend-py/app/services/engine/ ✓）
    found: list[tuple[Path, str]] = [(repo / item, item) for item in H3_TOKENIZER_CANDIDATES]
    try:                                            # 懒导入 ✓：inventory 会拉起 h3_form/safetensors 一串 ✓
        from . import inventory  # noqa: PLC0415

        root = inventory.models_dir()
    except Exception:  # noqa: BLE001 —— 探测失败只该「找不到」，不该炸 ✓
        root = None
    if root is not None:
        found.append((root / H3_TOKENIZER_SUBDIR, f"<模型根>/{H3_TOKENIZER_SUBDIR}"))
    return found


def find_h3_tokenizer() -> dict[str, Any] | None:
    """找一份**确实属于 Qwen 系**的词表 ⇒ ``{"path", "tokenizerClass", "source"}``；没有 ⇒ ``None`` ✓。

    返回值带 ``source``（**落点来历**：仓内相对路径 / ``<模型根>/tokenizers/…`` ✓）⇒ 调用方能
    **如实说**「这份词表是自动找到的 ✓」，而不是让人以为是自己传的 ✗。

    ⚠️ 落点**只认两处**：仓内 ``models/tokenizers/qwen3vl_32b`` ✓ 与**模型根**下的同名子目录 ✓。
    **不**拿开发期脚手架 `reference/` 当落点 ✗（用户 2026-09-26：那是要删的 ⇒ 写死 = 悬空引用 ✗，
    `tests/local_models_test.py` 有静态守卫盯着 ✓）。

    ⚠️⚠️ **为什么不按「装得上」找** ✗：脚手架里的 ``llama_tokenizer``（vocab=128000 ✓
    **是 Llama-3 的词表** ✗）挂在 Qwen3-VL 的 TE 上**不会报错** ✗ —— 它满足
    ``required <= 嵌入行数`` ✓、``describe()`` 一切正常 ✗，但 token id 与 Qwen 词表
    **整体错位** ⇒ 出片与提示词**无关** ✓✗（这类静默错比装不上难查得多 ✓）。
    ⇒ 判据取**词表自带的 ``tokenizer_class``** ✓（``tokenizer_config.json`` ✓ 是发布方写的事实 ✓，
    不是我猜的 ✓）；**没有该文件的目录一律不用** ✗（宁可不自动 ✓）。
    """
    for directory, source in _tokenizer_dirs():
        try:
            declared = json.loads(
                (directory / "tokenizer_config.json").read_text(encoding="utf-8")
            ).get("tokenizer_class")
        except (OSError, ValueError):
            continue
        if isinstance(declared, str) and declared in H3_TOKENIZER_CLASSES:
            return {"path": str(directory), "tokenizerClass": declared, "source": source}
    return None


def _load_from_reference(cfg: HubConfig, target: str, *, form: dict[str, Any] | None = None,
                         hint: str = "") -> TokenizerHub:
    """走 `transformers` ✓（**用法层优化由 :mod:`.tokenizers_tuning` 统一落** ✓：离线 ✓
    关遥测 ✓ 缓存目录 ✓，且**给它的 `from_pretrained` 打了受控补丁** ✓）。"""
    tokenizers_tuning.apply_tuning(offline=cfg.offline, cache_dir=cfg.cache_dir)
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415 —— 懒导入 ✓
    except ImportError as err:  # pragma: no cover - 上面已探测过 ✓
        raise TokenizerHubError(f"未安装 `transformers`（{err} ✓）") from err
    try:
        impl = AutoTokenizer.from_pretrained(
            target, local_files_only=bool(cfg.offline),
            cache_dir=cfg.cache_dir or _default_cache_dir())
    except Exception as err:  # noqa: BLE001 —— 外部库的裸异常**不许**漏出去 ✗
        raise TokenizerHubError(
            f"参考实现装载失败（`{target}` ✓）：{err} ✓ ⇒ 核对路径/仓库名 ✓"
            f"（离线时请给**本地目录** ✓）；或改用一个 ByteLevel-BPE 词表走**自研**路径 ✓") from err
    vocab_size = int(getattr(impl, "vocab_size", 0) or 0)
    required = int(len(impl)) if hasattr(impl, "__len__") else vocab_size
    notes = [f"参考实现 `transformers` 回退 ✓（形态 "
             f"`{(form or {}).get('modelType', '非本地')}` + `{(form or {}).get('preTokenizer', '')}` ✗"
             f"{('；' + hint) if hint else ''}）",
             "离线 ✓ 关遥测 ✓ 缓存目录 = " + str(cfg.cache_dir or _default_cache_dir() or "HF 默认 ✓")]
    hub = TokenizerHub("transformers", impl, config=cfg, vocab_size=vocab_size,
                       required_vocab_size=required,
                       detail={"form": form, "repo": getattr(impl, "name_or_path", target)},
                       notes=notes)
    hub._source_path = target                                                        # noqa: SLF001
    hub._install_cache()                                                             # noqa: SLF001
    return hub


# ⚠️ 这里**原本**有一份 `_apply_offline_env` ✓✗ —— 现已**删掉** ✓：
#    做法已经**收口到 `tokenizers_tuning.apply_tuning`** ✓（单一来源 ✗ —— 两份实现会在某次
#    改动里漂移 ✓，而"离线开关落在哪"是会咬人的那种 ✓）。
