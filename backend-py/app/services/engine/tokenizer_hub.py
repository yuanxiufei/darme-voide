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
* ⭐ **预分词器也扩到八种** ✓（2026-09-21 补 ✓）：`ByteLevel` / `Metaspace` / `BertPreTokenizer` ✓
  + `Whitespace`（按**字母数字↔非字母数字边界**切 ✓ 连续非字母数字**留一起** ✓）/
  `WhitespaceSplit` / `Punctuation`（**5 种 behavior** ✓ 含 `removed`/`contiguous` ✓）/
  `Digits`（两种口径 ✓）/ `CharDelimiterSplit` ✓ ⇒ 覆盖 **3 种模型 × 8 种预分词器** ✓；
* ⭐ **参考实现回退** ✓：只剩自研**明确拒绝**的形态（`byte_fallback` ✗ —— 实测参考实现该配置下
  **没走**字节回退 ✓ 语义未核清 ✓、`Precompiled` 等未实现 normalizer ✗、`Split` / `FixedLength` /
  `UnicodeScripts` 等预分词器 ✗ —— 理由由 `tokenizer_own.own_support` 给出 ✓）才走 `transformers` ✓
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
                 notes: Sequence[str] = ()) -> None:
        self.backend = backend
        self._impl = impl
        self.config = config
        self._vocab_size = int(vocab_size)
        self._required = int(required_vocab_size)
        self.detail = dict(detail or {})
        self.notes = list(notes)

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
        if self.backend == "own-bpe":
            return list(self._impl.encode(text, add_special_tokens=add_special_tokens))
        return [int(item) for item in self._impl.encode(text, add_special_tokens=add_special_tokens)]

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
            hub = TokenizerHub(
                backend, impl, config=cfg, vocab_size=impl.vocab_size,
                required_vocab_size=impl.required_vocab_size,
                detail={"form": form, **impl.describe()},
                notes=[f"{support.reason} ⇒ **不需要 `transformers`** ✗（零依赖 ✓ 离线 ✓）"])
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
        return None, "参考实现（`tokenizers`）**没装** ✓"
    if not path:
        return None, "没有词表路径 ✓（无从造参考侧 ✓）"
    try:
        from tokenizers import Tokenizer as ReferenceTokenizer  # noqa: PLC0415

        target = Path(path)
        json_path = target if target.is_file() else target / "tokenizer.json"
        tokenizer = ReferenceTokenizer.from_file(str(json_path))
        return (lambda text: [int(item) for item in tokenizer.encode(text).ids]), ""
    except Exception as err:  # noqa: BLE001 —— 如实回报原因 ✓（不是"没装"✗）
        return None, f"参考实现**装不了这份词表** ✓：{type(err).__name__}: {err} ✓"


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
