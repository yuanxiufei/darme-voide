r"""**分词器基准**（自研实现 vs 参考实现 ✓ 吞吐与规模一起报 ✓）。

## 为什么要有它（2026-09-21）

自研分词器现在的判据全是**正确性**（与 `tokenizers` Rust 那份逐例同 id ✓）—— 但**一个数字都没有** ✗：
「够不够快」「优化有没有效」全靠感觉 ✓✗。本脚本把这两件事变成可复现的测量 ✓：

* 造一份**接近真实规模**的词表（默认 32k 条 ✓ 由参考实现程序化写出 ✓ 不下载 ✓）；
* 分别测 **自研** 与 **参考实现**的吞吐（tokens/s ✓）——⚠️ 对手是 **Rust** ✗ 我们**必然慢** ✓，
  所以**别把"比它慢"当失败** ✗：这份数字的用途是（a）量级判断 ✓（b）**自身优化前后对比** ✓；
* 报**词表规模 × 文本长度**的交叉，避免"只测了一句话"的假象 ✓。

用法::

    ./.venv/Scripts/python.exe app/scripts/tokenizer_bench.py            # 默认 32k 词表
    ./.venv/Scripts/python.exe app/scripts/tokenizer_bench.py --vocab 8000 --rounds 3
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[2]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import tokenizer_own  # noqa: E402

#: 语料 ✓（混中西文 / 标点 / 空白 —— 让预分词与 Viterbi 都真跑起来 ✓）
CORPUS: tuple[str, ...] = (
    "雨夜霓虹街头，一个人站在便利店门口，手里捏着一张旧照片。",
    "a lone figure stands outside a neon-lit convenience store, holding a faded photo.",
    "镜头缓慢推近，霓虹在潮湿路面上拉出长长的倒影；远处传来地铁的低鸣。",
    "The camera slowly pushes in as rain streaks across the glass; subway rumble in the distance.",
    "中英混排 mixed 1234 with punctuation!? and   double   spaces.",
)


def build_vocab(size: int) -> list[tuple[str, float]]:
    """程序化造一份**规模可控**的 Unigram 词表 ✓（不需要下载 ✓ 分布也不刻意友好 ✓）。"""
    seeds = ["雨", "夜", "霓", "虹", "街", "头", "人", "照", "片", "镜", "头", "地", "铁",
             "a", "b", "c", "d", "e", "f", "g", "h", "i", "l", "m", "n", "o", "p", "r", "s",
             "t", "u", "w", "y", "in", "the", "and", "ing", "ly", "ed", "er", "ion", "ation",
             "▁", "▁a", "▁the", "▁in", "▁ing", "▁雨", "▁夜", "▁霓", "▁虹"]
    vocab: list[tuple[str, float]] = [("<unk>", 0.0), ("<s>", 0.0), ("</s>", 0.0)]
    score = -1.0
    index = 0
    while len(vocab) < size:
        left = seeds[index % len(seeds)]
        right = seeds[(index * 7 + 3) % len(seeds)]
        vocab.append((left + right, score))
        score -= 0.0005
        index += 1
    return vocab


def build_tokenizer(root: Path, size: int) -> tuple[Any, Any]:
    """造出「自研实现」与「参考实现」两份 ✓（**同一份词表** ✓ 否则比了没意义 ✗）。"""
    from tokenizers import Tokenizer as RefTokenizer  # noqa: PLC0415
    from tokenizers import models, pre_tokenizers  # noqa: PLC0415

    target = root / f"bench{size}"
    target.mkdir(parents=True, exist_ok=True)
    reference = RefTokenizer(models.Unigram(build_vocab(size), unk_id=0))
    reference.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁")
    reference.save(str(target / "tokenizer.json"))
    own = tokenizer_own.load_own_tokenizer(target)
    if own is None:
        raise SystemExit("自研实现装载失败 ✗（形态应当被覆盖 ✓）")
    return own, reference


def measure(encode: Any, texts: list[str], rounds: int) -> dict[str, float]:
    """测吞吐 ✓（**先热身** ✓ —— 不然第一次的导入/建表开销会污染数字 ✓）。"""
    for text in texts:
        encode(text)
    tokens = 0
    start = time.perf_counter()
    for _ in range(rounds):
        for text in texts:
            tokens += len(encode(text))
    elapsed = time.perf_counter() - start
    return {"seconds": elapsed, "tokens": tokens,
            "tokensPerSecond": tokens / elapsed if elapsed else float("inf")}


def main() -> int:
    parser = argparse.ArgumentParser(description="分词器基准")
    parser.add_argument("--vocab", type=int, default=32000, help="词表条目数（默认 32000）")
    parser.add_argument("--rounds", type=int, default=5, help="语料重复轮数（默认 5）")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        own, reference = build_tokenizer(Path(tmp), args.vocab)
        print(f"词表：{args.vocab} 条 ✓　语料：{len(CORPUS)} 段 × {args.rounds} 轮 ✓")
        # ⭐ ⚠️⚠️ **冷 / 热分开测** ✗（2026-09-21 ✓）：
        #    第一版只测「热」的 ✓✗ ⇒ 报出 **2.4M tokens/s、比 Rust 快 5.5 倍** ✓✗ ——
        #    那是**片段缓存全命中**的假象 ✓（同一批文本重复 3 轮 ✓）。数字必须**分开说** ✓：
        #    * 冷（每轮清空缓存 ✓）= 真实首次吞吐 ✓；
        #    * 热（不清 ✓）= 提示词反复出现时的吞吐 ✓。
        cold = measure(lambda text: (own.clear_cache(), own.encode(text))[1],
                       list(CORPUS), args.rounds)
        own_stats = measure(lambda text: own.encode(text), list(CORPUS), args.rounds)
        ref_stats = measure(lambda text: list(reference.encode(text).ids), list(CORPUS), args.rounds)
        scans = getattr(own, "trieScans", 0)
        stats = own.cache_stats() if hasattr(own, "cache_stats") else None

    print(f"自研·冷：{cold['tokensPerSecond']:>12,.0f} tokens/s"
          f"　（{cold['tokens']} tokens / {cold['seconds']:.3f}s ✓ 每轮清缓存 ✓）")
    print(f"自研·热：{own_stats['tokensPerSecond']:>12,.0f} tokens/s"
          f"　（{own_stats['tokens']} tokens / {own_stats['seconds']:.3f}s ✓ 不清缓存 ✓）")
    print(f"参考  ：{ref_stats['tokensPerSecond']:>12,.0f} tokens/s"
          f"　（{ref_stats['tokens']} tokens / {ref_stats['seconds']:.3f}s ✓ Rust ✓）")
    print(f"比值　：冷 / 参考 = {cold['tokensPerSecond'] / ref_stats['tokensPerSecond']:.3f} ✓"
          f"（⚠️ 对手是 Rust ✗ ⇒ 冷的时候慢是预期内的 ✓；"
          f"这份数字主要用来比**自研自己的优化前后** ✓）")
    print(f"前缀树扫描次数：{scans}（**每个起点只该扫一次** ✓ ⇒ 应 ≈ 字符数 ✓；"
          f"随长度平方增长就是退化 ✓✗）")
    if stats:
        print(f"片段缓存：{stats['entries']} 条 ✓ 命中 {stats['hits']} / 未命中 {stats['misses']}"
              f" ⇒ 命中率 {stats['hitRate']:.1%} ✓（热数字就是它堆出来的 ✓ 别拿它当「单次成本」 ✗）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
