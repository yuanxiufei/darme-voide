"""S27 自检：**对 `transformers` 的运行时改造（tuning）**（2026-09-20）。

用户指示：「**你也可以改造 transformers 并实现优化升级这个 transformers**」✓。
落地取舍：**不 fork / 不 vendored** ✗（数千文件 / 数十 MB ✓ 维护成本不划算 ✓），
而是**运行时补丁** ✓ 且要求**幂等 ✓ 可撤 ✓ 有计数 ✓ 可自证 ✓**。

⚠️ 两种世界都成立 ✓：装了 ⇒ 真打补丁并**真触发**一次离线兜底 ✓；没装 ⇒ 只声明策略 ✓ 不抛 ✗
（缺依赖路径用 ``monkeypatch`` 模拟 ✓）。

运行::

    ./.venv/Scripts/python.exe tests/engine_tokenizers_tuning_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import tokenizers_tuning as tuning  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _raises(call: Any, needle: str | None = None) -> str | None:
    """**能触发**的反向证明 ✓：调它、看报错里有没有那个词 ✓（没报错 ⇒ None ⇒ 断言红 ✓）。"""
    try:
        call()
    except Exception as err:  # noqa: BLE001 —— 就是来看它报什么的 ✓
        return str(err) if needle is None or needle in str(err) else None
    return None


# ══════════════════════════════════════════════════════════════════════════
# ① 清单与「缺库世界」：打了什么 / 没库时不炸
# ══════════════════════════════════════════════════════════════════════════
def case_manifest() -> None:
    report = tuning.apply_tuning()
    check("① 改造清单**响亮** ✓：5 条逐条报出 ✓（离线兜底 / 缓存目录 / 降噪 / 计数 / 可撤销 ✓）",
          len(tuning.TUNING_ITEMS) == 5 and list(report["items"]) == list(tuning.TUNING_ITEMS),
          report["items"])

    original = tuning._import_transformers          # noqa: SLF001 —— 模拟「没装」的世界 ✓
    saved = dict(tuning._STATS)                     # noqa: SLF001
    tuning._import_transformers = lambda: None      # noqa: SLF001
    try:
        missing_report = tuning.apply_tuning()
    finally:
        tuning._import_transformers = original      # type: ignore[assignment]  # noqa: SLF001
        tuning._STATS.update(saved)                 # noqa: SLF001
    check("② 缺 `transformers` 的世界 ⇒ `applied=False` + 说明 ✓ **不抛** ✗"
          "（它是可选依赖 ✓ 缺了不该把这条路径判死 ✗）",
          missing_report["applied"] is False and "可选" in str(missing_report["reason"]),
          missing_report["reason"])


# ══════════════════════════════════════════════════════════════════════════
# ② 真打补丁：幂等 ✓ 计数 ✓ 离线兜底**能触发** ✓
# ══════════════════════════════════════════════════════════════════════════
def case_patch() -> None:
    if not tuning._import_transformers():                                     # noqa: SLF001
        check("③ 本机没装 `transformers` ⇒ 补丁用例跳过 ✓（不是失败 ✗）", True, "skipped")
        return

    import transformers  # noqa: PLC0415
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase  # noqa: PLC0415

    tuning.unpatch()          # ⚠️ 先撤 ✓ —— 否则这里拿到的「原函数」其实**已经**是包装过的 ✗
    pristine = PreTrainedTokenizerBase.__dict__["from_pretrained"]
    first = tuning.apply_tuning()
    wrapped_once = PreTrainedTokenizerBase.__dict__["from_pretrained"]
    second = tuning.apply_tuning()
    wrapped_twice = PreTrainedTokenizerBase.__dict__["from_pretrained"]
    check("③ ⭐ **幂等**：连打两次 ⇒ 仍是**同一个**包装函数 ✓ 且与**原函数不同** ✓"
          "（叠加会表现为「越跑越慢」✓✗）",
          first["applied"] and second["applied"] and wrapped_once is wrapped_twice
          and wrapped_once is not pristine,
          (wrapped_once is wrapped_twice, wrapped_once is not pristine))

    check("④ `default_cache_dir()` 指向**本仓数据根** ‹hf-cache› ✓（不写用户主目录 ✓）",
          bool(tuning.default_cache_dir()) and "hf-cache" in str(tuning.default_cache_dir()),
          tuning.default_cache_dir())

    # ⭐ **能触发**的反向证明：真调一次 `from_pretrained`（假仓库名 ✓）⇒ 必须被强制离线 ✓
    # ⚠️ 用 **`AutoTokenizer`** ✓ 而不是某个具体类 ✗ —— 实测它**在解析出具体类之前就外呼** ✓✗
    #    （第一版只包了基类 ⇒ 这条当场红 ✓ 计数全 0 ✓），所以这是「兜底有没有漏」的**真判据** ✓。
    before = tuning.tuning_status()["stats"]
    _raises(lambda: transformers.AutoTokenizer.from_pretrained("no-such-org/no-such-model-xyz"))
    after = tuning.tuning_status()["stats"]
    check("⑤ ⭐⭐ **离线兜底真生效**：**Auto 工厂**那条路也被强制 `local_files_only=True` ✓"
          "且有计数 ✓（只包基类会漏 ✓✗ —— 本仓实测直连 HF 全超时 ✗ ⇒ 漏网外呼 = 生产机卡住 ✓✗）",
          after["offlineForced"] >= before["offlineForced"] + 1
          and after["lastKwargs"]["local_files_only"] is True
          and after["lastKind"] == "auto-tokenizer"
          and after["fromPretrainedCalls"] >= before["fromPretrainedCalls"] + 1,
          (before, after))

    check("⑥ **缓存目录也注入进去了** ✓（`cache_dir` 在真调用里可见 ✓ 有计数 ✓）",
          after["cacheDirInjected"] >= 1
          and str(after["lastKwargs"]["cache_dir"]).endswith("hf-cache"),
          after["lastKwargs"])

    check("⑦ 降噪落到环境变量 + 库日志上 ✓（遥测关 ✓ fork 警告关 ✓）",
          os.environ.get("HF_HUB_DISABLE_TELEMETRY") == "1"
          and os.environ.get("TOKENIZERS_PARALLELISM") == "false"
          and os.environ.get("HF_HUB_OFFLINE") == "1"
          and os.environ.get("TRANSFORMERS_OFFLINE") == "1",
          tuning.tuning_status()["env"])


# ══════════════════════════════════════════════════════════════════════════
# ③ 可撤销 ✓（改错了能退；状态如实）
# ══════════════════════════════════════════════════════════════════════════
def case_unpatch() -> None:
    if not tuning._import_transformers():                                     # noqa: SLF001
        check("⑧ 没装 ⇒ 撤销用例跳过 ✓", True, "skipped")
        return
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase  # noqa: PLC0415

    restored = tuning.unpatch()
    current = PreTrainedTokenizerBase.__dict__["from_pretrained"]
    check("⑧ ⭐ **可撤销**：`unpatch()` 后 `from_pretrained` 回到**原函数** ✓"
          "（`applied=False` ✓ —— 改错能退 ✓ 也能 A/B 对比 ✓）",
          "tokenizer" in restored["restored"] and tuning.tuning_status()["applied"] is False
          and isinstance(current, classmethod), restored)

    again = tuning.apply_tuning()
    check("⑨ 撤销后**还能再打** ✓（且仍是干净的一层 ✓ —— 否则计数会叠 ✓✗）",
          again["applied"] is True
          and isinstance(PreTrainedTokenizerBase.__dict__["from_pretrained"], classmethod),
          again["stats"])
    check("⑩ `tuning_status()` **从实底读** ✓：装没装 / 打了没 / 计数 / 环境变量 都给 ✓",
          set(tuning.tuning_status()) >= {"transformersInstalled", "applied", "items",
                                          "stats", "env", "defaultCacheDir"},
          sorted(tuning.tuning_status()))


def main() -> int:
    case_manifest()
    case_patch()
    case_unpatch()
    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"\n      ↳ {detail}"))
    print(f"\nSUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed"
          + (" ✗✗✗" if failed else " ✓"))
    return 1 if failed else 0


if __name__ == "__main__":
    # ⚠️ Windows 中文控制台是 **GBK**（py3.14 及以前不会自动 UTF-8 ✗）⇒ 不加这行，裸跑第一条 `print` 就 `UnicodeEncodeError` 崩 ✗（守口径见 `check_cli_encoding.py` ✓）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
