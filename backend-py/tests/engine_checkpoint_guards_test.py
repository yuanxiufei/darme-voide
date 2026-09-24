"""S7 自检：**权重侧的两条守卫**（内嵌元数据契约 + 未实现布局族的**具名**拒绝 ✓ 零依赖 ✓ 2026-09-24）。

为什么这两条要一起钉 ✗：它们守的都是**同一类错** ✓ —— 「**松读**」✗：

* 内嵌契约松读 ⇒ 拿**旧格式当新格式**用 ✓✗（所以 ``format`` 必须**等于**期望 ✓、报错要把两个值都印出来 ✓；
  布尔开关**必须真是 True** ✓、「有键」不算 ✓；``"3"`` 不算整数 ✓、「有值」不算对类型 ✓）。
  另：**没有** ``metadata`` ⇒ 返回 ``None`` ✓ —— ⚠️ 那**不是通过** ✗（与 ``unchecked`` 同理 ✓）。
* 布局族松读 ⇒ 撞上 ``convrot`` 这种**未实现**的族时只说一句「判不出布局」✗ ⇒ 下一个人
  会以为是文件坏了、或者以为「多试几次就好」✓✗。⇒ 必须**点名**「是什么 + 为什么不做」✓。

运行::

    ./.venv/Scripts/python.exe tests/engine_checkpoint_guards_test.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import checkpoint_meta as meta_mod  # noqa: E402
from app.services.engine import quant as quant_mod  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    _SKIPS.append(reason)


def _raises(call: Any, needle: str = "") -> str | None:
    try:
        call()
    except Exception as err:  # noqa: BLE001
        text = str(err)
        return text if needle in text else None
    return None


def _outer(contract: Any, *, as_json: bool = True) -> dict[str, Any]:
    return {"metadata": json.dumps(contract) if as_json else contract}


def case_contract_ok() -> None:
    """① 正常读 ✓（顺带钉「额外键保留」✗ —— 前向兼容 ✓）。"""
    contract = {"format": "h3-latent-upscaler-v1", "strict_latent_only": True,
                "base_config": {"channels": 24}, "config": {"blocks": 4}, "step": 12000,
                "future_field": "别把读取方弄崩 ✓"}
    got = meta_mod.read_contract(_outer(contract), expect_format="h3-latent-upscaler-v1",
                                 required=("base_config", "config"),
                                 truthy=("strict_latent_only",), integers=("step",))
    check("① 契约读写往返 ✓（含**额外键保留** ✓ —— 产物多写字段不该把读取方弄崩 ✓）",
          got is not None and got == contract, got)
    check("①′ 三个校验族都过 ⇒ 不抛 ✓（正向对照：下面那些报错不是「必然抛」的套套逻辑 ✓）",
          got is not None and got["step"] == 12000)


def case_contract_absent() -> None:
    """② **没有**契约 ⇒ ``None`` ✓ —— ⚠️ 那**不是通过** ✗（与 ``unchecked`` 同理 ✓）。"""
    check("② 没有 ``metadata`` ⇒ ``None`` ✓（**不抛错、也不自称通过** ✓ ⇒ 要不要拒由调用方定 ✓）",
          meta_mod.read_contract({}) is None and meta_mod.read_contract(None) is None
          and meta_mod.read_contract({"other": 1}) is None)


def case_contract_broken() -> None:
    """③ 坏契约 ⇒ **分开报** ✓（排查方向不同 ✓ —— 别一律说「格式不对」✗）。"""
    check("③ ``metadata`` 不是字符串（如直接塞 dict ✓）⇒ 报错且点名**类型** ✓",
          _raises(lambda: meta_mod.read_contract(_outer({"format": "x"}, as_json=False)),
                  "必须是**字符串**") is not None)
    bad_json = _raises(lambda: meta_mod.read_contract({"metadata": "{不是 JSON"}), "JSON")
    check("③′ JSON 坏了 ⇒ 报错点名 **JSON** ✓（并提示可能是被改过/写入截断 ✓）",
          bad_json is not None and "截断" in bad_json, bad_json)
    check("③″ 契约是**数组**（JSON 合法但不是对象 ✓）⇒ 报错 ✓",
          _raises(lambda: meta_mod.read_contract({"metadata": "[1, 2]"}), "JSON **对象**") is not None)


def case_contract_strict() -> None:
    """④ **严格**：format 对不上 / 缺字段 / 开关不是真 True / 整数不是整数 ⇒ 各自报 ✓。"""
    got = _raises(lambda: meta_mod.read_contract(_outer({"format": "v2"}), expect_format="v1"), "v2")
    check("④ ``format`` 对不上 ⇒ 报错且**两个值都印出来** ✓（否则没法判断是谁的问题 ✓）",
          got is not None and "'v1'" in got, got)
    check("④′ 缺 ``required`` 里的键 ⇒ 报错**点名缺哪个** ✓ 且印出契约里现有键 ✓",
          _raises(lambda: meta_mod.read_contract(_outer({"format": "v1"}),
                                                 required=("base_config",)), "base_config") is not None)
    off = _raises(lambda: meta_mod.read_contract(_outer({"format": "v1", "strict_latent_only": False}),
                                                 truthy=("strict_latent_only",)), "有键 ≠ 为真")
    check("④″ ⭐ 布尔开关存在但为 ``False`` ⇒ 报错 ✓（**有键 ≠ 为真** ✓✗ —— 松读会把它当已声明 ✗）",
          off is not None, off)
    check("④‴ ``truthy`` 的键**缺失** ⇒ 也报错 ✓（拿不到就是没声明 ✓）",
          _raises(lambda: meta_mod.read_contract(_outer({"format": "v1"}),
                                                 truthy=("strict_latent_only",)), "") is not None)
    check("④⁴ 整数字段：``\"3\"`` / ``-1`` / ``True`` ⇒ 都报错 ✓（**有值 ≠ 是对的类型** ✓✗）",
          all(_raises(call) is not None for call in (
              lambda: meta_mod.read_contract(_outer({"step": "3"}), integers=("step",)),
              lambda: meta_mod.read_contract(_outer({"step": -1}), integers=("step",)),
              lambda: meta_mod.read_contract(_outer({"step": True}), integers=("step",)))))
    check("④⁵ ``integers`` 的键缺失 ⇒ **放过** ✓（缺了不算错 ✗ —— 要它就必须写进 ``required`` ✓）",
          meta_mod.read_contract(_outer({"format": "v1"}), integers=("step",)) is not None
          and meta_mod.read_contract(_outer({"format": "v1", "step": None}),
                                     integers=("step",)) is not None)
    check("④⁶ 不给 ``expect_format`` ⇒ 不查 ``format`` ✓（调用方按自己的契约决定 ✓）",
          meta_mod.read_contract(_outer({"anything": 1})) == {"anything": 1})


def case_unsupported_family() -> None:
    """⑤ ⭐ 未实现的布局族要**具名拒绝** ✓（撞上 ``convrot`` 时别只说「判不出布局」✗）。"""
    msg = quant_mod.unsupported_family(
        source="minimax_h3_ref2va_pruned_int8_convrot.safetensors")
    check("⑤ 文件名带 ``convrot`` ⇒ 返回理由 ✓ 且**点名族名** ✓、说清「要旋转参数」✓、"
          "并强调**不猜** ✓",
          msg is not None and "convrot" in msg and "旋转参数" in msg and "不猜" in msg, msg)
    # ⚠️⚠️ 这里**不能**写「不许出现『判不出布局』这个词」✗ —— 文案为澄清而**提到**了它 ✓✗
    #    （本套第一次就栽在这上面 ✓；`engine_dual_stream_test` 的 55″ 同款 ✓）。
    #    ⇒ 只断言**语义**：必须说清「不是那个 ✓」+「是本仓没实现 ✓」✓。
    check("⑤′ ⭐ 且必须把话**说透** ✓：这是「不是文件坏了 ✓、是本仓没实现 ✓」"
          "（⚠️ 只断言「某个词不出现」会被**为澄清而提到它**的文案打碎 ✗✗）",
          msg is not None and "不是文件坏了" in msg and "没实现" in msg, msg)
    check("⑤″ 干净的模型名 ⇒ ``None`` ✓（正向对照：上面的命中不是「永远命中」的套套逻辑 ✓）",
          quant_mod.unsupported_family(source="minimax_h3_fl2va_int8.safetensors") is None
          and quant_mod.unsupported_family() is None)
    check("⑤‴ 张量名里出现也算命中 ✓（大小写不敏感 ✓）",
          quant_mod.unsupported_family(tensor_names=["x.weight_CONVROT_scale"]) is not None)
    check("⑤⁴ ``UNSUPPORTED_FAMILIES`` 是**公开**的 ✓（别让人只能看报错才知道有哪些族 ✗）",
          "convrot" in quant_mod.UNSUPPORTED_FAMILIES)


def main() -> int:
    case_contract_ok()
    case_contract_absent()
    case_contract_broken()
    case_contract_strict()
    case_unsupported_family()
    failures = [(name, detail) for name, passed, detail in _RESULTS if not passed]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    for reason in _SKIPS:
        print("SKIP  " + reason)
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed"
          + (f"（skip {len(_SKIPS)}）" if _SKIPS else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
