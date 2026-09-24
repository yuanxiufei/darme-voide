"""S7 自检：**加速链**（配置解析 + 提交前校验 + 接线 ✓ 纯逻辑、零依赖、零网络 ✓ 2026-09-24）。

判据按「**改完就知道能不能连上**」写 ✓：

* 没给注册表 ⇒ 一律 ``unchecked`` ✓（**不是** ``ok`` ✗ —— 本仓纪律：没查 ≠ 通过 ✗）；
* 未注册节点 / ``kind`` 接错 / 输入类型不对 ⇒ ``error`` ✓ 且 ``blocks_submit`` 为真 ✗（不许提交 ✓）；
* 参数名不存在 / 缺必填 ⇒ ``warn`` ✓（能提交，但会被忽略 / 被拒 ✓ —— 与 ``error`` **必须分开** ✗）；
* 接线：串行指针**逐级前移** ✓、``error`` 项**自动跳过且给理由** ✓、
  输出槽位**从注册表数出来**（没注册表 ⇒ 报错 ✓ **不猜** ✗）。

运行::

    ./.venv/Scripts/python.exe tests/engine_accel_chain_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import accel_chain as chain_mod  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    _SKIPS.append(reason)


def _raises(call: Any, needle: str = "") -> str | None:
    """抛错且消息含 ``needle`` ⇒ 返回消息 ✓；否则 ``None`` ✓（**抛了别的错也算没抛对** ✗）。"""
    try:
        call()
    except Exception as err:  # noqa: BLE001
        text = str(err)
        return text if needle in text else None
    return None


def full_node_types() -> dict[str, Any]:
    """合成注册表 ✓（形状照 ComfyUI ``/object_info`` ✓：``input.required`` / ``optional`` + ``output`` 类型表 ✓）。"""
    return {
        "LoraLoader": {"name": "LoraLoader",
                       "input": {"required": {"model": ["MODEL"], "clip": ["CLIP"],
                                              "lora_name": ["STRING", {"default": ""}],
                                              "strength_model": ["FLOAT", {"default": 1.0}],
                                              "strength_clip": ["FLOAT", {"default": 1.0}]}},
                       "output": ["MODEL", "CLIP"]},
        "TESpeedMiniMaxH3": {"name": "TESpeedMiniMaxH3",
                             "input": {"required": {"model": ["MODEL"]}},
                             "output": ["MODEL"]},
        "PathchSageAttentionKJ": {"name": "PathchSageAttentionKJ",
                                  "input": {"required": {"model": ["MODEL"]}},
                                  "output": ["MODEL"]},
        "SpectrumApplyMiniMaxH3": {"name": "SpectrumApplyMiniMaxH3",
                                   "input": {"required": {"model": ["MODEL"]},
                                             "optional": {"enabled": ["BOOLEAN", {"default": True}],
                                                          "blend_weight": ["FLOAT", {"default": 0.5}]}},
                                   "output": ["MODEL"]},
    }


def by_id(statuses: list[chain_mod.ChainStatus]) -> dict[str, chain_mod.ChainStatus]:
    return {status.id: status for status in statuses}


def joined(status: chain_mod.ChainStatus) -> str:
    """一环的全部理由拼起来 ✓（断言「**说过这句话**」✓ —— 只断言第 0 条会被**理由顺序**

    的变化打碎 ✓✗：本套就真撞过一次 ✓）。
    """
    return " ".join(issue.message for issue in status.issues)


def case_parse() -> None:
    """① 配置解析 ✓（合同锚点 + 负对照：每类错都必须**报错** ✗）。"""
    default = chain_mod.DEFAULT_CHAIN
    check("① 内置默认链：4 环 ✓、顺序固定 ✓、只有 LoRA 那环不是第三方 ✓",
          [entry.id for entry in default] == ["lora", "tespeed", "sage", "spectrum"]
          and [entry.third_party for entry in default] == [False, True, True, True],
          [entry.to_dict() for entry in default])
    check("①′ 默认链的 kind：LoRA 是 ``lora`` ✓（接管 model+clip ✓）、其余是 ``model_only`` ✓",
          [entry.kind for entry in default] == ["lora", "model_only", "model_only", "model_only"],
          [entry.kind for entry in default])

    entries = chain_mod.parse_chain({"chain": [{"id": "a", "class_type": "X"}]})
    check("①″ 省略项：kind 缺省 ``model_only`` ✓、``default_on`` 缺省为真 ✓（⚠️ 缺省值要有据 ✓）",
          entries[0].kind == "model_only" and entries[0].default_on is True, entries[0].to_dict())
    check("①‴ 裸数组也收 ✓（两种形态都认 ✓ —— 只认一种会让调用方悄悄拿到空链 ✗）",
          len(chain_mod.parse_chain([{"id": "a", "class_type": "X"}])) == 1)

    check("①⁴ kind 不认识 ⇒ 报错并列出合法的两个 ✓",
          _raises(lambda: chain_mod.parse_chain([{"id": "a", "class_type": "X", "kind": "video"}]), "model_only")
          is not None)
    check("①⁵ id 重复 ⇒ 报错 ✓（重复会让前端勾选串行 ✗）",
          _raises(lambda: chain_mod.parse_chain([{"id": "a", "class_type": "X"},
                                                 {"id": "a", "class_type": "Y"}]), "两次") is not None)
    check("①⁶ ``default_on`` 给了字符串 ``\"false\"`` ⇒ **报错** ✗（它有值 ≠ 它为真 ✓）",
          _raises(lambda: chain_mod.parse_chain(
              [{"id": "a", "class_type": "X", "default_on": "false"}]), "不是布尔值") is not None)
    check("①⁷ 缺 ``class_type`` / 缺 ``id`` / ``inputs`` 不是对象 / 没有 ``chain`` 键 ⇒ 各自报错 ✓",
          all(_raises(call) is not None for call in (
              lambda: chain_mod.parse_chain([{"id": "a"}]),
              lambda: chain_mod.parse_chain([{"class_type": "X"}]),
              lambda: chain_mod.parse_chain([{"id": "a", "class_type": "X", "inputs": 5}]),
              lambda: chain_mod.parse_chain({"items": []}))))


def case_validate_unchecked() -> None:
    """② 没给注册表 ⇒ **``unchecked``** ✓（守卫：这**不是**通过 ✗）。"""
    statuses = chain_mod.validate_chain(chain_mod.DEFAULT_CHAIN, None)
    check("② 没给注册表 ⇒ 4 环全是 ``unchecked`` ✓ 且**一个都不许当通过** ✗（``ok`` 计数为 0 ✓）",
          len(statuses) == 4 and all(status.level == "unchecked" for status in statuses)
          and sum(1 for status in statuses if status.level == "ok") == 0,
          [status.level for status in statuses])
    check("②′ ``unchecked`` **不拦提交** ✓ 但也不自称通过 ✓（拦不拦由调用方按纪律决定 ✓）",
          all(not status.blocks_submit for status in statuses))


def case_validate_ok() -> None:
    """③ 注册表齐全 ⇒ 全 ``ok`` ✓（正向对照：与 ② 的「没查」分得开 ✓）。"""
    statuses = chain_mod.validate_chain(chain_mod.DEFAULT_CHAIN, full_node_types())
    check("③ 合成注册表齐全 ⇒ 4 环全 ``ok`` ✓（⇒ ②的 unchecked 不是「必然不 ok」的套套逻辑 ✓）",
          all(status.level == "ok" and not status.issues for status in statuses),
          [status.to_dict() for status in statuses])


def case_validate_error() -> None:
    """④ 三类**硬错** ⇒ ``error`` ✓ 且 ``blocks_submit`` 为真 ✗。"""
    types = full_node_types()
    types.pop("TESpeedMiniMaxH3")
    statuses = by_id(chain_mod.validate_chain(chain_mod.DEFAULT_CHAIN, types))
    check("④ 节点未注册 ⇒ ``error`` ✓ 且理由**可行动**（点名「完全重启 ComfyUI」✓、"
          "并说清刷新网页不算 ✗）",
          statuses["tespeed"].level == "error"
          and "完全重启" in statuses["tespeed"].issues[0].message
          and "刷新" in statuses["tespeed"].issues[0].message,
          statuses["tespeed"].to_dict())
    check("④′ 未注册那环 ``blocks_submit`` 为真 ✗（不许把接不上的项提交上去 ✓）",
          statuses["tespeed"].blocks_submit and statuses["lora"].blocks_submit is False)

    no_clip = {"LoraLoader": {"name": "LoraLoader",
                              "input": {"required": {"model": ["MODEL"], "lora_name": ["STRING"]}},
                              "output": ["MODEL"]}}
    got = chain_mod.validate_chain(chain_mod.parse_chain(
        [{"id": "lora", "class_type": "LoraLoader", "kind": "lora"}]), no_clip)[0]
    check("④″ ``kind=lora`` 而节点没有 ``clip`` 输入 ⇒ ``error`` ✓（理由点名缺哪条流 ✓）",
          got.level == "error" and "clip" in joined(got), got.to_dict())

    # ⚠️ 夹具要**干净** ✓：这个合成节点还有个必填 ``clip`` ⇒ 必须把它给上 ✓
    #    否则混进来的「缺必填」warn 会让这条断言验的是别的东西 ✓✗（本套真撞过一次 ✓）。
    no_model = {"X": {"name": "X", "input": {"required": {"clip": ["CLIP"]}}, "output": ["MODEL"]}}
    got = chain_mod.validate_chain(chain_mod.parse_chain(
        [{"id": "x", "class_type": "X", "kind": "model_only", "inputs": {"clip": ["7", 0]}}]),
        no_model)[0]
    check("④‴ ``model_only`` 而节点没有 ``model`` ⇒ ``error`` ✓（**不是** warn ✗ —— 接错进不了链 ✓）"
          "—— ⚠️ 且**不许**再叠一句「缺必填」✗（接管流不算缺 ✓）",
          got.level == "error" and "model" in joined(got) and "缺少必填" not in joined(got),
          got.to_dict())

    wrong_type = {"X": {"name": "X", "input": {"required": {"model": ["IMAGE"]}}, "output": ["MODEL"]}}
    got = chain_mod.validate_chain(chain_mod.parse_chain(
        [{"id": "x", "class_type": "X", "kind": "model_only"}]), wrong_type)[0]
    check("④⁴ ``model`` 输入声明的类型是 IMAGE ⇒ ``error`` ✓（拿去接 MODEL 流会**拿错流** ✓✗）",
          got.level == "error" and "IMAGE" in joined(got) and "缺少必填" not in joined(got),
          got.to_dict())


def case_validate_warn() -> None:
    """⑤ 软问题 ⇒ ``warn`` ✓（**能提交** ✓ —— 与 ``error`` 分得开 ✓）。

    ⚠️ **带默认值的 required 不算缺** ✗✗（2026-09-24 接真注册表时踩到 ✓）：ComfyUI 里
    ``["FLOAT", {"default": 1.0}]`` 这种参数**不传也行**（节点自己填 ✓）⇒ 算成缺必填会让
    **每条真链都多出一堆噪音 warn** ✓✗。
    """
    types = full_node_types()
    entries = chain_mod.parse_chain([
        {"id": "lora", "class_type": "LoraLoader", "kind": "lora",
         "inputs": {"lora_name": "x.safetensors", "wrong_extra": 1}},
        {"id": "te", "class_type": "TESpeedMiniMaxH3", "kind": "model_only",
         "inputs": {"model_should_not_be_here": 1}},
    ])
    statuses = by_id(chain_mod.validate_chain(entries, types))
    check("⑤ 参数名不存在 ⇒ ``warn`` ✓ 且**点名**那个参数 ✓（写错会被节点默默忽略 ✓✗）",
          statuses["lora"].level == "warn" and "wrong_extra" in statuses["lora"].issues[0].message,
          statuses["lora"].to_dict())
    types2 = {"Y": {"name": "Y", "input": {"required": {"model": ["MODEL"], "strength": ["FLOAT"]}},
                    "output": ["MODEL"]}}
    got = chain_mod.validate_chain(chain_mod.parse_chain(
        [{"id": "y", "class_type": "Y", "kind": "model_only"}]), types2)[0]
    check("⑤′ 缺必填（**非接管流**的参数 ``strength`` ✓）⇒ ``warn`` ✓ 且措辞说清"
          "「没有默认值时才被拒」✓（别一律说必错 ✗）",
          got.level == "warn" and any("缺少必填" in issue.message and "strength" in issue.message
                                      for issue in got.issues), got.to_dict())
    check("⑤‴ ⭐ 接管流（``model`` ✓）不在 ``inputs`` 里**不算缺必填** ✗ —— 否则每一环都报 warn、"
          "真问题（没注册 / 接错流 ✓）被淹没 ✓✗",
          "缺少必填" not in joined(statuses["te"]), statuses["te"].to_dict())
    with_default = {"Z": {"name": "Z", "input": {"required": {"model": ["MODEL"],
                                                            "strength": ["FLOAT", {"default": 1.0}]}},
                          "output": ["MODEL"]}}
    got = chain_mod.validate_chain(chain_mod.parse_chain(
        [{"id": "z", "class_type": "Z", "kind": "model_only"}]), with_default)[0]
    check("⑤‴ ⭐ **带 ``default`` 的 required 不算缺** ✗✗：不传它 ⇒ **不报 warn** ✓"
          "（否则真链上每条都会多出噪音 warn ✓✗ —— 接真注册表时踩到过 ✓）",
          got.level == "ok" and not got.issues, got.to_dict())
    check("⑤⁴ 自己往 inputs 里写 ``model`` ⇒ **校验期**就判 ``error`` ✓（别等到接线才报 ✗）",
          chain_mod.validate_chain(chain_mod.parse_chain(
              [{"id": "z", "class_type": "PathchSageAttentionKJ", "kind": "model_only",
                "inputs": {"model": ["9", 0]}}]), full_node_types())[0].level == "error")
    check("⑤″ ``warn`` **不拦提交** ✓（与 ``error`` 的区别正在这里 ✓）",
          all(not status.blocks_submit for status in statuses.values()))


def case_build() -> None:
    """⑥ 接线 ✓：串行指针逐级前移 ✓、LoRA 那级接管 clip ✓、输出槽位**从注册表数** ✓。"""
    types = full_node_types()
    build = chain_mod.build_chain_nodes(chain_mod.DEFAULT_CHAIN, node_types=types,
                                        model=["4", 0], clip=["5", 0])
    check("⑥ 四环全接上 ✓ 且 id 带前缀 ✓",
          list(build.nodes) == ["accel_lora", "accel_tespeed", "accel_sage", "accel_spectrum"]
          and build.enabled == ("lora", "tespeed", "sage", "spectrum"), list(build.nodes))
    check("⑥′ 第一环的 model 指上游 ✓、第二环指第一环的 **MODEL 槽（0 ✓）** ⇒ 逐级前移 ✓",
          build.nodes["accel_lora"]["inputs"]["model"] == ["4", 0]
          and build.nodes["accel_tespeed"]["inputs"]["model"] == ["accel_lora", 0],
          {key: node["inputs"].get("model") for key, node in build.nodes.items()})
    check("⑥″ LoRA 那一环**同时接管 clip** ✓ 且槽位是 1 ✓（注册表 ``output=[MODEL, CLIP]`` ✓）"
          "⇒ 最终 clip 指针指向它 ✓、别的环**不动 clip** ✓",
          build.nodes["accel_lora"]["inputs"]["clip"] == ["5", 0]
          and build.clip == ["accel_lora", 1]
          and all("clip" not in build.nodes[key]["inputs"]
                  for key in ("accel_tespeed", "accel_sage", "accel_spectrum")),
          build.clip)
    check("⑥‴ 最终 model 指针 = 最后一环 ✓（上游拿它接采样 ✓）",
          build.model == ["accel_spectrum", 0], build.model)

    build2 = chain_mod.build_chain_nodes(
        chain_mod.DEFAULT_CHAIN, node_types=types, model=["4", 0], clip=["5", 0],
        statuses=chain_mod.validate_chain(chain_mod.DEFAULT_CHAIN, types),
        enabled={"sage": False})
    check("⑥⁴ 显式关掉一环 ⇒ 它不进图 ✓ 但**其余照样串上** ✓（且不被跳过项断开 ✓）",
          "accel_sage" not in build2.nodes and build2.model == ["accel_spectrum", 0]
          and ("sage", "未勾选 ✓") in build2.skipped, build2.skipped)

    bad_types = dict(types)
    bad_types.pop("TESpeedMiniMaxH3")
    build3 = chain_mod.build_chain_nodes(
        chain_mod.DEFAULT_CHAIN, node_types=bad_types, model=["4", 0], clip=["5", 0],
        statuses=chain_mod.validate_chain(chain_mod.DEFAULT_CHAIN, bad_types))
    check("⑥⁵ 判 ``error`` 的项**自动跳过且给理由** ✓（不是静默少一环 ✗）",
          "accel_tespeed" not in build3.nodes
          and any(key == "tespeed" and "error" in reason for key, reason in build3.skipped),
          build3.skipped)

    check("⑥⁶ 自己往 inputs 里写 model ⇒ 报错 ✓（否则会**绕开前一级** ✓✗）",
          _raises(lambda: chain_mod.build_chain_nodes(
              chain_mod.parse_chain([{"id": "a", "class_type": "LoraLoader", "kind": "lora",
                                      "inputs": {"model": ["9", 0]}}]),
              node_types=types, model=["4", 0], clip=["5", 0]), "绕开") is not None)

    check("⑥⁷ 没给注册表 ⇒ **报错** ✓（输出槽位不猜 ✗ —— 猜错就是把线接到别的输出上 ✓✗）",
          _raises(lambda: chain_mod.build_chain_nodes(
              chain_mod.DEFAULT_CHAIN, node_types=None, model=["4", 0], clip=["5", 0]), "不猜") is not None)

    no_model_out = dict(types)
    no_model_out["PathchSageAttentionKJ"] = {"name": "PathchSageAttentionKJ",
                                             "input": {"required": {"model": ["MODEL"]}},
                                             "output": ["LATENT"]}
    check("⑥⁸ 注册表里该节点**没有 MODEL 输出** ⇒ 报错 ✓（不许照着「第一个输出」硬接 ✗）",
          _raises(lambda: chain_mod.build_chain_nodes(
              [chain_mod.DEFAULT_CHAIN[2]], node_types=no_model_out,
              model=["4", 0], clip=["5", 0]), "槽") is not None)


def main() -> int:
    case_parse()
    case_validate_unchecked()
    case_validate_ok()
    case_validate_error()
    case_validate_warn()
    case_build()
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
