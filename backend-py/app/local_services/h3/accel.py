"""把**加速链**接到**真 ComfyUI** 上 ✓（2026-09-24 补 ✓ —— 「能力接不出去不算功能」✗）。

判据在引擎侧（:mod:`app.services.engine.accel_chain` ✓ 纯逻辑、可自检 ✓）；本模块只做**接缝** ✓：

* 拿 ``ComfyUIClient.object_info_all()`` 的**真实注册表** ✓ ⇒ 逐项校验（注册 / 参数名 / 缺必填 /
  **kind 接线** ✓）；
* 判 ``error`` 的项**不进图** ✗ 且报告里**带理由** ✓（静默少一环会让图悄悄变样 ✓✗）；
* ⭐ **ComfyUI 连不上 ⇒ 判「没查」（``unchecked``）** ✗ 而**不是**判错 ✓✗ —— 连不上和「节点真没装」
  是两回事 ✓；此时也**没法接线** ✓（输出槽位要从注册表数出来 ✓，本仓不猜 ✗）。

⚠️ 本模块**不自己造客户端、不联网** ✗：客户端由调用方给 ✓（这样才能用 stub 测 ✓ ——
见 ``tests/h3_comfyui_test.py`` 那套真 HTTP stub 的写法 ✓）。
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from app.services.engine import accel_chain as chain_mod

__all__ = ["check_chain"]


def check_chain(client: Any, raw_chain: Any, *, model: Sequence[Any], clip: Sequence[Any],
                enabled: Mapping[str, bool] | None = None,
                prefix: str = "accel") -> dict[str, Any]:
    """校验并（能过就）**接线** ✓ ⇒ 一份可给前端/日志的报告 ✓。

    返回键 ✓：

    * ``statuses``：每环的结论（``ok`` / ``warn`` / ``error`` / ``unchecked`` ✓ + 理由 ✓）；
    * ``counts``：各级数量 ✓（前端画徽标用 ✓）；
    * ``blocked``：有没有**不许提交**的项 ✗；
    * ``unreachable``：ComfyUI **连不上**时的原因 ✓（此时 ``unchecked`` ✓ **不是**判错 ✓）；
    * ``build``：``{nodes, model, clip, enabled, skipped}`` ✓ —— ⚠️ 只有在**不 blocked 且注册表可用**
      时才有 ✓（否则 ``None`` ✓，**不猜**槽位 ✗）。
    """
    entries = chain_mod.parse_chain(raw_chain)
    unreachable: str | None = None
    node_types: Mapping[str, Any] | None
    try:
        node_types = client.object_info_all()
        if not isinstance(node_types, Mapping) or not node_types:
            unreachable = "ComfyUI 返回的 ``/object_info`` 是空的 ✗ ⇒ 一个节点都查不到 ✓"
            node_types = None
    except Exception as err:  # noqa: BLE001 —— 连不上是**环境**问题 ✓ 不该在这里炸 ✗
        unreachable = f"连不上 ComfyUI（{type(err).__name__}: {err} ✓）"
        node_types = None

    statuses = chain_mod.validate_chain(entries, node_types)
    counts = {level: 0 for level in ("ok", "warn", "error", "unchecked")}
    for status in statuses:
        counts[status.level] = counts.get(status.level, 0) + 1
    blocked = any(status.blocks_submit for status in statuses)
    build = None
    if node_types is not None and not blocked:
        build = chain_mod.build_chain_nodes(entries, node_types=node_types, model=model, clip=clip,
                                            statuses=statuses, enabled=enabled, prefix=prefix)
    return {"statuses": [status.to_dict() for status in statuses], "counts": counts,
            "blocked": blocked, "unreachable": unreachable,
            "build": None if build is None else build.to_dict()}
