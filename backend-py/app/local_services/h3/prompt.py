"""把 **H3 的 prompt 契约**接到**提交前那一步** ✓（2026-09-24 补 ✓ —— 「能力接不出去不算功能」✗）。

判据在引擎侧（:mod:`app.services.engine.conditioning` ✓ 纯逻辑、可自检 ✓）；本模块只做**接缝** ✓，
把「一段文案 ⇒ 真正提交给 H3 的那段 prompt」这条路**收口成一处** ✓：

1. **全局提示词**（全片统一的风格/世界观 ✓）拼在最前 ✓ —— 单段提示词不用重复抄常量 ✓；
2. **文本写法 → 标签** ✓（``@图1`` / ``【图2】`` / ``@音1`` ⇒ ``<Picture 1>`` / ``<Audio 1>`` ✓）；
3. ⭐ **提交前核编号** ✗：引用了 ``<Picture 3>`` 却只给 2 张 ⇒ **当场拒** ✓✗
   （错误编号最常见的落点就是这里 ✓ —— 续接尾帧排最前 ⇒ 换场景后编号会**前移** ✓✗）；
4. ⭐ **该追加的声明必须追加** ✓：只写 ``<Audio 1>`` 那种绑定句**不算声明** ✗✗
   （照标签跳过 ⇒ 模型自由发挥、成片音轨与配音相关性≈0 ✓✗）。

⚠️ 本模块**不调模型、不联网** ✗：产出的是「该提交的那段文字 + 一份计划」✓
（真正提交由调用方走 ``queue_prompt`` ✓）。
"""
from __future__ import annotations

from typing import Any, Sequence

from app.services.engine import conditioning as cond

__all__ = ["build_prompt"]


def build_prompt(text: str, *, pictures: int = 0, audios: int = 0, videos: int = 0,
                 audio_refs: Sequence[cond.AudioReference] = (), video_count: int = 0,
                 global_prompt: str = "") -> dict[str, Any]:
    """一段文案 ⇒ **该提交给 H3 的 prompt** ✓ + 一份「做了什么」的计划 ✓。

    返回键 ✓：``prompt``（最终文本 ✓）、``markup``（转换后出现的标签编号 ✓）、
    ``plan``（声明计划 ✓）、``prefixed``（有没有拼全局提示词 ✓）。
    """
    head = str(global_prompt or "").strip()
    body = str(text or "").strip()
    combined = f"{head}\n{body}" if head and body else (head or body)

    marked = cond.h3_markup(combined)
    # ⚠️⚠️ 编号校验必须在**声明拼接之前** ✓（声明里也会出现 ``<Audio N>`` 标签 ✓）。
    #    ⚠️ 但**可用数**要把「声明将要引入的那些标签」算进去 ✗✗：本函数是**幂等**的 ✓
    #    ⇒ 二次进入时输入里**已经带着**上次追加的声明 ✓（含 ``<Audio 1>`` ✓）——
    #    若可用数仍按「只是音频素材」算，就会把**自己上次写的声明**判成越界 ✓✗
    #    （2026-09-24 本套第一次跑就栽在这上面 ✓）。
    #    ⇒ 参考素材本身就是那几路标签的**合法来源** ✓（一路参考音频 = 一个 ``<Audio N>`` ✓）。
    usable_audios = max(int(audios), len(audio_refs))
    usable_videos = max(int(videos), int(video_count))
    cond.validate_markup(marked, pictures=int(pictures), audios=usable_audios, videos=usable_videos)
    final, plan = cond.apply_declarations(marked, audio_refs=audio_refs, video_count=video_count)
    return {"prompt": final, "markup": cond.markup_references(marked), "plan": plan.to_dict(),
            "prefixed": bool(head and body)}
