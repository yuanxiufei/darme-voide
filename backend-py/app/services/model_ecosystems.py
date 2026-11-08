"""本机模型生态落点 —— 「世界各大模型」在电脑里分别落在哪。

为什么需要（2026-09-25 用户口径 ✓）：扫描要覆盖**世界各大模型** ✓，不能只认 Ollama ✗ ——
``local_model_scan`` 原来只扫本仓 ``models/``、``local_services/`` 与 ComfyUI ✗ ⇒
HuggingFace / ModelScope / LM Studio / GPT4All / Jan / llama.cpp 里躺着的模型**扫不到**（而且不报错 ✗），
用户会以为「本机没模型」✗✗。

本模块把「各生态的本地落点」收成**一张表** ✓，供扫描器当默认根目录，并给每个命中文件标注来源生态 ✓。

三条口径（都在扫描结果里如实体现 ✓，不许含糊 ✗）：

1. **零外部依赖** ✓：全部是**读目录** ✓ —— 不要求任何生态的服务在跑（``ollama serve`` /
   LM Studio / vLLM…）✗，这正是用户口径的原话 ✓；
2. **缓存 ≠ 能推理** ✓：读到的只是权重文件 ✓ ⇒ 默认 ``callable=False`` + ``note`` 说清**怎么才真能用** ✓
   （本仓铁律：**没查 ≠ 通过** ✓）；
3. **只读** ✓：只列文件，不删/不改别人生态的库 ✓（删除要由那个生态自己的服务做 ✓）。

⚠️ 与 ``_RULES``（启发式识别）**分工不同** ✗：规则回答「这是**什么**权重」✓，
本模块回答「它**在哪**、能不能**直接调**」✓。两者冲突时**规则优先** ✓（见 :func:`apply_ecosystem`）。

## ⚠️ 实现已搬 ``app/core/model_ecosystems.py``（2026-11-08 ✓）

本文件**只剩再导出** ✓ —— 下面每一个名字都是**同一个对象** ✓
（``model_ecosystems.ECOSYSTEMS is app.core.model_ecosystems.ECOSYSTEMS`` ✓），不是副本 ✗。

搬家的理由：引擎的 ``inventory.candidate_roots()`` 要「盘上还有没有别处的副本」⇒ 要这份落点表 ✓，
而引擎顶层只许 import ``app.core.*`` ✗（``tests/engine_independence_test.py`` 逐条判 ✓）。
这张表读的**只有生态约定与环境变量** ✓（零业务数据 ✓）；唯一的服务层牵连是 Ollama 的库根候选，
那段也已搬进 :mod:`app.core.ollama_roots` ✓（读环境变量 + 一个 ``mode=ro`` 打开的 sqlite ✓）。

⚠️ 新增生态请改 ``app/core/model_ecosystems.py`` ✗✗：改在这一份会**静默失效** ✓✗
（调用方拿的是 core 那个对象 ✓）。
"""

from __future__ import annotations

from ..core.model_ecosystems import (
    ECOSYSTEMS,
    OVERRIDE_RUNTIMES,
    Ecosystem,
    all_roots,
    apply_ecosystem,
    build_ollama_manifest_suggestion,
    build_suggestion,
    by_ecosystem,
    detect,
    get,
    labels,
    roots,
    skip_prefixes,
)

__all__ = [
    "ECOSYSTEMS",
    "OVERRIDE_RUNTIMES",
    "Ecosystem",
    "all_roots",
    "apply_ecosystem",
    "build_ollama_manifest_suggestion",
    "build_suggestion",
    "by_ecosystem",
    "detect",
    "get",
    "labels",
    "roots",
    "skip_prefixes",
]
