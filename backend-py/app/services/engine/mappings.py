"""**权重命名映射**（零依赖 ✓）—— 把"别家的张量名"对到"本仓的模块名" ✓。

## 为什么要有"预设 + 报告"（而不是让工作站上现研究 ✗）

各家 checkpoint 的命名差异**高度重复** ✓（外层前缀 ``model.`` / ``model.diffusion_model.`` ✓、
把 q/k/v 分成三个张量 ✓、把注意力叫 ``attn`` / ``attention`` ✓ ……）。真正麻烦的只有两类：

1. **改名**（正则替换 ✓）—— 例如 ``model.diffusion_model.blocks.0.attn.wq.weight``
   → ``blocks.0.attn.in_proj_weight`` ✓（本仓 DiT 用 ``nn.MultiheadAttention`` 的**融合**权重 ✓）；
2. **合并**（q/k/v 三段**拼**成一段 ✓ —— 这不只是改名 ✗，是**张量运算** ✓）。

所以本模块给两样东西 ✓：

* :data:`PRESETS` —— 常见约定的**预设表** ✓（真权重到手时先套预设看报告 ✓）；
* :func:`plan_mapping` —— **干跑报告** ✓✓：每条规则命中多少 ✓、哪些**目标键没人填** ✗、
  哪些**源键没被用上** ✗ ⇒ 于是"填表"变成**按报告补规则** ✓，而不是对着上千个名字硬看 ✗。

⚠️ 底线不变：**映射不能猜** ✗ —— 预设只是"常见写法" ✓，报告会如实说"还差这些" ✓；
**没匹配上的键不会静默丢弃** ✗，而会进 ``missingTargets`` / ``unusedSources`` ✓。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "MergeRule",
    "MappingPreset",
    "PRESETS",
    "RenameRule",
    "apply_merges",
    "build_key_map",
    "plan_mapping",
    "preset",
]


@dataclass(frozen=True)
class RenameRule:
    """一条**改名**规则 ✓（``pattern`` 命中就替换 ✓；``note`` 写清"这条治什么"✓）。"""

    pattern: str
    replacement: str
    note: str = ""


@dataclass(frozen=True)
class MergeRule:
    """把**同一层**的多个张量**按顺序拼**成一个 ✓（q/k/v → 融合权重 ✓）。

    ``block_pattern`` 用**第 1 个捕获组**圈出"层前缀" ✓（例如 ``blocks\\.(\\d+)\\.`` ✓），
    ``parts`` 是各段的后缀 ✓（顺序即拼接顺序 ✓ —— **顺序错了模型不会报错但结果全错** ✗，
    所以这里必须显式写清 ✓），``target`` 是目标后缀 ✓。
    """

    block_pattern: str
    parts: tuple[str, ...]
    target: str
    note: str = ""


@dataclass(frozen=True)
class MappingPreset:
    """一套预设 ✓（改名 + 合并 ✓）。"""

    name: str
    renames: tuple[RenameRule, ...] = ()
    merges: tuple[MergeRule, ...] = ()
    note: str = ""


#: 常见约定 ✓（**不是**穷举 ✗；真权重到手后按 :func:`plan_mapping` 的报告补 ✓）
PRESETS: dict[str, MappingPreset] = {
    "dit": MappingPreset(
        name="dit",
        note="DiT / 视频主干：去外层前缀 + 注意力后缀归一 + q/k/v 融合 ✓",
        renames=(
            RenameRule(r"^model\.diffusion_model\.", "", "去 ComfyUI 风格外层前缀 ✓"),
            RenameRule(r"^model\.", "", "去 `model.` 前缀 ✓"),
            RenameRule(r"^diffusion_model\.", "", "去 `diffusion_model.` 前缀 ✓"),
            RenameRule(r"^transformer\.", "", "去 `transformer.` 前缀 ✓"),
            RenameRule(r"^net\.", "", "去 `net.` 前缀 ✓"),
            RenameRule(r"^unet\.", "", "去 `unet.` 前缀 ✓"),
            RenameRule(r"\.attention\.", ".attn.", "`attention` → `attn` ✓"),
            RenameRule(r"\.self_attn\.", ".attn.", "`self_attn` → `attn` ✓"),
            RenameRule(r"^layers\.(\d+)\.", r"blocks.\1.", "`layers.N` → `blocks.N` ✓"),
            RenameRule(r"^transformer_blocks\.(\d+)\.", r"blocks.\1.", "`transformer_blocks` → `blocks` ✓"),
            RenameRule(r"^blocks\.(\d+)\.mlp\.fc1\.", r"blocks.\1.mlp.0.", "MLP fc1 → 本仓 `mlp.0` ✓"),
            RenameRule(r"^blocks\.(\d+)\.mlp\.fc2\.", r"blocks.\1.mlp.2.", "MLP fc2 → 本仓 `mlp.2` ✓"),
        ),
        merges=(
            MergeRule(r"^blocks\.(\d+)\.", ("attn.wq.weight", "attn.wk.weight", "attn.wv.weight"),
                      "attn.in_proj_weight", "q/k/v 三段 → 融合权重 ✓（**顺序 q,k,v** ✓）"),
            MergeRule(r"^blocks\.(\d+)\.", ("attn.wq.bias", "attn.wk.bias", "attn.wv.bias"),
                      "attn.in_proj_bias", "q/k/v 三个 bias → 融合 bias ✓"),
            MergeRule(r"^blocks\.(\d+)\.", ("attn.q_proj.weight", "attn.k_proj.weight",
                                            "attn.v_proj.weight"),
                      "attn.in_proj_weight", "`q_proj/k_proj/v_proj` 写法 ✓"),
            MergeRule(r"^blocks\.(\d+)\.", ("attn.to_q.weight", "attn.to_k.weight",
                                            "attn.to_v.weight"),
                      "attn.in_proj_weight", "`to_q/to_k/to_v` 写法 ✓"),
            MergeRule(r"^blocks\.(\d+)\.", ("attn.qkv.weight",),
                      "attn.in_proj_weight", "已是融合名 ⇒ 只改名 ✓"),
        ),
    ),
    # ⚠️ H3 专用预设（2026-09-20 抄自 `reference/ComfyUI` 的实测键名 ✓，出处逐条写在 note 里 ✓）：
    #    与通用 `dit` 预设的**关键差别有两条**：
    #    ① **不做 q/k/v 合并** ✗ —— H3 本身就是融合权重 `attn.qkv_proj`（`model.py:133-515` ✓）；
    #    ② H3 的很多键**在本仓 DiT 里没有对应物** ✗（见 note ✓）⇒ 本预设只做"能安全改的"，
    #       剩下的**如实列出来** ✓（不硬塞成看似对的名字 ✗ —— 那会变成"名字对、形状错"✓✗）。
    "minimax-h3": MappingPreset(
        name="minimax-h3",
        note=("MiniMax H3 DiT：**只去外层前缀 + MLP 落位** ✓；⚠️ 以下键本仓 DiT **暂无对应物** ✗："
              "`token_refiner.*`（2 层 refiner ✓）、`condition_proj.*`（5120→5376 ✓）、"
              "`rope.inv_freq`（3 轴 16→96 ✓）、`final_layer.video_out/audio_out`（**双输出** ✓✗）、"
              "`blocks.N.adaln_proj.linear`（expand=6×模态3=18 ✓ ✗）。"
              "键名与结构出处：`reference/ComfyUI/comfy/model_detection.py:390-418`、"
              "`comfy/ldm/minimax/model.py:474-481` ✓"),
        renames=(
            RenameRule(r"^model\.diffusion_model\.", "", "去 ComfyUI 外层前缀 ✓（`lora.py:384-388` ✓）"),
            RenameRule(r"^model\.model\.", "", "另一候选前缀 ✓"),
            RenameRule(r"^model\.", "", "回退前缀 ✓"),
            RenameRule(r"^net\.", "", "候选前缀 ✓"),
            RenameRule(r"^diffusion_model\.", "", "LoRA 侧前缀 ✓"),
            RenameRule(r"^blocks\.(\d+)\.mlp\.fc1\.", r"blocks.\1.mlp.0.", "MLP fc1 → 本仓 `mlp.0` ✓"),
            RenameRule(r"^blocks\.(\d+)\.mlp\.fc2\.", r"blocks.\1.mlp.2.", "MLP fc2 → 本仓 `mlp.2` ✓"),
            RenameRule(r"^blocks\.(\d+)\.attn\.qkv_proj\.", r"blocks.\1.attn.in_proj_",
                       "融合 qkv → 本仓 `in_proj_*` ✓；⚠️ **形状仍会对不上** ✗（H3 是 56 头×128 维=7168 ✓，"
                       "本仓 MHA 是 hidden/heads=96 ✓ ⇒ 要先把 DiT 的注意力维度改成可显式给 ✓）"),
            RenameRule(r"^blocks\.(\d+)\.attn\.out_proj\.", r"blocks.\1.attn.out_proj.",
                       "输出投影：名字一致 ✓"),
        ),
        #: ⚠️ **故意不给合并规则** ✗：H3 权重里 q/k/v 已经是融合的 ✓（与通用 `dit` 预设相反 ✓）
        merges=(),
    ),
    "text-encoder": MappingPreset(
        name="text-encoder",
        note="文本编码器：去前缀 + 词表/位置表命名归一 ✓",
        renames=(
            RenameRule(r"^model\.", "", "去 `model.` 前缀 ✓"),
            RenameRule(r"^text_model\.", "", "去 `text_model.` 前缀 ✓"),
            RenameRule(r"^encoder\.embeddings\.token_embedding\.", "embed.", "词表 → `embed` ✓"),
            RenameRule(r"^embeddings\.token_embedding\.", "embed.", "词表（无 `encoder.`）✓"),
            RenameRule(r"^embeddings\.position_embedding\.weight", "pos", "位置表 → `pos` ✓"),
            RenameRule(r"^encoder\.embeddings\.position_embedding\.weight", "pos", "位置表（HF 风格）✓"),
            RenameRule(r"^encoder\.layers\.(\d+)\.", r"blocks.\1.", "`encoder.layers.N` → `blocks.N` ✓"),
            RenameRule(r"^encoder\.final_layer_norm\.", "norm.", "末层归一 ✓"),
        ),
    ),
    "vae": MappingPreset(
        name="vae",
        note="VAE：去前缀 + 编解码命名归一 ✓",
        renames=(
            RenameRule(r"^first_stage_model\.", "", "去 `first_stage_model.` 前缀 ✓"),
            RenameRule(r"^vae\.", "", "去 `vae.` 前缀 ✓"),
            RenameRule(r"^model\.", "", "去 `model.` 前缀 ✓"),
            RenameRule(r"^encoder\.down\.(\d+)\.", r"encoder.\1.", "`down.N` → 顺序编号 ✓"),
            RenameRule(r"^decoder\.up\.(\d+)\.", r"decoder.\1.", "`up.N` → 顺序编号 ✓"),
        ),
    ),
}


def preset(name: str) -> MappingPreset:
    """按名字取预设 ✓（未知名字**报错并列出可用项** ✓ —— 不静默给个空的 ✗）。"""
    key = str(name or "").strip()
    if key not in PRESETS:
        raise KeyError(f"未知预设 {name!r}；可用：{sorted(PRESETS)} ✓")
    return PRESETS[key]


def build_key_map(name: str, *, extra: dict[str, str] | None = None) -> dict[str, str]:
    """预设 → ``key_map`` ✓（可直接喂 :func:`weights.load_module_weights` ✓）；``extra`` 覆盖/补充 ✓。"""
    table = {rule.pattern: rule.replacement for rule in preset(name).renames}
    table.update(extra or {})
    return table


def apply_renames(keys: list[str], rules: tuple[RenameRule, ...]) -> tuple[list[str], dict[str, int]]:
    """改名 ✓ ⇒ ``(新键列表, 每条规则命中数)`` ✓（命中数为 0 的规则会被**显式标出** ✓）。

    ⚠️ **按顺序应用全部规则** ✓（**不是**命中一条就停 ✗）—— 这条是往返自检抓出来的 ✓：
    ``model.diffusion_model.layers.0.attention.wq.weight`` 需要**三条**规则接力 ✓
    （去前缀 ✓ → ``layers.N`` → ``blocks.N`` ✓ → ``attention`` → ``attn`` ✓）；
    初版命中第一条就 ``break`` ✗ ⇒ 后面两条**永远不跑** ⇒ 装载时整片 ``layers.N`` 对不上 ✓。
    ⇒ 规则必须写成**按顺序可叠加、且不会互相打架**（各自的锚点要够紧 ✓）。
    """
    compiled = [(re.compile(rule.pattern), rule.replacement, rule.pattern) for rule in rules]
    hits: dict[str, int] = {rule.pattern: 0 for rule in rules}
    out: list[str] = []
    for key in keys:
        new_key = str(key)
        for pattern, replacement, label in compiled:
            if pattern.search(new_key):
                new_key = pattern.sub(replacement, new_key)
                hits[label] += 1
        out.append(new_key)
    return out, hits


def _merge_groups(keys: list[str], rules: tuple[MergeRule, ...]) -> tuple[dict[str, list[str]], dict[str, int]]:
    """按合并规则分组 ✓ ⇒ ``({目标键: [源键按序]}, 每条规则命中数)`` ✓。"""
    groups: dict[str, list[str]] = {}
    hits: dict[str, int] = {}
    present = set(keys)
    for rule in rules:
        label = f"{rule.parts[0]} → {rule.target}"
        hits[label] = 0
        for key in keys:
            match = re.match(rule.block_pattern, key)
            if not match:
                continue
            prefix = key[: match.end()]
            sources = [f"{prefix}{part}" for part in rule.parts]
            if not all(source in present for source in sources):
                continue
            target = f"{prefix}{rule.target}"
            groups.setdefault(target, list(sources))
            hits[label] += 1
    return groups, hits


def plan_mapping(keys: list[str], targets: list[str], *,
                 name: str = "dit") -> dict[str, Any]:
    """**干跑报告** ✓✓ —— 工作站上"填表"就靠它 ✓。

    给出：① 每条改名/合并规则的**命中数**（0 命中的规则是"这条预设在这儿用不上"✓）；
    ② 映射后能对上的目标键 ✓；③ **没人填的目标键** ✗（= 还缺规则 ✓）；
    ④ **没被用上的源键** ✗（= 规则没覆盖到 ✓ 或本身不需要 ✓）。
    """
    spec = preset(name)
    renamed, rename_hits = apply_renames(list(keys), spec.renames)
    groups, merge_hits = _merge_groups(renamed, spec.merges)

    produced = {key for key in renamed if key not in {source for sources in groups.values()
                                                     for source in sources}}
    produced |= set(groups)
    target_set = set(targets)
    missing = sorted(target_set - produced)
    unused = sorted(set(renamed) - produced)
    return {
        "preset": spec.name,
        "note": spec.note,
        "sourceKeys": len(keys),
        "targetKeys": len(targets),
        "renameHits": rename_hits,
        "mergeHits": merge_hits,
        "renamedKeys": renamed,
        "mergeGroups": {key: value for key, value in groups.items()},
        "matched": len(target_set & produced),
        "missingTargets": missing,
        "unusedSources": unused,
        # ⚠️ **能不能装**由"目标键有没有人填"决定 ✓ —— 缺键 **不**静默放过 ✗
        "complete": not missing,
        "verdict": ("可以装载 ✓" if not missing else
                    f"还差 {len(missing)} 个目标键 ✗ ⇒ 按报告补规则（或确认这些键本就该随机初始化 ✓）"),
    }


def apply_merges(state: dict[str, Any], groups: dict[str, list[str]]) -> dict[str, Any]:
    """按分组把张量**拼**起来 ✓（torch 懒导入 ✓）。

    沿**第 0 维**拼接 ✓（q/k/v 的 out 维在 0 ✓ 这是各家融合权重的通用布局 ✓）；
    **顺序严格按 ``groups`` 里的列表** ✓（顺序错 ⇒ 不报错但结果全错 ✗ ⇒ 由 :func:`plan_mapping`
    的 ``mergeGroups`` 把它**显式记录**下来以便核对 ✓）。
    """
    import torch  # noqa: PLC0415

    merged: dict[str, Any] = {}
    for target, sources in groups.items():
        tensors = [state[source] for source in sources if source in state]
        if len(tensors) != len(sources):      # pragma: no cover - plan 阶段已保证 ✓
            continue
        merged[target] = torch.cat(tensors, dim=0)
    return merged
