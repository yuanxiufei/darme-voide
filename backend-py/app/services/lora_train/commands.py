"""**训练命令组装** —— 把一张参数表变成**能直接跑的 argv** ✓（预缓存 + 训练 ✓）。

## 这个模块的形状为什么是"纯函数" ✗✓

参考实现把这些拼接**写在 nicegui 按钮回调里** ✓（``run_wan_training()`` 里既拼命令又起线程 ✓）。
那样写有两个后果 ✓✗：命令拼错要到**点下按钮、跑起来**才知道 ✓；同一段逻辑没法单测 ✗。
⇒ 本仓把「拼 argv」与「起进程」**切开** ✓：本模块**只拼 argv** ✓（纯函数、无 IO、可单测 ✓），
起进程在 ``runner.py`` ✓。⇒ ``tests/lora_train_test.py`` 能离线断言"这条配置该拼出什么 argv"✓。

## 事实来源（别凭记忆改 ✗）

各模型的参数→命令行映射**逐条对着参考实现核过** ✓：

* ``wan`` → ``wan_lora_train.py`` 第 109~448 行 ✓
* ``hunyuan`` → ``hunyuan_lora_train.py`` ✓（musubi ``hv_train_network`` ✓）
* ``kontext`` → ``flux_kontext_*`` ✓（编辑任务的样图控制图是 ``--ci`` ✓ 不是 ``--i`` ✓）
* ``flux`` → ⚠️ **走 sd-scripts** 的 ``flux_train_network`` ✓（其余四个走 musubi ✓）
* ``qwen`` → ``qwen_image_*`` ✓（``train_type`` 还会改脚本名 ✓）

## 三处**刻意与参考不同**的地方 ✓（都是"照抄会坑人"✓，diff 写在这里 ✓）

1. **``custom_params`` 按词拆开** ✓ —— 参考实现是 ``command.extend([custom_params])`` ✓，
   即**整串当成一个 argv** ✓✗ ⇒ 用户写 ``--min_timestep 100`` 会被当成一个畸形参数、
   报一个看不懂的错 ✓。本仓用 :func:`shlex.split` 拆 ✓（Windows 下 ``posix=False`` ✓
   以保住反斜杠路径 ✓）。
2. **prompt 文件写到任务目录** ✓ —— 参考写死相对路径 ``./wan_prompt_file.txt`` ✓，
   落点随**进程工作目录**漂移 ✓✗（本仓后端是被 IDE/服务拉起来的 ✓，工作目录不是仓库根 ✓）。
   本仓写进该任务自己的目录 ✓（``<日志目录>/<任务 id>/prompt.txt`` ✓）。
3. **启动前先把必填与路径查一遍** ✓ —— 参考是"点下去才发现少了权重"✓✗。
   本仓 :func:`validate` 一次报**全部**问题 ✓（本仓既定风格：错误一次说全 ✓）。

⚠️ 另有一处**照原样保留**的可疑条件 ✓：wan 的预缓存第 2 步里，``--fp8_t5`` 挂在
``vae_cache_cpu`` 这个开关上 ✓（参考实现如此 ✓，看着像笔误 ✓）。本仓**不擅自改** ✗，
只在 :func:`build_precache` 里标注出处 ✓ —— 改与不改都会影响显存占用 ✓，由用户定 ✓。
"""
from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from .errors import LoraTrainConfigError
from .options import KIND_INT, KIND_PATH_DIR, KIND_PATH_FILE, qwen_model_version, spec
from .paths import (
    TOOLKIT_SD_SCRIPTS,
    ResolvedTool,
    python_executable,
    resolve_tool,
)

# ---------------------------------------------------------------------------
# 机制
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommandPlan:
    """一个训练任务的**全部 argv** ✓。"""

    #: 预缓存命令序列 ✓（**空 = 这个模型没有预缓存步骤** ✓，如 FLUX ✓）
    cache: tuple[tuple[str, ...], ...] = ()
    #: 训练命令 ✓（已含 accelerate 前缀 ✓）
    train: tuple[str, ...] = ()
    #: 该模型用到的工具脚本 ✓（诊断用 ✓）
    tools: dict[str, str] = field(default_factory=dict)
    #: 生成的 prompt 文件 ✓（没生成则 ``None`` ✓）
    prompt_file: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "cache": [list(c) for c in self.cache],
            "train": list(self.train),
            "tools": dict(self.tools),
            "promptFile": str(self.prompt_file) if self.prompt_file else None,
        }


def _text(values: Mapping[str, Any], key: str) -> str:
    """取字符串值 ✓（缺键 = ``""`` ✓ —— 缺键说明调用方没过 :func:`options.merge` ✓，
    本模块**不替它兜底默认值** ✗：默认值只有一个权威来源 ✓）。"""
    value = values.get(key, "")
    return "" if value is None else str(value).strip()


def _flag_if(cmd: list[str], values: Mapping[str, Any], key: str, flag: str) -> None:
    """非空就 ``--flag value`` ✓。"""
    value = _text(values, key)
    if value:
        cmd.extend([flag, value])


def _switch_if(cmd: list[str], values: Mapping[str, Any], key: str, flag: str) -> None:
    """真就加开关 ✓。"""
    if values.get(key):
        cmd.append(flag)


def _truthy(values: Mapping[str, Any], key: str) -> bool:
    return bool(values.get(key))


def _custom_params(values: Mapping[str, Any]) -> list[str]:
    """**逃生口** ✓ —— 见模块头「刻意不同」第 1 条 ✓。"""
    raw = _text(values, "custom_params")
    if not raw:
        return []
    return shlex.split(raw, posix=False)


def _accelerate_prefix(values: Mapping[str, Any]) -> list[str]:
    """``accelerate launch`` 前缀 ✓。

    ⚠️ ``--gpu_ids 0`` **写死** ✓ —— 与参考实现一致 ✓。单卡机器的语义就是"用那张卡" ✓；
    多卡要改这里的写法（不是改参数值 ✓）⇒ 留在这里并标清楚 ✓。
    """
    return [
        python_executable(), "-m", "accelerate.commands.launch",
        "--num_cpu_threads_per_process", str(values.get("num_cpu_threads_per_process", 1)),
        "--mixed_precision", _text(values, "mixed_precision"),
        "--num_processes", str(values.get("num_processes", 1)),
        "--gpu_ids", "0",
    ]


def _attention_flags(cmd: list[str], values: Mapping[str, Any]) -> None:
    """注意力实现 ✓ —— xformers 分支连 ``--split_attn`` 一起给 ✓（参考实现如此 ✓）。"""
    impl = _text(values, "attention_implementation")
    if impl == "sdpa":
        cmd.append("--sdpa")
    elif impl == "xformers":
        cmd.extend(["--xformers", "--split_attn"])
    else:
        raise LoraTrainConfigError(
            f"attention_implementation 取值非法：{impl!r} ✗；合法值：['sdpa', 'xformers']"
        )


def _log_flags(cmd: list[str], values: Mapping[str, Any]) -> None:
    """日志三项 ✓ —— **各自非空才加** ✓（参考实现同样三个独立 ``if`` ✓）。"""
    log_dir = _text(values, "log_dir")
    if log_dir:
        cmd.extend(["--logging_dir", log_dir])
    log_prefix = _text(values, "log_prefix")
    if log_prefix:
        cmd.extend(["--log_prefix", log_prefix])
    tracker = _text(values, "log_tracker_name")
    if tracker:
        cmd.extend(["--log_tracker_name", tracker])


def _lr_scheduler_tail(cmd: list[str], values: Mapping[str, Any]) -> None:
    """调度器的**条件参数** ✓ —— 只在对应调度器下才拼 ✓（参考实现如此 ✓）。"""
    scheduler = _text(values, "lr_scheduler")
    if scheduler == "constant_with_warmup":
        cmd.extend(["--lr_warmup_steps", _text(values, "lr_warmup_steps")])
    elif scheduler == "cosine_with_restarts":
        cmd.extend(["--lr_scheduler_num_cycles", _text(values, "lr_scheduler_num_cycles")])


# ---------------------------------------------------------------------------
# 校验 —— 启动前一次报全 ✓
# ---------------------------------------------------------------------------

#: 这些键虽然控件类型是"文本" ✓，但**必须能解析成数** ✓ —— 参考实现把它们存成 ``"2e-4"``
#: 这样的字符串 ✓，值域校验全靠下游 argparse ✓✗：打错成 ``"2e-4 "`` 或 ``"2e4-"``
#: 会在**加载完底模之后**才炸 ✓。这里先验 ✓，代价是几行代码 ✓。
_NUMERIC_KEYS: frozenset[str] = frozenset({
    "batch_size", "max_train_epochs", "learning_rate", "network_dim",
    "gradient_accumulation_steps", "save_every_n_epochs", "save_every_n_steps",
    "sample_every_n_epochs", "sample_every_n_steps", "sample_w", "sample_h",
    "sample_frames", "sample_seed", "sample_steps", "discrete_flow_shift",
    "blocks_to_swap", "min_timestep", "max_timestep", "timestep_boundary",
    "lr_warmup_steps", "lr_scheduler_num_cycles",
})

#: 各模型的**必填**项 ✓ —— 少一个都不可能跑 ✓（拼进报错文案里 ✓）。
_REQUIRED_BY_MODEL: dict[str, tuple[str, ...]] = {
    "wan": ("dataset_config", "vae_path", "t5_path", "dit_weights_path"),
    "hunyuan": ("dataset_config", "vae_path", "text_encoder_model_path", "dit_weights_path"),
    "kontext": ("dataset_config", "vae_path", "t5_path", "clip_model_path", "dit_weights_path"),
    "flux": ("dataset_config", "vae_path", "t5_path", "clip_model_path", "dit_weights_path"),
    "qwen": ("dataset_config", "vae_path", "text_encoder_model_path", "dit_weights_path"),
}

#: 路径类参数里**必须真实存在**的 ✓ —— 提前拦掉"跑起来才发现少文件"✓。
_MUST_EXIST: frozenset[str] = frozenset({"dataset_config"})


def validate(model_key: str, values: Mapping[str, Any]) -> None:
    """启动前**一次报全**所有问题 ✓；全部通过则**返回 ``None``** ✓。

    ⚠️ 报错文案**聚合** ✓，不"发现一个抛一个"✗：一次报全，用户改一次就能跑 ✓。
    """
    resolved = spec(model_key)
    problems: list[str] = []

    for key in _REQUIRED_BY_MODEL[model_key]:
        if not _text(values, key):
            opt = resolved.option(key)
            problems.append(f"  · {key}（{opt.label if opt else key}）必填，当前为空")

    for opt in resolved.options:
        value = values.get(opt.key, opt.default)
        if opt.kind == KIND_INT:
            try:
                int(str(value).strip())
            except (TypeError, ValueError):
                problems.append(f"  · {opt.key} 需要整数，当前是 {value!r}")
        elif opt.key in _NUMERIC_KEYS:
            try:
                float(str(value).strip())
            except (TypeError, ValueError):
                problems.append(
                    f"  · {opt.key}（{opt.label}）需要数字，当前是 {value!r}"
                    "（支持 2e-4 这类写法）")

    if _truthy(values, "use_network_weights") and not _text(values, "network_weights_path"):
        problems.append("  · 勾了「从已有 LoRA 续训」，但 network_weights_path 为空")
    if _truthy(values, "custom_prompt_txt"):
        custom = _text(values, "custom_prompt_path")
        if not custom:
            problems.append("  · 勾了「用自定义 prompt 文件」，但 custom_prompt_path 为空")
        elif not Path(custom).is_file():
            problems.append(f"  · custom_prompt_path 不存在：{custom}")
    # 样图用控制图/首帧：只有**开着出样图**且填了路径时才要求存在 ✓
    if _truthy(values, "generate_samples"):
        image = _text(values, "sample_image_path")
        if image and not Path(image).is_file():
            problems.append(f"  · sample_image_path 不存在：{image}")

    for opt in resolved.options:
        if opt.kind not in (KIND_PATH_FILE, KIND_PATH_DIR):
            continue
        value = _text(values, opt.key)
        if not value:
            continue
        path = Path(value)
        if opt.key in _MUST_EXIST and not path.is_file():
            problems.append(f"  · {opt.key}（{opt.label}）指向的文件不存在：{value}")
        elif opt.kind == KIND_PATH_FILE and not path.is_file() and opt.key not in _MUST_EXIST:
            # ⚠️ 权重类**只警告不阻断**是错的 ✗ ⇒ 这里同样**阻断** ✓：
            # 「文件不存在」没有"也许能跑"的可能 ✓（占位符/软链接会在加载时才炸 ✓）。
            problems.append(f"  · {opt.key}（{opt.label}）指向的文件不存在：{value}")
        elif opt.kind == KIND_PATH_DIR and path.exists() and not path.is_dir():
            problems.append(f"  · {opt.key}（{opt.label}）不是目录：{value}")

    if problems:
        raise LoraTrainConfigError(
            f"模型 {model_key} 的参数没通过校验（共 {len(problems)} 处）✗：\n" + "\n".join(problems)
        )


# ---------------------------------------------------------------------------
# prompt 文件 —— 出样图时要交给训练脚本的那个 txt ✓
# ---------------------------------------------------------------------------

#: 各模型的**样图参数后缀** ✓ —— 一行 prompt 后面挂哪些开关 ✓。
#: ``--w/--h`` 宽高；``--f`` 帧数（视频）；``--d`` 种子；``--s`` 步数；
#: ``--i`` 首帧（图生视频/编辑的输入图）；``--ci`` 控制图（Kontext 专用 ✓）。
def _prompt_line(model_key: str, values: Mapping[str, Any]) -> str:
    """一行样图 prompt ✓（不含换行 ✓）。"""
    if model_key == "flux":
        # ⚠️ FLUX 走 sd-scripts ✓ ⇒ 它的 prompt 文件是**一行纯文本** ✓ ——
        # 尺寸/种子/步数都不在文件里（由数据集分桶与命令行决定 ✓）。
        return _text(values, "sample_prompt_text")
    parts = [_text(values, "sample_prompt_text")]
    if model_key == "kontext":
        parts += ["--w", _text(values, "sample_w"), "--h", _text(values, "sample_h")]
    else:
        parts += ["--w", _text(values, "sample_w"), "--h", _text(values, "sample_h")]
        if model_key in ("wan", "hunyuan"):
            parts += ["--f", _text(values, "sample_frames")]
    parts += ["--d", _text(values, "sample_seed"), "--s", _text(values, "sample_steps")]
    image = _text(values, "sample_image_path")
    if image:
        # ⚠️ ``--ci`` 是 **Kontext 与 Qwen-Image 共用**的写法 ✓（两处参考源码都写 ``--ci`` ✓：
        # kontext_lora_train.py 第 121 行 ✓、qwen_image_lora_train.py 第 265 行 ✓），
        # 只有 Wan / HunyuanVideo 用 ``--i``（首帧 ✓）。
        parts += ["--ci" if model_key in ("kontext", "qwen") else "--i", image]
    return " ".join(parts)


def prompt_content(model_key: str, values: Mapping[str, Any]) -> str:
    """prompt 文件的**完整内容** ✓ —— 首行那句注释是参考实现就有的 ✓（训练脚本靠它认格式 ✓）。"""
    return "# prompt 1: for generating a sample\n" + _prompt_line(model_key, values) + "\n"


def _needs_prompt_file(values: Mapping[str, Any]) -> bool:
    """要不要生成 prompt 文件 ✓。

    三种情况 ✓：没开样图 ⇒ 不要 ✗；开了 + 用自定义文件 ⇒ 用**用户那个** ✓（不覆盖 ✓）；
    开了 + 没自定义 ⇒ 生成 ✓。
    """
    if not _truthy(values, "generate_samples"):
        return False
    return not (_truthy(values, "custom_prompt_txt") and _text(values, "custom_prompt_path"))


def prompt_file_for(model_key: str, values: Mapping[str, Any], workspace: Path) -> Path | None:
    """**落盘** prompt 文件并返回路径 ✓；不需要则 ``None`` ✓。

    ⚠️ 这是本模块**唯一有副作用**的函数 ✓（写一个几百字节的 txt ✓）——
    落点由调用方给（``runner`` 给任务目录 ✓），**不写相对路径** ✗：见模块头「刻意不同」第 2 条 ✓。
    """
    if not _needs_prompt_file(values):
        return None
    workspace.mkdir(parents=True, exist_ok=True)
    target = workspace / "prompt.txt"
    target.write_text(prompt_content(model_key, values), encoding="utf-8")
    return target


def _sample_prompt_path(model_key: str, values: Mapping[str, Any], workspace: Path) -> Path | None:
    """**将要**使用的 prompt 文件路径 ✓ —— **不写盘** ✗，给"预报命令"用 ✓。"""
    if not _truthy(values, "generate_samples"):
        return None
    if _truthy(values, "custom_prompt_txt") and _text(values, "custom_prompt_path"):
        return Path(_text(values, "custom_prompt_path"))
    return workspace / "prompt.txt"


def _prompt_arg(model_key: str, values: Mapping[str, Any], workspace: Path) -> Path | None:
    """交给命令行的 ``--sample_prompts`` 参数 ✓（自定义文件优先 ✓），**该写盘的就写** ✓。"""
    path = _sample_prompt_path(model_key, values, workspace)
    if path is None:
        return None
    if path == workspace / "prompt.txt":
        returned = prompt_file_for(model_key, values, workspace)
        # 走到这里 ``prompt_file_for`` 必然返回同一个路径 ✓；不同就是逻辑分叉了 ⇒ 报错 ✗
        if returned != path:
            raise LoraTrainConfigError(
                f"prompt 文件路径不一致 ✗：预期 {path}，实际 {returned}（本包内部逻辑错误）"
            )
    return path


# ---------------------------------------------------------------------------
# wan —— 对照 wan_lora_train.py 第 109~448 行 ✓
# ---------------------------------------------------------------------------

_WAN_22_TASKS = ("t2v-A14B", "i2v-A14B")


def _wan_precache(values: Mapping[str, Any]) -> list[tuple[str, ...]]:
    """Wan 预缓存：**两步** ✓（潜变量 → 文本编码 ✓），参考实现分两次 ``Popen`` 串行跑 ✓。"""
    latents = resolve_tool("wan_cache_latents")
    text_encoder = resolve_tool("wan_cache_text_encoder_outputs")
    task = _text(values, "task")

    cmd1: list[str] = [
        python_executable(), str(latents.path),
        "--dataset_config", _text(values, "dataset_config"),
        "--vae", _text(values, "vae_path"),
    ]
    _switch_if(cmd1, values, "vae_cache_cpu", "--vae_cache_cpu")
    _switch_if(cmd1, values, "skip_existing", "--skip_existing")
    if _truthy(values, "use_clip") and _text(values, "clip_model_path"):
        cmd1.extend(["--clip", _text(values, "clip_model_path")])
    if "i2v" in task:
        cmd1.append("--i2v")

    cmd2: list[str] = [
        python_executable(), str(text_encoder.path),
        "--dataset_config", _text(values, "dataset_config"),
        "--t5", _text(values, "t5_path"),
        "--batch_size", _text(values, "batch_size"),
    ]
    # ⚠️ 下面这个条件**照参考实现原样保留** ✓（``vae_cache_cpu`` ✓ 而不是 ``fp8`` ✓）：
    # 看着像笔误 ✓，但"改对"会实际改变显存占用与缓存精度 ✓ ⇒ 不擅自动 ✗，标注于此 ✓。
    _switch_if(cmd2, values, "vae_cache_cpu", "--fp8_t5")
    return [tuple(cmd1), tuple(cmd2)]


def _wan_train(values: Mapping[str, Any], workspace: Path) -> tuple[str, ...]:
    """Wan 训练命令 ✓（顺序照参考实现排 ✓，便于和它对照 ✓）。"""
    tool = resolve_tool("wan_train_network")
    task = _text(values, "task")
    cmd: list[str] = [
        *_accelerate_prefix(values),
        str(tool.path),
        "--task", task,
        "--dit", _text(values, "dit_weights_path"),
        "--dataset_config", _text(values, "dataset_config"),
        "--mixed_precision", _text(values, "mixed_precision"),
        "--optimizer_type", _text(values, "optimizer_type"),
        "--learning_rate", _text(values, "learning_rate"),
        "--gradient_checkpointing",
        f"--gradient_accumulation_steps={_text(values, 'gradient_accumulation_steps')}",
        "--max_data_loader_n_workers", str(values.get("max_data_loader_n_workers", 2)),
        "--persistent_data_loader_workers",
        "--network_module", "networks.lora_wan",
        "--network_dim", _text(values, "network_dim"),
        "--timestep_sampling", _text(values, "timestep_sampling"),
        "--discrete_flow_shift", _text(values, "discrete_flow_shift"),
        "--max_train_epochs", _text(values, "max_train_epochs"),
        "--save_every_n_epochs", _text(values, "save_every_n_epochs"),
        "--save_every_n_steps", _text(values, "save_every_n_steps"),
        "--output_dir", _text(values, "output_dir"),
        "--output_name", _text(values, "output_name"),
        "--seed", "42",  # ⚠️ 写死 ✓：参考实现同样写死 42 且**没暴露**这个参数 ✓
        "--log_with", _text(values, "log_type"),
        "--lr_scheduler", _text(values, "lr_scheduler"),
    ]
    _attention_flags(cmd, values)
    _switch_if(cmd, values, "offload_inactive_dit", "--offload_inactive_dit")
    if _truthy(values, "enable_low_vram"):
        cmd.extend(["--blocks_to_swap", _text(values, "blocks_to_swap")])
    if _truthy(values, "use_network_weights") and _text(values, "network_weights_path"):
        cmd.extend(["--network_weights", _text(values, "network_weights_path")])
    if _truthy(values, "use_clip") and _text(values, "clip_model_path"):
        cmd.extend(["--clip", _text(values, "clip_model_path")])
    _switch_if(cmd, values, "fp8", "--fp8_base")
    _lr_scheduler_tail(cmd, values)

    sample_prompt = _prompt_arg("wan", values, workspace)
    if sample_prompt is not None:
        cmd.extend([
            "--sample_prompts", str(sample_prompt),
            "--sample_every_n_epochs", _text(values, "sample_every_n_epochs"),
            "--sample_every_n_steps", _text(values, "sample_every_n_steps"),
            # ⚠️ 样图用的是**训练同一套** vae/t5 ✓（参考实现也是取同一份配置 ✓）
            "--vae", _text(values, "vae_path"),
            "--t5", _text(values, "t5_path"),
        ])
        _switch_if(cmd, values, "sample_at_first", "--sample_at_first")

    _log_flags(cmd, values)

    if task in _WAN_22_TASKS:
        high_noise = _text(values, "dit_high_noise_path")
        if high_noise:
            cmd.extend(["--dit_high_noise", high_noise])
            cmd.extend(["--timestep_boundary", _text(values, "timestep_boundary")])
        if _truthy(values, "timestep_custom"):
            cmd.extend([
                "--min_timestep", _text(values, "min_timestep"),
                "--max_timestep", _text(values, "max_timestep"),
            ])
        _switch_if(cmd, values, "lazy_loading", "--lazy_loading")

    cmd.extend(_custom_params(values))
    return tuple(cmd)


# ---------------------------------------------------------------------------
# hunyuan —— 对照 hunyuan_lora_train.py 第 109~430 行 ✓
# ---------------------------------------------------------------------------


def _hunyuan_precache(values: Mapping[str, Any]) -> list[tuple[str, ...]]:
    """HunyuanVideo 预缓存：**两步** ✓。

    ⚠️ 与 Wan 的两点不同 ✓（照参考实现 ✓，不是漏写 ✗）：
    1. 第 1 步**没有任何可选开关** ✓（Wan 那边有 ``--vae_cache_cpu`` / ``--skip_existing`` ✓）；
    2. 第 2 步的 ``--fp8_llm`` 挂在 **``fp8``** 上 ✓（Wan 那边挂的是 ``vae_cache_cpu`` ✓，
       见 :func:`_wan_precache` 的注释 ✓）—— 同一个人写的两条链条件不一致 ✓，本仓各按各的抄 ✓。
    """
    latents = resolve_tool("cache_latents")  # ⚠️ 通用名 ✓，见 options._hunyuan 的说明 ✓
    text_encoder = resolve_tool("cache_text_encoder_outputs")

    cmd1: list[str] = [
        python_executable(), str(latents.path),
        "--dataset_config", _text(values, "dataset_config"),
        "--vae", _text(values, "vae_path"),
    ]
    cmd2: list[str] = [
        python_executable(), str(text_encoder.path),
        "--dataset_config", _text(values, "dataset_config"),
        # HunyuanVideo 有**两个**文本编码器 ✓（llava-llama3 + CLIP ✓）⇒ encoder1 / encoder2 ✓
        "--text_encoder1", _text(values, "text_encoder_model_path"),
        "--text_encoder2", _text(values, "clip_model_path"),
        "--batch_size", _text(values, "batch_size"),
    ]
    _switch_if(cmd2, values, "fp8", "--fp8_llm")
    return [tuple(cmd1), tuple(cmd2)]


def _hunyuan_train(values: Mapping[str, Any], workspace: Path) -> tuple[str, ...]:
    """HunyuanVideo 训练命令 ✓。"""
    tool = resolve_tool("hv_train_network")
    cmd: list[str] = [
        *_accelerate_prefix(values),
        str(tool.path),
        "--text_encoder1", _text(values, "text_encoder_model_path"),
        "--text_encoder2", _text(values, "clip_model_path"),
        "--dit", _text(values, "dit_weights_path"),
        "--dataset_config", _text(values, "dataset_config"),
        "--mixed_precision", _text(values, "mixed_precision"),
        "--optimizer_type", _text(values, "optimizer_type"),
        "--learning_rate", _text(values, "learning_rate"),
        "--gradient_checkpointing",
        f"--gradient_accumulation_steps={_text(values, 'gradient_accumulation_steps')}",
        "--max_data_loader_n_workers", str(values.get("max_data_loader_n_workers", 2)),
        "--persistent_data_loader_workers",
        "--network_module", "networks.lora",
        "--network_dim", _text(values, "network_dim"),
        "--timestep_sampling", _text(values, "timestep_sampling"),
        "--discrete_flow_shift", _text(values, "discrete_flow_shift"),
        "--max_train_epochs", _text(values, "max_train_epochs"),
        "--save_every_n_epochs", _text(values, "save_every_n_epochs"),
        "--save_every_n_steps", _text(values, "save_every_n_steps"),
        "--output_dir", _text(values, "output_dir"),
        "--output_name", _text(values, "output_name"),
        "--seed", "42",
        "--log_with", _text(values, "log_type"),
    ]
    _attention_flags(cmd, values)
    _switch_if(cmd, values, "offload_inactive_dit", "--offload_inactive_dit")
    if _truthy(values, "enable_low_vram"):
        cmd.extend(["--blocks_to_swap", _text(values, "blocks_to_swap")])
    if _truthy(values, "use_network_weights") and _text(values, "network_weights_path"):
        cmd.extend(["--network_weights", _text(values, "network_weights_path")])
    _switch_if(cmd, values, "fp8", "--fp8_base")

    # ⚠️ **本仓增补** ✓：``--lr_scheduler`` 是这四个 musubi 模型里**唯一**参考实现没拼的 ✓
    # （其余三个都有 ✓）。已核对 musubi-tuner **0.3.5** 源码 ✓：
    # ``src/musubi_tuner/hv_train_network.py:489`` 调 ``setup_parser_common()`` ✓，
    # 而 ``--lr_scheduler`` 定义在 ``training/parser_common.py`` ✓ ⇒ **这个参数真实存在** ✓，
    # 不拼它只会让 HunyuanVideo 少一档能力、不会报错 ✓。
    cmd.extend(["--lr_scheduler", _text(values, "lr_scheduler")])

    sample_prompt = _prompt_arg("hunyuan", values, workspace)
    if sample_prompt is not None:
        cmd.extend([
            "--sample_prompts", str(sample_prompt),
            "--sample_every_n_epochs", _text(values, "sample_every_n_epochs"),
            "--sample_every_n_steps", _text(values, "sample_every_n_steps"),
            # ⚠️ 只有 ``--vae`` ✓（没有 ``--t5`` ✓）—— 参考实现如此 ✓，
            # 因为 HunyuanVideo 的文本编码器已经在上面 --text_encoder1/2 给过了 ✓。
            "--vae", _text(values, "vae_path"),
        ])
        _switch_if(cmd, values, "sample_at_first", "--sample_at_first")

    _log_flags(cmd, values)
    _lr_scheduler_tail(cmd, values)
    cmd.extend(_custom_params(values))
    return tuple(cmd)


# ---------------------------------------------------------------------------
# kontext —— 对照 kontext_lora_train.py 第 108~399 行 ✓
# ---------------------------------------------------------------------------


def _kontext_precache(values: Mapping[str, Any]) -> list[tuple[str, ...]]:
    """Kontext 预缓存：**两步** ✓（同样是 ``--text_encoder1/2`` ✓，1 = T5 ✓，2 = CLIP ✓）。"""
    latents = resolve_tool("flux_kontext_cache_latents")
    text_encoder = resolve_tool("flux_kontext_cache_text_encoder_outputs")

    cmd1: list[str] = [
        python_executable(), str(latents.path),
        "--dataset_config", _text(values, "dataset_config"),
        "--vae", _text(values, "vae_path"),
    ]
    cmd2: list[str] = [
        python_executable(), str(text_encoder.path),
        "--dataset_config", _text(values, "dataset_config"),
        "--text_encoder1", _text(values, "t5_path"),
        "--text_encoder2", _text(values, "clip_model_path"),
        "--batch_size", _text(values, "batch_size"),
    ]
    return [tuple(cmd1), tuple(cmd2)]


def _kontext_train(values: Mapping[str, Any], workspace: Path) -> tuple[str, ...]:
    """Kontext 训练命令 ✓。

    ⚠️ 三处与 Wan 的**结构差异** ✓（照参考 ✓）：
    * 给了 ``--vae`` ✓（Wan 那条没给 ✓，靠 dataset config 里的 vae ✓）；
    * **没有** ``--discrete_flow_shift`` ✓（Kontext 只有 ``--timestep_sampling flux_shift`` ✓）；
    * 样图段**不带** ``--vae/--t5`` ✓（Wan 带 ✓）。
    """
    tool = resolve_tool("flux_kontext_train_network")
    cmd: list[str] = [
        *_accelerate_prefix(values),
        str(tool.path),
        "--dit", _text(values, "dit_weights_path"),
        "--vae", _text(values, "vae_path"),
        "--text_encoder1", _text(values, "t5_path"),
        "--text_encoder2", _text(values, "clip_model_path"),
        "--dataset_config", _text(values, "dataset_config"),
        "--mixed_precision", _text(values, "mixed_precision"),
        "--optimizer_type", _text(values, "optimizer_type"),
        "--learning_rate", _text(values, "learning_rate"),
        "--gradient_checkpointing",
        f"--gradient_accumulation_steps={_text(values, 'gradient_accumulation_steps')}",
        "--max_data_loader_n_workers", str(values.get("max_data_loader_n_workers", 2)),
        "--persistent_data_loader_workers",
        "--network_module", "networks.lora_flux",
        "--network_dim", _text(values, "network_dim"),
        "--timestep_sampling", _text(values, "timestep_sampling"),
        "--max_train_epochs", _text(values, "max_train_epochs"),
        "--save_every_n_epochs", _text(values, "save_every_n_epochs"),
        "--save_every_n_steps", _text(values, "save_every_n_steps"),
        "--output_dir", _text(values, "output_dir"),
        "--output_name", _text(values, "output_name"),
        "--seed", "42",
        "--log_with", _text(values, "log_type"),
        "--lr_scheduler", _text(values, "lr_scheduler"),
    ]
    _attention_flags(cmd, values)
    _switch_if(cmd, values, "offload_inactive_dit", "--offload_inactive_dit")
    if _truthy(values, "enable_low_vram"):
        cmd.extend(["--blocks_to_swap", _text(values, "blocks_to_swap")])
    if _truthy(values, "use_network_weights") and _text(values, "network_weights_path"):
        cmd.extend(["--network_weights", _text(values, "network_weights_path")])

    sample_prompt = _prompt_arg("kontext", values, workspace)
    if sample_prompt is not None:
        cmd.extend([
            "--sample_prompts", str(sample_prompt),
            "--sample_every_n_epochs", _text(values, "sample_every_n_epochs"),
            "--sample_every_n_steps", _text(values, "sample_every_n_steps"),
        ])
        _switch_if(cmd, values, "sample_at_first", "--sample_at_first")

    _log_flags(cmd, values)
    _lr_scheduler_tail(cmd, values)
    cmd.extend(_custom_params(values))
    return tuple(cmd)


# ---------------------------------------------------------------------------
# flux —— 对照 flux_lora_train.py 第 138~291 行 ✓（⚠️ 走 sd-scripts ✓）
# ---------------------------------------------------------------------------


def _flux_train(values: Mapping[str, Any], workspace: Path) -> tuple[str, ...]:
    """FLUX.1 训练命令 ✓。

    ⚠️ **参数名整套都不一样** ✓ —— 因为这边是 sd-scripts ✓ 不是 musubi ✓：
    ``--pretrained_model_name_or_path``（不是 ``--dit`` ✓）、``--ae``（不是 ``--vae`` ✓）、
    ``--t5xxl`` / ``--clip_l``（不是 ``--text_encoder1/2`` ✓）✓。
    还多一个 ``--model_prediction_type raw`` ✓（sd-scripts 的 FLUX 链特有 ✓，参考实现写死 ✓）。

    ⚠️ **``--discrete_flow_shift`` 不拼** ✓ —— 虽然参考那份 ``flux_settings.toml`` 里有
    ``discrete_flow_shift = 3.1582`` ✓，但**整段命令里从未用到它** ✓✗。
    本仓照实情办 ✓：``options._flux`` 把它标成 ``used=False`` ✓，前端会显示"当前不生效"✓。
    """
    tool = resolve_tool("flux_train_network", toolkit=TOOLKIT_SD_SCRIPTS)
    cmd: list[str] = [
        *_accelerate_prefix(values),
        str(tool.path),
        "--pretrained_model_name_or_path", _text(values, "dit_weights_path"),
        "--ae", _text(values, "vae_path"),
        "--t5xxl", _text(values, "t5_path"),
        "--clip_l", _text(values, "clip_model_path"),
        "--dataset_config", _text(values, "dataset_config"),
        "--save_model_as", "safetensors",
        "--mixed_precision", _text(values, "mixed_precision"),
        "--optimizer_type", _text(values, "optimizer_type"),
        "--learning_rate", _text(values, "learning_rate"),
        "--gradient_checkpointing",
        f"--gradient_accumulation_steps={_text(values, 'gradient_accumulation_steps')}",
        "--max_data_loader_n_workers", str(values.get("max_data_loader_n_workers", 2)),
        "--persistent_data_loader_workers",
        "--network_module", "networks.lora_flux",
        "--network_dim", _text(values, "network_dim"),
        "--timestep_sampling", _text(values, "timestep_sampling"),
        "--max_train_epochs", _text(values, "max_train_epochs"),
        "--save_every_n_epochs", _text(values, "save_every_n_epochs"),
        "--save_every_n_steps", _text(values, "save_every_n_steps"),
        "--output_dir", _text(values, "output_dir"),
        "--output_name", _text(values, "output_name"),
        "--seed", "42",
        "--log_with", _text(values, "log_type"),
        "--model_prediction_type", "raw",
        "--lr_scheduler", _text(values, "lr_scheduler"),
    ]
    # ⚠️ 缓存三个开关是**一组** ✓（参考实现一次性全加 ✓）：开 ``cache_to_disk`` ⇒
    # 潜变量与文本编码器输出**都**落盘 ✓，这也是 FLUX 没有独立预缓存步骤的原因 ✓。
    if _truthy(values, "cache_to_disk"):
        cmd.extend(["--cache_latents_to_disk", "--cache_text_encoder_outputs",
                    "--cache_text_encoder_outputs_to_disk"])
    _attention_flags(cmd, values)
    _switch_if(cmd, values, "offload_inactive_dit", "--offload_inactive_dit")
    if _truthy(values, "enable_low_vram"):
        cmd.extend(["--blocks_to_swap", _text(values, "blocks_to_swap")])
    if _truthy(values, "use_network_weights") and _text(values, "network_weights_path"):
        cmd.extend(["--network_weights", _text(values, "network_weights_path")])
    _switch_if(cmd, values, "fp8", "--fp8_base")

    sample_prompt = _prompt_arg("flux", values, workspace)
    if sample_prompt is not None:
        cmd.extend([
            "--sample_prompts", str(sample_prompt),
            # ⚠️ 只有 ``--sample_every_n_steps`` ✓ —— ``--sample_every_n_epochs`` 在参考实现里
            # 被**注释掉**了 ✓（flux_lora_train.py 第 250 行 ✓）⇒ 本仓不拼 ✓。
            "--sample_every_n_steps", _text(values, "sample_every_n_steps"),
        ])

    _log_flags(cmd, values)
    _lr_scheduler_tail(cmd, values)
    cmd.extend(_custom_params(values))
    return tuple(cmd)


# ---------------------------------------------------------------------------
# qwen —— 对照 qwen_image_lora_train.py 第 109~451 行 ✓
# ---------------------------------------------------------------------------


def _qwen_model_version(values: Mapping[str, Any]) -> str | None:
    """``train_type`` → ``--model_version`` ✓；``qwen_image_2512`` ⇒ ``None`` ✓（不加这个参数 ✓）。

    ⚠️ 这里**只映射 model_version** ✓ —— 训练脚本名**不随 train_type 变** ✓：
    参考实现的 ``run_wan_training()`` 无论哪种 train_type 拼的都是
    ``qwen_image_train_network.py`` ✓（第 326 行 ✓）。
    本模块早先版本的注释曾写过"edit 会换脚本" ✗，那是**猜的** ✗，已按源码更正 ✓。
    """
    # 取值域归 ``options`` ✓（单一来源 ✓），这里只做转发 ✓
    return qwen_model_version(values)


def _qwen_precache(values: Mapping[str, Any]) -> list[tuple[str, ...]]:
    """Qwen-Image 预缓存：**两步** ✓。

    ⚠️ 第 2 步的 ``--fp8_vl`` 挂在 ``fp8`` 上 ✓（不是 ``--fp8_t5`` ✓ ——
    Wan 那条叫 t5 ✓、这里叫 vl ✓：Qwen-Image 的文本编码器是 Qwen2.5-VL ✓）。
    """
    latents = resolve_tool("qwen_image_cache_latents")
    text_encoder = resolve_tool("qwen_image_cache_text_encoder_outputs")
    model_version = _qwen_model_version(values)

    cmd1: list[str] = [
        python_executable(), str(latents.path),
        "--dataset_config", _text(values, "dataset_config"),
        "--vae", _text(values, "vae_path"),
    ]
    cmd2: list[str] = [
        python_executable(), str(text_encoder.path),
        "--dataset_config", _text(values, "dataset_config"),
        "--text_encoder", _text(values, "text_encoder_model_path"),
        "--batch_size", _text(values, "batch_size"),
    ]
    _switch_if(cmd2, values, "fp8", "--fp8_vl")
    if model_version:
        # 顺序照参考 ✓：cmd1 加在末尾 ✓，cmd2 加在 --fp8_vl **之后** ✓
        cmd1.extend(["--model_version", model_version])
        cmd2.extend(["--model_version", model_version])
    return [tuple(cmd1), tuple(cmd2)]


def _qwen_train(values: Mapping[str, Any], workspace: Path) -> tuple[str, ...]:
    """Qwen-Image 训练命令 ✓。"""
    tool = resolve_tool("qwen_image_train_network")
    cmd: list[str] = [
        *_accelerate_prefix(values),
        str(tool.path),
        "--dit", _text(values, "dit_weights_path"),
        "--dataset_config", _text(values, "dataset_config"),
        "--vae", _text(values, "vae_path"),
        "--text_encoder", _text(values, "text_encoder_model_path"),
        "--mixed_precision", _text(values, "mixed_precision"),
        "--optimizer_type", _text(values, "optimizer_type"),
        "--learning_rate", _text(values, "learning_rate"),
        "--gradient_checkpointing",
        f"--gradient_accumulation_steps={_text(values, 'gradient_accumulation_steps')}",
        "--max_data_loader_n_workers", str(values.get("max_data_loader_n_workers", 2)),
        "--persistent_data_loader_workers",
        "--network_module", "networks.lora_qwen_image",
        "--network_dim", _text(values, "network_dim"),
        "--timestep_sampling", _text(values, "timestep_sampling"),
        "--discrete_flow_shift", _text(values, "discrete_flow_shift"),
        "--max_train_epochs", _text(values, "max_train_epochs"),
        "--save_every_n_epochs", _text(values, "save_every_n_epochs"),
        "--save_every_n_steps", _text(values, "save_every_n_steps"),
        "--output_dir", _text(values, "output_dir"),
        "--output_name", _text(values, "output_name"),
        "--seed", "42",
        "--log_with", _text(values, "log_type"),
        "--lr_scheduler", _text(values, "lr_scheduler"),
    ]
    _attention_flags(cmd, values)
    _switch_if(cmd, values, "offload_inactive_dit", "--offload_inactive_dit")
    if _truthy(values, "enable_low_vram"):
        cmd.extend(["--blocks_to_swap", _text(values, "blocks_to_swap")])
    if _truthy(values, "use_network_weights") and _text(values, "network_weights_path"):
        cmd.extend(["--network_weights", _text(values, "network_weights_path")])
    _switch_if(cmd, values, "fp8", "--fp8_base")

    model_version = _qwen_model_version(values)
    if model_version:
        cmd.extend(["--model_version", model_version])

    sample_prompt = _prompt_arg("qwen", values, workspace)
    if sample_prompt is not None:
        cmd.extend([
            "--sample_prompts", str(sample_prompt),
            "--sample_every_n_epochs", _text(values, "sample_every_n_epochs"),
            "--sample_every_n_steps", _text(values, "sample_every_n_steps"),
            # ⚠️ 样图这里给的是 ``--vae`` ✓（参考实现的 ``sample_vae_path`` 就是**同一个** vae ✓）
            "--vae", _text(values, "vae_path"),
        ])
        _switch_if(cmd, values, "sample_at_first", "--sample_at_first")

    _log_flags(cmd, values)
    _lr_scheduler_tail(cmd, values)
    cmd.extend(_custom_params(values))
    return tuple(cmd)


# ---------------------------------------------------------------------------
# 公开入口
# ---------------------------------------------------------------------------

#: 有独立预缓存步骤的模型 ✓ —— **FLUX 不在其中** ✓（见 ``options._flux`` 的说明 ✓）。
_PRECACHE_BUILDERS: dict[str, "PrecacheBuilder"] = {
    "wan": _wan_precache,
    "hunyuan": _hunyuan_precache,
    "kontext": _kontext_precache,
    "qwen": _qwen_precache,
}

_TRAIN_BUILDERS: dict[str, "TrainBuilder"] = {
    "wan": _wan_train,
    "hunyuan": _hunyuan_train,
    "kontext": _kontext_train,
    "flux": _flux_train,
    "qwen": _qwen_train,
}


def build_precache(model_key: str, values: Mapping[str, Any]) -> list[tuple[str, ...]]:
    """预缓存命令序列 ✓；没有这一步的模型返回**空列表** ✓（不是报错 ✓）。"""
    resolved = spec(model_key)
    builder = _PRECACHE_BUILDERS.get(resolved.key)
    if builder is None:
        return []
    return builder(values)


def build_train(model_key: str, values: Mapping[str, Any], *, workspace: Path) -> tuple[str, ...]:
    """训练命令 ✓（已含 ``accelerate`` 前缀 ✓）。

    ⚠️ **会**在 ``workspace`` 里落一个 ``prompt.txt`` ✓（当且仅当要出样图且用户没用自定义文件 ✓）。
    """
    resolved = spec(model_key)
    return _TRAIN_BUILDERS[resolved.key](values, workspace)


def tool_report(model_key: str) -> dict[str, str]:
    """该模型用到的工具脚本**落点** ✓（诊断/前端展示 ✓）；缺了就直接抛环境错 ✓。"""
    resolved = spec(model_key)
    report: dict[str, str] = {}
    for name in (resolved.train_script, *resolved.cache_scripts):
        report[name] = str(resolve_tool(name, toolkit=resolved.toolkit).path)
    return report


def build_plan(model_key: str, values: Mapping[str, Any], *, workspace: Path) -> CommandPlan:
    """**一条龙的入口** ✓ —— 校验 ⇒ 定位工具 ⇒ 拼预缓存 ⇒ 拼训练 ✓。

    ⚠️ 顺序是刻意的 ✓：``validate`` **在最前** ✓ ⇒ 参数不对时**不会**先在 ``workspace`` 里
    留下半拉子文件、也不会因为"本地没装 musubi"而盖掉"你参数填错了"这个更有用的错 ✓。
    """
    validate(model_key, values)
    resolved = spec(model_key)
    workspace = Path(workspace)

    tools = tool_report(model_key)
    cache = build_precache(model_key, values)
    train = build_train(model_key, values, workspace=workspace)
    return CommandPlan(
        cache=tuple(cache),
        train=train,
        tools=tools,
        prompt_file=_sample_prompt_path(model_key, values, workspace),
    )


#: 两个 ``Callable`` 别名 ✓ —— 写成 ``TypeAlias`` 只是为了让上面两个映射表可读 ✓。
PrecacheBuilder = Callable[[Mapping[str, Any]], list[tuple[str, ...]]]
TrainBuilder = Callable[[Mapping[str, Any], Path], tuple[str, ...]]
