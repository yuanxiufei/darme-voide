"""**LoRA 训练的参数体系** —— 每个模型有哪些参数 ✓、默认值多少 ✓、前端该给什么控件 ✓。

参考实现把参数表写在 **nicegui 界面代码**里 ✓（``ui.input`` 顺带绑定参数名 ✓）+ 一份随附的
``*_settings.toml`` ✓。本仓前端是 Nuxt ✗ ⇒ 界面不能靠读 nicegui 代码生成 ✗，
所以把「参数名 / 类型 / 默认值 / 中文标签 / 取值域」提成**数据** ✓：
后端据此校验 ✓，前端据此渲染表单 ✓（一份口径两处用 ✓）。

## 事实来源（别凭记忆改 ✗）

* **参数名与默认值** = 参考实现随附的 ``*_settings.toml`` ✓（逐项转录 ✓）；
* **参数是否真的进命令行** = 由 ``commands.py`` 决定 ✓，本文件只描述取值 ✓；
* ⚠️ **两个刻意改动** ✓（照抄会害人 ✗）：
  1. **不搬作者机器上的绝对路径** ✗ —— 参考那份 toml 里 ``I:\\train_models\\...`` /
     ``E:\\train\\...`` 是作者本机的盘符 ✓✗，抄进来只会变成「默认值指向不存在的盘」✓。本仓一律留空 ✓。
  2. **``qwen.num_processes`` 用 1 而非参考里的 ``"100"``** ✓ —— 单卡起 100 个 accelerate 进程
     是参数写错 ✗，不是要抄的口径 ✓。
* ⚠️ 参考里 ``qwen.edit`` 这个键**读了但没进命令** ✓（真正决定 ``--model_version`` 的是 ``train_type`` ✓）；
  本仓保留它以便老配置原样读回 ✓，标签里写明「当前不生效」✓。

## 不许静默兜底 ✗

:func:`merge` 遇到**不认识的键**直接报错 ✓✗（唯一例外是 ``_`` 开头的注释键 ✓）。
「悄悄丢掉一个拼错的参数、然后拿默认值跑 200 个 epoch」是本类工具最贵的失败方式 ✓。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .errors import LoraTrainConfigError
from .paths import TOOLKIT_MUSUBI, TOOLKIT_SD_SCRIPTS

KIND_TEXT = "text"
KIND_TEXTAREA = "textarea"
KIND_PATH_FILE = "pathFile"
KIND_PATH_DIR = "pathDir"
KIND_INT = "int"
KIND_FLOAT = "float"
KIND_BOOL = "bool"
KIND_SELECT = "select"


@dataclass(frozen=True)
class Option:
    """一个训练参数 ✓。"""

    key: str
    kind: str
    default: Any
    label: str
    choices: tuple[str, ...] = ()
    help: str = ""
    #: 是否**进命令行** ✓ —— ``False`` 表示参考实现读了但没用 ✓（本仓照样保留以便读回老配置 ✓）
    used: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "kind": self.kind, "default": self.default,
            "label": self.label, "choices": list(self.choices), "help": self.help,
            "used": self.used,
        }


@dataclass(frozen=True)
class ModelSpec:
    """一个可训练模型的**全部静态口径** ✓。"""

    key: str
    label: str
    toolkit: str
    train_script: str
    #: 预缓存脚本 ✓（**空元组 = 没有独立预缓存步骤** ✓ —— FLUX 走 sd-scripts 的内联缓存开关 ✓）
    cache_scripts: tuple[str, ...]
    network_module: str
    options: tuple[Option, ...]
    note: str = ""

    def option(self, key: str) -> Option | None:
        return next((o for o in self.options if o.key == key), None)

    def defaults(self) -> dict[str, Any]:
        return {o.key: o.default for o in self.options}

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "label": self.label, "toolkit": self.toolkit,
            "trainScript": self.train_script, "cacheScripts": list(self.cache_scripts),
            "networkModule": self.network_module, "note": self.note,
            "options": [o.as_dict() for o in self.options],
        }


#: 共用标签 ✓：``key -> (中文标签, 说明)`` ✓ —— 五个模型的同名参数大多是同一个意思 ✓，
#: 写一份就不会出现「同一个 learning_rate 在两个模型里叫两个名字」✓。
_LABELS: dict[str, tuple[str, str]] = {
    "dataset_config": ("数据集配置 (toml)", "dataset config 文件路径；[general]/[[datasets]] 结构见文档"),
    "vae_path": ("VAE 权重", "预缓存与训练都要用的 VAE"),
    "vae_cache_cpu": ("VAE 缓存放 CPU", "显存紧张时把 VAE 编码放 CPU 跑，慢但省显存"),
    "skip_existing": ("跳过已缓存的", "缓存文件已存在就不重算，续跑时省大量时间"),
    "use_clip": ("使用 CLIP 编码器", "部分 Wan2.1 任务需要 CLIP 编码（Wan2.2 一般不需要）"),
    "clip_model_path": ("CLIP 权重", "CLIP-L / open-clip 权重路径"),
    "t5_path": ("T5 权重", "T5 文本编码器权重（t5xxl 系）"),
    "text_encoder_model_path": ("文本编码器权重", "该模型主文本编码器（HunyuanVideo 用 llava-llama3，Qwen-Image 用 Qwen2.5-VL）"),
    "dit_weights_path": ("DiT 主权重", "要训 LoRA 的底模（扩散主干）"),
    "dit_high_noise_path": ("高噪 DiT 权重", "Wan2.2 双模型结构：低噪/高噪各一份，这里给高噪那份"),
    "batch_size": ("批大小", "每步喂多少条样本；受显存限制"),
    "max_train_epochs": ("训练轮数", "跑多少轮数据；单卡上这个值决定总时长"),
    "learning_rate": ("学习率", "可用 2e-4 这类科学计数法"),
    "network_dim": ("网络维度 (dim)", "LoRA 秩；越大容量越高、显存与体积也更大"),
    "gradient_accumulation_steps": ("梯度累积步数", "等效放大批大小，几乎不额外吃显存"),
    "timestep_sampling": ("时间步采样", "决定在扩散时间轴上怎么采训练点"),
    "discrete_flow_shift": ("flow shift", "Flow-matching 的 shift 值，与采样步数口径相关"),
    "enable_low_vram": ("低显存模式", "开则把部分层换到内存（配合 blocks_to_swap）"),
    "blocks_to_swap": ("换出块数", "低显存模式下换出到内存的 DiT 块数，越大越省显存越慢"),
    "output_dir": ("输出目录", "LoRA 权重落盘目录"),
    "output_name": ("输出名", "权重文件名前缀，通常起成数据集/角色名"),
    "save_every_n_epochs": ("每 N 轮存一次", ""),
    "save_every_n_steps": ("每 N 步存一次", ""),
    "use_network_weights": ("从已有 LoRA 续训", "开则从 network_weights_path 读入既有 LoRA 继续训"),
    "network_weights_path": ("已有 LoRA 路径", "续训起点权重"),
    "generate_samples": ("训练中出样图", "按下面参数每隔一段生成样图，直接看训练效果"),
    "sample_every_n_epochs": ("每 N 轮出样图", ""),
    "sample_every_n_steps": ("每 N 步出样图", ""),
    "sample_prompt_text": ("样图提示词", "训练中途出样图用的提示词（应带上触发词）"),
    "sample_image_path": ("样图首帧/控制图", "图生视频或编辑类任务作为输入图；纯文生图不用"),
    "sample_w": ("样图宽", ""),
    "sample_h": ("样图高", ""),
    "sample_frames": ("样图帧数", "视频模型有效；图像模型忽略"),
    "sample_seed": ("样图种子", ""),
    "sample_steps": ("样图步数", ""),
    "sample_at_first": ("训练前先出一张", "开始训练前立刻出样图，便于确认参数对不对"),
    "custom_prompt_txt": ("用自定义 prompt 文件", "开则改用 custom_prompt_path，忽略上面的提示词"),
    "custom_prompt_path": ("自定义 prompt 文件", "自带 --w/--h/--s 等后缀的 prompt 文件"),
    "fp8": ("fp8 权重", "底模按 fp8 加载，显存占用明显下降"),
    "num_cpu_threads_per_process": ("每进程 CPU 线程", "accelerate 启动参数；给多核机器用"),
    "num_processes": ("进程数", "单卡保持 1；多卡才加"),
    "timestep_custom": ("自定义时间步范围", "手动限制训练用的时间步区间（min/max）"),
    "min_timestep": ("最小时间步", "越小越偏结构/构图"),
    "max_timestep": ("最大时间步", "越大越偏细节/纹理"),
    "timestep_boundary": ("双模型分界时间步", "Wan2.2 高噪/低噪切换点"),
    "lazy_loading": ("懒加载", "按需把权重换进换出，省显存"),
    "attention_implementation": ("注意力实现", "sdpa 通用稳妥；xformers 需要装对应轮子"),
    "optimizer_type": ("优化器", "adamw8bit 省显存，是单卡主力选择"),
    "max_data_loader_n_workers": ("DataLoader 进程数", "Windows 上给大值容易启动卡住，2 是稳的"),
    "auto_shutdown": ("跑完自动关机", "训练结束后关机器，适合睡前挂长训"),
    "log_type": ("日志后端", "tensorboard 才能看到曲线"),
    "log_prefix": ("日志前缀", ""),
    "log_dir": ("日志目录", ""),
    "log_tracker_name": ("日志任务名", ""),
    "offload_inactive_dit": ("换出闲置 DiT", "Wan2.2 双模型时把当前不用的那份换出去"),
    "mixed_precision": ("混合精度", "fp16 快但部分模型会溢出；bf16 更稳"),
    "lr_scheduler": ("学习率调度", ""),
    "lr_warmup_steps": ("预热步数", "constant_with_warmup 时生效"),
    "lr_scheduler_num_cycles": ("重启周期数", "cosine_with_restarts 时生效"),
    "custom_params": ("自定义追加参数", "原样拼到命令末尾的逃生口；写这里的参数绕过前面的表单"),
}


def _o(key: str, kind: str, default: Any, *, choices: tuple[str, ...] = (),
       label: str = "", help_: str = "", used: bool = True) -> Option:
    """建一个 :class:`Option` ✓ —— 标签/说明默认取共用表 ✓，个别模型要改就显式传 ✓。"""
    base_label, base_help = _LABELS.get(key, (key, ""))
    return Option(key=key, kind=kind, default=default,
                  label=label or base_label, help=help_ or base_help,
                  choices=choices, used=used)


def _shared_tail(*, models_with_lr: bool = True, sample_frames: bool = False,
                 sample_image: bool = True) -> tuple[Option, ...]:
    """「采样 / 启动 / 优化器 / 日志」这一整段 ✓ —— 各模型的差异全在这几个开关上 ✓。"""
    items: list[Option] = [
        _o("generate_samples", KIND_BOOL, True),
        _o("sample_every_n_epochs", KIND_TEXT, "1000"),
        _o("sample_every_n_steps", KIND_TEXT, "1000"),
        _o("sample_prompt_text", KIND_TEXTAREA, ""),
    ]
    if sample_image:
        items.append(_o("sample_image_path", KIND_PATH_FILE, ""))
    items += [
        _o("sample_w", KIND_TEXT, "512"),
        _o("sample_h", KIND_TEXT, "512"),
    ]
    if sample_frames:
        items.append(_o("sample_frames", KIND_TEXT, "33"))
    items += [
        _o("sample_seed", KIND_TEXT, "666666"),
        _o("sample_steps", KIND_TEXT, "10"),
        _o("sample_at_first", KIND_BOOL, False),
        _o("custom_prompt_txt", KIND_BOOL, False),
        _o("custom_prompt_path", KIND_PATH_FILE, ""),
        _o("num_cpu_threads_per_process", KIND_INT, 1),
        _o("num_processes", KIND_INT, 1),
        _o("attention_implementation", KIND_SELECT, "sdpa", choices=("sdpa", "xformers")),
        _o("optimizer_type", KIND_TEXT, "adamw8bit"),
        _o("max_data_loader_n_workers", KIND_INT, 2),
        _o("auto_shutdown", KIND_BOOL, True),
        _o("log_type", KIND_TEXT, "tensorboard"),
        _o("log_prefix", KIND_TEXT, ""),
        _o("log_dir", KIND_PATH_DIR, "./logs"),
        _o("log_tracker_name", KIND_TEXT, ""),
        _o("offload_inactive_dit", KIND_BOOL, False),
        _o("mixed_precision", KIND_SELECT, "bf16", choices=("bf16", "fp16", "no")),
    ]
    if models_with_lr:
        items += [
            _o("lr_scheduler", KIND_SELECT, "constant",
               choices=("constant", "constant_with_warmup", "cosine", "cosine_with_restarts",
                        "linear", "adafactor")),
            _o("lr_warmup_steps", KIND_TEXT, "0"),
            _o("lr_scheduler_num_cycles", KIND_TEXT, "1"),
            _o("custom_params", KIND_TEXTAREA, ""),
        ]
    return tuple(items)


def _wan() -> ModelSpec:
    """Wan2.1 / Wan2.2 视频 LoRA ✓（musubi-tuner ✓）。"""
    options = (
        _o("dataset_config", KIND_PATH_FILE, ""),
        _o("task", KIND_SELECT, "i2v-A14B",
           choices=("t2v-A14B", "i2v-A14B", "t2v-1.3B", "i2v-1.3B", "t2v-14B", "i2v-14B"),
           label="任务类型",
           help_="决定是否按图生视频走（i2v 会给缓存脚本加 --i2v）；A14B 是 Wan2.2 双模型结构，"
                 "会触发高噪权重与时间步分界那一组参数"),
        _o("vae_cache_cpu", KIND_BOOL, False),
        _o("skip_existing", KIND_BOOL, True),
        _o("vae_path", KIND_PATH_FILE, ""),
        _o("t5_path", KIND_PATH_FILE, ""),
        _o("use_clip", KIND_BOOL, False),
        _o("clip_model_path", KIND_PATH_FILE, ""),
        _o("dit_weights_path", KIND_PATH_FILE, ""),
        _o("dit_high_noise_path", KIND_PATH_FILE, ""),
        _o("batch_size", KIND_TEXT, "16"),
        _o("max_train_epochs", KIND_TEXT, "200"),
        _o("learning_rate", KIND_TEXT, "2e-4"),
        _o("network_dim", KIND_TEXT, "32"),
        _o("gradient_accumulation_steps", KIND_TEXT, "1"),
        _o("timestep_sampling", KIND_SELECT, "shift",
           choices=("shift", "sigmoid", "uniform", "flux_shift")),
        _o("discrete_flow_shift", KIND_TEXT, "5"),
        _o("enable_low_vram", KIND_BOOL, True),
        _o("blocks_to_swap", KIND_TEXT, "20"),
        _o("output_dir", KIND_PATH_DIR, "./output"),
        _o("output_name", KIND_TEXT, ""),
        _o("save_every_n_epochs", KIND_TEXT, "1"),
        _o("save_every_n_steps", KIND_TEXT, "1000"),
        _o("use_network_weights", KIND_BOOL, False),
        _o("network_weights_path", KIND_PATH_FILE, ""),
        _o("fp8", KIND_BOOL, True),
        _o("timestep_custom", KIND_BOOL, True),
        _o("min_timestep", KIND_TEXT, "900"),
        _o("timestep_boundary", KIND_TEXT, "1000"),
        _o("max_timestep", KIND_TEXT, "1000"),
        _o("lazy_loading", KIND_BOOL, False),
        *_shared_tail(sample_frames=True),
    )
    return ModelSpec(
        key="wan", label="Wan 视频 LoRA（Wan2.1 / Wan2.2）",
        toolkit=TOOLKIT_MUSUBI, train_script="wan_train_network",
        cache_scripts=("wan_cache_latents", "wan_cache_text_encoder_outputs"),
        network_module="networks.lora_wan", options=options,
        note="A14B 走双 DiT（高噪/低噪）：需给高噪权重与时间步分界；单卡建议 fp8 + 低显存",
    )


def _hunyuan() -> ModelSpec:
    """HunyuanVideo LoRA ✓（musubi-tuner 的 ``hv_*`` ✓）。

    ⚠️ 预缓存脚本用的是**通用名** ``cache_latents`` / ``cache_text_encoder_outputs`` ✓
    （不是 ``hv_cache_*`` ✓）—— 参考实现里 hunyuan 那条链拼的就是通用名 ✓，实测 musubi-tuner
    根目录也同时存在通用名与专用名两套 ✓，所以这里**保持参考的写法** ✓。
    """
    options = (
        _o("dataset_config", KIND_PATH_FILE, ""),
        _o("vae_cache_cpu", KIND_BOOL, False),
        _o("skip_existing", KIND_BOOL, True),
        _o("vae_path", KIND_PATH_FILE, ""),
        _o("text_encoder_model_path", KIND_PATH_FILE, ""),
        _o("clip_model_path", KIND_PATH_FILE, ""),
        _o("dit_weights_path", KIND_PATH_FILE, ""),
        _o("batch_size", KIND_TEXT, "16"),
        _o("max_train_epochs", KIND_TEXT, "200"),
        _o("learning_rate", KIND_TEXT, "2e-4"),
        _o("network_dim", KIND_TEXT, "32"),
        _o("gradient_accumulation_steps", KIND_TEXT, "1"),
        _o("timestep_sampling", KIND_SELECT, "shift",
           choices=("shift", "sigmoid", "uniform", "flux_shift")),
        _o("discrete_flow_shift", KIND_TEXT, "5"),
        _o("enable_low_vram", KIND_BOOL, True),
        _o("blocks_to_swap", KIND_TEXT, "20"),
        _o("output_dir", KIND_PATH_DIR, "./output"),
        _o("output_name", KIND_TEXT, ""),
        _o("save_every_n_epochs", KIND_TEXT, "10"),
        _o("save_every_n_steps", KIND_TEXT, "500"),
        _o("use_network_weights", KIND_BOOL, False),
        _o("network_weights_path", KIND_PATH_FILE, ""),
        _o("fp8", KIND_BOOL, True),
        _o("timestep_custom", KIND_BOOL, True),
        _o("min_timestep", KIND_TEXT, "900"),
        _o("timestep_boundary", KIND_TEXT, "1000"),
        _o("max_timestep", KIND_TEXT, "1000"),
        _o("lazy_loading", KIND_BOOL, False),
        # ⚠️ 共用段里 ``models_with_lr`` 对 hunyuan 传 True ✓ —— 这是**本仓增补** ✓：
        # 参考实现的 hunyuan 链没暴露 lr_scheduler（其余四个模型都有 ✓），
        # 而 musubi 的 hv_train_network 本身支持 ✓ ⇒ 漏掉它会让 hunyuan 缺一档常用能力 ✗。
        # 增补只有这一处 ✓，值也取与其它模型一致的 constant ✓。
        #
        # ⚠️ ``sample_image=False`` ✓ —— **HunyuanVideo 没有"样图输入图"这个参数** ✓：
        # 参考实现里那几行 ``sample_image_path`` 是被**注释掉**的 ✓（ui.row 整段注释 ✓），
        # 且它的 make_prompt_file 里也没有 ``--i`` 分支 ✓ ⇒ 本仓不列这个键 ✓。
        *_shared_tail(sample_frames=True, sample_image=False),
    )
    return ModelSpec(
        key="hunyuan", label="HunyuanVideo LoRA",
        toolkit=TOOLKIT_MUSUBI, train_script="hv_train_network",
        cache_scripts=("cache_latents", "cache_text_encoder_outputs"),
        network_module="networks.lora", options=options,
        note="预缓存用通用名脚本；文本编码器是 llava-llama3（文本编码器权重那一项）",
    )


def _kontext() -> ModelSpec:
    """FLUX.1 Kontext（图像编辑）LoRA ✓（musubi-tuner 的 ``flux_kontext_*`` ✓）。"""
    options = (
        _o("dataset_config", KIND_PATH_FILE, ""),
        _o("vae_path", KIND_PATH_FILE, ""),
        _o("t5_path", KIND_PATH_FILE, ""),
        _o("clip_model_path", KIND_PATH_FILE, ""),
        _o("dit_weights_path", KIND_PATH_FILE, ""),
        _o("batch_size", KIND_TEXT, "16"),
        _o("max_train_epochs", KIND_TEXT, "200"),
        _o("learning_rate", KIND_TEXT, "2e-4"),
        _o("network_dim", KIND_TEXT, "32"),
        _o("gradient_accumulation_steps", KIND_TEXT, "1"),
        _o("timestep_sampling", KIND_SELECT, "flux_shift",
           choices=("flux_shift", "shift", "sigmoid", "uniform"),
           help_="Kontext 链**没有** discrete_flow_shift 这一项 ✓；shift 的口径由 timestep_sampling 决定 ✓"),
        _o("enable_low_vram", KIND_BOOL, True),
        _o("blocks_to_swap", KIND_TEXT, "20"),
        _o("output_dir", KIND_PATH_DIR, "./output"),
        _o("output_name", KIND_TEXT, ""),
        _o("save_every_n_epochs", KIND_TEXT, "1"),
        _o("save_every_n_steps", KIND_TEXT, "1000"),
        _o("use_network_weights", KIND_BOOL, False),
        _o("network_weights_path", KIND_PATH_FILE, ""),
        _o("lazy_loading", KIND_BOOL, False),
        # Kontext 是图像编辑 ✓ ⇒ 无 sample_frames ✗；有控制图（sample_image_path ✓，
        # 进命令行时是 ``--ci`` ✓ 不是 ``--i`` ✓，见 commands ✓）
        *_shared_tail(sample_frames=False),
    )
    return ModelSpec(
        key="kontext", label="FLUX.1 Kontext 编辑 LoRA",
        toolkit=TOOLKIT_MUSUBI, train_script="flux_kontext_train_network",
        cache_scripts=("flux_kontext_cache_latents", "flux_kontext_cache_text_encoder_outputs"),
        network_module="networks.lora_flux", options=options,
        note="图像编辑任务：样图的控制图走 --ci；没有 discrete_flow_shift 这一项",
    )


def _flux() -> ModelSpec:
    """FLUX.1 图像 LoRA ✓ —— ⚠️ **这一条链走 sd-scripts，不走 musubi-tuner** ✓。

    ⚠️ 与另外四个模型的**结构差异**（都是实测差异 ✓，不是遗漏 ✗）：

    * 预缓存脚本**不存在** ✓ —— sd-scripts 的 ``flux_train_network`` 自己是
      ``--cache_latents_to_disk`` 内联缓存的 ✓ ⇒ ``cache_scripts=()`` ✓，
      ``commands.build_precache`` 会直接返回空列表 ✓（不是报错 ✓）。
    * 采样参数**只有**「每 N 步/轮 + 提示词」✓ —— 没有起手尺寸/种子/步数 ✓
      （sd-scripts 侧采样尺寸由数据集分桶决定 ✓）⇒ 这里不列那些键 ✓，
      而 :func:`merge` 会把它们当**未知键**报错 ✓ —— 这是有意为之 ✓：
      从 Kontext 复制参数过来时会立刻发现"这些键在 FLUX 上不生效"✓，而不是静默忽略 ✗。
    * 有 ``cache_to_disk`` ✓ 与 ``fp8`` ✓；``discrete_flow_shift`` 默认 ``3.1582`` ✓
      （参考里就是这个位数 ✓，不是随手写的 ✓）。
    """
    options = (
        _o("dataset_config", KIND_PATH_FILE, ""),
        _o("vae_path", KIND_PATH_FILE, ""),
        _o("t5_path", KIND_PATH_FILE, ""),
        _o("clip_model_path", KIND_PATH_FILE, ""),
        _o("dit_weights_path", KIND_PATH_FILE, ""),
        _o("batch_size", KIND_TEXT, "16"),
        _o("max_train_epochs", KIND_TEXT, "200"),
        _o("learning_rate", KIND_TEXT, "2e-4"),
        _o("network_dim", KIND_TEXT, "64",
           help_="FLUX 参考默认给 64（其余模型是 32）"),
        _o("gradient_accumulation_steps", KIND_TEXT, "1"),
        _o("timestep_sampling", KIND_SELECT, "shift",
           choices=("shift", "sigmoid", "uniform", "flux_shift")),
        _o("enable_low_vram", KIND_BOOL, True),
        _o("blocks_to_swap", KIND_TEXT, "20"),
        _o("output_dir", KIND_PATH_DIR, "./output"),
        _o("output_name", KIND_TEXT, ""),
        _o("save_every_n_epochs", KIND_TEXT, "10"),
        _o("save_every_n_steps", KIND_TEXT, "500"),
        _o("use_network_weights", KIND_BOOL, False),
        _o("network_weights_path", KIND_PATH_FILE, ""),
        _o("generate_samples", KIND_BOOL, True),
        _o("sample_every_n_epochs", KIND_TEXT, "1",
           help_="⚠️ 参考实现读了它但**没有把它拼进命令** ✓（``used=False`` ✓）"),
        _o("sample_every_n_steps", KIND_TEXT, "100"),
        _o("sample_prompt_text", KIND_TEXTAREA, ""),
        _o("custom_prompt_txt", KIND_BOOL, False),
        _o("custom_prompt_path", KIND_PATH_FILE, ""),
        _o("num_cpu_threads_per_process", KIND_INT, 1),
        _o("num_processes", KIND_INT, 1),
        _o("lazy_loading", KIND_BOOL, False),
        _o("attention_implementation", KIND_SELECT, "sdpa", choices=("sdpa", "xformers")),
        _o("optimizer_type", KIND_TEXT, "adamw8bit"),
        _o("max_data_loader_n_workers", KIND_INT, 2),
        _o("auto_shutdown", KIND_BOOL, True),
        _o("log_type", KIND_TEXT, "tensorboard"),
        _o("log_prefix", KIND_TEXT, ""),
        _o("log_dir", KIND_PATH_DIR, "./logs"),
        _o("log_tracker_name", KIND_TEXT, ""),
        _o("offload_inactive_dit", KIND_BOOL, False),
        _o("mixed_precision", KIND_SELECT, "bf16", choices=("bf16", "fp16", "no")),
        _o("discrete_flow_shift", KIND_TEXT, "3.1582", used=False,
           label="flow shift（当前不生效）",
           help_="⚠️ 参考那份 flux 配置里有这个键 ✓，但整段训练命令**从未用到** ✓✗ ⇒ 标为不生效。"
                 "要真的改 flow shift，得走下面的「自定义追加参数」自己写 --discrete_flow_shift ✓"),
        _o("fp8", KIND_BOOL, True),
        _o("cache_to_disk", KIND_BOOL, True,
           label="潜变量缓存到磁盘",
           help_="sd-scripts 侧的内联缓存开关；同时控制是否 --cache_text_encoder_outputs_to_disk"),
        _o("lr_scheduler", KIND_SELECT, "constant",
           choices=("constant", "constant_with_warmup", "cosine", "cosine_with_restarts",
                    "linear", "adafactor")),
        _o("lr_warmup_steps", KIND_TEXT, "0"),
        _o("lr_scheduler_num_cycles", KIND_TEXT, "1"),
        _o("custom_params", KIND_TEXTAREA, ""),
    )
    return ModelSpec(
        key="flux", label="FLUX.1 图像 LoRA",
        toolkit=TOOLKIT_SD_SCRIPTS, train_script="flux_train_network",
        cache_scripts=(), network_module="networks.lora_flux", options=options,
        note="这一条链走 sd-scripts：没有独立预缓存步骤；采样参数只有每 N 步/轮 + 提示词",
    )


def _qwen() -> ModelSpec:
    """Qwen-Image / Qwen-Image-Edit LoRA ✓（musubi-tuner ✓）。

    ⚠️ ``train_type`` **只决定** ``--model_version`` ✓ —— 训练脚本名**不随它变** ✗：
    参考实现的 ``run_wan_training()`` 无论哪种 train_type 拼的都是
    ``qwen_image_train_network.py`` ✓（``qwen_image_lora_train.py`` 第 326 行 ✓）。
    本仓早先的注释曾写过"edit 会换脚本" ✗，那是**猜的** ✗，已按源码更正 ✓。
    """
    options = (
        _o("train_type", KIND_SELECT, "qwen_image_2512",
           choices=tuple(QWEN_MODEL_VERSIONS),
           label="训练类型",
           help_="决定缓存脚本与训练脚本上的 --model_version；qwen_image_2512 不加该参数"),
        _o("edit", KIND_BOOL, True, used=False,
           label="编辑模式（当前不生效）",
           help_="⚠️ 参考实现读了它但**没有拼进任何命令** ✓；真正决定 model_version 的是 train_type。"
                 "保留它只为让参考口径的老配置能原样读回"),
        _o("dataset_config", KIND_PATH_FILE, ""),
        _o("vae_path", KIND_PATH_FILE, ""),
        _o("text_encoder_model_path", KIND_PATH_FILE, ""),
        _o("dit_weights_path", KIND_PATH_FILE, ""),
        _o("batch_size", KIND_TEXT, "1",
           help_="Qwen-Image 参考默认就是 1（底模大、分辨率高）"),
        _o("max_train_epochs", KIND_TEXT, "500"),
        _o("learning_rate", KIND_TEXT, "5e-5"),
        _o("network_dim", KIND_TEXT, "32"),
        _o("gradient_accumulation_steps", KIND_TEXT, "1"),
        _o("timestep_sampling", KIND_SELECT, "shift",
           choices=("shift", "sigmoid", "uniform", "flux_shift")),
        _o("discrete_flow_shift", KIND_TEXT, "2.2"),
        _o("enable_low_vram", KIND_BOOL, True),
        _o("blocks_to_swap", KIND_TEXT, "30"),
        _o("output_dir", KIND_PATH_DIR, "./output"),
        _o("output_name", KIND_TEXT, ""),
        _o("save_every_n_epochs", KIND_TEXT, "10"),
        _o("save_every_n_steps", KIND_TEXT, "500"),
        _o("use_network_weights", KIND_BOOL, False),
        _o("network_weights_path", KIND_PATH_FILE, ""),
        _o("fp8", KIND_BOOL, True),
        _o("generate_samples", KIND_BOOL, False),
        _o("sample_every_n_epochs", KIND_TEXT, "1"),
        _o("sample_every_n_steps", KIND_TEXT, "100"),
        _o("sample_prompt_text", KIND_TEXTAREA, ""),
        _o("sample_image_path", KIND_PATH_FILE, ""),
        _o("sample_w", KIND_TEXT, "512"),
        _o("sample_h", KIND_TEXT, "512"),
        _o("sample_seed", KIND_TEXT, "666666"),
        _o("sample_steps", KIND_TEXT, "10"),
        _o("sample_at_first", KIND_BOOL, False),
        _o("custom_prompt_txt", KIND_BOOL, False),
        _o("custom_prompt_path", KIND_PATH_FILE, ""),
        _o("num_cpu_threads_per_process", KIND_INT, 1),
        _o("num_processes", KIND_INT, 1,
           help_="⚠️ 参考那份配置里写的是 \"100\" ✓ —— 那是**参数填错** ✓，本仓用 1（单卡）；"
                 "多卡再按实际卡数加"),
        _o("attention_implementation", KIND_SELECT, "sdpa", choices=("sdpa", "xformers")),
        _o("optimizer_type", KIND_TEXT, "adamw8bit"),
        _o("max_data_loader_n_workers", KIND_INT, 2),
        _o("auto_shutdown", KIND_BOOL, True),
        _o("log_type", KIND_TEXT, "tensorboard"),
        _o("log_prefix", KIND_TEXT, ""),
        _o("log_dir", KIND_PATH_DIR, "./logs"),
        _o("log_tracker_name", KIND_TEXT, ""),
        _o("offload_inactive_dit", KIND_BOOL, False),
        _o("mixed_precision", KIND_SELECT, "bf16", choices=("bf16", "fp16", "no")),
        _o("lr_scheduler", KIND_SELECT, "constant",
           choices=("constant", "constant_with_warmup", "cosine", "cosine_with_restarts",
                    "linear", "adafactor")),
        _o("lr_warmup_steps", KIND_TEXT, "0"),
        _o("lr_scheduler_num_cycles", KIND_TEXT, "1"),
        _o("custom_params", KIND_TEXTAREA, ""),
    )
    return ModelSpec(
        key="qwen", label="Qwen-Image / Edit LoRA",
        toolkit=TOOLKIT_MUSUBI, train_script="qwen_image_train_network",
        cache_scripts=("qwen_image_cache_latents", "qwen_image_cache_text_encoder_outputs"),
        network_module="networks.lora_qwen_image", options=options,
        note="train_type 决定缓存/训练命令上的 --model_version（2512 不加该参数）；edit 键当前不生效",
    )


#: ``train_type`` → ``--model_version`` 取值 ✓（``None`` = 不加这个参数 ✓）。
#:
#: ⚠️ 三个 edit 分支的取值 **不是**照抄键名 ✓：参考实现把 ``qwen_image_edit`` 映射成 ``edit`` ✓、
#: ``qwen_image_edit_2511`` 映射成 ``edit-2511`` ✓（``qwen_image_lora_train.py`` 第 155~163 行 ✓）。
#: 训练脚本名**不随** ``train_type`` 变 ✓（固定 ``qwen_image_train_network`` ✓，第 326 行 ✓）。
QWEN_MODEL_VERSIONS: dict[str, str | None] = {
    "qwen_image_2512": None,
    "qwen_image_edit": "edit",
    "qwen_image_edit_2509": "edit-2509",
    "qwen_image_edit_2511": "edit-2511",
}

MODELS: dict[str, ModelSpec] = {m.key: m for m in (_wan(), _hunyuan(), _kontext(), _flux(), _qwen())}

DEFAULT_MODEL = "wan"


def model_keys() -> tuple[str, ...]:
    """全部可训练模型 key ✓（顺序 = 前端展示顺序 ✓）。"""
    return tuple(MODELS)


def spec(model_key: str) -> ModelSpec:
    """取模型规格 ✓；不认识 ⇒ 报错并列出合法值 ✓（**不回落默认** ✗）。"""
    if model_key not in MODELS:
        raise LoraTrainConfigError(
            f"不认识的模型 {model_key!r} ✗；合法值：{list(MODELS)}"
        )
    return MODELS[model_key]


def qwen_model_version(values: Mapping[str, Any]) -> str | None:
    """qwen 的 ``--model_version`` 取值 ✓（``None`` = 不加 ✓）；非法取值 ⇒ 报错 ✓。

    放在这里而不是 ``commands.py`` ✓：这是**参数取值域**的事 ✓（归 options ✓）；
    ``commands`` 只管把它拼进 argv ✓。
    """
    train_type = str(values.get("train_type", "")).strip()
    if train_type not in QWEN_MODEL_VERSIONS:
        raise LoraTrainConfigError(
            f"qwen 的 train_type 取值非法：{train_type!r} ✗；合法值：{list(QWEN_MODEL_VERSIONS)}"
        )
    return QWEN_MODEL_VERSIONS[train_type]


def _coerce(model_key: str, opt: Option, value: Any) -> Any:
    """按 :class:`Option` 的口径收敛取值 ✓ —— 只做「能确定的转换」✓，转不了就报错 ✓。

    ⚠️ 数值类参数**保持字符串** ✓：``learning_rate`` 的 ``"2e-4"`` 原样透传给命令行 ✓，
    转成 float 再拼回去会变成 ``0.0002`` ✓ —— 值等价但**日志/配置回读**就对不上原始口径了 ✓✗。
    """
    if opt.kind in (KIND_BOOL,):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            low = value.strip().lower()
            if low in ("true", "1", "yes", "on"):
                return True
            if low in ("false", "0", "no", "off", ""):
                return False
        raise LoraTrainConfigError(f"{model_key}.{opt.key} 需要布尔值 ✗，收到 {value!r}")
    if opt.kind == KIND_INT:
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            raise LoraTrainConfigError(f"{model_key}.{opt.key} 需要整数 ✗，收到 {value!r}") from None
    if opt.kind == KIND_SELECT and opt.choices:
        text = str(value).strip()
        if text not in opt.choices:
            raise LoraTrainConfigError(
                f"{model_key}.{opt.key} 取值非法：{text!r} ✗；合法值：{list(opt.choices)}"
            )
        return text
    if value is None:
        return ""
    return str(value)


def merge(model_key: str, payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """**默认值 + 覆盖** ✓ ⇒ 一份可交给 ``commands`` 的完整取值 ✓。

    * 未知键 ⇒ :class:`LoraTrainConfigError` ✓（**不静默丢弃** ✗✗，理由见模块头 ✓）；
    * ``_`` 开头的键当注释放过 ✓（参考实现的 toml 里出现过 ``_comment`` 这类写法 ✓）；
    * 每个值都过 :func:`_coerce` ✓。
    """
    resolved = spec(model_key)
    values = resolved.defaults()
    for key, raw in (payload or {}).items():
        if not isinstance(key, str):
            raise LoraTrainConfigError(f"参数名必须是字符串 ✗，收到 {key!r}")
        if key.startswith("_"):
            continue
        opt = resolved.option(key)
        if opt is None:
            legal = ", ".join(sorted(o.key for o in resolved.options))
            raise LoraTrainConfigError(
                f"模型 {model_key} 没有参数 {key!r} ✗；该模型可用参数：{legal}"
            )
        values[key] = _coerce(model_key, opt, raw)
    return values


def catalog() -> list[dict[str, Any]]:
    """给前端的一份完整目录 ✓（``GET /lora-train/models`` 直接回它 ✓）。"""
    return [MODELS[key].as_dict() for key in model_keys()]

