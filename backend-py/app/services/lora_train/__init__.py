"""**LoRA 训练** ✓ —— LoRAMaster（https://github.com/wochenlong/LoRAMaster ✓）的移植 ✓。

本仓原本**只有推理侧**的 LoRA 支持 ✓（把 LoRA 当加速插件挂上去 ✓），
这一包是第一次引入**训练** ✓（见 ``docs`` 里那句"按顺序抄"的口径 ✓）。

## 抄的是什么 ✓ / 抄到哪 ✓

| LoRAMaster 的东西 ✓ | 落点 |
| --- | --- |
| 5 个 ``*_lora_train.py`` 的参数表与命令行组装 ✓ | :mod:`.options` + :mod:`.commands` ✓ |
| ``*_settings.toml`` 的读写 ✓ | :mod:`.config` ✓ |
| ``subprocess.Popen`` 起训练 + 杀进程树 ✓ | :mod:`.runner` ✓ |
| GUI 的日志面板 / 进度 ✓ | :mod:`.progress` ✓ + :mod:`.runtime` ✓ |
| ``auto_shutdown.py`` ✓ | :mod:`.shutdown` ✓ |
| ``dataset_manager/`` 全套 ✓ | :mod:`.dataset` ✓ |
| 5 个 NiceGUI 页面 ✓ | 前端页面 ✓（见 ``routers/lora_train.py`` ✓） |
| 工具目录都是写死的绝对路径 ✓✗ | :mod:`.paths` 动态探测 ✓（本仓纪律，见该模块头 ✓） |

## 五条训练链 ✓

``wan`` / ``hunyuan`` / ``kontext`` / ``flux`` / ``qwen-image`` ✓，
前四个走 **musubi-tuner** ✓、``flux`` 走 **sd-scripts** ✓（:data:`.paths.TOOLKIT_*` ✓）。

⚠️ **导入本包不会起任何线程、不会加载任何模型** ✓：
两个运行时的守护线程都是**第一次 ``submit`` 时才起** ✓（:meth:`LoraTrainRuntime.submit` ✓），
torch / transformers / Pillow 全是**惰性导入** ✓（见 ``runner``、``caption`` 模块头 ✓）。
"""
from __future__ import annotations

from . import commands, config, dataset, errors, options, paths, progress, runner, runtime, shutdown
from .errors import (
    LoraTrainBusyError,
    LoraTrainCancelled,
    LoraTrainConfigError,
    LoraTrainError,
    LoraTrainNotFoundError,
    LoraTrainRunError,
)
from .options import DEFAULT_MODEL, MODELS, catalog, merge, model_keys, spec
from .runtime import LoraTrainRuntime, LoraTrainTask, lora_train_runtime, resolve_values

__all__ = [
    "DEFAULT_MODEL",
    "LoraTrainBusyError",
    "LoraTrainCancelled",
    "LoraTrainConfigError",
    "LoraTrainError",
    "LoraTrainNotFoundError",
    "LoraTrainRunError",
    "LoraTrainRuntime",
    "LoraTrainTask",
    "MODELS",
    "catalog",
    "commands",
    "config",
    "dataset",
    "errors",
    "lora_train_runtime",
    "merge",
    "model_keys",
    "options",
    "paths",
    "progress",
    "resolve_values",
    "runner",
    "runtime",
    "shutdown",
    "spec",
]
