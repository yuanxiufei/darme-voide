"""**素材工具** ✓ —— LoRAMaster ``dataset_manager`` 的移植 ✓。

| 参考实现（``reference/lora/LoRAMaster/dataset_manager/`` ✓） | 本仓 |
| --- | --- |
| ``image_rename.py`` ✓ | :mod:`~app.services.lora_train.dataset.files` 的 ``run_rename`` ✓ |
| ``image_convert.py`` ✓ | 同上的 ``run_convert`` ✓ |
| ``AutoCaptioning.py`` ✓ | :mod:`~app.services.lora_train.dataset.caption` ✓ |
| 三个 ``*_settings.toml`` ✓ | :mod:`~app.services.lora_train.dataset.fields` ✓ |
| ``JoyTest.py``（一个 52 行的独立小脚本 ✓，与本包无关 ✓） | **不搬** ✗（它是作者的一次性试验 ✓） |
| NiceGUI 界面（``AutoCaptioning.py`` 第 509 行往后 ✓） | 前端页面 ✓（见 ``routers/lora_train.py`` 给的清单 ✓） |

⚠️ 各模块头都逐条写了「哪里照抄 ✓、哪里刻意改了并给出理由 ✓」✓ —— 读之前**先读模块头** ✓，
不然会把"改过的地方"当成抄漏了 ✓✗。
"""
from __future__ import annotations

from . import caption, fields, files, prompts, runtime
from .runtime import DatasetRuntime, dataset_runtime

__all__ = [
    "DatasetRuntime",
    "caption",
    "dataset_runtime",
    "fields",
    "files",
    "prompts",
    "runtime",
]
