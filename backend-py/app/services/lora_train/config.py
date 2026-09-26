"""LoRA 训练参数配置层：``settings.toml`` 的读写与校验 ✓（LoRAMaster 移植 ✓）。

出处（参考实现 ✓）：``reference/lora/LoRAMaster/<模型>_lora_train.py`` 里的
``load_settings()`` / ``save_settings()`` ✓ —— 参数表**平铺**在 ``<模型>_settings.toml`` ✓
（键名 = 训练参数名 ✓，值以**字符串**为主 ✓，只有少量 ``true``/``false`` 是真布尔 ✓）。

⚠️ 本仓与参考实现的**三处刻意不同** ✓（都不是遗漏 ✗）：
1. **落点**：参考实现把 toml 放在仓库目录里 ✗ ⇒ 本仓放**数据根**下
   （``<dataRoot>/lora-train/settings/<模型>_settings.toml`` ✓）✓ —— 配置是运行时数据 ✓，
   不该随代码升级被覆盖 ✓，也不该进 git ✓。
2. **读坏文件的行为**：参考实现的 ``load_settings()`` 用 ``except Exception: return {}`` ✗✗
   ⇒ 文件写坏了会**静默回到默认值** ✗（用户会以为自己的配置"没生效" ✓ 其实是被丢了 ✓）。
   本仓按纪律**响亮报错** ✓：带上文件路径与原由 ✓（本仓判据「不许静默兜底」✓）。
3. **键的校验**：参考实现不校验 ✗（toml 里多一个键，命令组装时被无声忽略 ✗）。
   本仓统一交给 :func:`options.merge` ✓ —— 未知键当场点名并列出该模型的合法参数 ✓。

读用标准库 ``tomllib``（Python 3.11+ ✓）；写**不引第三方包** ✓ ⇒ 自己实现一个
只覆盖本场景（平铺标量 ✓）的极小 TOML 书写器 ✓。
"""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Mapping

from ...core.config import get_data_root
from .errors import LoraTrainConfigError
from .options import merge, spec

#: 数据根下的子目录名（相对 ``get_data_root()``）✓
DATA_SUBDIR = "lora-train"
#: 训练参数文件所在子目录 ✓
SETTINGS_SUBDIR = "settings"


def data_dir() -> Path:
    """本模块的数据根 ✓（``<dataRoot>/lora-train`` ✓；不存在就建 ✓）。"""
    path = Path(get_data_root()) / DATA_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def settings_dir() -> Path:
    """训练参数文件目录 ✓（``<dataRoot>/lora-train/settings`` ✓）。"""
    path = data_dir() / SETTINGS_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def settings_path(model_key: str) -> Path:
    """某模型的参数文件路径 ✓；**模型名先过 :func:`options.spec` 校验** ✓（不认就报错 ✗）。"""
    return settings_dir() / f"{spec(model_key).key}_settings.toml"


def read_toml(path: Path) -> dict[str, Any]:
    """读一个平铺 TOML ✓ ⇒ 字典 ✓；文件不存在 ⇒ 空字典 ✓（**这不是兜底**：没配过就是没配过 ✓）。

    ⚠️ 文件存在但**读不动**（语法错 / 不是 dict）⇒ 显式报错 ✓（**绝不静默吃成空字典** ✗）。
    """
    target = Path(path)
    if not target.is_file():
        return {}
    try:
        with target.open("rb") as handle:
            loaded = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as err:
        raise LoraTrainConfigError(f"训练参数文件读不动 ✗：{target} —— {err}") from err
    if not isinstance(loaded, dict):  # pragma: no cover —— tomllib 只会给 dict ✓
        raise LoraTrainConfigError(f"训练参数文件顶层不是键值表 ✗：{target}")
    return loaded


def load_values(model_key: str) -> dict[str, Any]:
    """读某模型**完整**参数 ✓（默认值 + 文件覆盖 ✓，逐键过校验 ✓）。"""
    return merge(model_key, read_toml(settings_path(model_key)))


def save_values(model_key: str, payload: Mapping[str, Any] | None) -> Path:
    """把参数写成 TOML ✓ ⇒ 落盘路径 ✓。

    ⚠️ 先 ``merge`` 再落盘 ✓（顺序不能反 ✗）：参数错就**当场报错且不写字** ✓ ——
    否则会留下一个"半对"的文件 ✓，下次读回来又是另一个错 ✓。
    """
    values = merge(model_key, payload)
    target = settings_path(model_key)
    target.write_text(dumps_toml(values), encoding="utf-8", newline="\n")
    return target


def reset_values(model_key: str) -> bool:
    """删掉某模型的参数文件 ✓ ⇒ 是否真删了 ✓（本来就没了 ⇒ ``False`` ✓，不报错 ✓）。"""
    target = settings_path(model_key)
    if not target.is_file():
        return False
    target.unlink()
    return True


def _scalar(value: Any) -> str:
    """把一个标量写成 TOML 字面量 ✓（本场景只有 str / bool / int / float ✓）。"""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    text = "" if value is None else str(value)
    if "\n" in text or "\r" in text or '"""' in text:
        # 多行字符串 ✓ —— 把三引号拆开 ✓，否则会提前结束字面量 ✓
        body = text.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
        if body.endswith('"'):
            body = body[:-1] + '\\"'
        return f'"""\n{body}\n"""'
    escaped = (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def dumps_toml(values: Mapping[str, Any]) -> str:
    """平铺字典 ⇒ TOML 文本 ✓（键按字典序 ✓ ⇒ **同内容必得同文本** ✓，落盘可比对 ✓）。"""
    lines = [f"{key} = {_scalar(values[key])}" for key in sorted(values)]
    return "\n".join(lines) + "\n"
