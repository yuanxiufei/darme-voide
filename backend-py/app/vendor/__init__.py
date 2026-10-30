"""外部官方源码的 vendor 区（**原样移植，不做业务改动**）。

⚠️ 为什么 vendor 而不是 ``pip install``：本机 PyPI 直连不通（见 ``requirements.txt`` 的说明），
且后端要的是**确定的源码版本**而不是浮动的最新版。放进仓内以后，``ollama`` 这个包名
不会和机器上可能装着的同名第三方包冲突。

当前内容：

* ``app.vendor.ollama`` —— Ollama 官方 Python SDK（``reference/ollama-python``，MIT）。
  相对上游只做了**一处**机械改动：把 4 行 ``from ollama.xxx import`` 改成相对导入
  （``from .xxx import``），否则以 ``app.vendor.ollama`` 为模块路径导入时会去找顶层
  ``ollama`` 包。依赖 ``httpx`` / ``pydantic>=2.9`` / ``anyio`` / ``typing_extensions``，
  本项目全部已具备 ⇒ **零新依赖**。

用法（一律走这个路径，不要 ``import ollama``）::

    from app.vendor.ollama import AsyncClient, Client

    client = AsyncClient(host="http://localhost:11434")
    resp = await client.chat(model="qwen3:14b", messages=[...])
"""

from __future__ import annotations

from . import ollama

__all__ = ["ollama"]
