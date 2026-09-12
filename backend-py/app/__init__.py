"""Drama Studio —— Python 后端（绞杀者迁移）。

首个迁移域：``/api/v1/dramas``。其余路径在 ``PROXY_TO_NODE=1`` 时反代到 Node 后端，
未开启时返回 501 并说明该域尚未迁移（避免静默 404 难以定位）。
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
