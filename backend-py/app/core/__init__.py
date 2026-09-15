"""平台层 —— 与业务无关的基础设施：配置 / 数据库 / 表定义 / 响应层 / 请求工具。

层级约定（2026-09-15 收口，对齐前端 `pages` / `composables` / `utils` 的分层）：

    main.py     装配入口（include_router / 中间件 / SPA / 接缝）
    core/       平台层：**不 import routers/ 与 services/**
    routers/    HTTP 层：可 import core/ 与 services/
    services/   业务层：可 import core/，**不 import routers/**

⚠️ 该单向依赖由 `tests/layering_test.py` 机械守卫（2026-09-15 起）。
"""
from __future__ import annotations
