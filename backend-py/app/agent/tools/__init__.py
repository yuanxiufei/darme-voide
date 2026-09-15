"""Agent 工具集（S5）—— 对应 Node 侧 ``backend/src/agents/tools/``。

六组工具、共 ≈1,362 行，形状统一为工厂函数：

    create_xxx_tools(conn, episode_id, drama_id) -> dict[str, Tool]

⚠️ 这一层的定位是**薄封装**：真正干活的逻辑全在已迁的服务里（prompt 构建器、库操作、
生成服务、连续性状态机…）。Node 侧每个工具用 ``createTool`` + zod 声明入参；这里换成
JSON Schema（见 :mod:`app.services.agents.tool`），并显式传 ``conn``（Node 用全局 db）。
"""
