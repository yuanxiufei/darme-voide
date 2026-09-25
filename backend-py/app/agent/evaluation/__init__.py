"""评测闭环（``backend/src/evaluation/`` 的 Python 侧）。

✅ **已全量落地**（2026-09-15 收口；本包 + ``agent/evaluation_scheduler.py`` + ``routers/evaluation.py``）：

* ✅ ``types`` / ``catalog`` / ``scorer`` —— 数据契约、基准目录、确定性评分器；
* ✅ ``evaluator``：seed 临时剧组 + 跑 Agent + 取工具入参 + 交给 scorer + 物理清理；
* ✅ ``optimizer``：提示词迭代优化状态机（依赖的 ``agents/skills`` 与 ``creator`` 均已就位）；
* ✅ ``evaluation_scheduler``：定时调度（读 ``config.yaml`` 的 ``evaluation.auto_optimize``，
  **默认关闭**，需显式 ``enabled: true`` 才启动，避免意外烧钱）；
* ✅ ``routers/evaluation.py``：5 个端点全部注册 —— ⚠️ 前端**无 UI 入口**（只经 HTTP/CLI 使用）；
* ✅ ``cli``：开发者 CLI 已作为**独立脚本**重写，而不是照搬 TS 版本。
"""
