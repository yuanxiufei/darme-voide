"""评测闭环（``backend/src/evaluation/`` 的 Python 侧）。

⚠️ 迁移进度（分步落地，未迁部分见各自说明）：

* ✅ ``types``(151) / ``catalog``(49) / ``scorer``(327) —— 数据契约、基准目录、确定性评分器；
* ⬜ ``evaluator``(210)：seed 数据 + 跑 Agent + 取工具入参 + 交给 scorer（**依赖已就位的 runtime**）；
* ⬜ ``optimizer``(192)：提示词迭代优化（**还依赖未迁的** ``agents/skills.ts`` 与 ``agents/creator.ts``）；
* ⬜ ``evaluation-scheduler``(126)：定时调度；
* ⬜ ``routes/evaluation.ts``(62)：5 个端点；
* ⬜ ``cli.ts``(56)：**开发者 CLI，不是 HTTP 接口** ⇒ 不打算迁（删 ``backend/`` 后若仍需要，
  应作为独立脚本重写，而不是照搬）。
"""
