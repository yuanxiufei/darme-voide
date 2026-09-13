"""评测域 —— 与 ``backend/src/routes/evaluation.ts``（62 行，5 端点）对齐。

**已迁 2 个**：

* ``GET  /api/v1/evaluation/cases``  列出可用基准 case（id / kind / agentType）；
* ``POST /api/v1/evaluation/evaluate/{case_id}``  用 **Reference 提示词**评测指定 case。

**未迁 3 个**（依赖未迁件 ⇒ **交给 catch-all 兜底**：`PROXY_TO_NODE=1` 时反代到 Node，否则 501）：

* ``POST /optimize/{case_id}`` —— 依赖 ``optimizer``（还需 ``creator.ts``）；
* ``GET  /scheduler``         —— 依赖 ``evaluation-scheduler``；
* ``POST /run``               —— 同上。

⚠️ 这 3 条**不需要显式委派**：显式委派只用于「会被已注册的**参数路由遮蔽**」的路径
（如 ``GET /agent-configs/defaults`` 被 ``/{config_id}`` 吞掉）。这里 ``/optimize/{id}``、
``/run``、``/scheduler`` 的首段字面量与已注册路径都不同，不会被吞。

⚠️ 两处保真点：

1. ``GET /cases`` 在原 TS 里**没有 try** ⇒ 基准目录里有坏 JSON 会直接 500（不是 400）。
   这里同样不兜底（保持一致的失败形态）；
2. ``POST /evaluate`` 的三个失败分支**状态码不同**：case 不存在 → **404**、
   kind 未知 → **400**、运行期异常 → **400**；且提示文案带**全角冒号**（``未知基准 case：x``）。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy.engine import Connection

from ..db import get_conn
from ..request_utils import read_json
from ..response import bad_request, not_found, success
from ..services.agent_prompts import get_default_instructions
from ..services.evaluation.catalog import list_benchmark_cases, load_case_by_id
from ..services.evaluation.evaluator import evaluate_case
from ..services.evaluation.optimizer import optimize_agent_prompt
from ..services.evaluation_scheduler import get_scheduler_state, run_evaluation_cycle
from ..services.evaluation.types import AGENT_BY_KIND
from ..services.task_logger import log_task_error

router = APIRouter(prefix="/api/v1/evaluation", tags=["evaluation"])


@router.get("/cases")
def list_cases():
    """列出可用基准 case（⚠️ 无 try，坏 JSON ⇒ 500，与原 TS 一致）。"""
    return success(list_benchmark_cases())


@router.post("/evaluate/{case_id}")
async def evaluate_one(case_id: str, conn: Connection = Depends(get_conn)):
    """用 ``getDefaultInstructions(agentType)`` 作为 Reference 提示词评测指定 case。"""
    try:
        case_def = load_case_by_id(case_id)
        if case_def is None:
            return not_found(f"未知基准 case：{case_id}")
        agent_type = AGENT_BY_KIND.get(case_def.get("kind") or "")
        if not agent_type:
            return bad_request(f"未知 case kind：{case_def.get('kind')}")

        report = await evaluate_case(conn, case_def, get_default_instructions(agent_type))
        return success(report)
    except Exception as err:  # noqa: BLE001 —— 与 TS 的 catch 等价
        log_task_error("EvaluationAPI", "evaluate", {"error": str(err)})
        return bad_request(str(err) or str(err.__class__.__name__))


@router.post("/optimize/{case_id}")
async def optimize_one(case_id: str, request: Request, conn: Connection = Depends(get_conn)):
    """状态机优化（best **严格高于** Reference 时自动落库）。

    body（可选）：``{iterations?: number, auto_persist?: boolean}``。

    ⚠️ 三处细节：① body 用**宽容**读取（原 TS 是 ``c.req.json().catch(() => ({}))``）；
    ② ``iterations`` 判据是 ``Number.isInteger && > 0`` ⇒ JS 眼里 ``2.0`` 也是整数，
    这里用「整数值的浮点」等价判定；③ ``auto_persist !== false`` ⇒ **只有显式 false 才关**。
    """
    try:
        case_def = load_case_by_id(case_id)
        if case_def is None:
            return not_found(f"未知基准 case：{case_id}")
        agent_type = AGENT_BY_KIND.get(case_def.get("kind") or "")
        if not agent_type:
            return bad_request(f"未知 case kind：{case_def.get('kind')}")

        body = await read_json(request)
        raw_iterations = body.get("iterations")
        is_integer_like = (
            isinstance(raw_iterations, (int, float))
            and not isinstance(raw_iterations, bool)
            and float(raw_iterations).is_integer()
        )
        iterations = int(raw_iterations) if is_integer_like and raw_iterations > 0 else 3
        auto_persist = body.get("auto_persist") is not False

        history = await optimize_agent_prompt(
            conn, agent_type, case_def,
            {"iterations": iterations, "autoPersist": auto_persist},
        )
        return success(history)
    except Exception as err:  # noqa: BLE001 —— 与 TS 的 catch 等价
        log_task_error("EvaluationAPI", "optimize", {"error": str(err)})
        return bad_request(str(err) or str(err.__class__.__name__))


@router.get("/scheduler")
def scheduler_state():
    """定时调度器状态（``enabled``/``running``/``lastRunAt``/``nextRunAt``/``lastResults``）。"""
    return success(get_scheduler_state())


@router.post("/run")
async def run_cycle():
    """手动触发一轮全量优化（与定时**共用同一执行入口**，靠防重入保证不叠加）。"""
    try:
        return success(await run_evaluation_cycle())
    except Exception as err:  # noqa: BLE001 —— 与 TS 的 catch 等价
        log_task_error("EvaluationAPI", "run", {"error": str(err)})
        return bad_request(str(err) or str(err.__class__.__name__))
