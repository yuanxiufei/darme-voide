"""评测域 —— 与 ``backend/src/routes/evaluation.ts``（62 行，5 端点）对齐。

**已整域迁移（5/5）** —— 2026-09-15 校正：早先此处只列了 2 个已迁、3 个「未迁交给 catch-all 兜底」，
但那 3 个**后来都已迁**（``optimizer`` / ``creator`` / scheduler 链路均已落地）：

* ``GET  /api/v1/evaluation/cases``  列出可用基准 case（id / kind / agentType）；
* ``POST /api/v1/evaluation/evaluate/{case_id}``  用 **Reference 提示词**评测指定 case；
* ``POST /optimize/{case_id}``、``GET /scheduler``、``POST /run`` —— 优化器与调度器均已实现。
* ``POST /tune/{case_id}`` —— **本仓新增**（Node 侧没有）：参数调优（温度），见
  ``agent/evaluation/param_optimizer.py``。缺口 4 第一格，不是 TS 对齐项。
* ``POST /tune-skills/{case_id}`` —— **本仓新增**：技能绑定调优（摘掉哪个 skill 不损失分数），
  见 ``agent/evaluation/skill_optimizer.py``。缺口 4 第二格，不是 TS 对齐项。
* ``POST /tune-prompt/{case_id}`` —— **本仓新增**：提示词优化（改写提示词并评测），见
  ``agent/evaluation/optimizer.py`` 的 ``optimize_agent_prompt``。缺口 4 **第四格**，不是 TS 对齐项。
  ``POST /optimize/{case_id}`` 那个同步端点与 TS 是保真对拍项、**一字未动**；这条是它的入队版
  （每轮含一次 LLM 改写 + 一次 85–165 秒评测 ⇒ 十几分钟，不该压在请求里等）。
  四个调优（``tune`` / ``tune-skills`` / ``tune-tools`` / ``tune-prompt``）共用同一张后台任务表
  （``tune_runtime``），进度/取消/列表都是 ``/tune/{taskId}`` 那三条 ——
  路径前缀写成 ``tune-skills``/``tune-tools``/``tune-prompt`` 是为了不与 ``GET /tune/{taskId}``
  那条参数路由撞名。
* ``POST /tune-tools/{case_id}`` —— **本仓新增**：工具集调优（暴露哪几个工具），
  见 ``agent/evaluation/tool_optimizer.py``。缺口 4 **第三格**，不是 TS 对齐项。
  判据要两把尺子（整轮分 + 逐步账里的 ``unknownTool``），所以它依赖
  ``evaluator.step_trace`` 那份「单步对照」；落库恒 false（工具集真源是代码，不是配置列）。
* ``POST /film/{episode_id}`` —— **本仓新增**：成片评测（评"这一集能不能交片"），
  见 ``agent/evaluation/film_check.py``。缺口 8 的另一半，不是 TS 对齐项。

⚠️ 「哪些没迁」以 ``tests/route_parity_test.py`` 的机械扫描为准（当前**未注册 0 条** ——
早先此处写「未注册仅 1 条：``storage/change``」，那条后来已迁）。

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

from ..core.db import get_conn
from ..core.request_utils import read_json
from ..core.response import bad_request, not_found, success
from ..services.agent_prompts import get_default_instructions
from app.agent.evaluation.catalog import list_benchmark_cases, load_case_by_id
from app.agent.evaluation.evaluator import evaluate_case
from app.agent.evaluation.film_check import score_episode
from app.agent.evaluation.optimizer import normalize_prompt_options, optimize_agent_prompt
from app.agent.evaluation import tune_runtime
from app.agent.evaluation.param_optimizer import normalize_options
from app.agent.evaluation.skill_optimizer import (
    current_binding,
    normalize_skill_options,
    skill_candidates,
)
from app.agent.evaluation.tool_optimizer import normalize_tool_options
from app.agent.evaluation_scheduler import get_scheduler_state, run_evaluation_cycle
from app.agent.evaluation.types import AGENT_BY_KIND
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


@router.post("/film/{episode_id}")
def evaluate_film(episode_id: str, record: bool = True, conn: Connection = Depends(get_conn)):
    """**成片评测**（知识闭环缺口 8 的另一半，**本仓新增**，不是 TS 对齐项）。

    与上面四条的分工：那四条评的是**产出层**（提示词写得好不好、结构化输出够不够），
    这条评的是**成片**（这一集真的能交片吗）。判据全部来自已经发生的事实 ——
    分镜计划、最新成片、技术 QC 已量过的规格与 issue、人工复核已判过的结论，
    **不重跑 ffmpeg**、**不调模型**、**不自造主观分**（理由见 ``film_check`` 模块抬头）。

    ``record=false`` 时只出报告、不落 ``kind="eval"`` 跑次（用于"先看一眼"）。
    ⚠️ 两处失败形态：``episode_id`` 不是正整数 → **400**（不是 404 —— 这是参数错，
    不是"这一集不存在"）；成片/质检一条都没有 → **200 但 verdict=empty**，
    因为"这集还没开工"是**事实**，不是错误。
    """
    try:
        text_id = str(episode_id or "").strip()
        if not text_id.isdigit() or int(text_id) <= 0:
            return bad_request(f"episodeId 必须是正整数：{episode_id!r}")
        report = score_episode(conn, int(text_id), record=record)
        return success(report)
    except Exception as err:  # noqa: BLE001 —— 与 TS 的 catch 等价
        log_task_error("EvaluationAPI", "film", {"error": str(err)})
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


@router.post("/tune/{case_id}")
async def tune_params(case_id: str, request: Request):
    """参数调优（缺口 4 第一格）：**同一份提示词**只换温度，按「均值涨幅 > 抖动带」判采纳。

    body（可选）：``{repeat?, max_candidates?, auto_persist?, wait?, timeoutSeconds?}``。

    ⚠️ **默认入队，不当场跑**（2026-11-07 改）：真机实测一轮 4 跑 = **423.9 秒**，挂在同步返回链上
    的结果是「代理先超时、后台还在烧卡」—— 界面报失败而日志显示跑完，是最难查的一种。故：

    * 默认 ⇒ 立刻返回 ``{taskId, status:"queued", queuePosition, pollUrl}``，进度查
      ``GET /tune/{taskId}``（``doneRuns`` / ``totalRuns`` / ``stage`` / 跑完的 ``receipt``）；
    * ``wait=true`` ⇒ 等同旧行为：等它跑完再返回回执（``timeoutSeconds``，默认 900 秒，
      超时回当时快照并标 ``timedOut``，**任务本身不会因此停**）。脚本/探针用它最省事；
    * 同一个 (agent, case) 已在队列里 ⇒ 不重复入队，返回 ``deduped=true``（同一份提示词跑第二遍
      只是再烧七分钟）。

    ⚠️ 三条参数口径（**在入队前**就验完，验不过根本不会进队列）：
    ① ``repeat`` 默认 2，``< 2`` 是**硬错**（400）而不是悄悄抬到 2 —— 单次评测分不清「换温度了」
    与「这次恰好抖高了」，替调用方改参数等于替他撒谎；
    ② ``max_candidates`` 可以是 0（=只测基线，不下候选，用于先量抖动带）；
    ③ ``auto_persist !== false`` ⇒ 只有显式 false 才关。
    """
    try:
        case_def = load_case_by_id(case_id)
        if case_def is None:
            return not_found(f"未知基准 case：{case_id}")
        agent_type = AGENT_BY_KIND.get(case_def.get("kind") or "")
        if not agent_type:
            return bad_request(f"未知 case kind：{case_def.get('kind')}")

        body = await read_json(request)
        options: dict[str, object] = {"autoPersist": body.get("auto_persist") is not False}
        for src, dst in (("repeat", "repeat"), ("max_candidates", "maxCandidates")):
            value = body.get(src)
            is_integer_like = (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and float(value).is_integer()
            )
            # 给了非法值就**不认**（回默认），而不是 400：与 /optimize 的 iterations 同一口径。
            # 唯一例外是 repeat<2 —— 那是口径错不是笔误，交给 normalize_options 当场拒。
            if is_integer_like and value >= 0:
                options[dst] = int(value)
        # ⚠️ 校验只此一处（``normalize_options``）：入队时放行、排到才炸的那种错要烧掉前面的评测
        options = normalize_options(options)

        if not bool(body.get("wait")):
            queued = tune_runtime.submit(
                case_id=case_id, case_def=case_def, agent_type=agent_type, options=options,
            )
            return success(queued)

        try:
            timeout = float(body.get("timeoutSeconds") or body.get("timeout_seconds") or 0) or 900.0
        except (TypeError, ValueError):
            return bad_request(f"timeoutSeconds 不是数字（收到 {body.get('timeoutSeconds')!r} ✗）")
        started = tune_runtime.submit(
            case_id=case_id, case_def=case_def, agent_type=agent_type, options=options,
        )
        done = tune_runtime.wait(str(started["taskId"]), timeout=timeout)
        if done is None:
            return not_found(f"未知调优任务：{started['taskId']}")
        if done.get("status") == "failed":
            return bad_request(str(done.get("error") or "调优失败"))
        return success({**done, "waited": True})
    except Exception as err:  # noqa: BLE001 —— 与 TS 的 catch 等价
        log_task_error("EvaluationAPI", "tune", {"error": str(err)})
        return bad_request(str(err) or str(err.__class__.__name__))


@router.post("/tune-skills/{case_id}")
async def tune_skills(case_id: str, request: Request, conn: Connection = Depends(get_conn)):
    """技能绑定调优（缺口 4 第二格）：**同一份提示词、同一个温度**，只换「注入哪几个 skill」。

    body（可选）：``{repeat?, max_candidates?, auto_persist?, wait?, timeoutSeconds?}``。

    判据与温度**相反**（这是本端点最容易被误读的一处）：温度是「涨过抖动带才采纳」，
    这一格是「**摘掉后跌幅不超过抖动带 ⇒ 没有证据说明它有用 ⇒ 采纳这一摘**」——
    动的是注入量，少注入本身就是收益。两个理由码分别是 ``pruned`` / ``skill_needed``。

    ⚠️ 三处口径（与 ``POST /tune/{case_id}`` 的差异都在这里）：
    ① ``auto_persist`` 默认 **False**：温度调低最坏是更死板，删一个 skill 最坏是「该知道的规范
    不知道了」⇒ 删减默认只出结论、由人点头（``true`` 才落库）；
    ② ``max_candidates`` 是「最多试几个 skill」，实际候选还可能更少 —— 候选数取决于当前绑定
    有几个 skill、且**剩一个时没有候选**（不许摘成空）；
    ③ 同样**默认入队**：一轮也是 ``repeat × (1 + 候选数)`` 次真模型评测（几分钟到十几分钟）。

    返回值多带一份 ``preview``（当前绑定与本次会摘的候选），让调用方在开跑前就知道「要试什么」；
    真正的判决在 ``GET /tune/{taskId}`` 的 ``receipt`` 里。
    """
    try:
        case_def = load_case_by_id(case_id)
        if case_def is None:
            return not_found(f"未知基准 case：{case_id}")
        agent_type = AGENT_BY_KIND.get(case_def.get("kind") or "")
        if not agent_type:
            return bad_request(f"未知 case kind：{case_def.get('kind')}")

        body = await read_json(request)
        options: dict[str, object] = {"autoPersist": body.get("auto_persist") is True}
        for src, dst in (("repeat", "repeat"), ("max_candidates", "maxCandidates")):
            value = body.get(src)
            is_integer_like = (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and float(value).is_integer()
            )
            if is_integer_like and value >= 0:
                options[dst] = int(value)
        # 校验只此一处（``normalize_skill_options``，与温度那格共用 ``normalize_options`` 的实现）
        options = normalize_skill_options(options)

        # 候选数是「预演」而不是决策：真候选由优化器在跑起来时按当时的绑定再算一遍。
        # 这里算只是为了①入队回执的 totalRuns 别估得太离谱②把 preview 给调用方看。
        current, source = current_binding(conn, agent_type)
        candidates = skill_candidates(
            current, agent_type=agent_type, max_candidates=int(options["maxCandidates"]),
        )
        preview = {"current": current, "bindingSource": source, "candidates": candidates,
                   "tunable": "skills", "target": "skills"}

        if not bool(body.get("wait")):
            queued = tune_runtime.submit(
                case_id=case_id, case_def=case_def, agent_type=agent_type, options=options,
                target="skills", candidate_count=len(candidates),
            )
            return success({**queued, "preview": preview})

        try:
            timeout = float(body.get("timeoutSeconds") or body.get("timeout_seconds") or 0) or 900.0
        except (TypeError, ValueError):
            return bad_request(f"timeoutSeconds 不是数字（收到 {body.get('timeoutSeconds')!r} ✗）")
        started = tune_runtime.submit(
            case_id=case_id, case_def=case_def, agent_type=agent_type, options=options,
            target="skills", candidate_count=len(candidates),
        )
        done = tune_runtime.wait(str(started["taskId"]), timeout=timeout)
        if done is None:
            return not_found(f"未知调优任务：{started['taskId']}")
        if done.get("status") == "failed":
            return bad_request(str(done.get("error") or "调优失败"))
        return success({**done, "waited": True, "preview": preview})
    except Exception as err:  # noqa: BLE001 —— 与 TS 的 catch 等价
        log_task_error("EvaluationAPI", "tune-skills", {"error": str(err)})
        return bad_request(str(err) or str(err.__class__.__name__))


@router.post("/tune-tools/{case_id}")
async def tune_tools(case_id: str, request: Request, conn: Connection = Depends(get_conn)):
    """工具集调优（缺口 4 **第三格**）：同一份提示词、同一个温度、同一份绑定，只换「暴露哪几个工具」。

    body（可选）：``{repeat?, max_candidates?, wait?, timeoutSeconds?}``。

    判据**要两把尺子**（这是本格与前三格最根本的差别，也是它为什么必须等「单步对照」先落地）：
    ① 整轮分 —— 摘掉之后活还干不干得成；
    ② 逐步账里的 ``unknownTool`` 计数 —— 摘掉之后模型**还有没有再伸手去调它**。
    只有「分数没掉 **且** 模型不再调它」才判 ``pruned``（采纳这一摘）；分数没掉但轨迹里出现了
    ``Unknown tool`` ⇒ ``tool_wanted``，**拒绝**这一摘（别的工具补位而已，不是不需要）。

    ⚠️ 三处与其他三格的差异：
    ① **没有 ``auto_persist``**：``agent_configs`` 没有 tools 列，工具集真源是代码里的装配语句
    ⇒ 回执恒 ``persisted=false`` 且带 ``persistSkip``，结论只落 ``agent_optimizations(target="tools")``；
    ② **没有 ``preview.candidates``**：候选源是「基线这一跑真被调用过的工具」，而基线要先跑
    （优化器拿不到装配全集 —— 装配要真 ``episodeId``，seed 在 ``evaluate_case`` 内部）；
    ③ 候选数一样受 ``max_candidates`` 限制，且**剩一个工具时没有候选**（不许摘成空）。

    返回值仍是入队快照（默认不等），``preview`` 只给口径说明；真正的判决在 ``GET /tune/{taskId}``。
    """
    try:
        case_def = load_case_by_id(case_id)
        if case_def is None:
            return not_found(f"未知基准 case：{case_id}")
        agent_type = AGENT_BY_KIND.get(case_def.get("kind") or "")
        if not agent_type:
            return bad_request(f"未知 case kind：{case_def.get('kind')}")

        body = await read_json(request)
        options: dict[str, object] = {}
        for src, dst in (("repeat", "repeat"), ("max_candidates", "maxCandidates")):
            value = body.get(src)
            is_integer_like = (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and float(value).is_integer()
            )
            if is_integer_like and value >= 0:
                options[dst] = int(value)
        options = normalize_tool_options(options)

        preview = {
            "tunable": "tools",
            "target": "tools",
            "candidateSource": "trace",
            "note": ("候选在基线跑完之后才定：源头是这一跑真被调用过的工具 —— 装配全集要真 "
                     "episodeId，评估时才有"),
            "persist": False,
            "persistSkip": ("agent_configs 没有 tools 列，工具集真源是代码装配语句 ⇒ "
                            "本格只出结论、由人照着改代码"),
        }

        if not bool(body.get("wait")):
            queued = tune_runtime.submit(
                case_id=case_id, case_def=case_def, agent_type=agent_type, options=options,
                target="tools",
            )
            return success({**queued, "preview": preview})

        try:
            timeout = float(body.get("timeoutSeconds") or body.get("timeout_seconds") or 0) or 900.0
        except (TypeError, ValueError):
            return bad_request(f"timeoutSeconds 不是数字（收到 {body.get('timeoutSeconds')!r} ✗）")
        started = tune_runtime.submit(
            case_id=case_id, case_def=case_def, agent_type=agent_type, options=options,
            target="tools",
        )
        done = tune_runtime.wait(str(started["taskId"]), timeout=timeout)
        if done is None:
            return not_found(f"未知调优任务：{started['taskId']}")
        if done.get("status") == "failed":
            return bad_request(str(done.get("error") or "调优失败"))
        return success({**done, "waited": True, "preview": preview})
    except Exception as err:  # noqa: BLE001 —— 与 TS 的 catch 等价
        log_task_error("EvaluationAPI", "tune-tools", {"error": str(err)})
        return bad_request(str(err) or str(err.__class__.__name__))


@router.post("/tune-prompt/{case_id}")
async def tune_prompt(case_id: str, request: Request, conn: Connection = Depends(get_conn)):
    """提示词优化（缺口 4 **第四格**）：先改写提示词、再评测，严格更高才采纳。

    body（可选）：``{iterations?, repeat?, auto_persist?, wait?, timeoutSeconds?}``。

    ⚠️ 这是**入队版**；``POST /optimize/{case_id}`` 那个同步端点只同步了一处口径（基线改成
    「DB 里正在生效的那份」），两条**共用同一个** ``optimize_agent_prompt``，区别只在投递方式。
    同步那条留给 CLI/脚本（不受浏览器/代理超时约束）；
    而这里一轮 =「一次 LLM 改写 + 一次 85–165 秒的评测」× ``iterations``，**十几分钟起步**，
    走请求面必然会先被代理掐断，所以补这条入队口。

    ⚠️ 三处与另三格的差异（照抄任何一条都会得到错的读数）：
    ① **``repeat`` 是「Reference 复测几次」**，不是「同一档位重跑几次」：候选每轮只评一次 ⇒
    分母 ``repeat + iterations``（那 ``repeat`` 跑全在 Reference 上）。照抄 ``repeat × (1 + n)``
    会报出 2 倍分母、任务 done 了页面还停在半程；
    ② **是四格里唯一会真改配置的一格**：``auto_persist`` 默认 **True**（与 ``/optimize`` 一致），
    best 严格高于 Reference **+ 抖动带** 才写进 ``agent_configs.system_prompt`` ⇒ 传 ``false``
    只出结论。Reference 是**运行时在用的那份**提示词（DB 优先、无配置行才回退默认）
    ⇒ 这条判据读作「要赢过正在跑的版本」，不会把已经更好的配置覆盖成更差的；
    ③ **抖动带可选**：``repeat=1``（默认）⇒ 没测抖动带、判据是「严格高于当前 best」
    （``jumpKind="none"``）；``repeat=2–5`` ⇒ 用 Reference 样本极差当抖动带
    （``jumpKind="within"``），「碰巧高 0.5 分」不再被采纳。实测同配置重复测量极差 **8 分**
    （见 ``docs/knowledge-loop.md`` §4.2），所以真有噪声顾虑就传 ``repeat=3``，代价是
    Reference 多跑 2 次（47–165 秒/次）。**没有固定阈值**：阈值只能来自本次实测的极差，
    拍一个「8 分」会把真实的 5 分改进也判成无效。

    两条边界**越界即拒（400）**，不夹逼、也不静默回默认：``iterations > 20``、
    ``repeat`` 不在 1–5（见 ``optimizer.normalize_prompt_options`` 的依据）。

    返回值仍是入队快照（默认不等），真正的判决在 ``GET /tune/{taskId}`` 的 ``receipt`` 里。
    """
    try:
        case_def = load_case_by_id(case_id)
        if case_def is None:
            return not_found(f"未知基准 case：{case_id}")
        agent_type = AGENT_BY_KIND.get(case_def.get("kind") or "")
        if not agent_type:
            return bad_request(f"未知 case kind：{case_def.get('kind')}")
        body = await read_json(request)
        options: dict[str, object] = {"autoPersist": body.get("auto_persist") is not False}
        value = body.get("iterations")
        is_integer_like = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and float(value).is_integer()
        )
        # 与 ``/optimize`` 同一口径：非法值**不认**、回默认 3，而不是 400；
        # `iterations=0` 是合法值（只评一次 Reference 就收工，用来测通路）。
        # ⚠️ 「非法值回默认」只盖**类型**错（`"abc"`、`1.5`、`True`）；**越界**（>20）不在此列，
        #    由下面 `normalize_prompt_options` 当场 400 —— 悄悄按默认跑，用户会以为跑了 30 轮。
        if is_integer_like and value >= 0:
            options["iterations"] = int(value)
        value = body.get("repeat")
        repeat_is_integer_like = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and float(value).is_integer()
        )
        if repeat_is_integer_like:
            options["repeat"] = int(value)

        # 唯一校验点（执行端共用同一个函数）：入队前就拒，别让用户等到轮询才看见「排队了、然后失败」
        normalized = normalize_prompt_options(options)
        iterations = int(normalized["iterations"])
        repeat = int(normalized["repeat"])
        options = {"iterations": iterations, "repeat": repeat,
                   "autoPersist": normalized["autoPersist"]}

        # repeat=1 与 repeat≥2 是**两种判据**，别用一句「复测 1 次取抖动带」糊过去
        rule = (f"Reference 复测 {repeat} 次取抖动带（极差待测），候选要超过「当前 best + 抖动带」才采纳"
                if repeat >= 2 else
                "本格没测抖动带（repeat=1，零容差：候选严格高于当前 best 才采纳）")
        preview = {
            "tunable": "system_prompt",
            "target": "system_prompt",
            "iterations": iterations,
            # 分母的真值是 repeat + iterations（Reference 那 repeat 跑也要钱）
            "runs": repeat + iterations,
            "repeat": repeat,
            "jumpKind": "within" if repeat >= 2 else "none",
            "note": rule + "；且是四格里唯一会真写 agent_configs.system_prompt 的一格",
            "persist": bool(normalized["autoPersist"]),
        }

        if not bool(body.get("wait")):
            queued = tune_runtime.submit(
                case_id=case_id, case_def=case_def, agent_type=agent_type, options=options,
                target="system_prompt", candidate_count=iterations,
                total_runs=repeat + iterations,
            )
            return success({**queued, "preview": preview})

        try:
            timeout = float(body.get("timeoutSeconds") or body.get("timeout_seconds") or 0) or 900.0
        except (TypeError, ValueError):
            return bad_request(f"timeoutSeconds 不是数字（收到 {body.get('timeoutSeconds')!r} ✗）")
        started = tune_runtime.submit(
            case_id=case_id, case_def=case_def, agent_type=agent_type, options=options,
            target="system_prompt", candidate_count=iterations,
            total_runs=repeat + iterations,
        )
        done = tune_runtime.wait(str(started["taskId"]), timeout=timeout)
        if done is None:
            return not_found(f"未知调优任务：{started['taskId']}")
        if done.get("status") == "failed":
            return bad_request(str(done.get("error") or "调优失败"))
        return success({**done, "waited": True, "preview": preview})
    except Exception as err:  # noqa: BLE001 —— 与 TS 的 catch 等价
        log_task_error("EvaluationAPI", "tune-prompt", {"error": str(err)})
        return bad_request(str(err) or str(err.__class__.__name__))


@router.get("/tune")
def list_tune_tasks(limit: int = 20):
    """最近的调优任务（新的在前 —— 进程重启即清空，见 ``tune_runtime`` 模块头）。"""
    return success({"tasks": tune_runtime.list_tasks(limit=limit)})


@router.get("/tune/{task_id}")
def tune_task(task_id: str):
    """调优任务进度 / 回执：``status`` + ``doneRuns``/``totalRuns`` + ``stage``；跑完带 ``receipt``。"""
    snapshot = tune_runtime.get(task_id)
    if snapshot is None:
        return not_found(f"未知调优任务：{task_id}")
    return success(snapshot)


@router.post("/tune/{task_id}/cancel")
def cancel_tune_task(task_id: str):
    """取消（**只在两次评测之间生效** —— 一次真模型生成从外面掐不断，见 ``tune_runtime`` 模块头）。"""
    snapshot = tune_runtime.cancel(task_id)
    if snapshot is None:
        return not_found(f"未知调优任务：{task_id}")
    return success(snapshot)


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
