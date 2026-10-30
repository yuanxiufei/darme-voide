"""S5 运行时自检（``agents/runtime.py`` ← ``agents/index.ts`` 745 行）。

运行时是「一堆失败路径」构成的：真正跑通的那条很短，值钱的是**失败怎么分类、怎么退避、
什么时候换模型、什么时候立刻放弃**。所以本测试的重点全在失败路径上，用**假驱动 + 假时钟**
跑，不打网络、不真等 2 秒。

四块容易静默写错的逻辑：

1. **失败分类**：401/403/400、上下文超限、审核拦截 ⇒ ``fatal``（重试无意义，立即失败）；
   限流/过载/超时/网络 ⇒ ``transient``；**兜底是 transient**（宁重试，别误伤可用模型）；
2. **模型 fallback**：DB 指定模型**前置**（去重保序）；同一模型内瞬态最多退避 3 次
   （2s/4s/8s），耗尽才换下一个；全挂才抛最后一个错；
3. **风格 Profile 注入有两个前置条件**（缺一个就整段不注入）：类型必须是
   ``storyboard_breaker/extractor/script_rewriter``，**且必须存在激活的 Profile** ——
   视觉图谱引导也在 profile 分支内 ⇒ 没有 profile 时连图谱都不注入（与原 TS 一致）；
4. **``maxOutputTokens`` / ``temperature`` 必须显式下发**（原 TS 注释：不传则 DB 里是死配置，
   长回复被截断时排在文本之后的 tool call 整段丢失）。

运行::

    ./.venv/Scripts/python.exe tests/agent_runtime_test.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="agentrt_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.core.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.core.models import agent_configs, dramas, episodes, style_profiles  # noqa: E402
from app.core.response import now  # noqa: E402
from app.agent import context_budget as cb  # noqa: E402
from app.agent import runtime as rt  # noqa: E402
from app.agent.tool import Tool, ToolRegistry, json_string, object_schema  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


class FakeDriver:
    """假驱动：按预设脚本返回或抛错，并记录每次调用。"""

    def __init__(self, script: list[object]) -> None:
        self.script = script
        self.calls: list[dict] = []

    async def __call__(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self.script) - 1)
        outcome = self.script[index]
        if isinstance(outcome, BaseException):
            raise outcome
        return dict(outcome)  # type: ignore[arg-type]


_SLEPT: list[float] = []


async def _fake_sleep(seconds: float) -> None:
    _SLEPT.append(seconds)


def _ok(text: str = "done", **extra) -> dict:
    payload = {"text": text, "tool_calls": [], "tool_results": [], "steps": []}
    payload.update(extra)
    return payload


def _status_error(status: int, message: str) -> Exception:
    err = Exception(message)
    err.status = status  # type: ignore[attr-defined]
    return err


def main() -> int:  # noqa: C901
    # ================= 失败分类 =================
    check("分类: 401/403/400 -> fatal（认证/参数错，重试无意义）",
          rt.classify_llm_error(_status_error(401, "unauthorized")) == "fatal"
          and rt.classify_llm_error(_status_error(403, "forbidden")) == "fatal"
          and rt.classify_llm_error(_status_error(400, "bad request")) == "fatal")
    check("分类: 上下文超限 -> fatal",
          rt.classify_llm_error(Exception("This model's maximum context length is 8192"))
          == "fatal"
          and rt.classify_llm_error(Exception("token limit exceeded")) == "fatal")
    check("分类: 审核拦截 -> fatal",
          rt.classify_llm_error(Exception("blocked by content filter")) == "fatal"
          and rt.classify_llm_error(Exception("moderation: unsafe")) == "fatal")
    check("分类: 429/5xx -> transient",
          all(rt.classify_llm_error(_status_error(code, "err")) == "transient"
              for code in (429, 500, 502, 503, 504)))
    check("分类: 限流/配额/超时/网络 -> transient",
          rt.classify_llm_error(Exception("rate limit reached")) == "transient"
          and rt.classify_llm_error(Exception("insufficient_quota")) == "transient"
          and rt.classify_llm_error(Exception("request timed out")) == "transient"
          and rt.classify_llm_error(Exception("fetch failed")) == "transient")
    check("分类: **兜底是 transient**（宁重试，别误伤可用模型）",
          rt.classify_llm_error(Exception("某种没见过的错")) == "transient")
    check("分类: 兼容 response.status（非顶层 status）",
          rt.classify_llm_error(Exception("boom")) == "transient")
    resp_err = Exception("boom")
    resp_err.response = type("R", (), {"status": 401})()  # type: ignore[attr-defined]
    check("分类: 从 err.response.status 取状态码（401 -> fatal）",
          rt.classify_llm_error(resp_err) == "fatal")

    # ================= 退避 / token / 归一 =================
    check("退避: 2s 起、×2、30s 封顶",
          [rt.backoff_delay(i) for i in range(6)] == [2000, 4000, 8000, 16000, 30000, 30000],
          [rt.backoff_delay(i) for i in range(6)])
    usage = rt.extract_token_usage({"steps": [
        {"usage": {"promptTokens": 10, "completionTokens": 5}},
        {"usage": {"inputTokens": 1, "outputTokens": 2, "totalTokens": 9}},
    ]})
    check("用量: 累加所有 step，且 v4/v5 两套字段名都认",
          usage is not None and (usage.input_tokens, usage.output_tokens, usage.total_tokens)
          == (11, 7, 24), usage)
    check("用量: 没有 steps 时回退到顶层 usage；都没有 -> None",
          rt.extract_token_usage({"usage": {"inputTokens": 3}}) is not None
          and rt.extract_token_usage({"usage": {"inputTokens": 3}}).input_tokens == 3
          and rt.extract_token_usage({}) is None)
    check("用量: totalTokens 缺失时用 input+output 补",
          rt.extract_token_usage({"usage": {"inputTokens": 4, "outputTokens": 6}}).total_tokens
          == 10)
    check("归一: 工具名从 payload.toolName 取（Mastra 形状）",
          rt.normalize_tool_nick({"payload": {"toolName": "save_script"}}) == "save_script"
          and rt.normalize_tool_nick({"toolName": "x"}) == "x"
          and rt.normalize_tool_nick({"tool": {"id": "y"}}) == "y")
    check("归一: 工具结果 —— 字符串原样、对象紧凑 JSON（无空格）",
          rt.normalize_tool_output("abc") == "abc"
          and rt.normalize_tool_output({"a": 1, "b": [2]}) == '{"a":1,"b":[2]}'
          and rt.normalize_tool_output({"payload": {"result": "inner"}}) == "inner")

    # ================= 指令组装 / 模型候选 =================
    assembled = rt.assemble_instructions("基础指令", "技能指令")
    check("组装: base + skill + 协议契约，`\\n\\n` 连接",
          assembled.split("\n\n")[0] == "基础指令" and "技能指令" in assembled
          and "协议" in assembled, assembled[:60])
    check("组装: 空 base / None skill **被丢掉**（不产生空段）",
          rt.assemble_instructions("", None).startswith("##"), rt.assemble_instructions("", None)[:20])
    check("模型: DB 指定模型**前置**且去重保序",
          rt.model_candidates({"models": ["a", "b", "c"]}, "b") == ["b", "a", "c"])
    check("模型: 无 DB 模型 -> 直接用配置里的；模型列表为空 -> 回退单 model",
          rt.model_candidates({"models": ["a"]}, None) == ["a"]
          and rt.model_candidates({"model": "solo"}, None) == ["solo"])

    # ================= 风格 Profile 注入 =================
    client = TestClient(app)
    client.post("/api/v1/ai-configs", json={
        "service_type": "text", "provider": "openai", "base_url": "https://api.example.com",
        "api_key": "k", "model": ["gpt-4o-mini"], "is_active": True,
    })
    drama_id = client.post("/api/v1/dramas", json={"title": "SMOKE 运行时"}).json()["data"]["id"]
    with engine.begin() as conn:
        conn.execute(dramas.update().where(dramas.c.id == drama_id)
                     .values(genre="都市", style="写实"))

    with engine.begin() as conn:
        check("风格: 非注入类型（voice_assigner）原样返回",
              rt.append_style_profile(conn, "voice_assigner", 1, drama_id, "原指令") == "原指令")
        check("风格: 注入类型但**没有激活 Profile** -> 原样返回（连视觉图谱都不注入）",
              rt.append_style_profile(conn, "storyboard_breaker", 1, drama_id, "原指令")
              == "原指令")

    with engine.begin() as conn:
        conn.execute(style_profiles.insert().values(
            drama_id=drama_id, name="本剧风格", is_active=True,
            shot_patterns="多用过肩镜头", storytelling="三幕式",
            audio_captions="字幕居中", preferences="禁用漫画风", qc_rules="曝光不过曝",
            created_at=now(), updated_at=now(),
        ))
    with engine.begin() as conn:
        injected = rt.append_style_profile(conn, "storyboard_breaker", 1, drama_id, "原指令")
    # ⚠️ 分节标题**自带**前导 `\n\n`，外层再用 `\n` 连接 ⇒ 实际是 `原指令\n\n\n【项目风格…】`
    #（三个换行，与原 TS 的 `[instructions, '\n\n【…】'].join('\n')` 完全一致）
    check("风格: 五个分节按序注入，且用 `\\n` 连接（不是 \\n\\n）",
          injected.startswith("原指令\n\n\n【项目风格 Profile")
          and injected.index("shot_patterns") < injected.index("storytelling")
          < injected.index("audio_captions") < injected.index("preferences")
          < injected.index("qc_rules")
          and "多用过肩镜头" in injected and "三幕式" in injected
          and "曝光不过曝" in injected,
          injected[:120])
    # ⚠️ 判据必须用独特标识 `【视觉图谱引导】`：原先写的是宽泛的 `"视觉"` 子串 ✗，
    #    而 skill 正文里本来就有"视觉"二字 ⇒ 该 check 即使视觉图谱**完全没注入**也会 PASS（假阳性）。
    check("风格: storyboard_breaker 额外带**视觉图谱引导**",
          "【视觉图谱引导】" in injected, injected[-160:])
    with engine.begin() as conn:
        other = rt.append_style_profile(conn, "extractor", 1, drama_id, "原指令")
    check("风格: 非分镜类型**不带**视觉图谱（只在 storyboard_breaker 分支）",
          "视觉图谱" not in other and "多用过肩镜头" in other, other[-80:])

    # 全局 Profile 兜底（本剧没有专属激活 Profile 时）
    drama2 = client.post("/api/v1/dramas", json={"title": "第二部"}).json()["data"]["id"]
    with engine.begin() as conn:
        conn.execute(style_profiles.insert().values(
            drama_id=None, name="全局风格", is_active=True, shot_patterns="全局镜头偏好",
            created_at=now(), updated_at=now(),
        ))
        global_prof = rt.append_style_profile(conn, "script_rewriter", 1, drama2, "原指令")
    check("风格: 本剧无专属 -> **回退全局** Profile（drama_id IS NULL 且激活）",
          "全局镜头偏好" in global_prof, global_prof[:80])

    # ================= 多集节奏相位注入（2026-09-25 接线 ✓） =================
    # ⭐ 早先这里是「`rhythm-phase.ts` 未迁」的 warn 占位 ✗，而 `services/rhythm_phase.py` 早已迁好
    # ⇒ `storyboard_breaker` 一直少一段跨集节奏引导 ✓✗（「能力已有、没接出去」同族 ✓）。
    with engine.begin() as conn:
        ep_id = int(conn.execute(episodes.insert().values(
            drama_id=drama_id, episode_number=1, title="节奏集", content="x",
            created_at=now(), updated_at=now())).lastrowid)
    with engine.begin() as conn:
        with_rhythm = rt.append_style_profile(conn, "storyboard_breaker", ep_id, drama_id, "原指令")
        no_rhythm = rt.append_style_profile(conn, "extractor", ep_id, drama_id, "原指令")
        ghost = rt.append_style_profile(conn, "storyboard_breaker", 999999, drama_id, "原指令")
    check("节奏: storyboard_breaker **注入跨集节奏引导**（不再是「未迁」warn ✓）",
          "【多集节奏相位要求】" in with_rhythm and "climax(高潮" in with_rhythm, with_rhythm[-240:])
    check("节奏: 非分镜类型**不带**节奏引导（只在 storyboard_breaker 分支 ✓）",
          "多集节奏相位" not in no_rhythm, no_rhythm[-80:])
    check("节奏: episode 不存在 -> **不注入、不炸**（空串被跳过 ✓ 不产生空段 ✗）",
          "多集节奏相位" not in ghost and "原指令" in ghost, ghost[:60])

    # ================= ⭐ 回归：风格注入必须**真的进模型**（2026-09-25 修） =================
    # 历史缺陷：生产入口 `run_agent_with_retry` 只把**纯 base** 交给
    # `run_agent_with_instructions`，后者自行组装时**不含风格注入** ⇒
    # `append_style_profile` 的产出（风格 Profile / 跨集节奏 / 视觉图谱）**从未进过模型**，
    # 全仓唯一读取者是测试。上面那几条 check 全部只单测 `append_style_profile` **本身**，
    # 所以**测不出来** —— 这里改为断言**假驱动真实收到**的 instructions，才咬得住这条链。
    style_driver = FakeDriver([_ok(text="ok")])
    with engine.begin() as conn:
        asyncio.run(rt.run_agent_with_instructions(
            conn, "storyboard_breaker", ep_id, drama_id, "候选提示词", "hi",
            generate=style_driver, sleep=_fake_sleep))
    sent = style_driver.calls[0]["instructions"]
    check("⭐回归: 风格 Profile **经运行路径**进入模型（不再只在 build_agent_config 里算完就丢）",
          "【项目风格 Profile" in sent and "多用过肩镜头" in sent, sent[:120])
    check("⭐回归: 跨集节奏相位同样经运行路径进入模型",
          "【多集节奏相位要求】" in sent, sent[-200:])
    check("⭐回归: 视觉图谱引导同样经运行路径进入模型（storyboard_breaker 专属）",
          "【视觉图谱引导】" in sent, sent[-200:])
    check("回归: 评测语义未被破坏 —— 形参 instructions 仍是 base（开头就是候选词）",
          sent.startswith("候选提示词"), sent[:24])

    # ================= 工具装配 =================
    # ⚠️ `create_agent_tools` 自 2026-09-15 起需要 `conn`（orchestrator 的子 Agent 委托要跑真实 run）；
    #    这里装配阶段不触库，借一个只读连接即可。
    with engine.connect() as tools_conn:
        check("工具: 未知类型 -> None（该类型不可运行）",
              rt.create_agent_tools(tools_conn, "no_such_agent", 1, drama_id) is None)
        ids_by_type = {t: rt.create_agent_tools(tools_conn, t, 1, drama_id).ids()
                       for t in ("script_rewriter", "extractor", "voice_assigner",
                                 "grid_prompt_generator", "storyboard_breaker",
                                 "orchestrator")}
    check("工具: script_rewriter -> 剧本工具",
          "save_script" in ids_by_type["script_rewriter"], ids_by_type["script_rewriter"])
    check("工具: extractor -> 提取工具（含四个 read_existing_*）",
          {"save_dedup_characters", "save_dedup_scenes", "save_dedup_props",
           "read_existing_props"} <= set(ids_by_type["extractor"]),
          ids_by_type["extractor"])
    check("工具: voice_assigner -> 三个音色工具",
          {"list_voices", "get_characters", "assign_voice"}
          <= set(ids_by_type["voice_assigner"]), ids_by_type["voice_assigner"])
    check("工具: storyboard_breaker 与 grid **额外挂语料检索**",
          "search_reference_prompts" in ids_by_type["storyboard_breaker"]
          and "search_reference_prompts" in ids_by_type["grid_prompt_generator"]
          and "search_reference_prompts" not in ids_by_type["voice_assigner"],
          ids_by_type["storyboard_breaker"])
    check("工具: orchestrator -> **subagent 两件套**（2026-09-15 起已迁，不再是空占位）",
          sorted(ids_by_type["orchestrator"]) == ["list_available_agents", "run_subagent"],
          ids_by_type["orchestrator"])
    check("工具: 子 Agent 工具**只给 orchestrator**（领域 Agent 拿到会失去天然终止保证）",
          all("run_subagent" not in ids_by_type[t]
              for t in ids_by_type if t != "orchestrator"), ids_by_type)

    # ================= 配置装配 =================
    with engine.begin() as conn:
        check("装配: 未知类型 -> None",
              rt.build_agent_config(conn, "nope", 1, drama_id) is None)
        built = rt.build_agent_config(conn, "voice_assigner", 1, drama_id)
    check("装配: 无 DB 配置时 name 回落**登记表默认名**、base 回落**出厂提示词**",
          built.name == "角色音色分配" and built.db_config is None
          and built.base_instructions.startswith("你是配音导演"),
          (built.name, built.base_instructions[:30]))
    check("装配: 指令含协议契约（单点组装）",
          "## " in built.instructions and "yaml" in built.instructions,
          built.instructions[:40])
    check("装配: 文本配置与解析后的 baseUrl 都在",
          isinstance(built.text_config, dict) and isinstance(built.resolved_base_url, str),
          built.resolved_base_url)

    with engine.begin() as conn:
        conn.execute(agent_configs.insert().values(
            agent_type="voice_assigner", name="我的配音导演", system_prompt="自定义系统提示",
            model="db-model-x", max_tokens=1234, temperature=0.2, is_active=True,
            created_at=now(), updated_at=now(),
        ))
        built2 = rt.build_agent_config(conn, "voice_assigner", 1, drama_id)
    check("装配: DB 配置的 system_prompt / name / model 生效",
          built2.base_instructions == "自定义系统提示" and built2.name == "我的配音导演"
          and built2.db_config.model == "db-model-x", (built2.name, built2.base_instructions))
    check("装配: 模型候选把 DB 模型排在最前",
          rt.model_candidates(built2.text_config, built2.db_config.model)[0] == "db-model-x")

    # ================= run：成功路径 =================
    _SLEPT.clear()
    driver = FakeDriver([_ok(
        text="完成\n```yaml\nstatus: ok\nsummary: 已分配音色\n```",
        tool_calls=[{"toolName": "list_voices", "args": {"x": 1}}],
        tool_results=[{"toolName": "list_voices", "result": "[]"}],
        steps=[{"usage": {"inputTokens": 10, "outputTokens": 4, "totalTokens": 14}}],
    )])
    with engine.begin() as conn:
        result = asyncio.run(rt.run_agent_with_instructions(
            conn, "voice_assigner", 1, drama_id, "候选提示词", "给角色配音",
            generate=driver, sleep=_fake_sleep))
    check("run: 返回模型名 / 文本 / 工具调用 / 协议",
          result.model and result.text.startswith("完成")
          and result.tool_calls == [{"toolName": "list_voices", "args": {"x": 1}}]
          and result.protocol is not None and result.protocol["status"] == "ok"
          and result.protocol_errors == [], (result.model, result.protocol))
    check("run: token 用量已累加", result.usage.total_tokens == 14, result.usage)
    check("run: 驱动收到的是**传进来的候选提示词**（评测闭环语义）",
          driver.calls[0]["instructions"].startswith("候选提示词"), driver.calls[0]["instructions"][:20])
    check("run: maxSteps 默认 20；maxOutputTokens/temperature 来自 DB 配置",
          driver.calls[0]["max_steps"] == 20
          and driver.calls[0]["model_settings"] == {"maxOutputTokens": 1234, "temperature": 0.2},
          driver.calls[0]["model_settings"])
    check("run: 首个模型是 DB 指定的那个", driver.calls[0]["model"] == "db-model-x")

    # 协议缺失
    with engine.begin() as conn:
        no_protocol = asyncio.run(rt.run_agent_with_instructions(
            conn, "voice_assigner", 1, drama_id, "候选", "hi",
            generate=FakeDriver([_ok(text="没有协议块")]), sleep=_fake_sleep))
    check("run: 没有协议块 -> protocol=None 且 errors 非空（**不抛异常**）",
          no_protocol.protocol is None and no_protocol.protocol_errors,
          no_protocol.protocol_errors)

    # ================= run：失败路径 =================
    fatal_driver = FakeDriver([_status_error(401, "unauthorized")])
    with engine.begin() as conn:
        raised = ""
        try:
            asyncio.run(rt.run_agent_with_instructions(
                conn, "voice_assigner", 1, drama_id, "候选", "hi",
                generate=fatal_driver, sleep=_fake_sleep))
        except Exception as err:  # noqa: BLE001
            raised = str(err)
    check("run: **fatal 只调一次、不重试不换模型**（省时间省费用）",
          raised == "unauthorized" and len(fatal_driver.calls) == 1,
          (raised, len(fatal_driver.calls)))

    _SLEPT.clear()
    transient_driver = FakeDriver([
        _status_error(429, "rate limit"), _status_error(429, "rate limit"),
        _status_error(429, "rate limit"), _status_error(429, "rate limit"),
        _ok(text="第二个模型成功"),
    ])
    with engine.begin() as conn:
        fell_back = asyncio.run(rt.run_agent_with_instructions(
            conn, "voice_assigner", 1, drama_id, "候选", "hi",
            generate=transient_driver, sleep=_fake_sleep))
    check("run: 瞬态同模型退避 **3 次**（2s/4s/8s）后换下一个模型",
          _SLEPT == [2.0, 4.0, 8.0] and len(transient_driver.calls) == 5
          and fell_back.text == "第二个模型成功",
          (_SLEPT, len(transient_driver.calls)))
    check("run: 第 5 次调用**已换到下一个模型**（前 4 次都是 DB 模型）",
          transient_driver.calls[0]["model"] == "db-model-x"
          and transient_driver.calls[3]["model"] == "db-model-x"
          and transient_driver.calls[4]["model"] != "db-model-x",
          [c["model"] for c in transient_driver.calls])

    all_bad = FakeDriver([_status_error(503, "overloaded")])
    with engine.begin() as conn:
        err_text = ""
        try:
            asyncio.run(rt.run_agent_with_instructions(
                conn, "voice_assigner", 1, drama_id, "候选", "hi",
                generate=all_bad, sleep=_fake_sleep))
        except Exception as err:  # noqa: BLE001
            err_text = str(err)
    # 候选模型 = DB 模型 + 配置里的 gpt-4o-mini ⇒ 2 个模型 ×（初试 + 3 次重试）= 8 次
    check("run: 模型全挂 -> 抛**最后一个**错误（2 模型 ×（初试+3 重试）= 8 次调用）",
          err_text == "overloaded" and len(all_bad.calls) == 8, (err_text, len(all_bad.calls)))

    opts_driver = FakeDriver([_ok(text="ok")])
    with engine.begin() as conn:
        asyncio.run(rt.run_agent_with_instructions(
            conn, "voice_assigner", 1, drama_id, "候选", "hi", {"maxSteps": 5},
            generate=opts_driver, sleep=_fake_sleep))
    check("run: options.maxSteps 可覆盖默认 20", opts_driver.calls[0]["max_steps"] == 5)

    with engine.begin() as conn:
        invalid = ""
        try:
            asyncio.run(rt.run_agent_with_instructions(
                conn, "no_such_type", 1, drama_id, "候选", "hi",
                generate=FakeDriver([_ok()]), sleep=_fake_sleep))
        except ValueError as err:
            invalid = str(err)
    check("run: 非法类型 -> `Invalid agent type: ...`",
          invalid == "Invalid agent type: no_such_type", invalid)

    # run_agent_with_retry：用 DB 的 base instructions
    retry_driver = FakeDriver([_ok(text="ok")])
    with engine.begin() as conn:
        asyncio.run(rt.run_agent_with_retry(
            conn, "voice_assigner", 1, drama_id, "hi",
            generate=retry_driver, sleep=_fake_sleep))
    check("retry 入口: 传的是 DB base instructions（skill/契约由内层统一组装）",
          retry_driver.calls[0]["instructions"].startswith("自定义系统提示"),
          retry_driver.calls[0]["instructions"][:24])

    # ================= ⭐ Ollama 原生工具循环（2026-10-28 新增） =================
    # 缺陷背景：本机默认文本配置就是 `provider=ollama`，而运行时原先只认 OpenAI 兼容 ⇒
    # `POST /api/v1/agent/{type}/chat` 恒 400「provider 'ollama' … 尚未移植」。
    # 这里咬住**原生 `/api/chat` 的工具往返**：tools 真的下发、入参是**对象**也能吃、
    # 结果用 `tool_name` 回灌（没有 `tool_call_id`）、用量从 `prompt_eval_count/eval_count` 折算。
    seen_args: list[dict] = []

    async def _echo(args: dict) -> dict:
        seen_args.append(args)
        return {"voices": ["少女音"]}

    echo_tools = ToolRegistry({"echo_voice": Tool(
        id="echo_voice", description="回显音色",
        input_schema=object_schema({"gender": json_string("性别")}),
        execute=_echo,
    )})

    sent: list[dict] = []

    class _Resp:
        def __init__(self, payload: dict) -> None:
            self._payload = payload
            self.is_success = True
            self.status_code = 200
            self.text = "{}"

        def json(self) -> dict:
            return self._payload

    async def _fake_fetch(url: str, init: dict, kind: str, **kwargs) -> _Resp:
        sent.append({"url": url, "body": json.loads(init["body"]), "kind": kind})
        if len(sent) == 1:
            return _Resp({
                "message": {"role": "assistant", "content": "", "tool_calls": [
                    {"function": {"name": "echo_voice", "arguments": {"gender": "female"}}},
                ]},
                "prompt_eval_count": 30, "eval_count": 7,
            })
        return _Resp({"message": {"role": "assistant", "content": "已经分配好了"},
                      "prompt_eval_count": 40, "eval_count": 5})

    real_fetch = rt.fetch_with_retry
    rt.fetch_with_retry = _fake_fetch  # type: ignore[assignment]
    try:
        ollama_result = asyncio.run(rt.default_generate(
            config={"provider": "ollama", "baseUrl": "http://localhost:11434"},
            model="qwen3:14b", instructions="你是配音导演", tools=echo_tools,
            message="给角色分配音色", max_steps=4,
            model_settings={"maxOutputTokens": 512, "temperature": 0.2},
        ))
    finally:
        rt.fetch_with_retry = real_fetch  # type: ignore[assignment]

    check("ollama: 走**原生 /api/chat**，tools 以 OpenAI function 形状下发（不再 400）",
          sent and sent[0]["url"].endswith("/api/chat")
          and sent[0]["body"]["tools"][0]["function"]["name"] == "echo_voice",
          sent[0]["url"] if sent else "no-call")
    check("ollama: `arguments` 是**对象**也能解析（OpenAI 那条是 JSON 字符串）",
          seen_args == [{"gender": "female"}], seen_args)
    check("ollama: 工具结果用 `role=tool` + `tool_name` 回灌（**没有** tool_call_id）",
          any(m.get("role") == "tool" and m.get("tool_name") == "echo_voice"
              and "tool_call_id" not in m for m in sent[1]["body"]["messages"]),
          sent[1]["body"]["messages"][-1] if len(sent) > 1 else "no-second-call")
    check("ollama: 第二轮仍带着 assistant 的 tool_calls（模型看得到自己调过什么）",
          any(m.get("role") == "assistant" and m.get("tool_calls")
              for m in sent[1]["body"]["messages"]), len(sent))
    check("ollama: 用量折算自 prompt_eval_count/eval_count（30+7+40+5=82）",
          rt.extract_token_usage(ollama_result).total_tokens == 82, ollama_result["steps"])
    check("ollama: 最终文本 / tool_calls / tool_results 都齐",
          ollama_result["text"] == "已经分配好了"
          and ollama_result["tool_calls"] == [
              {"toolName": "echo_voice", "args": {"gender": "female"}}]
          and "少女音" in ollama_result["tool_results"][0]["result"],
          ollama_result["tool_results"])
    # ⚠️ 窗口检查（2026-10-28 实测的静默截断）：Ollama 默认 num_ctx=4096 且与 num_predict 共用，
    #    长 prompt 会被截到 ~2050 token 并**丢掉开头的 instructions**，接口毫无反应。
    #    默认 32768 的理由：storyboard_breaker 的固定开销自带 20372（> 16384）。
    check("ollama: 每轮请求都显式下发 num_ctx（默认 32768，不是 4096）",
          all(call["body"]["options"]["num_ctx"] == 32768 for call in sent),
          [call["body"]["options"] for call in sent])
    check("ollama: 反馈预算按 num_ctx − 输出预留算（32768−512=32256）",
          rt._ollama_context_budget(
              {"provider": "ollama"}, {"maxOutputTokens": 512}).limit_tokens == 32256,
          rt._ollama_context_budget({"provider": "ollama"}, {"maxOutputTokens": 512}))
    check("ollama: 预算跟随 settings.numCtx 覆盖 + 输出缺省回落 4096（32768−4096=28672）",
          rt._ollama_context_budget({"provider": "ollama", "settings": {"numCtx": 65536}},
                                    {}).limit_tokens == 61440
          and rt._ollama_context_budget({"provider": "ollama"}, {}).limit_tokens == 28672)

    # ================= 超预算：本地必须**当场失败**，不许静默截断 =================
    # ⚠️ 2026-10-28 实测：Ollama 超窗是 HTTP 200 + 从开头截断（system 被丢），接口无任何反应。
    #    机械层压不动（没有老 tool 输出可压）时，照发就是"安静地出一份错的稿"。
    over_messages = [{"role": "system", "content": "指令" * 300},
                     {"role": "user", "content": "剧本" * 300}]
    strict_raised = ""
    try:
        rt._apply_context_budget(over_messages, [], cb.Budget(limit_tokens=200), strict=True)
    except cb.ContextOverflowError as err:
        strict_raised = str(err)
    check("预算: strict=True（本地）超窗 -> 抛 ContextOverflowError（不照发）",
          "超出模型窗口" in strict_raised and "numCtx" in strict_raised, strict_raised)
    loose_steps: list = []
    loose = rt._apply_context_budget(over_messages, loose_steps, cb.Budget(limit_tokens=200))
    check("预算: strict=False（云端）超窗 -> 只留痕不中断（长度不变）",
          len(loose) == 2 and any("contextBudget" in s for s in loose_steps), loose_steps)
    plan = cb.CompressionPlan(overBudget=True, tokensBefore=900.0, tokensAfter=900.0,
                              needsSummarize=True)
    overflow_err = cb.ContextOverflowError(plan, 200)
    check("分类: ContextOverflowError -> fatal（重试/换模型都不会变好）",
          rt.classify_llm_error(overflow_err) == "fatal"
          and abs(overflow_err.overflow_tokens - 700.0) < 1e-6,
          (rt.classify_llm_error(overflow_err), overflow_err.overflow_tokens))

    # ================= Ollama HTTP 200 + {"error"} 不是成功 =================
    # ⚠️ 真实症状（显存不够装下 num_ctx 时）：body 里只有 error，没有 message ⇒
    #    旧代码取到空文本就 break，库一行没改，接口回 200。
    class _ErrResp:
        is_success = True
        status_code = 200
        text = '{"error":"model requires more system memory"}'

        def json(self) -> dict:
            return {"error": "model requires more system memory"}

    async def _err_fetch(url: str, init: dict, kind: str, **kwargs) -> _ErrResp:
        return _ErrResp()

    real_fetch2 = rt.fetch_with_retry
    rt.fetch_with_retry = _err_fetch  # type: ignore[assignment]
    ollama_body_error = ""
    try:
        asyncio.run(rt.default_generate(
            config={"provider": "ollama", "baseUrl": "http://localhost:11434"},
            model="qwen3:14b", instructions="你是配音导演", tools=echo_tools,
            message="给角色分配音色", max_steps=2,
            model_settings={"maxOutputTokens": 512},
        ))
    except ValueError as err:
        ollama_body_error = str(err)
    finally:
        rt.fetch_with_retry = real_fetch2  # type: ignore[assignment]
    check("ollama: 200 + {error} -> 抛错（不再当成空文本的空成功）",
          "model requires more system memory" in ollama_body_error, ollama_body_error)

    # ================= 汇总 =================
    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    # ⚠️ Windows 中文控制台是 **GBK**（py3.14 及以前不会自动 UTF-8 ✗）⇒ 不加这行，裸跑第一条 `print` 就 `UnicodeEncodeError` 崩 ✗（守口径见 `check_cli_encoding.py` ✓）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
