"""S6 自检：Agent 出厂默认提示词（``services/agent_prompts.py`` ← ``agents/index.ts`` 的 ``DEFAULT_PROMPTS``）。

这份资产**曾经刻意不搬**，2026-09-12 决策变更为「搬 + 逐字守卫」。本测试关心三件事：

1. **正文完整且活**：六个类型都有非空提示词、``${...}`` 插值全部已替换（没有残留占位符、
   也没有被截断）；
2. **不重复注入**：提示词正文里**不含**输出协议契约（那由 ``assemble_instructions`` 追加）；
3. **真的接上了运行时**：无 DB 配置时 ``build_agent_config`` 的 base instructions
   从「空串」变成出厂提示词 —— 这是本次搬迁要修的那个洞。

（逐字 vs TS 的比对在 ``route_parity_test.py`` 的 ``_agent_prompts_drift``。）

运行::

    ./.venv/Scripts/python.exe tests/agent_prompts_test.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="aprompt_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models import agent_configs, dramas  # noqa: E402
from app.response import now  # noqa: E402
from app.services import agent_prompts as ap  # noqa: E402
from app.services.agent_registry import (  # noqa: E402
    AGENT_DEFAULT_NAMES,
    VALID_AGENT_TYPES,
    get_default_name,
)
from app.services.agents import runtime as rt  # noqa: E402
from app.services.agents import skills as sk  # noqa: E402
from app.services.agents.protocol import build_protocol_contract  # noqa: E402
from app.services.prompt_blocks import (  # noqa: E402
    IMAGE_PROMPT_TEMPLATE_CHARACTER,
    IMAGE_PROMPT_TEMPLATE_SCENE,
    IMAGE_PROMPT_TEMPLATE_SHOT,
    SCREENPLAY_FORMAT_RULES,
)

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


class FakeDriver:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def __call__(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        return {"text": "ok", "tool_calls": [], "tool_results": [], "steps": []}


async def _no_sleep(_seconds: float) -> None:
    return None


def main() -> int:  # noqa: C901
    # ================= 正文完整性 =================
    check("清单: 六个类型都有非空提示词，未知类型 -> 空串",
          all(ap.get_default_instructions(t).strip() for t in VALID_AGENT_TYPES)
          and ap.get_default_instructions("no_such") == "",
          [t for t in VALID_AGENT_TYPES if not ap.get_default_instructions(t).strip()])
    check("清单: 键集合与顺序都等于 VALID_AGENT_TYPES（与登记表同源同序）",
          tuple(ap.DEFAULT_INSTRUCTIONS) == tuple(VALID_AGENT_TYPES),
          (list(ap.DEFAULT_INSTRUCTIONS), list(VALID_AGENT_TYPES)))
    check("清单: 显示名仍只在登记表里（本模块不存 name）",
          all(get_default_name(t) == AGENT_DEFAULT_NAMES[t] for t in VALID_AGENT_TYPES))
    check("插值: **没有残留 `${...}` 占位符**（漏替换就是被截断/拼错的信号）",
          not any("${" in ap.get_default_instructions(t) for t in VALID_AGENT_TYPES),
          [t for t in VALID_AGENT_TYPES if "${" in ap.get_default_instructions(t)])
    check("插值: 四段 prompt-blocks 常量都按原样嵌进去了",
          SCREENPLAY_FORMAT_RULES in ap.get_default_instructions("script_rewriter")
          and IMAGE_PROMPT_TEMPLATE_CHARACTER in ap.get_default_instructions("extractor")
          and IMAGE_PROMPT_TEMPLATE_SCENE in ap.get_default_instructions("extractor")
          and IMAGE_PROMPT_TEMPLATE_SHOT in ap.get_default_instructions("grid_prompt_generator"))
    extractor = ap.get_default_instructions("extractor")
    check("插值: extractor 里角色骨架在场景骨架**之前**（顺序反了提示词语义就变）",
          extractor.index(IMAGE_PROMPT_TEMPLATE_CHARACTER)
          < extractor.index(IMAGE_PROMPT_TEMPLATE_SCENE))
    check("组装: 提示词正文**不含**输出协议契约（那份由 assemble_instructions 追加）",
          not any("输出协议" in ap.get_default_instructions(t) for t in VALID_AGENT_TYPES))

    # ================= 关键内容抽查（防「差不多但缺一段」）=================
    storyboard = ap.get_default_instructions("storyboard_breaker")
    check("内容: 分镜提示词含七种 scene_type 全表 + ONE_SHOT_ONE_SPEAKER + 连续性状态机",
          all(token in storyboard for token in
              ("| single |", "| dialogue_2p |", "| meeting |", "| argument |",
               "| long_dialogue |", "| action |", "| silent |",
               "ONE_SHOT_ONE_SPEAKER", "连续性状态机", "save_continuity_states")))
    check("内容: 音色提示词含两条硬规则（一角色一音色 / 跨集锁定）",
          all(token in ap.get_default_instructions("voice_assigner")
              for token in ("一角色一音色", "跨集锁定", "speaker_id 全局唯一")))
    check("内容: 总调度提示词列出五个子 Agent + run_subagent 委托约定",
          all(token in ap.get_default_instructions("orchestrator")
              for token in ("list_available_agents", "run_subagent", "script_rewriter",
                            "grid_prompt_generator")))
    check("内容: 提取提示词覆盖角色/场景/物品三条去重链与独立字段要求",
          all(token in extractor for token in
              ("save_dedup_characters", "save_dedup_scenes", "save_dedup_props",
               "read_existing_props", "clothing", "core_features")))
    check("内容: 宫格提示词含 exactly N visible panels 与单镜骨架",
          "exactly N visible panels" in ap.get_default_instructions("grid_prompt_generator")
          and IMAGE_PROMPT_TEMPLATE_SHOT in ap.get_default_instructions("grid_prompt_generator"))
    check("内容: 剧本改写提示词含两入口（读原文 / 直接回传规范）+ 必须自己改写",
          all(token in ap.get_default_instructions("script_rewriter")
              for token in ("read_episode_script", "rewrite_to_screenplay", "save_script",
                            "你必须自己完成改写工作")))

    snap = ap.default_prompts_snapshot()
    check("快照: 顺序即声明序、每项只有 type/instructions",
          [item["type"] for item in snap] == list(VALID_AGENT_TYPES)
          and all(set(item) == {"type", "instructions"} for item in snap))

    # ================= 接上运行时（本次搬迁要修的洞）=================
    client = TestClient(app)
    client.post("/api/v1/ai-configs", json={
        "service_type": "text", "provider": "openai", "base_url": "https://api.example.com",
        "api_key": "k", "model": ["gpt-4o-mini"], "is_active": True,
    })
    drama_id = client.post("/api/v1/dramas", json={"title": "SMOKE 出厂提示词"}).json()["data"]["id"]
    with engine.begin() as conn:
        conn.execute(dramas.update().where(dramas.c.id == drama_id).values(genre="都市", style="写实"))

    with engine.begin() as conn:
        built = rt.build_agent_config(conn, "voice_assigner", 1, drama_id)
    check("运行时: **无 DB 配置时 base instructions 不再是空串**（回落到出厂提示词）",
          built.base_instructions == ap.get_default_instructions("voice_assigner")
          and len(built.base_instructions) > 200,
          len(built.base_instructions))
    # ⚠️ 组装顺序是 **base + skill + 协议契约**（skill 段在中间；voice_assigner 有默认绑定的 skill）
    expected_parts = [
        ap.get_default_instructions("voice_assigner"),
        sk.load_agent_skills("voice_assigner", None),
        build_protocol_contract(),
    ]
    expected = "\n\n".join(part for part in expected_parts if part not in (None, ""))
    check("运行时: 完整 instructions **精确等于** 出厂提示词 + skill 段 + 协议契约（不重复注入）",
          built.instructions == expected
          and built.instructions.count("## Available Skills") <= 1,
          built.instructions[-60:])

    with engine.begin() as conn:
        conn.execute(agent_configs.insert().values(
            agent_type="voice_assigner", name="我的配音导演", system_prompt="自定义系统提示",
            model="m", is_active=True, created_at=now(), updated_at=now(),
        ))
        overridden = rt.build_agent_config(conn, "voice_assigner", 1, drama_id)
    check("运行时: DB 有 system_prompt 时**它优先**（出厂提示词只是回落）",
          overridden.base_instructions == "自定义系统提示"
          and ap.get_default_instructions("voice_assigner") not in overridden.instructions,
          overridden.base_instructions)

    # 六个类型都能装配且拿到非空指令
    with engine.begin() as conn:
        built_all = {t: rt.build_agent_config(conn, t, 1, drama_id) for t in VALID_AGENT_TYPES}
    check("运行时: 六个类型都能装配，且非分镜类的 base 都等于各自出厂提示词",
          all(built_all[t] is not None for t in VALID_AGENT_TYPES)
          and built_all["orchestrator"].base_instructions
          == ap.get_default_instructions("orchestrator"),
          [t for t in VALID_AGENT_TYPES if built_all[t] is None])

    driver = FakeDriver()
    with engine.begin() as conn:
        asyncio.run(rt.run_agent_with_retry(
            conn, "orchestrator", 1, drama_id, "hi", generate=driver, sleep=_no_sleep))
    check("运行时: run_agent_with_retry 把**出厂提示词**交给驱动（orchestrator 无 DB 配置）",
          driver.calls[0]["instructions"].startswith(
              ap.get_default_instructions("orchestrator")[:40]),
          driver.calls[0]["instructions"][:40])

    # 幂等：多次取值一致（没有可变状态被就地改动）
    check("幂等: 反复取值内容不变",
          ap.get_default_instructions("extractor") == extractor
          and ap.default_prompts_snapshot()[1]["instructions"] == extractor)

    # ================= 汇总 =================
    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
