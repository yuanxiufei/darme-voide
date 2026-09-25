"""S7 自检：**后端 → 本地 H3 薄封装(8765) 的全栈活体链路**（2026-09-16 新增）。

链路（本自检覆盖「从后端配置到真服务落库」的**整条**）：

    ai_service_configs(video / provider=minimax / base_url=http://127.0.0.1:8765)
      → ``services/video_generation.generate_video``（配置解析 + prompt 富化 + 入队）
      → ``_process_video_generation``（GPU 租约 + 用量记账 + 适配器）
      → ``MiniMaxVideoAdapter`` 造请求 → **真 HTTP** → 8765 薄封装
      → 解析 ``taskId`` **落库** → ``_poll_video_task`` 真轮询 → 失败分支落 ``error_msg``

与既有两份自检的**分工**（三者不重复，各补一层）：

  * ``h3_chain_test.py``      —— 适配器 ↔ 薄封装 **in-process** 契约（TestClient，**不需要 8765 在跑** ✓）
  * ``local_services_live_test.py`` —— 适配器 ↔ 薄封装 **单点真 HTTP**（不经过后端服务层 ✓）
  * **本文件** —— **全栈**：配置 → 服务层 → 真服务 → 数据库落库；并用「**落库的 task_id 能被
    正在运行的 8765 认出来**」这一条**交叉证明**打的确实是那个**真进程**（排除了「其实是 mock/桩」✓）。

⚠️ 类设计：**live-aware** —— 8765 未在跑则**显式 SKIP**（不是「通过」✓）。CI 上不因此变红 ✓，
   但跳过条数会打印出来 ✓（沿用 ``local_services_live_test.py`` 的约定）。

✅ 阶段 2 已接线（2026-09-17 翻转）：薄封装 ``_run_h3`` 已真提交 ComfyUI 并取回产物，不再是
   占位桩 ⇒ ⑥ 断言已改为「终态必须是 ``succeeded``(+``video_url``) 或 ``failed`` + **真原因**，
   且**不得**再出现 ``not wired yet``」；⑦ 再反套套逻辑（error_msg 不能是网络层失败特征）。
   本文件跨了真网络与真数据库 ✓，与 ``h3_chain_test.py`` 的 in-process 契约互补。

运行::

    ./.venv/Scripts/python.exe tests/h3_backend_live_test.py
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="h3live_"))
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.core.db import engine  # noqa: E402
from app.core.models import api_usage, storyboards, video_generations  # noqa: E402
from app.core.response import now  # noqa: E402
from app.main import app  # noqa: E402
from app.services import ai_providers as ap  # noqa: E402
from app.services import video_generation as vg  # noqa: E402

#: 薄封装地址（与 ``local_services/h3/server.py`` 的默认端口一致）
H3_BASE = "http://127.0.0.1:8765"
#: 薄封装的 ``uuid.uuid4().hex`` 形状（落库的 taskId 必须长这样 ⇒ 它来自服务而不是我们编的）
HEX32 = re.compile(r"^[0-9a-f]{32}$")
#: 网络层失败特征 —— 出现它们说明**根本没连上**服务（那 ⑤ 的文案就失去意义了 ✗）
TRANSPORT_HINTS = ("connecterror", "connection", "refused", "timed out", "timeout", "allconnection")

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(name: str) -> None:
    _SKIPS.append(name)


def h3_alive() -> tuple[bool, str]:
    try:
        response = httpx.get(f"{H3_BASE}/healthz", timeout=3)
        return response.status_code == 200, response.text[:120]
    except Exception as exc:  # noqa: BLE001
        return False, type(exc).__name__


async def _drive(storyboard_id: int, drama_id: int) -> int:
    """真实驱动一次：入队 → 处理（含真 HTTP 提交与轮询）→ 返回 video_generations.id。"""
    with engine.begin() as conn:
        video_id = await vg.generate_video(conn, {
            "storyboardId": storyboard_id,
            "dramaId": drama_id,
            "prompt": "活体全栈：雨夜霓虹街头，主角回头",
            "duration": 5,
            "aspectRatio": "9:16",
            # ⚠️ force=True 是**刻意**的：本自检要测的是「后端 ↔ 8765」这条接缝，
            #    不是 take 预算/剧本指纹门禁（那两条各有自己的用例 ✓）⇒ 放行以免被门禁挡住 ✗。
            "force": True,
        })
    with engine.begin() as conn:
        config = ap.get_active_config(conn, "video")
    if not config:
        raise AssertionError("夹具失配：没有取到活跃 video 配置")
    await vg._process_video_generation(video_id, config)
    return video_id


def main() -> int:  # noqa: C901
    alive, how = h3_alive()
    if not alive:
        skip(f"h3-8765 未在跑（{how}）⇒ 全栈活体链路未验证"
             f"（in-process 契约仍由 h3_chain_test.py 覆盖 ✓）")
        print(f"SKIP  {_SKIPS[0]}")
        print()
        print("SUMMARY: 0/0 passed（skip 1：服务未启动）")
        return 0

    # 轮询节奏：真实配置是 300 次 × 10 秒（≈50 分钟）✗ —— 本自检只验**接线**，
    # 重试预算与节奏另有覆盖 ✓ ⇒ 缩短为「3 次 × 0.05 秒」，判据不变 ✓。
    vg._POLL_INTERVAL_SECONDS = 0.05  # type: ignore[attr-defined]
    vg._POLL_MAX_ATTEMPTS = 3  # type: ignore[attr-defined]

    client = TestClient(app)
    created = client.post("/api/v1/ai-configs", json={
        "service_type": "video", "provider": "minimax", "base_url": H3_BASE,
        "api_key": "local-key", "model": ["h3-fl2va"], "is_active": True,
    })
    check("① 配置: 活跃 video 配置写入成功（provider=minimax / base_url=8765）",
          created.status_code in (200, 201), created.text[:200])

    drama_id = client.post("/api/v1/dramas", json={"title": "SMOKE H3 全栈"}).json()["data"]["id"]
    episode_id = client.get(f"/api/v1/dramas/{drama_id}").json()["data"]["episodes"][0]["id"]
    storyboard_id = client.post("/api/v1/storyboards", json={
        "episode_id": episode_id, "title": "镜头",
    }).json()["data"]["id"]
    # 直接写库设置环境音（绕开路由白名单 ⇒ 夹具确定 ✓）；H3 路径会把它注入成 [background_audio]
    with engine.begin() as conn:
        conn.execute(storyboards.update().where(storyboards.c.id == storyboard_id)
                     .values(sound_effect="雨声、远处车流声", updated_at=now()))

    video_id = asyncio.run(_drive(storyboard_id, drama_id))

    with engine.begin() as conn:
        row = conn.execute(select(video_generations)
                           .where(video_generations.c.id == video_id)).first()
        usage = conn.execute(select(api_usage)
                             .where(api_usage.c.video_generation_id == video_id)).all()

    task_id = str(getattr(row, "task_id", "") or "")
    error_msg = str(getattr(row, "error_msg", "") or "")
    prompt = str(getattr(row, "prompt", "") or "")

    check("② 入队: 记录 provider=minimax 且状态已推进（processing/failed）",
          row is not None and row.provider == "minimax"
          and row.status in ("processing", "failed"), (getattr(row, "status", None),))
    check("③ prompt 富化: 分镜的 sound_effect 被注入成 H3 的 [background_audio] 标记",
          "[background_audio]" in prompt and "雨声" in prompt, prompt[:120])
    check("④ 真提交: 落库的 task_id 非空且是薄封装的 32 位 hex（uuid4().hex 形状）",
          bool(HEX32.match(task_id)), task_id)

    # ⑤ **交叉证明**：拿**落库的** task_id 去问**正在运行的**那个 8765 ⇒ 它必须认得这个任务。
    #    这一条是整份自检的核心：只有「后端真的把请求打到了这个进程」才可能成立 ✓
    #    （mock/桩、连不上、写死假 id 都会在这一条上暴露 ✗）。
    try:
        probe = httpx.get(f"{H3_BASE}/v1/video_generation/task/{task_id}", timeout=5)
        probe_body = probe.json() if probe.status_code == 200 else {}
        check("⭐ 交叉证明: 落库的 task_id 被运行中的 8765 认出（200 + status/video_url/error_msg）",
              probe.status_code == 200
              and {"status", "video_url", "error_msg"} <= set(probe_body), probe.text[:160])
    except Exception as exc:  # noqa: BLE001
        check("⭐ 交叉证明: 落库的 task_id 被运行中的 8765 认出", False,
              f"{type(exc).__name__}: {exc}"[:160])

    # ⑥ ✅ **阶段 2 已接线**（2026-09-17 翻转）：终态只可能是
    #    ① `succeeded` + video_url 非空（ComfyUI + 模型 + 节点都就绪 ⇒ 真出片 ✓）；或
    #    ② `failed` + **真原因**（ComfyUI 不可达 / 缺节点包 ✓）—— 但**绝不能**再是占位桩那句 ✗。
    check("⑥ 阶段2 已接线: 终态是 succeeded(+video_url) 或 failed(+真原因)，且不再有 'not wired yet' ✗",
          row is not None and "not wired yet" not in error_msg
          and (row.status == "succeeded" or (row.status == "failed" and bool(error_msg))),
          (getattr(row, "status", None), error_msg[:160]))

    # ⑦ 反套套逻辑：error_msg **不能**是网络层失败 —— 否则 ⑥ 里那句文案根本不会是服务给的 ✗
    check("⑦ 反套套逻辑: error_msg 不含网络层失败特征（否则说明其实没连上服务）",
          bool(error_msg) and not any(hint in error_msg.lower() for hint in TRANSPORT_HINTS),
          error_msg[:160])

    # ⑧ 本地语义: 用量记账把 127.0.0.1 的配置记为**本地**（is_local=1 ⇒ 不按云端计费口径）
    check("⑧ 记账: api_usage 有该任务的记录且 is_local=1（后端认下这是本地服务）",
          bool(usage) and all(bool(getattr(item, "is_local", False)) for item in usage),
          [(getattr(item, "service_type", None), getattr(item, "is_local", None)) for item in usage])

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    for name in _SKIPS:
        print("SKIP  " + name)
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
