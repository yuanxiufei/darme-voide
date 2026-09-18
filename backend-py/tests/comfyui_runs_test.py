"""S7 自检：**ComfyUI 工作流运行 = 项目自己的一等公民能力**（2026-09-17）。

盯的不是「能不能转发」✗（那是上游 8765 的事 ✓），而是**本项目的实现**：
持久化（DB 状态机 ✓）、崩溃恢复（不重提交 ✓）、串行租约 ✓、**产物落进我们数据根** ✓、
用量记账 ✓、错误归因 ✓。

做法：起**真 HTTP stub 上游**（8765 的形状 ✓）+ **真 DB**（临时数据根 ✓）⇒ 跑完整链路再核对落点 ✓。

运行::

    ./.venv/Scripts/python.exe tests/comfyui_runs_test.py
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="comfyruns_"))
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

import httpx  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import JSONResponse, Response  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.core.config import get_storage_root  # noqa: E402
from app.core.db import engine  # noqa: E402
from app.core.models import api_usage, comfyui_runs  # noqa: E402
from app.services import comfyui as svc  # noqa: E402
from app.core.response import now  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
#: stub 上游计数（**反套套逻辑** ✓）
SEEN: dict[str, Any] = {"workflow_runs": 0, "h3_runs": 0, "polls": 0, "files": 0}
#: 让 stub 按需返回失败 / 带 warnings ✓
MODE: dict[str, Any] = {"fail": False, "warnings": []}
ARTIFACT = b"STUB-MP4-ARTIFACT-BYTES"


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def build_stub() -> FastAPI:
    stub = FastAPI()

    @stub.post("/v1/workflows/run")
    async def run_workflow(payload: dict[str, Any]) -> JSONResponse:  # noqa: D103
        SEEN["workflow_runs"] += 1
        SEEN["workflow_body"] = json.dumps(payload, ensure_ascii=False)
        return JSONResponse({"task_id": f"wf-{SEEN['workflow_runs']}", "status": "queued"})

    @stub.post("/v1/video_generation")
    async def run_h3(payload: dict[str, Any]) -> JSONResponse:  # noqa: D103
        SEEN["h3_runs"] += 1
        SEEN["h3_body"] = json.dumps(payload, ensure_ascii=False)
        return JSONResponse({"task_id": f"h3-{SEEN['h3_runs']}", "status": "queued"})

    @stub.get("/v1/workflows/task/{task_id}")
    @stub.get("/v1/video_generation/task/{task_id}")
    def poll(task_id: str) -> dict[str, Any]:  # noqa: D103
        SEEN["polls"] += 1
        if MODE["fail"]:
            return {"status": "failed", "error_msg": "缺少节点包：ComfyUI-MiniMaxH3-Easy ✗",
                    "video_url": None, "warnings": []}
        if SEEN["polls"] < 2:  # 先给一次 processing ⇒ 证明轮询真的在跑 ✓
            return {"status": "processing", "video_url": None, "error_msg": None, "warnings": []}
        return {
            "status": "succeeded",
            "video_url": f"http://127.0.0.1:{SEEN['port']}/files/{task_id}.mp4",
            "error_msg": None,
            "warnings": MODE["warnings"],
            "expected_seconds": 5.167,
            "outputs": [{"kind": "videos", "source": "stub.mp4",
                         "url": f"http://127.0.0.1:{SEEN['port']}/files/{task_id}.mp4"}],
        }

    @stub.get("/files/{name}")
    def files(name: str) -> Response:  # noqa: D103
        SEEN["files"] += 1
        return Response(content=ARTIFACT, media_type="video/mp4")

    return stub


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def start_server(app: FastAPI) -> tuple[str, uvicorn.Server, threading.Thread]:
    port = free_port()
    SEEN["port"] = port
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 25
    while time.time() < deadline:
        try:
            if httpx.get(f"{base}/files/probe", timeout=1).status_code == 200:
                return base, server, thread
        except Exception:  # noqa: BLE001
            time.sleep(0.1)
    raise RuntimeError("stub 上游没能起来")


async def wait_terminal(run_id: int, timeout: float = 30) -> dict[str, Any]:
    deadline = time.time() + timeout
    last: dict[str, Any] = {}
    while time.time() < deadline:
        with engine.begin() as conn:
            row = svc.get_run(conn, run_id)
            if row is None:
                return {}
            if row.status in ("succeeded", "failed"):
                return svc.run_to_dict(row)
        await asyncio.sleep(0.1)
    return last


async def case_happy_path() -> None:
    """① 正常一次：DB 状态机 → 产物进我们数据根 → 记账 ✓。"""
    MODE["fail"] = False
    MODE["warnings"] = ["模板是 FL2VA 形态：参考图/参考音频不生效 ✗"]
    with engine.begin() as conn:
        run_id = await svc.submit_run(conn, {
            "kind": "workflow", "workflowName": "MiniMax_H3_Fast_T2V.json",
            "prompt": "自检：雨夜霓虹街头", "duration": 5,
            "workflow": {"1": {"class_type": "CapLoader", "inputs": {"text": "x"}}},
            "inject": [{"class_type": "CapLoader", "field": "text", "value": "注入值"}],
            "settings": {"checkpoint_map": {"fl2va": "m1"}},
        })
    with engine.begin() as conn:
        row = svc.get_run(conn, run_id)
    check("① 入队即返回 id，且**先落 DB**（status=queued，不是靠上游历史 ✓）",
          run_id > 0 and row is not None and row.status in ("queued", "processing"),
          (run_id, getattr(row, "status", None)))

    result = await wait_terminal(run_id)
    check("② 状态机跑到 succeeded（轮询 upstream 真在跑 ✓）",
          result.get("status") == "succeeded", result.get("errorMsg"))
    check("③ 上游任务 id 被记下（崩溃恢复靠它 ✓）",
          str(result.get("remoteTaskId") or "").startswith("wf-"), result.get("remoteTaskId"))

    # 产物必须**下载进我们自己的数据根** ✓（不是只留一个上游 URL ✗）
    local = str(result.get("localPath") or "")
    stored = Path(get_storage_root()) / local.replace("static/", "", 1) if local else None
    check("④ 产物落到**我们数据根** static/comfyui/ 下，且字节与上游一致 ✓",
          local.startswith("static/comfyui/") and stored is not None and stored.exists()
          and stored.read_bytes() == ARTIFACT,
          (local, str(stored), stored.read_bytes()[:24] if stored and stored.exists() else None))
    check("⑤ 上游 URL 与产物清单都保留（可追溯/可重取 ✓）",
          str(result.get("primaryUrl") or "").startswith("http") and result.get("outputs"),
          (result.get("primaryUrl"), result.get("outputs")))
    check("⑥ warnings 原样带出（不吞掉上游的诚实声明 ✓）",
          any("FL2VA" in w for w in result.get("warnings") or []), result.get("warnings"))
    check("⑦ 耗时被记录（elapsedMs > 0 ✓）", int(result.get("elapsedMs") or 0) > 0,
          result.get("elapsedMs"))

    with engine.begin() as conn:
        every = conn.execute(select(api_usage)).all()
        usage = [u for u in every if f'"comfyuiRunId":{run_id}' in str(u.meta or "")]
        final_row = svc.get_run(conn, run_id)
    check("⑧ 用量记账：api_usage 有该运行、provider=comfyui 且 is_local ✓（与图片/视频同口径 ✓）",
          bool(usage) and all(u.provider == "comfyui" and bool(u.is_local) for u in usage),
          {"runStatus": getattr(final_row, "status", None),
           "error": getattr(final_row, "error_msg", None),
           "usageRows": [(u.id, u.service_type, u.provider, u.is_local, u.units, u.meta)
                         for u in every][:6]})

    check("⑨ 反套套逻辑：上游真的被提交与轮询过 ✓",
          SEEN["workflow_runs"] >= 1 and SEEN["polls"] >= 2 and SEEN["files"] >= 1, dict(SEEN))


async def case_failure_and_h3() -> None:
    """② 失败要落 DB 且带**真原因** ✓；③ H3 走的是另一条上游协议 ✓。"""
    MODE["fail"] = True
    with engine.begin() as conn:
        run_id = await svc.submit_run(conn, {"kind": "workflow", "prompt": "失败用例",
                                            "workflow": {"1": {"class_type": "X", "inputs": {}}}})
    result = await wait_terminal(run_id)
    check("⑩ 上游失败 ⇒ DB 里 failed 且 error_msg 保留**上游真原因**（缺节点包 ✓）",
          result.get("status") == "failed" and "缺少节点包" in str(result.get("errorMsg")),
          result.get("errorMsg"))

    MODE["fail"] = False
    h3_before = SEEN["h3_runs"]
    with engine.begin() as conn:
        run_id2 = await svc.submit_run(conn, {"kind": "h3", "prompt": "H3 用例", "duration": 5,
                                              "aspectRatio": "9:16", "workflowName": "h3"})
    result2 = await wait_terminal(run_id2)
    check("⑪ kind=h3 走 **/v1/video_generation** 协议（翻译在服务层做 ✓）",
          SEEN["h3_runs"] == h3_before + 1 and "aspect_ratio" in str(SEEN.get("h3_body")),
          SEEN.get("h3_body"))
    check("⑫ H3 的运行同样落到 succeeded 且产物本地化 ✓",
          result2.get("status") == "succeeded"
          and str(result2.get("localPath") or "").startswith("static/comfyui/"),
          (result2.get("status"), result2.get("localPath")))


async def case_recovery() -> None:
    """④ 崩溃恢复：在途运行**绝不重提交**，只收尾并说明 ✓。"""
    ts = now()
    with engine.begin() as conn:
        # 一条「已提交拿到 remote id」的中断运行 + 一条「提交都没成功」的中断运行 ✓
        submitted = conn.execute(comfyui_runs.insert().values(
            kind="workflow", status="processing", step="poll", remote_task_id="wf-999",
            params="{}", is_local=True, created_at=ts, updated_at=ts)).inserted_primary_key[0]
        pending = conn.execute(comfyui_runs.insert().values(
            kind="workflow", status="queued", step="queued", params="{}",
            is_local=True, created_at=ts, updated_at=ts)).inserted_primary_key[0]
    runs_before = SEEN["workflow_runs"]
    svc.recover_runs_on_startup()
    with engine.begin() as conn:
        row_a = svc.get_run(conn, int(submitted))
        row_b = svc.get_run(conn, int(pending))
    check("⑬ 恢复：两条在途运行都收尾为 failed（不留「永远 processing」✗）",
          row_a.status == "failed" and row_b.status == "failed",
          (row_a.status, row_b.status))
    check("⑭ 恢复说明区分「已提交（可查上游）」与「未提交（未占 GPU）」✓",
          "已提交上游" in str(row_a.error_msg) and "提交前即中断" in str(row_b.error_msg),
          (row_a.error_msg, row_b.error_msg))
    check("⑮ 恢复**没有重提交**（绝不重复占用 GPU/算力 ✓）",
          SEEN["workflow_runs"] == runs_before, (runs_before, SEEN["workflow_runs"]))


async def case_listing() -> None:
    """⑤ 查询是**我们自己的**（不是转发 ComfyUI 历史 ✗）✓。"""
    with engine.begin() as conn:
        rows = [svc.run_to_dict(r) for r in svc.list_runs(conn, limit=50)]
    check("⑯ 能按自己的 DB 列出运行（含状态/产物/耗时 ✓ 不依赖上游 history ✓）",
          len(rows) >= 4 and all({"status", "outputs", "elapsedMs"} <= set(r) for r in rows),
          len(rows))
    with engine.begin() as conn:
        succeeded = svc.list_runs(conn, status="succeeded")
    check("⑰ 能按状态过滤 ✓", all(r.status == "succeeded" for r in succeeded), len(succeeded))


async def main() -> int:
    base, server, thread = start_server(build_stub())
    os.environ["COMFYUI_SERVICE_URL"] = base
    os.environ["COMFYUI_POLL_SECONDS"] = "0.05"
    os.environ["COMFYUI_RUN_TIMEOUT"] = "30"
    try:
        await case_happy_path()
        await case_failure_and_h3()
        await case_recovery()
        await case_listing()
    finally:
        server.should_exit = True
        thread.join(timeout=5)

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(asyncio.run(main()))
