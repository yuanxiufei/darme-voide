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
SEEN: dict[str, Any] = {"workflow_runs": 0, "h3_runs": 0, "polls": 0, "files": 0, "free": 0}
#: 让 stub 按需返回失败 / 带 warnings / 先来 N 次显存不足 ✓
MODE: dict[str, Any] = {"fail": False, "warnings": [], "oom": 0,
                        "failMsg": "缺少节点包：ComfyUI-MiniMaxH3-Easy ✗"}
#: 这一次提交「注定显存不足」的任务 id（模拟上游 alloc 失败 ✓）
OOM_TASKS: set[str] = set()
ARTIFACT = b"STUB-MP4-ARTIFACT-BYTES"


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def build_stub() -> FastAPI:
    stub = FastAPI()

    @stub.post("/v1/workflows/run")
    async def run_workflow(payload: dict[str, Any]) -> JSONResponse:  # noqa: D103
        SEEN["workflow_runs"] += 1
        SEEN["workflow_body"] = json.dumps(payload, ensure_ascii=False)
        task = f"wf-{SEEN['workflow_runs']}"
        if MODE["oom"] > 0:  # 前 N 次提交注定 alloc 失败（模拟「卸载后重试就过」✓）
            MODE["oom"] -= 1
            OOM_TASKS.add(task)
        return JSONResponse({"task_id": task, "status": "queued"})

    @stub.post("/v1/video_generation")
    async def run_h3(payload: dict[str, Any]) -> JSONResponse:  # noqa: D103
        SEEN["h3_runs"] += 1
        SEEN["h3_body"] = json.dumps(payload, ensure_ascii=False)
        task = f"h3-{SEEN['h3_runs']}"
        if MODE["oom"] > 0:
            MODE["oom"] -= 1
            OOM_TASKS.add(task)
        return JSONResponse({"task_id": task, "status": "queued"})

    @stub.post("/v1/free")
    def free_upstream() -> dict[str, Any]:  # noqa: D103
        SEEN["free"] += 1
        return {"ok": True}

    @stub.get("/v1/workflows/task/{task_id}")
    @stub.get("/v1/video_generation/task/{task_id}")
    def poll(task_id: str) -> dict[str, Any]:  # noqa: D103
        SEEN["polls"] += 1
        if task_id in OOM_TASKS:
            return {"status": "failed", "video_url": None, "warnings": [],
                    "error_msg": "CUDA error: allocation on device 0 out of memory. "
                                 "Tried to allocate 2.00 GiB"}
        if MODE["fail"]:
            return {"status": "failed", "error_msg": MODE["failMsg"],
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

    # ⚠️⚠️ 参考素材必须**原样过服务层** ✓✗（2026-10-27 补）：修前这一层**一个都不传** ✗
    #    ⇒ 产品侧挂着参考图/音频/视频跑 H3 时，素材在**服务层**就被静默丢掉 ✓✗
    #    （收端按纯文生跑完 ✓、回执里 ``reference_*`` 全是 0 ✓，从日志完全看不出是被谁丢的 ✓）。
    h3_before2 = SEEN["h3_runs"]
    with engine.begin() as conn:
        run_id3 = await svc.submit_run(conn, {
            "kind": "h3", "prompt": "H3 参考素材用例", "duration": 4, "workflowName": "h3",
            "referenceImageUrls": ["https://x/a.png"], "referenceAudioUrls": ["https://x/b.mp3"],
            "referenceVideoUrls": ["https://x/c.mp4"], "referenceImageSize": "max",
        })
    await wait_terminal(run_id3)
    body3 = json.loads(SEEN.get("h3_body") or "{}")
    check("⑫′ H3 的参考图/音频/视频/缩放档位**原样过服务层** ✓（用后端协议键、camelCase ✓）",
          SEEN["h3_runs"] == h3_before2 + 1
          and body3.get("referenceImageUrls") == ["https://x/a.png"]
          and body3.get("referenceAudioUrls") == ["https://x/b.mp3"]
          and body3.get("referenceVideoUrls") == ["https://x/c.mp4"]
          and body3.get("referenceImageSize") == "max",
          body3)
    # ⚠️ 空列表**不写进 body** ✓✗：``referenceImageUrls: []`` 会被收端读成「声明了但没内容」✓，
    #    与「压根没提这件事」不是一回事 ✓（收端要据此决定走 T2V 还是 R2V ✓）。
    empty_body = svc._build_upstream_body(
        "h3", {"prompt": "p", "referenceImageUrls": [], "referenceAudioUrls": "",
               "referenceVideoUrls": None, "referenceImageSize": ""})
    check("⑫″ 参考素材为空 ⇒ 这几个键**根本不出现** ✓（不是塞空数组/空串 ✗）",
          "referenceImageUrls" not in empty_body and "referenceAudioUrls" not in empty_body
          and "referenceVideoUrls" not in empty_body and "referenceImageSize" not in empty_body,
          empty_body)


async def case_oom_retry_and_hints() -> None:
    """⑬ 显存不足 ⇒ 卸载 + **重试一次**（不是硬失败、也不是无限重刷 ✓）；失败原因带可执行建议 ✓。

    2026-11-03 自 ``comfyui-gateway/src/runner.js`` 迁入的两条行为：
    ① ``isOOM`` ⇒ ``/free`` ⇒ 重提交一次（第二次仍不足才判失败 ✓）；
    ② ``failureHint`` ⇒ 上游英文堆栈 + 一句「据此改什么」（模型/节点包/SageAttention/VDN/显存 ✓）。
    """
    MODE["fail"] = False
    MODE["warnings"] = []
    os.environ["COMFYUI_OOM_RETRY_WAIT"] = "0"

    # ① 第一次 alloc 失败、卸载后第二次成功 ✓
    MODE["oom"] = 1
    before_runs, before_free = SEEN["workflow_runs"], SEEN["free"]
    with engine.begin() as conn:
        run_id = await svc.submit_run(conn, {"kind": "workflow", "prompt": "显存重试用例",
                                            "workflow": {"1": {"class_type": "X", "inputs": {}}}})
    result = await wait_terminal(run_id)
    with engine.begin() as conn:
        usage = [u for u in conn.execute(select(api_usage)).all()
                 if f'"comfyuiRunId":{run_id}' in str(u.meta or "")]
    check("⑬ 显存不足 ⇒ 自动 /v1/free 后**重提交一次**并成功 ✓（不是直接判失败 ✗）",
          result.get("status") == "succeeded" and SEEN["free"] == before_free + 1
          and SEEN["workflow_runs"] == before_runs + 2,
          (result.get("status"), result.get("errorMsg"), SEEN["free"], SEEN["workflow_runs"]))
    check("⑭ 重试成功也要**记账**：warnings 写明「已卸载重试」✓、用量 retryCount 如实 +1 ✓",
          any("重试一次" in w for w in result.get("warnings") or [])
          and any(int(u.retry_count or 0) == 1 for u in usage),
          (result.get("warnings"), [(u.units, u.retry_count) for u in usage]))

    # ② 两次都 alloc 失败 ⇒ failed，但原因里必须写明**已经重试过**（否则使用者会误判 ✗）
    MODE["oom"] = 5
    before_runs, before_free = SEEN["workflow_runs"], SEEN["free"]
    with engine.begin() as conn:
        run_id2 = await svc.submit_run(conn, {"kind": "workflow", "prompt": "显存重试仍失败",
                                             "workflow": {"1": {"class_type": "X", "inputs": {}}}})
    result2 = await wait_terminal(run_id2)
    msg2 = str(result2.get("errorMsg") or "")
    check("⑮ 重试一次仍不足 ⇒ failed，原因保留上游原文**且**写明已重试 + 给出降档建议 ✓",
          result2.get("status") == "failed" and "out of memory" in msg2
          and "已自动卸载上游模型并重试一次" in msg2 and "建议：" in msg2,
          msg2)
    check("⑯ **只重试一次**：上游总共被提交 2 次、free 1 次（绝不无限重刷占 GPU ✗）",
          SEEN["workflow_runs"] == before_runs + 2 and SEEN["free"] == before_free + 1,
          (SEEN["workflow_runs"] - before_runs, SEEN["free"] - before_free))

    # ③ 其它失败 ⇒ 同类建议（模型名对不上 / 缺节点包 ⇒ 各自点名该去哪改 ✓）
    MODE["oom"] = 0  # ← 上一段把接下来 5 次提交都标成「注定 alloc 失败」，这里要清干净 ✓
    MODE["fail"] = True
    MODE["failMsg"] = ("Prompt outputs failed validation: Value not in list: unet_name: "
                       "'qwen_image_2512_fp8.safetensors' not in [...]")
    with engine.begin() as conn:
        run_id3 = await svc.submit_run(conn, {"kind": "workflow", "prompt": "缺模型用例",
                                             "workflow": {"1": {"class_type": "X", "inputs": {}}}})
    msg3 = str((await wait_terminal(run_id3)).get("errorMsg") or "")
    check("⑰ 模型名对不上 ⇒ 原文照留 + 提示「到模型页核对/换本机已有那档」✓",
          "Value not in list" in msg3 and "建议：" in msg3 and "模型" in msg3, msg3)

    MODE["failMsg"] = "缺少节点包：ComfyUI-MiniMaxH3-Easy ✗"
    with engine.begin() as conn:
        run_id4 = await svc.submit_run(conn, {"kind": "workflow", "prompt": "缺节点用例",
                                             "workflow": {"1": {"class_type": "X", "inputs": {}}}})
    msg4 = str((await wait_terminal(run_id4)).get("errorMsg") or "")
    check("⑱ 缺节点包 ⇒ 提示装插件并重启 ComfyUI ✓（原文一字不改 ✓）",
          msg4.startswith("RuntimeError: 缺少节点包") and "节点插件" in msg4, msg4)
    MODE["fail"] = False

    # ④ 纯函数判据：认得出就认、认不出**不许编原因** ✓
    silent = "something odd happened"
    check("⑲ describe_failure：认不出的失败**只留原文**（不硬编原因 ✗）",
          svc.describe_failure(silent) == silent and svc.failure_hints(silent) == [], silent)
    sage = "Memory efficient sage attention is not new enough."
    check("⑳ H3 全族都挂该补丁 ⇒ SageAttention 一句要点明「没有可绕的变体」✓",
          any("SageAttention" in hint and "没有可绕" in hint for hint in svc.failure_hints(sage)),
          svc.failure_hints(sage))
    vdn = "ApplyVDNH3Advanced: stage-B adapter missing"
    check("㉑ VDN 适配器缺失 ⇒ 点名 video-h3-*-novdn 退路 ✓",
          any("novdn" in hint for hint in svc.failure_hints(vdn)), svc.failure_hints(vdn))
    check("㉒ is_out_of_memory 只看显存原文（英文/中文两种都认 ✓，业务失败不误判 ✓）",
          svc.is_out_of_memory("CUDA error: allocation on device 0 out of memory")
          and svc.is_out_of_memory("显存不足") and not svc.is_out_of_memory("缺少节点包 ✗"))


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
    check("㉓ 恢复：两条在途运行都收尾为 failed（不留「永远 processing」✗）",
          row_a.status == "failed" and row_b.status == "failed",
          (row_a.status, row_b.status))
    check("㉔ 恢复说明区分「已提交（可查上游）」与「未提交（未占 GPU）」✓",
          "已提交上游" in str(row_a.error_msg) and "提交前即中断" in str(row_b.error_msg),
          (row_a.error_msg, row_b.error_msg))
    check("㉕ 恢复**没有重提交**（绝不重复占用 GPU/算力 ✓）",
          SEEN["workflow_runs"] == runs_before, (runs_before, SEEN["workflow_runs"]))


async def case_listing() -> None:
    """⑤ 查询是**我们自己的**（不是转发 ComfyUI 历史 ✗）✓。"""
    with engine.begin() as conn:
        rows = [svc.run_to_dict(r) for r in svc.list_runs(conn, limit=50)]
    check("㉖ 能按自己的 DB 列出运行（含状态/产物/耗时 ✓ 不依赖上游 history ✓）",
          len(rows) >= 4 and all({"status", "outputs", "elapsedMs"} <= set(r) for r in rows),
          len(rows))
    with engine.begin() as conn:
        succeeded = svc.list_runs(conn, status="succeeded")
    check("㉗ 能按状态过滤 ✓", all(r.status == "succeeded" for r in succeeded), len(succeeded))


async def main() -> int:
    base, server, thread = start_server(build_stub())
    os.environ["COMFYUI_SERVICE_URL"] = base
    os.environ["COMFYUI_POLL_SECONDS"] = "0.05"
    os.environ["COMFYUI_RUN_TIMEOUT"] = "30"
    try:
        await case_happy_path()
        await case_failure_and_h3()
        await case_oom_retry_and_hints()
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
