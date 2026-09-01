"""
MiniMax H3 本地推理薄封装服务（端口 8765）。

对外协议与 backend/src/services/adapters/minimax-video.ts 对齐：
  POST /v1/video_generation           body={model,prompt,aspect_ratio,duration,...} -> {task_id}
  GET  /v1/video_generation/task/{id} -> {status, video_url, error_msg}

底层调用 ComfyUI(8188) 执行 H3 workflow（FL2VA / Ref2VA），完成后卸载 GPU。

启动：
  uvicorn server:app --host 0.0.0.0 --port 8765
  或  python server.py
"""
import os
import threading
import uuid
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")

app = FastAPI(title="MiniMax H3 Local Service", version="0.1.0")

# 任务内存存储：task_id -> {status, body, video_url, error_msg}
_TASKS: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()


def _run_h3(task: Dict[str, Any]) -> None:
    """调用 ComfyUI 执行 H3 推理。阶段 2 实现，此处为占位桩。"""
    # TODO(阶段2): 组装 workflow -> POST /prompt -> 轮询 /history -> 取视频 -> POST /free
    with _LOCK:
        task["status"] = "failed"
        task["error_msg"] = "H3 ComfyUI backend not wired yet"


@app.post("/v1/video_generation")
def create_video(body: Dict[str, Any]) -> JSONResponse:
    task_id = uuid.uuid4().hex
    task: Dict[str, Any] = {
        "task_id": task_id,
        "status": "queued",
        "body": body,
        "video_url": None,
        "error_msg": None,
    }
    with _LOCK:
        _TASKS[task_id] = task
    threading.Thread(target=_run_h3, args=(task,), daemon=True).start()
    return JSONResponse({"task_id": task_id, "status": "queued"})


@app.get("/v1/video_generation/task/{task_id}")
def poll_task(task_id: str) -> JSONResponse:
    with _LOCK:
        task = _TASKS.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return JSONResponse({
        "status": task["status"],
        "video_url": task["video_url"],
        "error_msg": task["error_msg"],
    })


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {"ok": True, "comfyui_url": COMFYUI_URL}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
