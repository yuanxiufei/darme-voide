"""S7 自检：**本地 H3 推理链的接缝契约**（2026-09-15 新增 —— 这条接缝此前从未被测过）。

链路：后端（`provider=minimax` + `base_url=http://localhost:8765` + `runtime=h3`）
      → `MiniMaxVideoAdapter`（`app/services/adapters/video_adapters.py`）
      → 8765 薄封装（`app/local_services/h3/server.py`）
      → ComfyUI(8188) 执行 H3 workflow

⚠️ 为什么必须有这个自检：两侧是**各自手写**的（适配器按 Node 侧 `minimax-video.ts` 移植，
薄封装是后来手写的）⇒ 端点名 / body 字段 / 响应字段**任何一处漂移**都会让「本地 H3」在真机上
以**运行时错误**暴露（而那要等模型下完、ComfyUI 起来才看得见 ✗）。这里用「适配器造请求 →
**薄封装真接收**」的零依赖闭环把它钉死：**不需要 GPU、不需要 8765 在跑**。

阶段状态：**⑦ 已于 2026-09-17 翻转** —— 薄封装的 `_run_h3` 已接线（组装 → `/prompt` → `/history` 轮询 →
`/view` 取片 → `/files` 回 URL → `/free` ✓），所以「占位桩文案」不该再出现 ✓。本自检不起 ComfyUI，
因此预期终态是 **failed + 真原因**（ComfyUI 不可达）✓；真跑通由 `h3_stage2_test.py`（stub ComfyUI）
与 `h3_backend_live_test.py`（真 8765）覆盖 ✓。

运行::

    ./.venv/Scripts/python.exe tests/h3_chain_test.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="h3chain_"))
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from fastapi.testclient import TestClient  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def load_wrapper_app():
    """按路径加载 8765 薄封装（它在 `app/local_services/` 下，**不是**应用分层代码，故按文件加载）。"""
    path = BACKEND_PY / "app" / "local_services" / "h3" / "server.py"
    spec = importlib.util.spec_from_file_location("h3_wrapper_server", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def main() -> int:
    from app.services.adapters.video_adapters import MiniMaxVideoAdapter

    module = load_wrapper_app()
    client = TestClient(module.app)
    adapter = MiniMaxVideoAdapter()
    config = {"baseUrl": "http://localhost:8765", "apiKey": "local-key"}
    record = {
        "prompt": "雨夜霓虹街头，主角回头",
        "aspectRatio": "9:16",
        "duration": 5,
        "referenceMode": "first_last",
        "firstFrameUrl": "http://x/first.png",
        "lastFrameUrl": "http://x/last.png",
    }

    # ① 适配器造的**生成请求**
    gen = adapter.build_generate_request(config, record)
    check("适配器: 生成请求 = POST {baseUrl}/v1/video_generation",
          gen["method"] == "POST" and urlsplit(gen["url"]).path == "/v1/video_generation", gen["url"])
    check("适配器: body 含 model/prompt/aspect_ratio/duration（H3 原生字段）",
          {"model", "prompt", "aspect_ratio", "duration"} <= set(gen["body"]), sorted(gen["body"]))
    check("适配器: first_last 模式带 first/last_frame_image",
          gen["body"].get("first_frame_image") == "http://x/first.png"
          and gen["body"].get("last_frame_image") == "http://x/last.png", gen["body"])

    # ② 薄封装的**路由表**必须含适配器用到的两条路径（两侧各自手写 ⇒ 这是最该钉的一条）
    declared = {getattr(route, "path", "") for route in module.app.routes}
    check("接缝: 薄封装声明了 /v1/video_generation", "/v1/video_generation" in declared, sorted(declared))
    check("接缝: 薄封装声明了 /v1/video_generation/task/{task_id}",
          "/v1/video_generation/task/{task_id}" in declared, sorted(declared))
    check("反套套逻辑: 伪造路径 /v1/videos 必须**不在**路由表里", "/v1/videos" not in declared)

    # ③ 闭环：拿适配器造的 body **真打**薄封装
    created = client.post(urlsplit(gen["url"]).path, json=gen["body"])
    check("闭环: 生成请求被薄封装接受（200 + task_id + queued）",
          created.status_code == 200 and created.json().get("task_id")
          and created.json().get("status") == "queued", created.text[:200])
    task_id = created.json().get("task_id", "")

    # ④ 适配器能解析薄封装的响应（字段名 must match）
    parsed = adapter.parse_generate_response(created.json())
    check("闭环: 适配器解析薄封装响应 -> isAsync + taskId",
          parsed.get("isAsync") is True and parsed.get("taskId") == task_id, parsed)

    # ⑤ 适配器造的**轮询请求**路径 == 薄封装声明的那条
    poll = adapter.build_poll_request(config, task_id)
    check("适配器: 轮询请求 = GET {baseUrl}/v1/video_generation/task/{id}",
          poll["method"] == "GET" and urlsplit(poll["url"]).path == f"/v1/video_generation/task/{task_id}",
          poll["url"])

    # ⑥ 闭环：轮询真打薄封装，字段名对齐（status / video_url / error_msg）
    polled = client.get(urlsplit(poll["url"]).path)
    check("闭环: 轮询返回 status/video_url/error_msg 三字段",
          polled.status_code == 200 and {"status", "video_url", "error_msg"} <= set(polled.json()),
          polled.text[:200])
    check("闭环: 未知 task_id 轮询 -> 404（不会假装成功）",
          client.get("/v1/video_generation/task/nope").status_code == 404)

    # ⑦ ✅ **阶段 2 已接线**（2026-09-17 翻转）：终态失败时必须是**真原因**（ComfyUI 不可达 / 缺节点包 ✓），
    #    而不再是那条占位桩文案 `H3 ComfyUI backend not wired yet` ✗。
    #    本自检**不起 ComfyUI**（in-process ✓）⇒ 预期就是「不可达」这类真原因 ✓。
    for _ in range(40):  # 后台线程要时间（连 ComfyUI / 探节点 ✓）
        body = polled.json()
        if body.get("status") in ("failed", "succeeded"):
            break
        time.sleep(0.2)
        polled = client.get(urlsplit(poll["url"]).path)
    body = polled.json()
    reason = str(body.get("error_msg") or "")
    # ⚠️ 「真原因」随环境而异（本自检用的是**假 URL** ⇒ 也可能是「参考媒体取不到」✓）⇒
    #    判据锚在**要点**上：失败 + 有具体文案 + **不是**那句占位桩 ✗。
    check("阶段2 已接线: 终态失败时给的是**具体原因**（不可达/缺节点/媒体取不到…），不再是 'not wired yet' ✗",
          body.get("status") == "failed" and bool(reason) and "not wired yet" not in reason,
          body)

    # ⑧ 运维面：healthz 回报 ComfyUI 地址（排障要用）
    health = client.get("/healthz")
    check("运维: /healthz 返回 ok + comfyui_url",
          health.status_code == 200 and health.json().get("ok") is True
          and health.json().get("comfyui_url", "").startswith("http"), health.text[:160])

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
