"""推断端点守卫（``POST /api/v1/comfyui/workflows/infer``，2026-11-03 立）。

这条路由的失败面很窄但很贵：**交错的图必须当场拒**，否则前端会把一份「看起来能用」的回执
写进 registry，然后每次提交都静默走样 ✗。三条必拒：

1. **UI 格式**（带 ``nodes`` 键）—— 那不是能提交的图，得先在 ComfyUI 里「导出 API 格式」；
2. **图上没有输出节点** —— 提交上去跑一遍什么都不落盘；
3. **``type`` 给了别名**（``picture`` / ``clip``）—— 静默回落成猜 = 前端拿到自己没要的类型。

外加两条「照常返回」：合法 API 图给 200 + 落点；``type`` 合法时覆盖猜测值 ✓。

运行::

    ./.venv/Scripts/python.exe tests/comfyui_infer_route_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="comfyinferroute_"))
os.environ.setdefault("PROXY_TO_NODE", "0")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(BACKEND_PY))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

_URL = "/api/v1/comfyui/workflows/infer"

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _api_graph() -> dict:
    """一张最小可用的 API 格式图（文生图）。"""
    return {
        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat"}, "_meta": {"title": "正向编码"}},
        "2": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "3": {"class_type": "KSampler", "inputs": {"positive": ["1", 0], "seed": 7, "steps": 8, "cfg": 3.5}},
        "4": {"class_type": "SaveImage", "inputs": {"images": []}},
    }


def main() -> None:
    client = TestClient(app)

    resp = client.post(_URL, json={"workflow": _api_graph()})
    body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    data = body.get("data") or {}
    check("合法 API 图 ⇒ 200", resp.status_code == 200, resp.text[:300])
    check("回执带推断类型", data.get("type") == "image", data.get("type"))
    check("回执带落点", (data.get("bindings") or {}).get("prompt", {}).get("node") == "1", data.get("bindings"))
    check("回执带 seeded 参数表", any(item.get("key") == "seed" for item in data.get("parameters") or []), data.get("parameters"))

    resp = client.post(_URL, json={"workflow": _api_graph(), "type": "video", "name": "demo"})
    data = resp.json().get("data") or {}
    check("type 给了就覆盖猜测", resp.status_code == 200 and data.get("type") == "video", data.get("type"))
    check("name 只进 suggested（不改调用方）", (data.get("suggested") or {}).get("name") == "demo", data.get("suggested"))

    ui = {"nodes": [], "links": []}
    resp = client.post(_URL, json={"workflow": ui})
    check("UI 格式 ⇒ 400", resp.status_code == 400, resp.status_code)
    check("UI 格式要给可判前缀", "[ui_format]" in resp.json().get("message", ""), resp.json().get("message"))

    no_out = {"1": {"class_type": "KSampler", "inputs": {"seed": 1}}}
    resp = client.post(_URL, json={"workflow": no_out})
    check("没有输出节点 ⇒ 400", resp.status_code == 400, resp.status_code)
    check("没输出要给可判前缀", "[no_output]" in resp.json().get("message", ""), resp.json().get("message"))

    resp = client.post(_URL, json={"workflow": _api_graph(), "type": "picture"})
    check("type 别名 ⇒ 400（不静默回落）", resp.status_code == 400 and "[bad_hint]" in resp.json().get("message", ""), resp.json().get("message"))

    resp = client.post(_URL, json={})
    check("缺 workflow ⇒ 400 且说清要什么", resp.status_code == 400 and "workflow" in resp.json().get("message", ""), resp.json().get("message"))

    failures = 0
    for name, ok, detail in _RESULTS:
        if ok:
            print(f"PASS  {name}")
        else:
            failures += 1
            print(f"FAIL  {name}\n      {detail!r}")
    print(f"\nSUMMARY: {len(_RESULTS) - failures}/{len(_RESULTS)} passed")


if __name__ == "__main__":
    main()
