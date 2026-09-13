"""S6 自检：``POST /episodes/{id}/continue-script``（AI 续写剧本，不落库）。

LLM 打桩 ⇒ 不打网络。四条容易抄错的语义：

1. 集查询**不带软删过滤**，但拿到后显式判 ``deleted_at`` ⇒ **软删集是 404**；
2. ``mode`` 只有 `'script'` 才是剧本模式，**其余一切值**（含缺省/非法）都是 `'raw'`；
3. ``text`` **必须是字符串**（数字/对象一律当空）⇒ `text is required`；
4. 服务抛错 → 400 且文案是 ``str(err)``。

运行::

    ./.venv/Scripts/python.exe tests/episodes_continue_script_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="epcs_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import update  # noqa: E402

from app.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models import dramas, episodes  # noqa: E402
from app.response import now  # noqa: E402
from app.routers import episodes as er  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


_CALLS: list[dict] = []


async def _fake_continue(conn, payload):  # noqa: ANN001
    _CALLS.append(payload)
    if payload.get("text") == "BOOM":
        raise RuntimeError("续写炸了")
    return f"续写结果({payload['mode']})"


er.continue_script = _fake_continue  # type: ignore[assignment]


def main() -> int:
    ts = now()
    with engine.begin() as conn:
        drama_id = int(conn.execute(dramas.insert().values(
            title="剧", created_at=ts, updated_at=ts)).lastrowid)
        episode_id = int(conn.execute(episodes.insert().values(
            drama_id=drama_id, episode_number=1, title="一", content="内容",
            status="draft", created_at=ts, updated_at=ts)).lastrowid)
        deleted_id = int(conn.execute(episodes.insert().values(
            drama_id=drama_id, episode_number=2, title="二", content="内容",
            status="draft", deleted_at=ts, created_at=ts, updated_at=ts)).lastrowid)

    client = TestClient(app)
    base = f"/api/v1/episodes/{episode_id}/continue-script"

    check("续写: id 非法 -> 404 Invalid episode id",
          client.post("/api/v1/episodes/abc/continue-script", json={"text": "x"}).json()
          == {"code": 404, "message": "Invalid episode id"})
    check("续写: 集不存在 -> 404 Episode not found",
          client.post("/api/v1/episodes/999999/continue-script", json={"text": "x"}).json()
          == {"code": 404, "message": "Episode not found"})
    check("续写: **软删集也是 404**（查到后显式判 deleted_at）",
          client.post(f"/api/v1/episodes/{deleted_id}/continue-script",
                      json={"text": "x"}).json()
          == {"code": 404, "message": "Episode not found"})
    check("续写: 缺 text / 空串 / 纯空白 -> 400 text is required",
          all(client.post(base, json=body).json()
              == {"code": 400, "message": "text is required"}
              for body in ({}, {"text": ""}, {"text": "   "})))
    check("续写: text **非字符串**（数字/对象/数组）一律当空 -> 400",
          all(client.post(base, json={"text": value}).status_code == 400
              for value in (123, {"a": 1}, ["x"], None)))

    _CALLS.clear()
    ok = client.post(base, json={"text": "  上一段剧情  "}).json()["data"]
    check("续写: 成功回执 {continuation}，text **原样透传**（不 trim）",
          ok == {"continuation": "续写结果(raw)"}
          and _CALLS[-1] == {"text": "  上一段剧情  ", "mode": "raw"}, (ok, _CALLS[-1]))
    check("续写: mode='script' -> 透传 script；其它值（含非法）一律 raw",
          client.post(base, json={"text": "x", "mode": "script"}).status_code == 200
          and _CALLS[-1]["mode"] == "script"
          and client.post(base, json={"text": "x", "mode": "SCRIPT"}).status_code == 200
          and _CALLS[-1]["mode"] == "raw")

    boom = client.post(base, json={"text": "BOOM"})
    check("续写: 服务抛错 -> 400 且文案是 str(err)",
          boom.status_code == 400 and boom.json()["message"] == "续写炸了", boom.json())

    failed_items = [item for item in _RESULTS if not item[1]]
    for name, ok_item, detail in _RESULTS:
        print(("PASS  " if ok_item else "FAIL  ") + name
              + ("" if ok_item else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed_items)}/{len(_RESULTS)} passed")
    return 1 if failed_items else 0


if __name__ == "__main__":
    raise SystemExit(main())
