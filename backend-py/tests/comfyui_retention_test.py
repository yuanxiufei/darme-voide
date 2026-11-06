"""ComfyUI 运行记录**保留清理**守卫（`comfyui.purge_runs`，2026-11-03 立）。

这条最容易出的事故是**不可逆的**：删错一个文件，产物就没了；删错一条记录，那个文件就再没人
知道该删（孤儿）。所以每条判据都盯一个"删错了不会报错"的口子：

1. **默认干跑** ✓：不传 `dry_run=False` 一个字节都不许动；
2. **在跑的绝不碰** ✗：`processing` / `queued` 过期了也不清（后台还在写产物）；
3. **行与产物同生共死** ✓：产物被占用删不掉 ⇒ **这条记录也留着** ✓（先删行会让"该删哪个文件"
   这条线索永久丢失 ✗）；反过来文件早就不在了 ⇒ 照删记录、照实计 `missingFiles` ✓；
4. **共享产物不删** ✗：同一路径还被别的记录引用（另一次运行 adopted 了同一份）⇒ 留着；
5. **不认识的路径不动** ✗：不在 `static/comfyui/` 下的整行留着（宁可不清，不许误删）；
6. **`days<=0` 什么都不做** ✓（别把 0 理解成"全清" ✗）；
7. **预演端点只读** ✓：`GET /runs/retention` 恒干跑、跑完文件与记录都还在，且它必须排在
   `/runs/{run_id}` **之前**（排后面会被当 id 吃掉 ⇒ 400「run_id 非法」✗）。

运行::

    ./.venv/Scripts/python.exe tests/comfyui_retention_test.py
"""
from __future__ import annotations

import os
import shutil
import stat
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="comfyretention_"))
os.environ.setdefault("PROXY_TO_NODE", "0")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(BACKEND_PY))

from app.core import db  # noqa: E402
from app.core.config import get_data_root  # noqa: E402
from app.core.models import comfyui_runs  # noqa: E402
from app.services import comfyui as svc  # noqa: E402

_NOW = datetime(2026, 11, 3, 12, 0, 0, tzinfo=timezone.utc)
_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def stamp(days_ago: int) -> str:
    return (_NOW - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def wipe() -> None:
    with db.engine.begin() as conn:
        conn.execute(comfyui_runs.delete())


def seed(status: str, *, days_ago: int, local_path: str | None, kind: str = "image") -> int:
    with db.engine.begin() as conn:
        result = conn.execute(
            comfyui_runs.insert().values(
                kind=kind,
                workflow_name="probe",
                status=status,
                local_path=local_path,
                created_at=stamp(days_ago),
                updated_at=stamp(days_ago),
            )
        )
        return int(result.inserted_primary_key[0])


def purge(**kwargs):
    with db.engine.begin() as conn:
        return svc.purge_runs(conn, now_ts=_NOW, **kwargs)


def run_ids() -> set[int]:
    with db.engine.begin() as conn:
        return {int(row.id) for row in conn.execute(comfyui_runs.select()).all()}


def make_artifact(name: str, body: bytes = b"x" * 2048) -> tuple[str, Path]:
    """在数据根下造一个真产物，返回 (相对路径, 绝对路径)。"""
    relative = f"static/comfyui/{name}"
    absolute = Path(get_data_root()) / relative
    absolute.parent.mkdir(parents=True, exist_ok=True)
    absolute.write_bytes(body)
    return relative, absolute


def case_dry_run() -> None:
    wipe()
    relative, absolute = make_artifact("dry.png")
    run = seed("succeeded", days_ago=90, local_path=relative)
    report = purge(days=30)
    check("① 默认干跑：文件还在 ✓", absolute.is_file())
    check("① 默认干跑：记录还在 ✓", run in run_ids())
    check("① 干跑也要算清账", report["dryRun"] is True and report["runs"] == 1 and report["files"] == 1, report)
    check("① 干跑报字节", report["bytes"] == 2048, report["bytes"])


def case_apply() -> None:
    wipe()
    relative, absolute = make_artifact("gone.png")
    run = seed("succeeded", days_ago=90, local_path=relative)
    report = purge(days=30, dry_run=False)
    check("② 真删：文件没了 ✓", not absolute.exists())
    check("② 真删：记录没了 ✓", run not in run_ids())
    check("② 真删账对得上", report["runs"] == 1 and report["files"] == 1 and report["failed"] == [], report)


def case_running_untouched() -> None:
    wipe()
    relative, _absolute = make_artifact("running.png")
    run = seed("processing", days_ago=90, local_path=relative)
    report = purge(days=30, dry_run=False)
    check("③ 在跑的（processing）过期也不清 ✗", run in run_ids())
    check("③ 在跑的进 keptRunning", [item["id"] for item in report["keptRunning"]] == [run], report["keptRunning"])

    wipe()
    relative, _absolute = make_artifact("queued.png")
    run = seed("queued", days_ago=90, local_path=relative)
    purge(days=30, dry_run=False)
    check("③ queued 同口径不清 ✗", run in run_ids())


def case_fresh_untouched() -> None:
    wipe()
    relative, _absolute = make_artifact("fresh.png")
    run = seed("succeeded", days_ago=1, local_path=relative)
    report = purge(days=30, dry_run=False)
    check("④ 没过期的一条不动 ✓", run in run_ids() and report["considered"] == 0, report["considered"])


def case_shared_artifact() -> None:
    wipe()
    relative, absolute = make_artifact("shared.png")
    old = seed("succeeded", days_ago=90, local_path=relative)
    keep = seed("succeeded", days_ago=1, local_path=relative)
    report = purge(days=30, dry_run=False)
    check("⑤ 共享产物不删 ✗", absolute.is_file())
    check("⑤ 共享的两条都留着 ✓", {old, keep} <= run_ids())
    check("⑤ 共享要报出来", any(item["localPath"] == relative for item in report["keptShared"]), report["keptShared"])


def case_outside_path() -> None:
    wipe()
    run = seed("succeeded", days_ago=90, local_path="models/loras/keep.safetensors")
    report = purge(days=30, dry_run=False)
    check("⑥ 不在 static/comfyui/ 下的整行留着 ✗", run in run_ids())
    check("⑥ 要报 keptOutside", any(item["id"] == run for item in report["keptOutside"]), report["keptOutside"])


def case_missing_file() -> None:
    wipe()
    run = seed("succeeded", days_ago=90, local_path="static/comfyui/never-existed.png")
    report = purge(days=30, dry_run=False)
    check("⑦ 产物已不在盘上 ⇒ 记录照删 ✓", run not in run_ids())
    check("⑦ 但要计 missingFiles", report["missingFiles"] == 1, report["missingFiles"])


def case_no_path() -> None:
    wipe()
    run = seed("failed", days_ago=90, local_path=None)
    purge(days=30, dry_run=False)
    check("⑦ 没有产物的失败记录直接清 ✓", run not in run_ids())


def case_undeletable_file() -> None:
    """产物删不掉 ⇒ **这条记录也留着** ✓（否则线索永久丢失 ✗）。

    ⚠️ 这条判据靠 **Windows 的独占打开**才有意义 ✓：POSIX 下 `unlink` 打开中的文件照样成功
    ⇒ 在那边强行断言只会得到一条假绿 ✗，所以明说跳过而不是硬跑 ✓。
    """
    if os.name != "nt":
        print("SKIP  ⑧ 删不掉的产物（这机制只有 Windows 独占打开才复现）")
        return
    wipe()
    relative, absolute = make_artifact("locked.png")
    run = seed("succeeded", days_ago=90, local_path=relative)
    # Windows 上把文件占住（独占打开）⇒ os.remove 抛 PermissionError ✓
    handle = open(absolute, "rb")
    try:
        report = purge(days=30, dry_run=False)
    finally:
        handle.close()
    check("⑧ 删不掉的产物报了错", bool(report["failed"]), report["failed"])
    check("⑧ 记录的产物没被扣掉", run in run_ids())
    check("⑧ 记进 keptFailed", any(item["id"] == run for item in report["keptFailed"]), report["keptFailed"])
    purge(days=30, dry_run=False)
    check("⑧ 放开占用后下一次能清掉 ✓", run not in run_ids() and not absolute.exists())


def case_days_zero() -> None:
    wipe()
    relative, absolute = make_artifact("zero.png")
    run = seed("succeeded", days_ago=900, local_path=relative)
    report = purge(days=0, dry_run=False)
    check("⑨ days=0 什么都不做（不是全清 ✗）", run in run_ids() and absolute.is_file())
    check("⑨ days=0 要说清为什么", bool(report.get("note")), report.get("note"))


def case_status_scope() -> None:
    wipe()
    a, _pa = make_artifact("a.png")
    b, _pb = make_artifact("b.png")
    failed_run = seed("failed", days_ago=90, local_path=a)
    ok_run = seed("succeeded", days_ago=90, local_path=b)
    report = purge(days=30, statuses=("failed",), dry_run=False)
    check("⑩ statuses 可收窄", failed_run not in run_ids() and ok_run in run_ids(), sorted(run_ids()))
    check("⑩ 只清了一种", report["runs"] == 1, report["runs"])


def case_env_days() -> None:
    """不传 `days` 时认环境变量 ✓；传了就**以参数为准**（不静默覆盖 ✗）。"""
    wipe()
    relative, absolute = make_artifact("env.png")
    run = seed("succeeded", days_ago=5, local_path=relative)
    os.environ["COMFYUI_RETENTION_DAYS"] = "2"
    try:
        report = purge()
        check("⑪ 环境变量当默认保留期 ✓", report["days"] == 2, report["days"])
        check("⑪ 5 天前的记录在 2 天保留期下算过期 ✓", report["considered"] == 1, report["considered"])
        report = purge(days=10)
        check("⑪ 显式 days 压过环境变量 ✓", report["days"] == 10 and report["considered"] == 0, report)
        check("⑪ 参数优先时文件不许动 ✗", absolute.is_file() and run in run_ids())
        os.environ["COMFYUI_RETENTION_DAYS"] = "not-a-number"
        report = purge()
        check("⑪ 环境变量写坏 ⇒ 退回默认值（不崩 ✗）", report["days"] == svc.RETENTION_DAYS, report["days"])
    finally:
        os.environ.pop("COMFYUI_RETENTION_DAYS", None)


def case_preview_route() -> None:
    """`GET /runs/retention` **只算不删** ✓，且必须**先于** `/runs/{run_id}` 匹配 ✓。

    第二个判据容易被当成小事：FastAPI 按声明顺序匹配，这条一旦被挪到 ``/runs/{run_id}``
    后面，``retention`` 会被当成 id 去 ``parse_param_id`` ⇒ 回 400「run_id 非法」✗。
    两个端点都返回 ``{data:…}``，只看状态码还会以为是"参数不对"，很难看出是路由顺序 ✓。
    """
    from fastapi.testclient import TestClient

    from app.main import app

    wipe()
    relative, absolute = make_artifact("preview.png")
    run = seed("succeeded", days_ago=400, local_path=relative)
    running = seed("processing", days_ago=400, local_path=None)

    client = TestClient(app)
    resp = client.get("/api/v1/comfyui/runs/retention", params={"days": 30})
    data = resp.json().get("data") or {}
    check("⑫ 预演端点回 200（被 /runs/{run_id} 抢走会回 400「run_id 非法」✗）",
          resp.status_code == 200 and data.get("days") == 30, resp.text[:300])
    check("⑫ 端点恒为干跑 ✓", data.get("dryRun") is True, data.get("dryRun"))
    check("⑫ 逾期一条算进去、在跑的那条不动 ✓",
          data.get("runs") == 1 and [item["id"] for item in data.get("keptRunning") or []] == [running],
          data)
    check("⑫ 会释放的字节照实算 ✓", data.get("bytes") == 2048, data.get("bytes"))
    check("⑫ 端点跑完文件还在 ✓（能删的口子只能有一个）", absolute.is_file())
    check("⑫ 端点跑完记录还在 ✓", {run, running} <= run_ids(), sorted(run_ids()))

    resp = client.get("/api/v1/comfyui/runs/retention", params={"days": 0})
    data = resp.json().get("data") or {}
    check("⑫ days=0 ⇒ 一条都不清 ✓", data.get("runs") == 0 and bool(data.get("note")), data)
    check("⑫ days=0 也不删文件 ✓", absolute.is_file())


def main() -> None:
    case_dry_run()
    case_apply()
    case_running_untouched()
    case_fresh_untouched()
    case_shared_artifact()
    case_outside_path()
    case_missing_file()
    case_no_path()
    case_undeletable_file()
    case_days_zero()
    case_status_scope()
    case_env_days()
    case_preview_route()

    failures = 0
    for name, ok, detail in _RESULTS:
        if ok:
            print(f"PASS  {name}")
        else:
            failures += 1
            print(f"FAIL  {name}\n      {detail!r}")
    print(f"\nSUMMARY: {len(_RESULTS) - failures}/{len(_RESULTS)} passed")
    shutil.rmtree(os.environ["DATA_ROOT"], ignore_errors=True)


if __name__ == "__main__":
    main()
