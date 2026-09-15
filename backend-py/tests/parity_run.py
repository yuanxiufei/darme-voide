"""把「删 ``backend/`` 前的等价性对拍」编排成一条命令。

⚠️ **本工具的生命周期与 `backend/` 绑定**：它要起 Node 才能对拍 ⇒ **删掉 `backend/` 后即失效**
（届时保留本文件作为「当时如何验证等价性」的历史证据，不要试图改造成单后端工具）。
日常回归请用 `run_all.py`：路径等价性由 `route_parity_test.py` 的**快照模式**继续保证，不需要 Node。

做四件事（全部带超时，失败会打印对应子进程的日志尾巴）：

1. 建一个**空 fixture 目录**当数据根，并检查项目根**没有** ``.data-root`` 标记
   （Node 侧标记优先级高于 ``DATA_ROOT`` 环境变量 ⇒ 有它就会读别处的库、对拍无意义）；
2. 起 **Node**（cwd=``backend``，5789），等它自检通过 —— 它会初始化库、种服务商；
3. 起 **Python**（cwd=``backend-py``，5790，**同一个 DATA_ROOT**），等就绪；
4. 跑 ``parity_diff``（14 条只读端点逐字段 diff），最后无论成败都**收掉两个子进程**。

用法::

    ./.venv/Scripts/python.exe tests/parity_run.py [--keep] [--report ../tmp/parity.json]

⚠️ Windows 上子进程是 ``cmd``/``node`` 套娃 ⇒ 结束用 ``taskkill /T``（否则 tsx 拉起的 node 会变孤儿占端口）。
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from parity_diff import main as parity_main  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
BACKEND = REPO / "backend"
BACKEND_PY = REPO / "backend-py"
NODE_URL = "http://127.0.0.1:5789"
PY_URL = "http://127.0.0.1:5790"


def _wait_ready(url: str, timeout: float = 90.0) -> bool:
    import httpx

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = httpx.get(url + "/api/v1/dramas", timeout=3)
            if response.status_code < 500:
                return True
        except Exception:  # noqa: BLE001 —— 还没起来
            pass
        time.sleep(1)
    return False


def _spawn(command: list[str], cwd: Path, log_path: Path, env: dict[str, str]) -> subprocess.Popen:
    handle = open(log_path, "w", encoding="utf-8", errors="replace")
    return subprocess.Popen(command, cwd=str(cwd), env=env, stdout=handle, stderr=subprocess.STDOUT)


def _kill(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True, check=False)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=15)
    except Exception:  # noqa: BLE001
        proc.kill()


def _tail(path: Path, count: int = 25) -> str:
    if not path.exists():
        return "(无日志)"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-count:])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true", help="对拍后不杀子进程（手工排查用）")
    parser.add_argument("--report", default=str(REPO / "tmp" / "parity.json"))
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args(argv)

    marker = REPO / ".data-root"
    if marker.exists():
        print(f"❌ 拒绝运行：{marker} 存在（标记文件优先于 DATA_ROOT，会让两侧读到别处的库）")
        print(f"   内容：{marker.read_text(encoding='utf-8', errors='replace').strip()}")
        return 2

    fixture = Path(tempfile.mkdtemp(prefix="parity_"))
    node_log = fixture / "node.log"
    py_log = fixture / "py.log"
    env = {**os.environ, "DATA_ROOT": str(fixture), "PYTHONIOENCODING": "utf-8"}
    print(f"fixture: {fixture}")

    # ⚠️ 不用 `npx`：它不在本机 PATH，且 Windows 上 Popen 解析不了 `.cmd`
    #    ⇒ 直接用 node 跑仓内 tsx CLI（`node_modules/tsx/dist/cli.mjs`）
    node_exe = shutil.which("node") or "node"
    tsx_cli = BACKEND / "node_modules" / "tsx" / "dist" / "cli.mjs"
    node = _spawn([node_exe, str(tsx_cli), "src/index.ts"], BACKEND, node_log, env)
    try:
        print("等 Node(5789) 就绪 …")
        if not _wait_ready(NODE_URL, args.timeout):
            print("❌ Node 没起来，日志尾巴：")
            print(_tail(node_log))
            return 1

        python = _spawn([str(BACKEND_PY / ".venv" / "Scripts" / "python.exe"),
                         "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "5790"],
                        BACKEND_PY, py_log, env)
        try:
            print("等 Python(5790) 就绪 …")
            if not _wait_ready(PY_URL, args.timeout):
                print("❌ Python 没起来，日志尾巴：")
                print(_tail(py_log))
                return 1

            print("\n=== 逐端点对拍 ===")
            code = parity_main(["--node", NODE_URL, "--py", PY_URL, "--report", args.report])
        finally:
            if not args.keep:
                _kill(python)
    finally:
        if not args.keep:
            _kill(node)

    print(f"\n退出码 {code}（0 = 无新差异）｜ fixture 与日志留在 {fixture}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
