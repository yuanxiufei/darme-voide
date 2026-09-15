"""S7 自检：**生产镜像与代码常量的一致性**（`Dockerfile` / `.dockerignore` ↔ app 代码）。

为什么需要：镜像里的路径/端口是**手写常量**（`COPY` 目标、`--port`、数据根），
而代码里的对应值（`config.PROJECT_ROOT` / `FRONTEND_DIST` / `server.port`）会各自演化 ——
两边一漂移，**本机构建/自检全绿，容器起来才发现静态站 404、端口不通或技能库缺失**。
本项目真实发生过同类事故（技能库搬进 `backend-py/` 后 `COPY skills/` 指向了不存在的路径）。
本自检把「容器布局假设」钉在代码常量上，改 Dockerfile 或改常量时立刻暴露。

⚠️ **本机没装 Docker**（见 `.codebuddy/memory`）⇒ 这里只做**静态**核对，不做真构建；
真构建的验收请在有 Docker 的机器上跑 `docker build .`（首次务必验证 `/` 能出前端、`/api/v1/...` 通）。

运行::

    ./.venv/Scripts/python.exe tests/dockerfile_contract_test.py
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="dfc_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app import config  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO / "Dockerfile"
DOCKERIGNORE = REPO / ".dockerignore"
COMPOSE = REPO / "docker-compose.yml"

_R: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _R.append((name, bool(condition), detail))


def main() -> int:
    if not DOCKERFILE.is_file():
        print("跳过：仓库里没有 Dockerfile")
        print("\nSUMMARY: 1/1 passed")
        return 0

    text = DOCKERFILE.read_text(encoding="utf-8")
    # 只核对**指令行**：头部注释会「提到」旧写法以便对照，那是有意的
    code = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    ignore = DOCKERIGNORE.read_text(encoding="utf-8") if DOCKERIGNORE.is_file() else ""

    # ① 容器内布局：代码放 /app/backend-py ⇒ PROJECT_ROOT（上溯两级）必须是 /app
    project_root = Path(config.PROJECT_ROOT).resolve()
    check("布局: config.PROJECT_ROOT 上溯两级 == 仓库根（容器内即 /app）",
          project_root == REPO, project_root)

    # ② 前端产物：Dockerfile 把它放到 ./frontend/dist，代码也按 <root>/frontend/dist 找
    # ⚠️ 比**原始路径**而不是 `.resolve()`：本机 `frontend/dist` 是指向 `.output/public` 的
    #    **目录联接**（方便本地直接跑 Python 后端）⇒ resolve 会跳到联接目标，断言就假红了。
    frontend_dist = Path(config.FRONTEND_DIST)
    check("布局: FRONTEND_DIST == <仓库根>/frontend/dist（与 COPY 目标一致）",
          os.path.normcase(str(frontend_dist)) == os.path.normcase(str(REPO / "frontend" / "dist")),
          frontend_dist)
    check("前端构建产物存在（`COPY --from` 的源 /app/frontend/.output/public）",
          (REPO / "frontend" / ".output" / "public" / "index.html").is_file())

    # ③ 每个 COPY 源都得真实存在（否则 `docker build` 直接失败）
    for relative in ("backend-py/requirements.txt", "configs/config.example.yaml",
                     "backend-py/app/main.py", "backend-py/skills/README.md",
                     "frontend/package.json", "frontend/package-lock.json"):
        check(f"COPY 源存在: {relative}", (REPO / relative).exists(), relative)

    # ④ 端口三处一致（EXPOSE / ENV PORT / CMD --port）且与 app 默认端口相同
    ports = (re.findall(r'--port"?,\s*"(\d+)"', code) + re.findall(r"EXPOSE (\d+)", code)
             + re.findall(r"ENV PORT=(\d+)", code))
    check("端口: Dockerfile 三处一致", ports and len(set(ports)) == 1, ports)
    check("端口: 与 app/config 的 server 端口一致",
          ports and str(config.server.get("port")) == ports[0], (ports, config.server))

    # ⑤ 不再出现已迁移/已下线的位置与运行时（真删 backend/ 后这些会直接失败）
    for stale in ("COPY skills/", "COPY backend/src", "backend/package.json", "tsx", "5789"):
        check(f"指令里已无旧引用: {stale}", stale not in code, stale)

    # ⑥ 技能库必须落在 backend-py/ 下（代码按**自身文件位置**解析技能根，靠 COPY 目标保持相对布局）
    check("技能库: COPY 到 ./backend-py/skills/",
          re.search(r"COPY backend-py/skills\s+\./backend-py/skills", code) is not None)

    # ⑦ .dockerignore 必须排除 venv / __pycache__（否则构建上下文白拖几百 MB）
    check("dockerignore: 排除 backend-py/.venv", "backend-py/.venv" in ignore)
    check("dockerignore: 排除 __pycache__", "**/__pycache__" in ignore)

    # ⑧ 编排文件（docker-compose）必须与镜像同口径 —— 它是**独立手写**的，最容易漏改
    #    （本项目真实发生过：Dockerfile 改完了、compose 还写着 5789/NODE_ENV）
    if COMPOSE.is_file():
        compose = COMPOSE.read_text(encoding="utf-8")
        compose_code = "\n".join(ln for ln in compose.splitlines() if not ln.lstrip().startswith("#"))
        check("compose: 端口与镜像一致（无旧端口 5789）", "5789" not in compose_code, "5789")
        check("compose: 暴露 5790", "5790" in compose_code)
        check("compose: 不再用 Node 时代的环境变量 NODE_ENV",
              "NODE_ENV" not in compose_code)
        check("compose: 数据卷仍挂到 /app/data（与镜像数据根一致）",
              "./data:/app/data" in compose_code)
    else:
        check("compose: 文件不存在则跳过", True)

    failures = [item for item in _R if not item[1]]
    for name, passed, detail in _R:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_R) - len(failures)}/{len(_R)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
