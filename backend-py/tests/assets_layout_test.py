"""S7 自检：**三类资产的位置守卫**（2026-09-15「三目录并入 app/」当天新增）。

为什么需要：三目录合并把 `skills/`、`scripts/`、`local_services/` 从 `backend-py/` 顶层搬进
`app/` 之后，出现过两类**静默失效**，两类都测不出来的原因都是「断言指向了一个**不存在的路径**」：

1. 冒烟里的兜底清理写成 `parents[2] / "skills"` ✗（搬库后不存在）⇒ `shutil.rmtree(..., ignore_errors=True)`
   **静默不删** ⇒ 残留 skill 让**下一轮**冒烟假红（实测 475/476，重跑即绿）。
2. 同文件的「无残留」断言也指向同一条坏路径 ✗ ⇒ **恒真**（假绿）⇒ 那条守卫等于被关掉了。

⇒ 本自检把「资产根到底在哪」变成**机械判据**：路径常量必须真的存在、且必须在 `app/` 之内；
`.dockerignore` 必须排除后两者（否则镜像会带上工具链与本机 clone ✗）。

运行::

    ./.venv/Scripts/python.exe tests/assets_layout_test.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BP = REPO / "backend-py"
APP = BP / "app"

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def main() -> int:
    sys.path.insert(0, str(BP))
    from app.core.config import APP_ROOT, BACKEND_PY_ROOT, skills_dir  # noqa: PLC0415

    # ① 权威路径其实存在（这是最容易被「搬家」打穿的一条）
    resolved = skills_dir()
    check("技能库: skills_dir() 指向的目录**真的存在**", resolved.is_dir(), resolved)
    check("技能库: 在 app/ 之内（三类资产已并入）",
          APP in resolved.parents and resolved.name == "skills", resolved)
    check("技能库: 至少含一个 SKILL.md（不是空壳）",
          any(resolved.glob("*/SKILL.md")), sorted(p.name for p in resolved.iterdir())[:6])

    # ② 另两类资产在 app/ 内（scripts/ 与 local_services/ 不进镜像，但必须在盘上）
    for name in ("scripts", "local_services"):
        path = APP / name
        check(f"资产根: app/{name}/ 在盘上", path.is_dir(), path)

    # ③ 根常量自洽（BACKEND_PY_ROOT = backend-py；APP_ROOT = backend-py/app）
    check("常量: BACKEND_PY_ROOT == <repo>/backend-py", BACKEND_PY_ROOT == BP, BACKEND_PY_ROOT)
    check("常量: APP_ROOT == <repo>/backend-py/app", APP_ROOT == APP, APP_ROOT)

    # ④ 冒烟那条「无残留」断言用的是权威路径（防它再退回分段拼接 ✗）
    # ⚠️ 只扫**代码行**：注释里正引用着那句旧写法（讲解这个坑）✗ —— 不排除注释就会「守卫匹配到
    #    自己的文档」（本守卫第一版就踩了：11/12，红在它自己身上）。
    smoke_lines = [
        line for line in (BP / "tests" / "smoke_test.py").read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    ]
    smoke_code = "\n".join(smoke_lines)
    check("冒烟: 兜底清理走 skills_dir()（不再 `parents[N] / \"skills\"`）",
          "shutil.rmtree(skills_dir() / SMOKE_SKILL" in smoke_code
          and not re.search(r'parents\[\d+\] / "skills"', smoke_code))

    # ⑤ .dockerignore 必须排除 app/scripts 与 app/local_services（否则镜像白胖且带本机 clone）
    ignore = (REPO / ".dockerignore").read_text(encoding="utf-8")
    for name in ("scripts", "local_services"):
        check(f"dockerignore: 排除 backend-py/app/{name}", f"backend-py/app/{name}" in ignore)

    # ⑥ 反套套逻辑：判定器必须会失败（造一条假路径喂给同一判据）
    def exists_and_in_app(p: Path) -> bool:
        return p.is_dir() and APP in p.parents

    check("反套套逻辑: 伪造 app/nope/ 必须判不存在", not exists_and_in_app(APP / "nope"))
    check("反套套逻辑: 正确路径必须放行", exists_and_in_app(APP / "scripts"))

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
