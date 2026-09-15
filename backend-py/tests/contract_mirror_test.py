"""S7 自检：**共享契约镜像**（`frontend/app/types/contracts.ts` ↔ Python 后端）不漂移。

为什么需要（2026-09-15 补）：那份文件自己写着「字段**权威在后端**，后端改字段时**必须同步这里**
（这条不变量和「画风词表两处同步」同类）」—— 但**没有任何守卫看着它** ✗。「画风词表」那条早在
`route_parity_test.py` 的镜像常量守卫覆盖之内，而契约镜像当时只靠人记性 ⇒ 后端改字段名、
前端 DTO 没跟 ⇒ **前端静默读 `undefined`**（不报错、只是值为空/功能失灵）。

判据（保守、可解释）：
* 从 `contracts.ts` 的 interface 声明里抽字段名（先剥注释，避免把注释里的词当声明）；
* 字段名本身、或其 **camelCase / PascalCase 孪生**（后端 `toSnakeCase` 是按列名生成的 ⇒
  源码里常只有 camelCase）出现在 `backend-py/app/**/*.py` 里 ⇒ 算「对得上」；
* 基线（2026-09-15 实测）：**31 个字段 / 5 个 interface ⇒ 0 未匹配**。**非零即真回归**：
  要么后端真的改了字段（那就同步前端镜像），要么镜像里新加了后端不存在的字段（那就是幻觉字段）。

另锁两条该文件自己写的规则：① 只允许 `export type/interface`（无运行时逻辑、无第三方 import）；
② nuxt 的 `~contracts` 别名仍在（别名丢了前端会构建失败）。

运行::

    ./.venv/Scripts/python.exe tests/contract_mirror_test.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONTRACTS = REPO / "frontend" / "app" / "types" / "contracts.ts"
NUXT_CONFIG = REPO / "frontend" / "nuxt.config.ts"
APP = REPO / "backend-py" / "app"

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def parse_contracts() -> dict[str, str]:
    """``字段名 -> 所属 interface``（剥注释；只看 interface 体里的声明行）。"""
    text = CONTRACTS.read_text(encoding="utf-8")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)

    fields: dict[str, str] = {}
    current = ""
    for line in text.splitlines():
        match = re.match(r"\s*export interface (\w+)", line)
        if match:
            current = match.group(1)
            continue
        match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\??\s*:", line)
        if match and current:
            fields[match.group(1)] = current
    return fields


def twin(name: str) -> set[str]:
    """字段名 + 它的 camel / Pascal 孪生（后端源码里常只有驼峰形态）。"""
    parts = name.split("_")
    camel = parts[0] + "".join(p.title() for p in parts[1:])
    return {name, camel, "".join(p.title() for p in parts)}


def backend_sources() -> dict[str, str]:
    return {p.as_posix(): p.read_text(encoding="utf-8", errors="replace")
            for p in APP.rglob("*.py")}


def found_in(name: str, sources: dict[str, str]) -> list[str]:
    variants = twin(name)
    return [rel for rel, src in sources.items() if any(v in src for v in variants)]


def main() -> int:
    fields = parse_contracts()
    sources = backend_sources()
    interfaces = sorted(set(fields.values()))

    # ① 防「解析器坏了 ⇒ 0 字段 ⇒ 假绿」
    check("解析: contracts.ts 里读到 ≥25 个字段、≥5 个 interface（基线 31/5）",
          len(fields) >= 25 and len(interfaces) >= 5, f"{len(fields)}/{len(interfaces)}")

    # ② ⭐ 核心：每个镜像字段都能在后端源码里找到
    missing = {name: owner for name, owner in fields.items() if not found_in(name, sources)}
    check("⭐ 镜像: 每个契约字段都能在后端源码里找到（snake/camel 孪生均算；基线 0 未匹配）",
          not missing, sorted(missing.items())[:8])

    # ③ 反套套逻辑：匹配器必须**会失败**
    check("反套套逻辑: 故意造一个字段名 -> 必须判「找不到」",
          not found_in("__definitely_not_a_backend_field__", sources))

    # ④ 该文件自己写的规则 1：只允许 export type / interface（无运行时逻辑、无第三方 import）
    raw = CONTRACTS.read_text(encoding="utf-8")
    body = re.sub(r"/\*.*?\*/", "", raw, flags=re.S)
    body = re.sub(r"//[^\n]*", "", body)
    # 顶层（顶格）语句只允许 `export …`；interface 体内的字段是缩进的，天然不算
    top = [line.strip() for line in body.splitlines()
           if line.strip() and not line.startswith((" ", "\t"))]
    bad_top = [line for line in top
               if not line.startswith(("export ", "|", "}", "];"))]
    check("规则1: 文件顶层只含 export type/interface（无运行时逻辑 / 无 import）",
          not bad_top and "import " not in body, bad_top[:3])

    # ⑤ nuxt 别名 `~contracts` 仍在（丢了前端直接构建失败）
    config_text = NUXT_CONFIG.read_text(encoding="utf-8") if NUXT_CONFIG.is_file() else ""
    check("规则2: nuxt.config.ts 里 `~contracts` 别名仍在", "~contracts" in config_text, NUXT_CONFIG)

    # ⑥ 前端 dev 代理必须指向 `backendTarget`（**不许硬编码端口**）
    #    ⚠️ 2026-09-15 真踩过：变量切到 5790，而 `vite.server.proxy` 里仍写死 5789（已删的 Node）
    #    ⇒ `npm run dev` 的 /api、/static 全打到空端口，且**不报错**、只是请求失败 ✗
    code = re.sub(r"/\*.*?\*/", "", config_text, flags=re.S)
    # ⚠️ 只切「不在 `:` 之后的 `//`」—— 否则会把 URL 的 `http://` 也切掉
    #    （我第一版就这么错：`http://localhost:5790` 变成 `http:` ⇒ 正则扑空 ✗）
    code = re.sub(r"(?<!:)//[^\n]*", "", code)
    target = re.search(r"backendTarget\s*=\s*process\.env\.NUXT_API_TARGET\s*\|\|\s*'([^']+)'", code)
    check("规则3: dev 代理默认目标 = Python 后端端口 5790",
          bool(target) and target.group(1).endswith(":5790"),
          target.group(1) if target else "未找到 backendTarget 定义")
    idx = code.find("proxy:")
    window = code[idx: idx + 400] if idx >= 0 else ""
    check("规则3: `vite.server.proxy` 用 `backendTarget`（无硬编码端口）",
          window.count("backendTarget") >= 2 and not re.search(r"localhost:\d{4}", window),
          window[:140] or "未找到 proxy 块")
    check("规则3: nuxt.config.ts 的**代码**里不再出现已删除的 5789", "5789" not in code)

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"（契约字段 {len(fields)} 个 / interface {len(interfaces)} 个；后端源码 {len(sources)} 个 .py）")
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
