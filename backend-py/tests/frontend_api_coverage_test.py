"""S7 自检：**前端调用点 ↔ 后端路由**覆盖 —— 删 `backend/` 之后唯一后端的安全网。

为什么需要（2026-09-15 删库当天补）：路由守卫（``route_parity_test.py``）证明的是
「**Node 曾有过的 224 条路径**都被覆盖」，它**看不见前端**。删库后 Python 是唯一服务方 ⇒
只要前端调了一个 Python 没实现的路径，就是**线上 501**（用户点一下按钮就报错）。
本自检把前端源码里的调用点抽出来，逐条对 ``app.openapi()`` 的路由表**反查**，锁住这条不变量。

提取规则（每条都是踩过坑之后定下来的）：
* ``api.get/post/put/patch/del('<path>')``（含泛型 ``api.get<T>('/x')``）⇒ 直接取方法 + 路径；
* ``fetch('/api/v1/…', { method: 'POST' })`` ⇒ **方法必须从调用现场读** —— 先按 GET 处理时，
  4 条 ``POST`` 动作端点被误报成「缺失」✗；
* 路径里 ``${...}`` 归一化成 ``{}``（后端模板是 ``{episode_id}``）；
* 含空格 / 残留 ``$`` 或 ``{`` 的串（模板三元表达式）直接丢弃 —— 它们不可能是真路径。
* **注释先被遮蔽成等长空格再扫描**（`//`、`/* */`、HTML `<!-- -->`，且**认引号**）——
  2026-09-22 实测踩到：`useApi.ts` 里一条"这里曾经写错成 …"的注释写了调用形状 ⇒
  被当成真调用点，反查后端**当然找不到** ⇒ 本套当场红。当时的处置是「写注释别用调用形状」，
  但那是**纪律代替机制**：任何人顺手在注释里记一句老路径就红，且红得莫名其妙。
  2026-11-08 改成遮蔽（与 `check-parity.mjs::maskComments` 同一套口径）。

  ⚠️ 遮蔽器**必须认引号**，否则更糟：正则看不见引号，字面量里的注释起始符会把后面的真代码
  一起吃掉（`'https://…'`、`'src/**/*.ts'`）⇒ 那段内容对判据完全隐身（**假绿**，无任何提示）。
  `.vue` 还要分块：模板区只遮 HTML 注释、**不跑 JS 引号状态机** —— 模板文本里一个 `Don't`
  的撇号会让状态机一路吃到文件尾，把真调用点也遮掉。

2026-11-08 收口三处盲区（都是「自检看不见」而不是「代码有错」）：
* **``patch`` 从未被扫**：``_CALL_RE`` 只列了 ``get|post|put|del``，而 ``useApi.updateEntry`` 用的是
  ``api.patch`` ⇒ 那条调用点零覆盖，后端真删掉 ``PATCH /entries/{id}`` 本套仍全绿 ✗；
* **只扫 Web 端**：``FRONT`` 只指 ``frontend/app``，``desktop/frontend/src`` 的调用点无任何保护
  （两端是复制关系，桌面端改动同样会 501）；
* **查询串被整体剥掉**：``?a=b`` 连同参数名一起丢弃 ⇒ 前端写 ``?minSamples=`` 而后端声明
  ``min_samples``（无 alias）时字段**静默失效**，比 404 更难查。现在参数名也核：
  前端字面量里的每个 query 键，必须是后端该 (method, path) 上声明过的参数名之一。

基线（2026-09-15 实测）：**166 个调用点 ｜ 166 已注册 ｜ 0 未匹配**。
基线（2026-11-08 双端合扫）：见运行输出尾部（两端各扫一遍，调用点按 (method,path) 去重合并）。

运行::

    ./.venv/Scripts/python.exe tests/frontend_api_coverage_test.py
"""
from __future__ import annotations

import ast
import os
import re
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="frontend_cov_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
#: 两端都要扫：桌面端 `desktop/frontend/src` 是从 `frontend/app` 复制的，改了同一个
#: `useApi.ts` 却打向不存在的路由，症状与 Web 端一模一样（点一下 501）。
FRONT_ROOTS = [REPO / "frontend" / "app", REPO / "desktop" / "frontend" / "src"]

_CALL_RE = re.compile(r"api\.(get|post|put|patch|del)\s*(?:<[^>]*>)?\s*\(\s*[`'\"]([^`'\"]+)[`'\"]")
_ABS_RE = re.compile(r"[`'\"](/api/v1/[^`'\"]+)[`'\"]")
_METHOD_RE = re.compile(r"method:\s*['\"]([A-Za-z]+)['\"]")
_TEMPLATE_PARAM = re.compile(r"\$\{[^}]*\}")
#: 被**嵌套反引号**截断的模板尾：`api.get(`/images${q.size ? `?size=${q.size}` : ''}`)` 里
#: `_CALL_RE` 的 `[^`'"]+` 停在内层反引号 ⇒ 拿到的只是 `/images${q.size ?`。这种尾巴接的
#: 一定是查询串（或更深一层模板），静态路径主体只到 `${` 为止。
_TEMPLATE_TAIL = re.compile(r"\$\{[^}]*$")
#: 查询串里的键名；`${...}` 先剔掉（它会引入 `?`/`&` 之外的噪声）。
_QUERY_KEY = re.compile(r"[?&]([A-Za-z_][A-Za-z0-9_]*)(?==|&|$)")
_METHODS = {"get": "GET", "post": "POST", "put": "PUT", "patch": "PATCH", "del": "DELETE"}

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def normalize(raw: str) -> str | None:
    """前端路径字面量 → 后端路由模板形态；不可能是真路径的返回 ``None``。

    2026-11-08 补两处（此前这 19 条 ×2 端调用点**整条被丢弃**：既不判红也不判绿 = 完全隐身）：

    * 嵌套模板被截断的尾巴切掉，只核静态路径主体 —— ``api.get(`/images${q.size ? `?size=${q.size}` : ''}`)``
      这种形态盘上有 19 条（`/images`、`/character-library`、`/knowledge-loop/entries/${encodeURIComponent(id)}`
      …），路径主体明明可以核，却因为尾巴上多了一层反引号被整条扔掉 ✗；
    * 切尾之后仍残留裸 ``{`` ``}`` 的（``${fn({a: 1})}`` 这类含对象字面量的模板）**仍旧丢弃** ——
      路径本身是运行时拼的，静态核不了，硬把 ``/x{})}`` 拿去比对只会制造假红。
    """
    path = raw.split("?")[0]
    path = _TEMPLATE_PARAM.sub("{}", path)
    path = _TEMPLATE_TAIL.sub("", path)
    bare = path.replace("{}", "")
    if not path.startswith("/") or " " in bare or "$" in bare or "{" in bare or "}" in bare:
        return None
    return path


def query_keys(raw: str) -> set[str]:
    """从路径字面量里抽出查询键名（`?a=b&c=d` ⇒ `{a, c}`）。"""
    bare = _TEMPLATE_PARAM.sub("", raw)
    return set(_QUERY_KEY.findall(bare))


def _blank(seg: str) -> str:
    """遮成**等长空格**（换行保留）—— 不删字，出错时报的行号/列号才不会错位。"""
    return re.sub(r"[^\n]", " ", seg)


def mask_comments(code: str, inline_slash: bool = True) -> str:
    """把 JS/TS 注释遮蔽成等长空格，**并且认引号**（见模块 docstring 那两段 ⚠️）。

    ``inline_slash=True`` 时行内 ``//`` 也算注释（扫描调用点要的就是这个：一行尾巴上
    ``api.get('/x')  // 以前是 '/y'`` 里那条旧路径不该被当真）；``False`` 时只遮行首注释。
    """
    out: list[str] = []
    n = len(code)
    i = 0
    quote: str | None = None
    home = True  # 当前是否位于「行首」（只剩空白）
    while i < n:
        c = code[i]
        nxt = code[i + 1] if i + 1 < n else ""
        if quote:
            out.append(c)
            if c == "\\":
                out.append(nxt)
                i += 2
                continue
            if c == quote:
                quote = None
                home = False
            i += 1
            continue
        if c == "\n":
            out.append(c)
            i += 1
            home = True
            continue
        if c in " \t":
            out.append(c)
            i += 1
            continue
        if code.startswith("<!--", i):
            end = code.find("-->", i + 4)
            to = n if end < 0 else end + 3
            seg = _blank(code[i:to])
            out.append(seg)
            home = "\n" in seg
            i = to
            continue
        if c == "/" and nxt == "*":
            end = code.find("*/", i + 2)
            to = n if end < 0 else end + 2
            seg = _blank(code[i:to])
            out.append(seg)
            home = "\n" in seg
            i = to
            continue
        if c == "/" and nxt == "/" and (inline_slash or home):
            to = code.find("\n", i)
            to = n if to < 0 else to
            out.append(_blank(code[i:to]))
            i = to
            continue
        if c in "\"'`":
            quote = c
            out.append(c)
            i += 1
            continue
        out.append(c)
        i += 1
        home = False
    return "".join(out)


def _mask_html_comments(text: str) -> str:
    """只遮 ``<!-- -->``（模板/样式区用：那里不跑 JS 引号状态机）。"""
    out: list[str] = []
    i = 0
    while True:
        start = text.find("<!--", i)
        if start < 0:
            out.append(text[i:])
            return "".join(out)
        out.append(text[i:start])
        end = text.find("-->", start + 4)
        to = len(text) if end < 0 else end + 3
        out.append(_blank(text[start:to]))
        i = to


#: `<script …>` … `</script>` 块（`.vue` 里只有这些区是 JS）。
_SCRIPT_BLOCK = re.compile(r"(<script\b[^>]*>)(.*?)(</script>)", re.S)


def mask_source(text: str, suffix: str) -> str:
    """按文件类型遮蔽注释：`.ts/.js` 整体走 JS 规则；`.vue` **分块**（模板区只遮 HTML 注释）。"""
    if suffix != ".vue":
        return mask_comments(text, True)
    parts: list[str] = []
    last = 0
    for match in _SCRIPT_BLOCK.finditer(text):
        parts.append(_mask_html_comments(text[last: match.start()]))
        parts.append(match.group(1) + mask_comments(match.group(2), True) + match.group(3))
        last = match.end()
    parts.append(_mask_html_comments(text[last:]))
    return "".join(parts)


def scan_source(text: str, suffix: str) -> list[tuple[str, str, bool]]:
    """从一个文件源码里抽出调用点 ``[(method, 原始路径字面量, 是否 fetch 绝对形)]``。"""
    code = mask_source(text, suffix)
    found: list[tuple[str, str, bool]] = []
    for match in _CALL_RE.finditer(code):
        found.append((_METHODS[match.group(1)], match.group(2), False))
    for match in _ABS_RE.finditer(code):
        raw = match.group(1)[len("/api/v1"):]
        tail = code[match.end(): match.end() + 240]
        method = _METHOD_RE.search(tail)
        found.append((method.group(1).upper() if method else "GET", raw, True))
    return found


def collect_calls() -> tuple[dict[tuple[str, str], set[str]], int, int, dict[tuple[str, str], set[tuple[str, str]]]]:
    """返回 ``(调用点, 文件数, 来自 fetch 字面量的条数, query 键用法)``。"""
    calls: dict[tuple[str, str], set[str]] = {}
    query_uses: dict[tuple[str, str], set[tuple[str, str]]] = {}
    absolute = 0
    files = [p for root in FRONT_ROOTS if root.is_dir()
             for p in root.rglob("*")
             if p.is_file() and p.suffix in (".ts", ".vue", ".js")]
    for file in files:
        try:
            text = file.read_text(encoding="utf-8")
        except OSError:
            continue
        rel = file.relative_to(REPO).as_posix()
        for method, raw, is_absolute in scan_source(text, file.suffix):
            if is_absolute:
                if not normalize(raw):
                    continue
                absolute += 1
            path = normalize(raw)
            if not path:
                continue
            calls.setdefault((method, path), set()).add(rel)
            for key in query_keys(raw):
                query_uses.setdefault((method, path), set()).add((key, rel))
    return calls, len(files), absolute, query_uses


def backend_patterns() -> list[tuple[str, re.Pattern[str]]]:
    patterns: list[tuple[str, re.Pattern[str]]] = []
    for route, ops in app.openapi()["paths"].items():
        for method in ops:
            body = re.sub(r"\{[^}]+\}", "[^/]+",
                          re.escape(route).replace(r"\{", "{").replace(r"\}", "}"))
            patterns.append((method.upper(), re.compile("^" + body + "$")))
    return patterns


def backend_param_names() -> dict[tuple[str, str], set[str]]:
    """``(method, 归一化路径) -> 该端点声明过的参数名``（path/query/header/cookie 全算）。

    路径归一化成与前端字面量同形（``/knowledge-loop/entries/{}``），否则两边对不上。
    名字取的是 openapi 里的对外名 —— FastAPI 的 ``Query(alias="minSamples")`` 报的就是
    ``minSamples`` ⇒ 能和前端字面量直接比。请求体参数在 ``requestBody`` 里，不是本条的范围。

    ⚠️ openapi **只报声明过的**：本仓有十几处是手工 ``request.query_params.get("k")``
    （`asset-versions` / `export` / `images` / `traces` …），openapi 里这些参数**一个都不出现** ⇒
    只认 openapi 会把**真在用**的参数判成「未声明」（假红 ✗，2026-11-08 实测 6 条）。
    故再并进 ``manual_query_names()`` 从源码 AST 抽出来的键。
    """
    table: dict[tuple[str, str], set[str]] = {}
    for route, ops in app.openapi()["paths"].items():
        key_path = _norm_path(route)
        for method, spec in ops.items():
            names = {p["name"] for p in spec.get("parameters", []) or []}
            table.setdefault((method.upper(), key_path), set()).update(names)
    for key, names in manual_query_names().items():
        table.setdefault(key, set()).update(names)
    return table


ROUTERS = Path(__file__).resolve().parents[1] / "app" / "routers"
_HTTP_METHODS = {"get", "post", "put", "patch", "delete", "head", "options"}


def _norm_path(route: str) -> str:
    """与前端字面量同形：去掉 `/api/v1`、`{xxx}`→`{}`、去尾斜杠（根路径除外）。"""
    path = re.sub(r"\{[^}]+\}", "{}", route).replace("/api/v1", "", 1)
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    return path


def _router_prefix(tree: ast.Module) -> str:
    """模块级 ``router = APIRouter(prefix="/api/v1/xxx")`` 的前缀。"""
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
            continue
        func = node.value.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        if name != "APIRouter":
            continue
        for kw in node.value.keywords:
            if kw.arg == "prefix" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                return kw.value.value
    return ""


def _decorated_routes(node: ast.AST) -> list[tuple[str, str]]:
    """``@router.get("/x")`` ⇒ ``[("GET", "/x")]``。"""
    out: list[tuple[str, str]] = []
    for dec in getattr(node, "decorator_list", []) or []:
        if not (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)):
            continue
        method = dec.func.attr.lower()
        if method not in _HTTP_METHODS:
            continue
        sub = dec.args[0] if dec.args else None
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            out.append((method.upper(), sub.value))
    return out


def _query_keys_of(node: ast.AST) -> set[str]:
    """函数体里手工读的 query 键：``request.query_params.get("k")``、``q["k"]``（含 `q = …query_params`）。"""
    holders: set[str] = set()
    for sub in ast.walk(node):
        if (isinstance(sub, ast.Assign) and isinstance(sub.value, ast.Attribute)
                and sub.value.attr == "query_params"):
            holders |= {t.id for t in sub.targets if isinstance(t, ast.Name)}

    def is_query_params(expr: ast.AST) -> bool:
        if isinstance(expr, ast.Attribute) and expr.attr == "query_params":
            return True
        return isinstance(expr, ast.Name) and expr.id in holders

    keys: set[str] = set()
    for sub in ast.walk(node):
        if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "get" and sub.args and is_query_params(sub.func.value)):
            arg = sub.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                keys.add(arg.value)
        elif isinstance(sub, ast.Subscript) and is_query_params(sub.value):
            sl = sub.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                keys.add(sl.value)
    return keys


def manual_query_names() -> dict[tuple[str, str], set[str]]:
    """从路由源码 AST 抽出「手工读的 query 参数」⇒ ``(method, 路径) -> 键集合``。

    为什么要 AST 而不是正则：要的是「**哪个端点**读了哪个键」，正则拿不到这层归属，
    只能全文件混成一个集合 ⇒ 把 A 端点读的键算到 B 端点上，判据就白做了。
    """
    out: dict[tuple[str, str], set[str]] = {}
    for file in sorted(ROUTERS.glob("*.py")):
        try:
            tree = ast.parse(file.read_text(encoding="utf-8"))
        except SyntaxError:  # 并行工作流的半成品也不该让本套崩
            continue
        prefix = _router_prefix(tree)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            routes = _decorated_routes(node)
            if not routes:
                continue
            keys = _query_keys_of(node)
            if not keys:
                continue
            for method, sub in routes:
                out.setdefault((method, _norm_path(prefix + sub)), set()).update(keys)
    return out


def matched(method: str, full: str, patterns: list[tuple[str, re.Pattern[str]]]) -> bool:
    return any(m == method and rx.match(full) for m, rx in patterns)


def unknown_query_keys(
    params: dict[tuple[str, str], set[str]],
    query_uses: dict[tuple[str, str], set[tuple[str, str]]],
) -> list[tuple[str, str, str, str]]:
    """前端字面量里写了、后端该端点却没声明的查询键 ⇒ 该字段**静默失效**。

    只在「该 (method, path) 确实是后端端点」时核 —— 端点不存在的已由 ③ 报过，别重复喊。
    """
    return [
        (method, path, key, rel)
        for (method, path), keys in sorted(query_uses.items())
        if (method, path) in params
        for key, rel in sorted(keys)
        if key not in params[(method, path)]
    ]


def main() -> int:
    calls, file_count, absolute, query_uses = collect_calls()
    patterns = backend_patterns()
    params = backend_param_names()

    unmatched = [(method, "/api/v1" + raw, sorted(where)[0])
                 for (method, raw), where in sorted(calls.items())
                 if not matched(method, "/api/v1" + raw, patterns)]

    #: 前端字面量里写了、后端该端点却没声明的查询键 ⇒ 字段静默失效（比 404 更难查）。
    unknown_query = unknown_query_keys(params, query_uses)


    # ① 防「提取规则坏了 ⇒ 0 个调用点 ⇒ 假绿」
    check("提取: 前端调用点数量正常（≥100）", len(calls) >= 100, len(calls))
    # ② 防「只扫到 useApi 一种形态」（fetch 绝对字面量那条路也得真在用）
    check("提取: 也扫到了 `fetch('/api/v1/…')` 形态（≥1）", absolute >= 1, absolute)
    check("提取: 两端都扫到了（文件数 ≥ 100，Web + 桌面）", file_count >= 100, file_count)
    check("提取: 后端路由表规模正常（≥200 组合）", len(patterns) >= 200, len(patterns))
    check("提取: 后端参数表规模正常（≥200 组合）", len(params) >= 200, len(params))
    # ②′ 防「patch 那条路又掉了」：`updateEntry` 是唯一的 PATCH 调用点
    check("提取: PATCH 调用点被扫到（≥1）", any(m == "PATCH" for m, _ in calls),
          [p for m, p in calls if m == "PATCH"])
    # ③ ⭐ 核心不变量：前端每一个调用点，后端都实现了
    check("⭐ 覆盖: 前端**全部**调用点都能在后端路由表里找到（0 未匹配）",
          not unmatched, unmatched[:6])
    # ③′ 查询键必须真的是后端声明的参数名（写错/改名的字段会**静默失效**）
    check("⭐ 参数名: 前端字面量里的每个 query 键都在后端声明过",
          not unknown_query, unknown_query[:6])
    # ④ 反套套逻辑：匹配器必须**会拒绝**（否则上面那条恒真、毫无意义）
    check("反套套逻辑: 故意造一条不存在的路径 -> 匹配器必须拒绝",
          not matched("GET", "/api/v1/__definitely_not_a_route__", patterns)
          and not matched("POST", "/api/v1/storyboards/1/__nope__", patterns))
    check("反套套逻辑: 故意造一个未声明的 query 键 -> 参数核对必须拒绝",
          bool(unknown_query_keys({("GET", "/probe"): {"declared"}},
                                  {("GET", "/probe"): {("__not_declared__", "probe.ts")}}))
          and not unknown_query_keys({("GET", "/probe"): {"declared"}},
                                     {("GET", "/probe"): {("declared", "probe.ts")}}))
    # ⑤ 注释遮蔽器自守：两个方向都要证明 —— 注释里的形状**被拒**、引号里的注释起始符**不吃掉**真代码。
    #    夹具先过一遍「不遮蔽」的正则，确认它**确实**是会被扫到的形状（否则这条判据恒真 = 白写）。
    probe = (
        "// api.get('/__c_line__')\n"
        "/* api.post('/__c_block__') */\n"
        "<!-- api.del('/__c_html__') -->\n"
        "const u = 'https://example.com//a'; api.get('/__keep_url__')\n"
        "const g = 'src/**/*.vue'; api.patch('/__keep_glob__')\n"
        "const t = `a // b`; fetch('/api/v1/__keep_fetch__', { method: 'POST' })\n"
    )
    naive = ({m.group(2) for m in _CALL_RE.finditer(probe)}
             | {r[len("/api/v1"):] for r in _ABS_RE.findall(probe)})
    masked = {raw for _, raw, _ in scan_source(probe, ".ts")}
    check("注释遮蔽: 注释里的调用形状不被当成调用点（且引号内的 `//`、`/*` 不吃真代码）",
          {"/__c_line__", "/__c_block__", "/__c_html__"} <= naive
          and {"/__keep_url__", "/__keep_glob__", "/__keep_fetch__"} <= masked
          and not {"/__c_line__", "/__c_block__", "/__c_html__"} & masked,
          {"naive": sorted(naive), "masked": sorted(masked)})
    # ⑤′ `.vue` 必须分块：模板里的撇号（`Don't`）不许让状态机吃到文件尾
    vue_probe = (
        "<template>\n  <p>Don't stop</p>\n  <!-- api.get('/__vue_html__') -->\n</template>\n"
        '<script setup lang="ts">\n  // api.get(\'/__vue_line__\')\n  api.get(\'/__vue_real__\')\n</script>\n'
    )
    vue_found = {raw for _, raw, _ in scan_source(vue_probe, ".vue")}
    check("注释遮蔽: `.vue` 分块（模板撇号不吃掉 script 里的真调用）",
          vue_found == {"/__vue_real__"}, sorted(vue_found))
    # ⑥ 截断尾（嵌套反引号）不许整条丢弃：路径主体照核，且改坏静态段必须报。
    #    双向夹具：一条盘上真实形态（`/knowledge-loop/entries/…`）、一条纯静态（`/images`）。
    tail_id = normalize("/knowledge-loop/entries/${encodeURIComponent(id)}${includeDeleted ?")
    tail_plain = normalize("/images${q.size ?")
    check("截断尾: 嵌套模板的静态路径主体被收回核（改坏静态段必须报）",
          tail_id == "/knowledge-loop/entries/{}" and tail_plain == "/images"
          and matched("GET", "/api/v1" + tail_id, patterns)
          and not matched("GET", "/api/v1/knowledge-loop/entrys/{}", patterns),
          {"tail_id": tail_id, "tail_plain": tail_plain})
    # ⑥′ 切尾仍留有裸括号的（`${fn({a: 1})}`）必须**继续丢弃**，不许造出 `/x{})}` 这种假路径；
    #     而「三元里带空格」的（`${a ? 'p q' : ''}`）只核静态主体 `/x` —— 原先因空格整条丢弃。
    check("截断尾: 含对象字面量的模板仍旧丢弃，带空格的三元只核静态主体",
          normalize("/x${fn({a: 1})}") is None
          and normalize("/x${a ? 'p q' : ''}") == "/x")

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"（扫描 {file_count} 个前端文件；调用点 {len(calls)} 个；"
          f"后端 {len(patterns)} 条 (method,path)；有 query 键的调用 {len(query_uses)} 个）")
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
