"""自检：**全仓生成物落点只有一棵树**（2026-11-07 ✓）。

## 为什么要有这一套

同类毛病在本仓犯过三次，每次症状都是「跑完了，文件不知道在哪」：

1. 引擎四个落盘入口各自 `tempfile.mkdtemp` ⇒ 每跑一次一个新目录，图散在 `%TEMP%` ✓✗
   （已收口到 :mod:`app.services.engine.output_paths` ✓，见 `engine_output_paths_test`）；
2. `video_interpolate` / `ffmpeg_compose` 在 `get_storage_root()` 后面**又拼了一次
   `"static"`** ✓✗ ⇒ 实落 `<dataRoot>/static/static/xxx`，而回执 URL 指
   `<dataRoot>/static/xxx` ⇒ **插帧成功的片子点开 404**（文件在，链接错）；
3. `local_services/h3/server.py` 默认落 `<代码目录>/outputs` ✓✗ ⇒ 跑一次在源码树里
   长出一个 `outputs/`，重装/打包/切分支就没了，主后端也找不到。

共同点：**落点分散在各自模块里，没人对账**。本文件把「散」变成「登记制」✓：

- `get_storage_root()` 后面只许跟**登记过**的子目录名 ✓（新增一处 ⇒ 本文件红 ⇒ 逼登记）；
- `get_data_root()` 同理 ✓；
- 任何字符串常量都**不许以 `static/` 开头** ✗（那是把 `static` 拼第二次的信号 ✓）；
- 不许拿 `__file__` 去拼 `output/outputs/tmp/temp` ✗（代码目录里的产物 ✓）；
- `backend-py/output/` **不存在** ✓（2026-11-07 退役：实验产物改落
  `<dataRoot>/experiments/<实验名>/` ✓、成品仍落 `<dataRoot>/static/` ✓ —— 代码目录里不留产物区 ✓✗）；
- `backend-py/tmp/` 里**不许出现媒体产物** ✗（那是现场证据区：全量日志 `run_all_*.log` 与
  一次性脚本；真产物落进去会随 `tmp/` 一起被当垃圾清掉 ✓✗ —— 2026-11-07 实测里面躺着
  `out_seedvr2/video_seed20260101.mp4` 与 `probe_image_out/sdxl-*.png`，已收进
  `<dataRoot>/experiments/{seedvr2,sdxl-probe}/`）✓。

⚠️ 本文件**不跑** ffmpeg / 不写业务数据 ✓：只做 AST 与一次 import ✓。
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
APP = BACKEND_PY / "app"
sys.path.insert(0, str(BACKEND_PY))

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


#: ★ 登记表 —— **`<dataRoot>/static/` 下允许出现的子目录** ✓
#: （新增落点 ⇒ 这里加名字 + `app/core/config.py` 的 docstring 表里加一行 ✓；
#: 两者都做才算"统一管理"，只改一处本文件就红 ✓）
STATIC_DIRS = {
    "engine",        # engine.output_paths（<kind>/ 与 <kind>-<id>/ 与其下 refs-*）
    "images",        # file_storage 家族
    "uploads",       # routers/upload.py
    "videos",        # webhooks / video_generation
    "audio",         # tts_generation / ai_voices / segment_audio
    "voices",        # routers/ai_voices.py
    "frames",        # frame_extractor
    "composed",      # ffmpeg_compose 成片
    "mixed",         # ffmpeg_compose 混音中间音轨（留作复盘）
    "subtitles",     # ffmpeg_compose 字幕
    "merged",        # ffmpeg_merge 拼接成片
    # ⚠️ 这里**没有** `temp` ✓：`ffmpeg_merge` 的 concat 清单是**一次调用的中转件**，
    # 2026-11-07 改用 `tempfile.TemporaryDirectory()` ⇒ 不再落 `static/temp` ✓。
    # `static/` 是**产物区**（要发 URL、随数据根备份）⇒ 中转件不住这里；谁再往
    # `get_storage_root() / "temp"` 写，① 会当场报「多出未登记」✓。
    "interpolate",   # video_interpolate
    "jianying",      # jianying_draft
    "h3",            # local_services/h3/server.py
    "comfyui",       # comfyui.download_file(url, "comfyui") 远端产物
    "grid-cells",    # grid_split 宫格切图
}

#: ★ 登记表 —— **`<dataRoot>/` 下允许出现的子目录** ✓
DATA_DIRS = {
    "db",             # config.get_db_dir
    "knowledge",      # knowledge_ledger / outcome_index / skill_usage / training_registry
    "repair",         # routers/repair.py
    "lora-train",     # lora_train.config.data_dir
    "engine-dryrun",  # engine.output_paths.dryrun_root
    "materialized",   # engine.materialize
    "models",         # local_model_scan
    "hf-cache",       # engine.tokenizers_tuning.default_cache_dir（它被设成 HF_HOME）
    "experiments",    # 实验/探针产物（不上线、不发 URL）—— `_bench/*.py` 探针的 OUT_DIR
    "canvas",         # canvas_plan_store.plans_dir（<dataRoot>/canvas/plans 画布草案）
}

#: 写文件时**不许**出现在 `__file__` 拼路径里的目录名 ✓（代码目录里的产物 ✓✗）
FORBIDDEN_CODE_DIRS = {"output", "outputs", "tmp", "temp", "out", "results"}

#: 什么算「媒体产物」✓（`tmp/` 里出现任何一个 ⇒ 那是产物跑错了区 ✓✗）
MEDIA_SUFFIXES = {
    ".mp4", ".mov", ".mkv", ".webm", ".avi",
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp",
    ".wav", ".mp3", ".flac", ".m4a", ".ogg",
}


def _iter_app_files() -> list[Path]:
    """`app/` 下的 py 文件 ✓（`app/skills/` 是随技能分发的独立工具 ✗ —— 它们的输出由 CLI 参数定 ✓）"""
    return [p for p in APP.rglob("*.py") if "skills" not in p.relative_to(APP).parts
            and "__pycache__" not in p.parts]


def _func_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_root_call(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Call) and _func_name(node.func) == name


def _is_root_expr(node: ast.AST, name: str) -> bool:
    """`get_xxx_root()` 或 `Path(get_xxx_root())` ✓（两种写法本仓都有 ✓）"""
    if _is_root_call(node, name):
        return True
    return (isinstance(node, ast.Call) and _func_name(node.func) == "Path"
            and len(node.args) == 1 and _is_root_call(node.args[0], name))


def _first_dir_of(root_name: str) -> dict[str, str]:
    """扫出 `X(get_root()) / "<字面量>"` 的**首段**目录名 ✓ ⇒ {目录名: 出处}"""
    found: dict[str, str] = {}
    for path in _iter_app_files():
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError:  # pragma: no cover
            continue
        rel = str(path.relative_to(BACKEND_PY))
        for node in ast.walk(tree):
            if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
                continue
            if not _is_root_expr(node.left, root_name):
                continue
            if isinstance(node.right, ast.Constant) and isinstance(node.right.value, str):
                found.setdefault(node.right.value, f"{rel}:{node.lineno}")
    return found


def case_static_dirs_registered() -> None:
    """`get_storage_root()` 字面量 + `get_absolute_path("x")` + 回执里的 `static/x/` 三路合起来对账 ✓。

    ⚠️ **不判「登记了但没人写」** ✗：多数落点是**参数化**的（`file_storage.sub_dir`、
    `_MIX_SUBDIR`）✓，AST 在调用点只看得到变量名 ✓ ⇒ 那一半只能靠人核 ✓；
    而「多出一个没登记的」是可判的 ✓，也正是最该拦的方向 ✓（偷偷加落点 ✓✗）。
    """
    found = _first_dir_of("get_storage_root")
    for path in _iter_app_files():
        rel = str(path.relative_to(BACKEND_PY))
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            # `get_absolute_path("grid-cells")` ⇒ `<staticRoot>/grid-cells` ✓
            if isinstance(node, ast.Call) and _func_name(node.func) == "get_absolute_path" \
                    and node.args and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.args[0].value, str):
                found.setdefault(node.args[0].value.strip("/").split("/")[0], f"{rel}:{node.lineno}")
            # 回执里的 `static/<dir>/…`（URL 相对数据根 ✓，等于宣告了一处落点 ✓）
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                text = node.value.strip().replace("\\", "/")
                if text.startswith("static/") and text.count("/") >= 2:
                    found.setdefault(text.split("/")[1], f"{rel}:{node.lineno}")

    extra = sorted(set(found) - STATIC_DIRS)
    check("① `<dataRoot>/static/` 下的子目录**全部登记过** ✓✗"
          "（字面量 / `get_absolute_path(\"x\")` / 回执 URL 三路合并对账 ✓；多一个 ⇒ 有人偷偷加了落点 ✓）",
          not extra, {name: found[name] for name in extra})

    bad = {name: where for name, where in found.items() if name.startswith("static")}
    check("①′ 没有第二个 `static` ✓✗（`get_storage_root()` 返回的**就是** `<dataRoot>/static` ✓ —— "
          "再拼一次会落 `static/static/xxx`、URL 指向 `static/xxx` ⇒ 文件在链接 404 ✓✗）",
          not bad, bad)

    # ⚠️ 已知盲区（2026-11-07 实测踩到）：`_first_dir_of` 只认「`get_data_root()` 调用 / 字面量」
    # 这一种拼法。`engine.tokenizers_tuning.default_cache_dir()` 写的是 `Path(root) / "hf-cache"`
    # （`root` 是局部变量）⇒ 既不会被判「未登记」，登记了也不会被判「多余」。「漏登记 + 没忽略」
    # 的 `data/hf-cache/` 就这么躲过一个多月。⇒ **本条不是全量扫描**，新增落点时别指望它一定喊。
    data_found = _first_dir_of("get_data_root")
    data_extra = sorted(set(data_found) - DATA_DIRS)
    check("② `<dataRoot>/` 下的子目录同样登记过 ✓（跟着数据根走的作品数据 ✓）",
          not data_extra, {name: data_found[name] for name in data_extra})


def case_no_static_prefix_constants() -> None:
    """两种「又拼一次 static」的写法 ✓✗：

    ⓐ 拼接右值：`Path(get_storage_root()) / "static/…"` ⇒ 实落 `static/static/…` ✓；
    ⓑ 目录名常量：`_SUBDIR = "static/mixed"` ⇒ 被 ⓐ 那种调用处拼一次就是两份 ✓
       （`static/mixed` 就是这么错的 ✓）。

    ⚠️ **放过**以 `/` 结尾的前缀（`"static/audio/"`）✓：那些是**回执 URL**（相对数据根 ✓），
    不是落点 ✓ —— 一律判红会把三十几处正确的 URL 拼装全标成错误 ✓✗。
    """
    bad: dict[str, str] = {}
    for path in _iter_app_files():
        rel = str(path.relative_to(BACKEND_PY))
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) \
                    and isinstance(node.right, ast.Constant) and isinstance(node.right.value, str):
                # ⚠️ 只看**在 `get_storage_root()` 之上**再拼 `static` 的那种 ✗；
                # `get_data_root()) / "static"` 是在**定义** storage root 本身 ✓（config / data_storage /
                # h3 的独立兜底都这么写 ✓），不是二次拼接 ✓，必须放行 ✓。
                if any(_is_root_expr(sub, "get_storage_root") for sub in ast.walk(node.left)):
                    text = node.right.value.replace("\\", "/")
                    if text == "static" or text.startswith("static/"):
                        bad[f"{rel}:{node.lineno}（拼接右值）"] = node.right.value
            elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                    and isinstance(node.value.value, str):
                text = node.value.value.replace("\\", "/")
                if text.startswith("static/") and not text.endswith("/"):
                    bad[f"{rel}:{node.lineno}（目录名常量）"] = node.value.value
    check("③ 代码里**没有**「二次 static」✓✗（拼接右值 `… / \"static/x\"` 与目录名常量 "
          "`\"static/mixed\"` 两种写法；以 `/` 结尾的回执 URL 前缀不算 ✓）",
          not bad, bad)


def case_no_code_dir_outputs() -> None:
    bad: dict[str, str] = {}
    for path in _iter_app_files():
        rel = str(path.relative_to(BACKEND_PY))
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
                continue
            # `Path(__file__)… / "outputs"`
            has_file = any(isinstance(sub, ast.Attribute) and sub.attr in {"parent", "parents"}
                           for sub in ast.walk(node.left)) or \
                any(isinstance(sub, ast.Name) and sub.id == "__file__" for sub in ast.walk(node.left))
            if not has_file:
                continue
            if isinstance(node.right, ast.Constant) and isinstance(node.right.value, str) \
                    and node.right.value in FORBIDDEN_CODE_DIRS:
                bad[f"{rel}:{node.lineno}"] = node.right.value
    check("④ 没有「拿 `__file__` 拼 output/outputs/tmp/temp」的落点 ✓✗"
          "（那会让产物住进源码树：重装/打包/切分支就没了 ✓；h3 服务曾这么写 ✓）",
          not bad, bad)


def case_h3_defaults_to_data_root() -> None:
    import importlib.util  # noqa: PLC0415

    path = APP / "local_services" / "h3" / "server.py"
    spec = importlib.util.spec_from_file_location("_layout_h3_server", path)
    if spec is None or spec.loader is None:  # pragma: no cover
        check("⑤ 本地 H3 服务的默认落点在数据根下 ✓", False, "加载不到 server.py")
        return
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:  # noqa: BLE001
        check("⑤ 本地 H3 服务的默认落点在数据根下 ✓", False, f"{type(error).__name__}: {error}")
        return

    from app.core.config import get_storage_root  # noqa: PLC0415

    out = Path(str(module.OUTPUT_DIR)).resolve()
    root = Path(get_storage_root()).resolve()
    inside = root == out.parent or root in out.parents
    check("⑤ 本地 H3 服务默认落 `<dataRoot>/static/h3` ✓（独立进程也跟数据根走 ✓✗ 不落代码目录 ✓）",
          inside and out.name == "h3", {"OUTPUT_DIR": str(out), "storageRoot": str(root)})


def case_file_storage_subdirs() -> None:
    allowed = STATIC_DIRS | {"engine/refs-audio", "engine/refs-video", "comfyui"}
    found: dict[str, str] = {}
    for path in _iter_app_files():
        rel = str(path.relative_to(BACKEND_PY))
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _func_name(node.func)
            if name not in {"save_uploaded_file", "save_base64_image", "download_file"}:
                continue
            for arg in node.args[1:2]:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    found.setdefault(arg.value, f"{rel}:{node.lineno}")
    bad = {name: where for name, where in found.items() if name not in allowed}
    check("⑥ `file_storage` 那三个写盘函数的 `sub_dir` 也在登记表内 ✓"
          "（它们是参数化的 ✓ ⇒ 只能查**调用点** ✓）",
          not bad, bad)


def case_output_dir_retired() -> None:
    """`backend-py/output/` 已退役 —— 产物统一进数据根 ✓✗

    2026-11-07 创作者裁定「生成的统一进 data 下」⇒ 代码目录里不再有产物区 ✓：
      · 要发 URL、随数据根备份的成品 ⇒ ``<dataRoot>/static/`` ✓
      · 不上线、不发 URL 的实验/探针产物 ⇒ ``<dataRoot>/experiments/<实验名>/`` ✓

    本判据**反着判**：`output/` 存在即红 ✓✗。旧口径是「顶层只许 `_bench/`」，
    那等于承认代码目录里可以住产物 ⇒ 备份面、忽略面、打包排除面都要多记一套 ✓✗，
    而它明明只是「本机探针跑出来的图」，跟作品数据不是一回事 ✓。
    """
    out = BACKEND_PY / "output"
    leftovers = sorted(p.name for p in out.iterdir()) if out.is_dir() else None
    check("⑦ `backend-py/output/` 不存在 ✓✗（产物统一进 `<dataRoot>/`：成品 `static/`、"
          "实验 `experiments/<实验名>/`；代码目录里不再留产物区）",
          not out.exists(), leftovers)


def case_tmp_has_no_media() -> None:
    """`tmp/` 是**现场证据区**（日志 ＋ 一次性脚本），不是产物区 ✓✗"""
    tmp = BACKEND_PY / "tmp"
    if not tmp.is_dir():
        check("⑨ `backend-py/tmp/` 里不出现媒体产物 ✓", True, "目录还没建")
    else:
        found: dict[str, str] = {}
        for path in tmp.rglob("*"):
            if path.is_file() and path.suffix.lower() in MEDIA_SUFFIXES:
                found[str(path.relative_to(BACKEND_PY))] = f"{path.stat().st_size} B"
        check("⑨ `backend-py/tmp/` 里不出现媒体产物 ✓✗（那是现场证据区：`run_all_*.log` 与一次性脚本 ✓；"
              "产物一律进 `<dataRoot>/experiments/<实验名>/` ✓ —— 落 tmp 会随 tmp 一起被当垃圾清 ✓✗）",
              not found, found)

    ignore_file = BACKEND_PY.parent / ".gitignore"
    rows = {line.strip() for line in ignore_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")}
    check("⑨′ `.gitignore` 有 `backend-py/tmp/` **专属**条目 ✓（通用 `tmp/` 顺带命中 ⇒ "
          "那条一改，日志与脚本就混进提交 ✓✗）",
          "backend-py/tmp/" in rows, sorted(r for r in rows if "tmp" in r))


def case_bench_root_has_no_media() -> None:
    """`_bench/` 是**一次性调试草稿区**（探针脚本 ＋ 命令输出 txt/log），不是产物区 ✓✗

    ⚠️ 只查**顶层**：子目录里允许住媒体 —— `_tune_sandbox/` 是探针把 `DATA_ROOT` 指向的
    整库沙盒（`static/` 结构与真数据根同构 ✓，产物落那儿合法），`onnx_backup/` 是包元数据
    备份 ✓。要管的是「图摊在 `_bench/` 根上」这种 —— 那既不在统一产物区、也不在任何沙盒里。

    2026-11-07 实测顶层摊着 `p1_full.png` / `p1_img.png` / `risk_sample.png`（1.7 MB），
    而本文件当时**全绿**：⑦ 只管 `output/` 顶层、⑨ 只管 `tmp/`，`_bench/` 根没人看 ✓✗。
    已收进 `<dataRoot>/experiments/_probe/` ✓。
    """
    bench = BACKEND_PY / "_bench"
    if not bench.is_dir():
        check("⑩ `backend-py/_bench/` 顶层不出现媒体产物 ✓", True, "目录还没建")
        return
    found = {p.name: f"{p.stat().st_size} B" for p in sorted(bench.iterdir())
             if p.is_file() and p.suffix.lower() in MEDIA_SUFFIXES}
    check("⑩ `backend-py/_bench/` **顶层**不出现媒体产物 ✓✗（那是调试草稿区：探针脚本与命令输出；"
          "图/片/音一律进 `<dataRoot>/experiments/<实验名>/` ✓、要发 URL 的成品进 `<dataRoot>/static/` ✓）",
          not found, found)


def case_tune_sandbox_uses_junction() -> None:
    """探针沙盒的 `materialized/` 必须是**联接** ✓✗（实体 = 凭空多占 7.6 GB）

    三个探针（`tune_probe` / `_tune_prompt_size` / `_tune_raw`）都把 `DATA_ROOT` 指向
    `_bench/_tune_sandbox/` ✓ —— 库是副本（合规 ✓），但 `materialized/` 不是库、是
    **真·7.6 GB 反量化缓存** ✓：沙盒里另存一份 = 同一份权重占两处磁盘 ✗。

    2026-11-07 这条漏过两次，两个成因都已修（收口在 `_bench/_sandbox.py`）：
    ① 只有 `tune_probe.py` 建联接，另两个探针不建 ⇒ 谁先跑谁就真物化；
    ② 旧写法 `if not link.exists(): mklink` ⇒ **删了实体只留空壳**时 `exists()` 照样为真
       ⇒ 跳过联接 ⇒ 下次跑又物化 7.6 GB（`Remove-Item` 之后正是这个状态）。

    ⚠️ 空壳也算红 ✓：它现在没接上真缓存，引擎一跑就会在里面物化。

    ⚠️ **「是联接」还不够** ✓✗：联接到 `_bench/` 里另一个目录同样多占一份 ✓ ⇒ 连同
    「解析后不能落在 `_bench/` 子树内」一起判 ✓。这里**不硬编码**真缓存叫
    `data/materialized` ✓ —— 数据根可以被 `DARME_DATA_ROOT` 换走 ✓，钉死目标只会在
    换根时误报 ✓。
    """
    link = BACKEND_PY / "_bench" / "_tune_sandbox" / "materialized"
    exists = link.exists() or link.is_symlink()
    if not exists:
        check("⑩′ 探针沙盒的 `materialized/` 复用真缓存 ✓", True, "沙盒还没建")
        return
    junction = link.is_symlink() or bool(getattr(link, "is_junction", lambda: False)())
    bench = (BACKEND_PY / "_bench").resolve()
    resolved = None
    if junction:
        try:
            resolved = link.resolve()
        except OSError:  # pragma: no cover - 目标盘掉线
            resolved = None
    escaped = resolved is not None and (resolved == bench or bench in resolved.parents)
    check("⑩′ 探针沙盒的 `materialized/` 是**联接且指向沙盒之外** ✓✗（实体目录凭空多占 7.6 GB；"
          "空壳同样不行 —— 它会挡住 `mklink` ⇒ 下次跑重新物化；联到 `_bench/` 内另一处也一样多占，"
          "见 `_bench/_sandbox.py`）",
          junction and not escaped,
          {"junction": junction, "path": str(link.relative_to(BACKEND_PY)),
           "resolved": str(resolved) if resolved else None})


def case_static_has_no_reports() -> None:
    """`<dataRoot>/static/` 下**不许**出现干跑报告 ✓✗（报告是凭据、不是产物 ✓）

    落在 `static` 下 ⇒ `/static` 把它服务成一个**能点开的链接** ✓，而点开是 JSON ✓✗。
    正确落点是 `<dataRoot>/engine-dryrun/` ✓（:func:`app.services.engine.output_paths.report_dir`
    —— 连调用方给的 `outputs_dir` 也只借最后一层名字 ✓）。

    2026-11-07 实测捞到 4 个：`static/engine/dryrun_seed{0,1}.json` 与
    `static/engine/video-eng-{436aead6d1f8,d5e25e9ffae8}/dryrun_seed3.json`。根因是
    `dryrun.write` **先看调用方给没给** `outputs_dir` ✓，而业务通路给的正是
    `static/engine/<kind>-<id>` ✓ —— 已收口 ✓。
    """
    from app.core.config import get_storage_root  # noqa: PLC0415

    root = Path(get_storage_root())
    if not root.is_dir():
        check("⑪ `<dataRoot>/static/` 下没有干跑报告 ✓", True, "目录还没建")
        return
    found = {str(p.relative_to(root)): f"{p.stat().st_size} B"
             for p in root.rglob("dryrun_*.json") if p.is_file()}
    check("⑪ `<dataRoot>/static/` 下没有 `dryrun_*.json` ✓✗（报告是凭据不是产物 ✓；"
          "落这儿 `/static` 会把 JSON 当成品链接服务出去 ✓）",
          not found, found)


def case_dryrun_has_no_temp_dirs() -> None:
    """`<dataRoot>/engine-dryrun/` 下**不许**出现 `mkdtemp` 式随机目录 ✓✗

    这里的子目录只该有两种名字 ✓：
    ⓐ `stage` 名（`krea2` / `h3` / `image` / `video` —— 探针与自检**没给** `outputs_dir` 时 ✓）；
    ⓑ `<kind>-<id>`（`video-eng-<12 位>` —— 业务通路 :func:`bridge.outputs_dir_for` 的
    `dry_run=True` 支 ✓）。

    以 `engine_` 开头的一律是 `tempfile.mkdtemp(prefix="engine_*_")` 的产物 ✓✗：
    2026-11-07 实测盘上 **31 个**（29 个 `engine_pipe_*` 同一秒跑出、2 个 `engine_runtime_*`）。
    两个成因都已修 ⇐ ① :func:`output_paths.report_dir` 现在只借**落在两棵树根下**的
    `outputs_dir` 名字 ✓（随机临时目录退回 `stage` ✓）；② `engine_pipeline_test` /
    `engine_runtime_test` 开始设 `DARME_ENGINE_DRYRUN` ✓ —— 此前它虽是登记在册的隔离口 ✓，
    却**全仓无人设** ✓✗（死配置），测试直写真数据根 ✓。
    """
    from app.core.config import get_data_root  # noqa: PLC0415

    root = Path(get_data_root()) / "engine-dryrun"
    if not root.is_dir():
        check("⑫ `<dataRoot>/engine-dryrun/` 下没有 `mkdtemp` 随机目录 ✓", True, "目录还没建")
        return
    found = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("engine_"))
    check("⑫ `<dataRoot>/engine-dryrun/` 下没有 `mkdtemp` 随机目录 ✓✗"
          "（`engine_pipe_*` / `engine_runtime_*` 每跑一次多一个 ⇒ 干跑树退化成垃圾场 ✓；"
          "`report_dir` 只借本仓落点的名字 ✓、测试须设 `DARME_ENGINE_DRYRUN` ✓）",
          not found, found)


def case_data_dirs_ignored() -> None:
    """`<dataRoot>` 下**每个登记目录**都必须在 `.gitignore` 里有专属条目 ✓✗

    `data/` 里的东西**全是**机器本地数据（库文件 / 下载的权重 / 训练集 / 干跑报告）⇒
    没有一样该入库 ✓。漏一条的后果不是"多提交几个小文件"，是 `git add -A` **整片吞进去**：
    `data/materialized/` 实测 7.7 GB、`data/models/` 是 model_manager 下到数据根的几十 GB
    出图权重 ✓✗。2026-11-07 实测漏的正是 `data/models/` ✓。

    ⚠️ 只认 `data/<名>/` 这一种写法：`.gitignore` 里带斜杠的规则**锚定仓库根** ✓，
    写成裸 `models/` 会同时命中仓根 `models/`（那是要入库的仓内素材）⇒ 反而错 ✓✗。
    """
    rows = {line.strip() for line in (BACKEND_PY.parent / ".gitignore")
            .read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")}
    missing = sorted(name for name in DATA_DIRS if f"data/{name}/" not in rows)
    check("②′ `<dataRoot>/` 的每个登记目录都在 `.gitignore` 里有 `data/<名>/` 条目 ✓✗"
          "（漏一条 ⇒ `git add -A` 整片入库：materialized 7.7 GB、models 几十 GB）",
          not missing, missing)


def case_engine_tree_unchanged() -> None:
    """引擎那一层已由 `engine_output_paths_test` 守着 ✓，这里只核「根还在 static 下」✓"""
    from app.services.engine import output_paths  # noqa: PLC0415

    root = Path(output_paths.outputs_root())
    dry = Path(output_paths.dryrun_root())
    check("⑧ 引擎产物根 `engine/` 与干跑根 `engine-dryrun/` 都在数据根下 ✓"
          "（前者可被 `/static` 服务 ✓、后者不进 `static` ✓）",
          root.name == "engine" and dry.name == "engine-dryrun", {"engine": str(root), "dryrun": str(dry)})


def main() -> int:
    case_static_dirs_registered()
    case_no_static_prefix_constants()
    case_no_code_dir_outputs()
    case_h3_defaults_to_data_root()
    case_file_storage_subdirs()
    case_output_dir_retired()
    case_tmp_has_no_media()
    case_bench_root_has_no_media()
    case_tune_sandbox_uses_junction()
    case_static_has_no_reports()
    case_dryrun_has_no_temp_dirs()
    case_data_dirs_ignored()
    case_engine_tree_unchanged()

    failed = [row for row in _RESULTS if not row[1]]
    for name, ok, detail in _RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            print(f"      实测: {detail}")
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
