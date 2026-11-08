"""「这台机器上装了哪些 ComfyUI 类目录」的**公共口** —— 引擎侧只许认这一份。

为什么这一整套要住在 ``app/core`` ✗✓：

引擎读权重时必须能回答两个不同的问题 ✓：
① 「扫模型」的根在哪 ⇒ ``<根>/models`` ✓（``local_model_scan.get_default_roots`` 的第一类）；
② 「安装根」在哪 ⇒ ``<根>/comfy/sd1_tokenizer`` 这类**安装内部**落点 ✓
（``engine/bridge.resolve_tokenizer_path`` 要找的就是它 ✓✗ —— SDXL 的 CLIP-BPE 词表
**不是权重**，靠「有没有 .safetensors」永远找不着 ✓）。

这两个问题的答案本来都只住 ``app/services/local_model_scan.py`` ✓，于是引擎为了知道自己
「该去哪找词表」就得 ``import app.services.local_model_scan`` ✓✗ —— 那正是
``tests/engine_independence_test.py`` 的 ``_SEAMS`` 要逐条解释的那类越界 ✓。

判据：这一整套读的**只有配置与盘** ✓（``configs/model-paths.json`` / ``COMFYUI_PATH`` /
盘符枚举 / 目录名），一个业务数据都不碰 ✓ ⇒ 与 ``app.core.model_paths`` 同一类 ✓，
搬进 ``app.core`` 零方向性问题 ✓。

⚠️ 分层**只搬到这里为止** ✗✗：``local_model_scan.get_default_roots`` 的另外几类根
（``extra_roots`` / ``model_ecosystems`` 的世界各大生态落点 / ``model_folders`` 的声明表）
仍留在服务层 ✓ —— 那些要读生态约定与用户的声明文件，属「模型库功能」而非「路径口径」✓。
``inventory`` 那条 ``_SEAMS`` 登记的就是剩下的这半 ✓（撤销办法同样写在表里 ✓）。

⚠️ 三条不许改的口径 ✓✗（``tests/local_models_test.py`` 逐条守着 ✓）：
* **不写死任何机器路径** ✗ —— 关键词是**产品名**（``COMFYUI_DIR_HINTS``）✓，盘符靠枚举 ✓；
* 探测要返回**全部**命中 ✓（只取第一个会把后面几个安装整份遮住 ✗✗ —— 本机就这么漏掉过
  ``ComfyUI-Shared/models`` 里那份 SDXL ✓）；
* 显式配置（``COMFYUI_PATH`` / ``comfyui_root``）**排在探测之前** ✓（显式意图优先 ✓）。
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from . import model_paths

__all__ = [
    "COMFYUI_BLANKET_DEPTH",
    "COMFYUI_DIR_HINTS",
    "COMFYUI_PROBE_DEPTH",
    "SKIP_DIRS",
    "SYSTEM_DIRS",
    "comfyui_roots",
    "detect_comfyui",
    "detect_comfyui_roots",
    "list_drives",
    "probe_bases",
]

#: 递归时跳过的目录名 —— **仅跳过「非模型」目录**。
#: ⚠️ clip / vae / controlnet / upscale_models / embeddings / photomaker / style_models /
#: ipadapter 等目录下**都是模型权重**，不能跳过，需被扫描识别。
SKIP_DIRS = frozenset([
    "node_modules", ".git", "venv", ".venv", "__pycache__", ".cache", ".huggingface",
    "output", "temp", "input", "assets", "metadata", "samples", "custom_nodes",
    ".codebuddy", "python", "lib", "site-packages",
])

#: 系统级目录（磁盘/全盘扫描时跳过）。这里用**目录名小写**匹配，模型几乎不可能放在这些目录下。
SYSTEM_DIRS = frozenset([
    "windows", "program files", "program files (x86)", "programdata",
    "$recycle.bin", "system volume information", "recovery", "perflogs",
    "msocache", "documents and settings", "windows.old", "appdata",
    "onedrive", "onedrivetemp", "intel", "amd", "nvidia", "drivers",
])

#: ⭐ 2026-09-26 用户口径 ✓：**不许写死路径** ✗ —— 模型既可以装在**用户指定的任意目录** ✓，
#: 也要能扫**电脑上任意目录**里已有的模型 ✓。所以这里**没有**任何**机器盘符字面量** ✗，
#: 只有「**按名字动态探测 + 全盘枚举**」的口径 ✓：
#:
#: * **权威来源**（优先于探测）：环境变量 ``COMFYUI_PATH`` / ``model-paths.json`` 的 ``comfyui_root`` ✓、
#:   用户的「额外扫描目录」``extra_roots`` ✓（**任意目录**都行 ✓，见 ``local_model_scan.get_default_roots`` ✓）；
#:   下载/存放的**主目录** = ``MODELS_DIR`` / ``model-paths.json`` 的 ``models_dir`` ✓（默认 ``<data_root>/models`` ✓）；
#: * **探测回退** = 家目录 + **各现存盘符**（``list_drives`` ✓）往下找「名字含关键词」的目录 ✓，
#:   判定「像 ComfyUI 根」只看**目录下有没有 ``models/``** ✓（与旧口径一致 ✓ ⇒ 桌面端的**共享模型目录**
#:   ``…/ComfyUI-Shared`` ✓ 也算一个根 ✓，本机那份 SDXL 正在它下面 ✓）；
#: * ⚠️ 关键词是**产品名**不是机器路径 ✓：用户把 ComfyUI 装在**任意盘、名字里带关键词**的目录里照样命中 ✓
#:   （名字毫不相干的目录 = 不瞎猜 ✗，靠用户显式配 ``extra_roots`` / ``comfyui_root`` ✓）。
COMFYUI_DIR_HINTS = ("comfyui", "comfy-desktop", "comfyui-desktop", "comfy")
#: 探测深度（层）：**关键词目录**最深看到这里 ✓。
#: ⚠️ 桌面端把安装藏在 ``<盘>/Comfy-Desktop/ComfyUI-Installs/<名字>/ComfyUI`` ✓ ⇒ **5 层**才够 ✓。
COMFYUI_PROBE_DEPTH = 5
#: 「顺便看一眼」的深度（层）：名字**不沾边**的目录只在最上面这几层进一下 ✓ ——
#: 因为开发/自建目录常挂在 ``<盘>/code/ComfyUI`` 这类**父目录名不含关键词**的地方 ✗
#: （只认关键词会漏掉它 ✗）；再深就**不乱翻** ✗（省时的同时避开无关目录树 ✓）。
COMFYUI_BLANKET_DEPTH = 2


def list_drives() -> list[dict[str, str]]:
    """枚举可用盘符（Windows 探测 A-Z；POSIX 上结果为空 —— 与原 TS 行为一致）。"""
    drives: list[dict[str, str]] = []
    for code in range(65, 91):
        letter = chr(code)
        root = f"{letter}:\\"
        try:
            if os.path.exists(root):
                drives.append({"letter": letter, "root": root})
        except OSError:
            pass
    return drives


def probe_bases() -> list[str]:
    """探测起点：家目录 + 各现存盘根（``list_drives`` ✓ —— POSIX 上取不到盘符 ⇒ 只剩家目录 ✓）。"""
    bases: list[str] = []
    home = os.path.expanduser("~")
    if home and os.path.isdir(home):
        bases.append(home)
    for drive in list_drives():
        root = str(drive.get("root") or "")
        if root and os.path.isdir(root):
            bases.append(root)
    return bases


def detect_comfyui_roots(bases: list[str] | None = None) -> list[str]:
    """**动态**探测本机全部「ComfyUI 类」目录（保序：起点顺序 + 浅→深 ✓）。

    ⚠️ 与旧 ``detect_comfyui()`` 的关键差别：这里返回**全部**命中 ✓ —— 旧实现只取**第一个** ⇒
    机器上装了两个以上 ComfyUI 时，**后面的全被遮住** ✗✗（本机就是这么漏掉那份 SDXL 的 ✓：
    候选表把 ``ComfyUI-Installs/…`` 排在前 ⇒ ``ComfyUI-Shared/models`` 永远扫不到 ✗）。

    做法：从起点往下走，**名字含关键词**（``COMFYUI_DIR_HINTS``）的目录一路下潜
    （最多 ``COMFYUI_PROBE_DEPTH`` 层 ✓），名字不沾边的只在最上面 ``COMFYUI_BLANKET_DEPTH`` 层
    「顺便看一眼」✓ —— 因为自建/开发目录常挂在 ``<盘>/code/ComfyUI`` 这种**父目录名不含关键词**的地方 ✗；
    命中判定 = **名字含关键词 + 目录下有 ``models/``** ✓（与旧口径一致 ✓）。``SKIP_DIRS`` / ``SYSTEM_DIRS``
    照旧跳过 ✓（⇒ 再怎么「顺便看」也不会进 ``Windows`` / ``AppData`` ✓）。

    ``bases=None`` ⇒ 家目录 + 各现存盘根 ✓；测试**注入临时目录** ✓（不碰真实磁盘 ✓）。
    """
    found: list[str] = []
    seen: set[str] = set()
    queue: list[tuple[str, int]] = [
        (base, 0) for base in (bases if bases is not None else probe_bases())
    ]
    while queue:
        current, depth = queue.pop(0)
        try:
            with os.scandir(current) as scanned:
                children: list[tuple[str, str, bool]] = []
                for entry in sorted(scanned, key=lambda item: item.name.lower()):
                    try:
                        children.append((entry.name.lower(), os.path.abspath(entry.path),
                                         entry.is_dir(follow_symlinks=False)))
                    except OSError:
                        continue
        except OSError:
            continue
        for name, path, is_dir in children:
            if not is_dir or name.startswith(".") or name in SKIP_DIRS or name in SYSTEM_DIRS:
                continue
            hinted = any(hint in name for hint in COMFYUI_DIR_HINTS)
            # ⚠️ 名字不沾边的**只在最上面几层**「顺便看一眼」✓（``<盘>/code/ComfyUI`` 这种 ✗）；
            #    再往下只跟着关键词走 ✗ —— 既不乱翻无关目录树 ✓、对普通名字也**不瞎猜**是根 ✓
            #    （用户的任意目录另有显式入口：``extra_roots`` / ``comfyui_root`` ✓）
            if not hinted and depth + 1 >= COMFYUI_BLANKET_DEPTH:
                continue
            key = os.path.normcase(path)
            if key in seen:
                continue
            seen.add(key)
            if hinted and os.path.isdir(os.path.join(path, "models")):
                found.append(path)
            if depth + 1 < COMFYUI_PROBE_DEPTH:
                queue.append((path, depth + 1))
    return found


def detect_comfyui() -> str:
    """探测**第一个** ComfyUI 安装位置（保留给「单值要用」的调用方 ✓）。

    ⚠️ **扫描一律走** :func:`detect_comfyui_roots` ✓（只取第一个 = 漏 ✗，2026-09-26 修的正是这条 ✓）。
    """
    roots = detect_comfyui_roots()
    return roots[0] if roots else ""


def comfyui_roots(*, path: Path | None = None,
                  detect: Callable[[], list[str]] | None = None) -> list[tuple[str, str]]:
    """ComfyUI 类**安装根** + 来源说明 ✓ ⇒ ``[(根, 来源)]`` ✓（**显式配的排在前** ✓）。

    三处来源（顺序固定 ✓）：``COMFYUI_PATH`` ✓ → 配置文件 ``comfyui_root``
    ✓ → 探测（默认 :func:`detect_comfyui_roots`）的**全部**命中 ✓。

    ``path`` 与 :func:`model_paths.read_paths_config` 同一个**隔离口** ✓✗：服务层那层薄委托
    ``local_model_scan.get_comfyui_roots()`` 必须把自己唯一可 patch 的 ``_paths_config_file()``
    传下来 ✓ —— 少传一个，自检里换掉的配置文件对它不生效 ⇒ 「换了文件还读到真配置」的假绿 ✓✗
    （``tests/local_models_test.py`` 的端点判据盯着这条 ✓）。引擎侧不传 = 真身 ✓。

    ``detect`` 是**第二个隔离口** ✓✗，同样是被真事故逼出来的：探测函数在本模块里叫
    ``detect_comfyui_roots`` ✓，而服务层的测试是去 patch **服务层那个名字**
    （``ls.detect_comfyui_roots = lambda: [...]`` ✓ —— 注入临时目录、不碰真实磁盘 ✓）。
    只把 ``path`` 传下去而**不传探测函数** ⇒ 本模块会走自己的那份全局 ⇒ patch 静默失效 ✓✗
    （实测症状：``get_default_roots`` 那条「每个 ComfyUI 根都要收进来」直接红 ✓）。
    ⇒ 委托方必须把**自己模块里那个名字**显式传进来 ✓，而不是指望跨模块的 monkeypatch 生效 ✗。
    """
    cfg = model_paths.read_paths_config(path=path)
    explicit = os.environ.get("COMFYUI_PATH") or cfg.get("comfyui_root") or ""
    probe = detect or detect_comfyui_roots
    found: list[tuple[str, str]] = []
    for candidate, source in (
        (explicit, "COMFYUI_PATH" if os.environ.get("COMFYUI_PATH") else "清单 configs/model-paths.json 的 comfyui_root"),
        *((item, "动态探测 detect_comfyui_roots()") for item in probe()),
    ):
        text = os.path.abspath(str(candidate)) if str(candidate or "").strip() else ""
        if text and os.path.normcase(text) not in {os.path.normcase(path_) for path_, _ in found}:
            found.append((text, source))
    return found
