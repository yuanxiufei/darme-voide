"""自检：**LoRA 训练**（LoRAMaster 移植 ✓ 2026-09-26）。

⚠️ 本套**不装模型、不跑训练** ✗（真跑一次 Wan 是几小时 + 几十 GB 权重 ✓）——
它钉的是**「跑之前那一半」** ✓：参数怎么收敛 ✓、argv 怎么拼 ✓、输出怎么切行 ✓、
日志怎么增量取 ✓、素材怎么改 ✗、错误码怎么分 ✓。这些正是「第一次真跑就翻车」的地方 ✓。

七组 ✓：① 参数层 ✓ ② 命令层 argv 结构 ✓ ③ 切行（tqdm 全靠它才不刷屏 ✓）
④ 日志缓冲 / 进度事实 ✓ ⑤ 真起子进程 + 真取消 ✓ ⑥ 素材工具 ✓ ⑦ 素材运行时 ✓
（每组的具体口径写在各自 `case_*` 的注释里 ✓）。

运行::

    ./.venv/Scripts/python.exe tests/lora_train_test.py
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
# ⚠️ 参数文件都落在**数据根**下 ⇒ 必须在 import 之前指到临时目录 ✓（别碰真实数据 ✓）
os.environ["DATA_ROOT"] = tempfile.mkdtemp(prefix="lora_train_")
sys.path.insert(0, str(BACKEND_PY))

from app.services.lora_train import commands, options, paths  # noqa: E402
from app.services.lora_train.dataset import caption as caption_mod  # noqa: E402
from app.services.lora_train.dataset import files as dataset_files  # noqa: E402
from app.services.lora_train.dataset.runtime import ACTIONS, DatasetRuntime  # noqa: E402
from app.services.lora_train.errors import (  # noqa: E402
    LoraTrainConfigError,
    LoraTrainNotFoundError,
)
from app.services.lora_train.progress import LogBuffer, parse_facts  # noqa: E402
from app.services.lora_train.runner import (  # noqa: E402
    CommandRunner,
    OutputLine,
    child_env,
    iter_output,
)

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    _SKIPS.append(reason)


def raises(call: object) -> bool:
    try:
        call()  # type: ignore[operator]
    except Exception:  # noqa: BLE001 - 只要它抛 ✓
        return True
    return False


# ---------------------------------------------------------------------------
# ① 参数层
# ---------------------------------------------------------------------------


def case_options() -> None:
    keys = options.model_keys()
    check("参数: 5 条训练链都在 ✓（顺序 = 前端展示顺序 ✓）",
          keys == ("wan", "hunyuan", "kontext", "flux", "qwen"), keys)
    check("参数: 默认落 wan ✓", options.DEFAULT_MODEL == "wan")
    check("参数: **FLUX 走 sd-scripts**、其余四条走 musubi-tuner ✓（实测口径，不是笔误 ✓）",
          options.spec("flux").toolkit == paths.TOOLKIT_SD_SCRIPTS
          and all(options.spec(k).toolkit == paths.TOOLKIT_MUSUBI
                  for k in keys if k != "flux"))

    try:
        options.merge("wan", {"not_a_real_option": 1})
    except LoraTrainConfigError as err:
        check("参数: 未知键 ⇒ 报错并列出合法键 ✓（不静默丢弃 ✗）",
              "not_a_real_option" in str(err), str(err)[:100])
    else:
        check("参数: 未知键居然过了（该红 ✗）", False, "no raise")

    merged = options.merge("wan", {"_comment": "注", "learning_rate": "2e-4"})
    check("参数: `_` 开头的键当注释放过 ✓ 且数值保持字符串 ✓（`2e-4` 要原样透传 ✓）",
          "_comment" not in merged and merged.get("learning_rate") == "2e-4")

    try:
        options.merge("nope", {})
    except LoraTrainConfigError as err:
        check("参数: 未知模型 ⇒ 报错并列出合法值 ✓（不回落默认 ✗）",
              "合法值" in str(err) and "wan" in str(err), str(err)[:100])
    else:
        check("参数: 未知模型居然回落了（该红 ✗）", False, "no raise")

    # ⚠️ 数值类参数在参考实现的 toml 里**就是字符串** ✓（``batch_size = "1"`` ✓）
    #    ⇒ 本仓照抄：KIND_TEXT ✓ 原样透传 ✓。真按整数收的是 `num_processes` 这类 ✓。
    check("参数: 真正的整数键写错 ⇒ 当场报错 ✓（KIND_INT ✓）",
          raises(lambda: options.merge("wan", {"num_processes": "abc"})))
    check("参数: 数值类参数是**字符串** ✓（照抄参考 toml ✓，`batch_size = \"16\"` ✓）",
          isinstance(options.merge("wan", {})["batch_size"], str))


# ---------------------------------------------------------------------------
# ② 命令层：argv 结构
# ---------------------------------------------------------------------------


def _dummy_payload(model_key: str, root: Path) -> dict[str, object]:
    """把该模型的**路径类**参数都填上哑文件 ✓（这里只验 argv 结构 ✓，不真跑 ✓）。"""
    payload: dict[str, object] = {}
    for opt in options.spec(model_key).options:
        if opt.kind not in (options.KIND_PATH_FILE, options.KIND_PATH_DIR):
            continue
        target = root / f"dummy_{model_key}_{opt.key}"
        if opt.kind == options.KIND_PATH_DIR:
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.write_text("", encoding="utf-8")
        payload[opt.key] = str(target)
    return payload


def _prompt_tokens(prompt_file: Path | None) -> list[str]:
    """prompt 文件切成 token ✓（``--ci`` / ``--i`` 是里面**单独一段** ✓ 不是 argv ✓）。"""
    if prompt_file is None:
        return []
    return prompt_file.read_text(encoding="utf-8").split()


def case_commands(root: Path) -> None:
    workspace = root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    for model_key in options.model_keys():
        resolved = options.spec(model_key)
        payload = _dummy_payload(model_key, root)
        try:
            plan = commands.build_plan(model_key, options.merge(model_key, payload),
                                       workspace=workspace)
        except LoraTrainConfigError as err:
            check(f"命令[{model_key}]: 参数填齐后组命令不该失败 ✗", False, str(err)[:180])
            continue
        joined = " ".join(plan.train)
        check(f"命令[{model_key}]: 预缓存条数 == 声明的缓存脚本数 ✓",
              len(plan.cache) == len(resolved.cache_scripts),
              f"{len(plan.cache)} vs {resolved.cache_scripts}")
        check(f"命令[{model_key}]: 训练脚本**真在** argv 里 ✓（不是空壳命令 ✓）",
              str(resolved.train_script) in joined, plan.train[-6:])
        check(f"命令[{model_key}]: 走 accelerate launch ✓",
              "-m" in plan.train and "accelerate.commands.launch" in plan.train, plan.train[:4])
        check(f"命令[{model_key}]: 工具落点报告齐 ✓",
              set(plan.tools) == {resolved.train_script, *resolved.cache_scripts},
              sorted(plan.tools))

    flux = commands.build_plan("flux", options.merge("flux", _dummy_payload("flux", root)),
                               workspace=workspace)
    check("命令[flux]: **一条预缓存都没有** ✓（sd-scripts 那条路不预缓存 ✓；多出来该红 ✗）",
          flux.cache == (), flux.cache)

    # ⚠️ 回归: `--ci` / `--i` 不许互换 ✗✗（参考实现两处源码实测：kontext/qwen 用 --ci ✓）
    kontext = options.merge("kontext", {
        **_dummy_payload("kontext", root), "generate_samples": True,
        "sample_image_path": _dummy_payload("kontext", root)["vae_path"]})
    kplan = commands.build_plan("kontext", kontext, workspace=workspace)
    check("命令[kontext]: 样图**控制图**用 `--ci` ✓（参考实现 flux_kontext_lora_train.py 就是这么写的 ✓）",
          kplan.prompt_file is not None and "--ci" in kplan.prompt_file.read_text(
              encoding="utf-8"), kplan.prompt_file)
    check("命令[kontext]: `--ci` 与 `--i` **不许互换** ✓（互换会让样图拿错输入图 ✗）",
          "--ci" in _prompt_tokens(kplan.prompt_file)
          and "--i" not in _prompt_tokens(kplan.prompt_file),
          _prompt_tokens(kplan.prompt_file))

    wan = options.merge("wan", {
        **_dummy_payload("wan", root), "generate_samples": True,
        "sample_image_path": _dummy_payload("wan", root)["vae_path"]})
    wplan = commands.build_plan("wan", wan, workspace=workspace)
    check("命令[wan]: 样图**首帧**用 `--i` ✓（wan_lora_train.py 实测 ✓）",
          "--i" in _prompt_tokens(wplan.prompt_file)
          and "--ci" not in _prompt_tokens(wplan.prompt_file),
          _prompt_tokens(wplan.prompt_file))
    check("命令[wan]: prompt 文件**真落盘** ✓ 且路径进了 argv ✓",
          wplan.prompt_file is not None and wplan.prompt_file.is_file()
          and str(wplan.prompt_file) in wplan.train, wplan.prompt_file)
    check("命令[wan]: prompt 首行注释照抄参考实现 ✓（训练脚本靠它认格式 ✓）",
          wplan.prompt_file is not None and wplan.prompt_file.read_text(
              encoding="utf-8").startswith("# prompt 1: for generating a sample"))

    try:
        commands.validate("wan", options.merge("wan", {}))
    except LoraTrainConfigError as err:
        check("命令: 必填项缺了 ⇒ **一次报全** ✓（改一次就能跑 ✓，不是发现一个抛一个 ✗）",
              all(key in str(err) for key in ("dataset_config", "vae_path", "t5_path",
                                              "dit_weights_path")), str(err)[:160])
    else:
        check("命令: 必填项缺了居然过了（该红 ✗）", False, "no raise")

    check("命令: 参数填齐 ⇒ 校验通过 ✓（不误报 ✓）",
          not raises(lambda: commands.validate("wan", options.merge(
              "wan", _dummy_payload("wan", root)))))


# ---------------------------------------------------------------------------
# ③ 切行（tqdm 全靠它才不刷屏）
# ---------------------------------------------------------------------------


def case_iter_output() -> None:
    raw = (b"epoch 1/10\n"
           b"steps:  10%|#         | 1/10 [00:01<00:09]\r"
           b"steps:  20%|##        | 2/10 [00:02<00:08]\r"
           + "中文行没有换行结尾".encode("utf-8"))
    lines = list(iter_output(io.BytesIO(raw)))
    check("切行: `\\n` 是普通行、`\\r` 认成**进度条刷新** ✓（`overwrite=True` ⇒ 上层替换 ✓）",
          lines[0].overwrite is False and lines[1].overwrite is True
          and lines[2].overwrite is True, [ln.overwrite for ln in lines])
    check("切行: 每次刷新都交出来 ✓（并成一条还是替换，由上层决定 ✓）",
          "10%" in lines[1].text and "20%" in lines[2].text)
    check("切行: 结尾没有换行的半行**也交出来** ✓（报错就挂在最后一行 ✓✗）",
          lines[-1].text == "中文行没有换行结尾", lines[-1].text)

    class _One(io.BytesIO):
        """每次只喂 1 字节 ✓ —— 专门制造「跨 chunk」 ✓。"""

        def read1(self, _n: int = -1) -> bytes:  # noqa: D102
            return self.read(1)

    check("切行: 跨 chunk 的 `\\r\\n` 当成**一个**换行 ✓（否则会多出一条空行 ✗）",
          [ln.text for ln in iter_output(_One(b"a\r\nb\n"))] == ["a", "b"])
    check("切行: UTF-8 多字节**被 chunk 切断**也要拼回来 ✓（否则中文全成 U+FFFD ✗）",
          list(iter_output(_One("标签：四".encode("utf-8") + b"\n")))[0].text == "标签：四")


# ---------------------------------------------------------------------------
# ④ 日志缓冲 + 进度事实
# ---------------------------------------------------------------------------


def case_log_buffer() -> None:
    buffer = LogBuffer()
    buffer.append(OutputLine("epoch 1/10"))
    buffer.append(OutputLine("steps:  10%|#  | 1/10", overwrite=True))
    buffer.append(OutputLine("steps:  50%|## | 5/10", overwrite=True))
    check("日志: 进度条只留**最新那条** ✓（前一次被替换掉 ✓）",
          "50%" in buffer.snapshot()["progress"] and "10%" not in buffer.snapshot()["progress"],
          buffer.snapshot()["progress"])
    buffer.append(OutputLine("epoch 2/10"))
    check("日志: 普通行一到 ⇒ 挂着的进度条**落进正文** ✓（顺序不错乱 ✓）",
          buffer.snapshot()["lines"] == ["epoch 1/10", "steps:  50%|## | 5/10", "epoch 2/10"],
          buffer.snapshot()["lines"])
    check("日志: 进度条**不占序号** ✓（序号只数进过正文的行 ⇒ 序号↔下标对得上 ✓）",
          buffer.sequence == 3, buffer.sequence)
    check("日志: 正文里没有进度条中间态 ✓（否则一小时几万行 ✗）",
          all("10%" not in row for row in buffer.snapshot()["lines"]))
    buffer.append(OutputLine("epoch 3/10"))
    check("日志: 进度**事实**认得出 ✓（epoch 3/10 ⇒ 3 / 10 / 30% ✓）",
          buffer.facts.get("epoch") == 3 and buffer.facts.get("epochs") == 10
          and buffer.facts.get("percent") == 30, buffer.facts)
    tail = buffer.snapshot(since=2)
    check("日志: `since=2` 只回序号**大于 2** 的行 ✓（真增量 ✓，不是每次整份 ✗）",
          tail["lines"] == ["epoch 2/10", "epoch 3/10"], tail["lines"])
    check("日志: 没追丢 ⇒ `truncated=False` ✓", tail["truncated"] is False)
    buffer.flush_progress()
    check("日志: `flush_progress` 把挂着的那条落进正文 ✓（不 flush 最后一条会丢 ✗）",
          any("50%" in row for row in buffer.snapshot()["lines"]))

    small = LogBuffer(limit=3)
    for index in range(10):
        small.append(OutputLine(f"line-{index}"))
    check("日志: 上限真起作用 ✓（只留最近 3 行 ✓）", len(small.snapshot()["lines"]) == 3)
    check("日志: 被挤掉 ⇒ `truncated=True` ✓（**明说**你漏了一段 ✓，不静默少给 ✗）",
          small.snapshot(since=1)["truncated"] is True)

    check("事实: 认得上游实测的 `epoch X/Y` ✓（musubi 与 sd-scripts 都是这个格式 ✓）",
          parse_facts("epoch 3/200") == {"epoch": 3, "epochs": 200, "percent": 1})
    check("事实: 认得上游 tqdm 的 `steps:  42%|…| 42/100 […]` ✓",
          parse_facts("steps:  42%|####| 42/100 [00:10<00:13,  4.04it/s]")
          == {"percent": 42, "step": 42, "totalSteps": 100})
    check("事实: **认不出就不编** ✓（没证据 ⇒ 空字典 ✓，不是猜个 0 ✗）",
          parse_facts("loading weights...") == {})


# ---------------------------------------------------------------------------
# ⑤ 真起子进程 + 真取消
# ---------------------------------------------------------------------------


def _fake_tool() -> paths.ResolvedTool:
    """一个**假的**已定位工具 ✓ —— 只为拿一份合规 env ✓（不真跑它 ✓）。"""
    root = Path(tempfile.gettempdir())
    return paths.ResolvedTool(path=root / "fake_train_network.py", pythonpath=(root,),
                              matched="fake.py", toolkit=paths.TOOLKIT_MUSUBI, root=root)


def case_runner() -> None:
    seen: list[OutputLine] = []
    runner = CommandRunner(on_line=seen.append)
    script = ("import sys, time\n"
              "sys.stdout.write('epoch 1/10\\n')\n"
              "sys.stdout.write('标签：中文不能丢\\n')\n"
              "sys.stdout.write('steps:  50%|##| 5/10\\r')\n"
              "sys.stdout.flush()\n"
              "time.sleep(0.05)\n"
              "sys.stdout.write('steps: 100%|##| 10/10\\r')\n"
              "sys.stdout.flush()\n")
    result = runner.run([sys.executable, "-c", script], env=child_env(_fake_tool()))
    check("运行器: 真起子进程并跑到正常结束 ✓", result.ok, result.as_dict())
    check("运行器: 中文按 UTF-8 收进来**没变问号** ✓",
          any("标签：中文不能丢" in ln.text for ln in seen), [ln.text for ln in seen][:4])
    check("运行器: 两次进度条刷新 ⇒ `overwrite=True` ✓（上层据此替换 ✓）",
          sum(1 for ln in seen if ln.overwrite) == 2,
          [(ln.text, ln.overwrite) for ln in seen])
    check("运行器: 行数如实统计 ✓", result.lines == len(seen), f"{result.lines} vs {len(seen)}")

    failed = runner.run([sys.executable, "-c", "import sys; sys.exit(3)"],
                        env=child_env(_fake_tool()))
    check("运行器: 返回码非 0 **不抛** ✓（调用方要把它记进任务里 ✓✗）",
          failed.returncode == 3 and not failed.ok, failed.as_dict())
    try:
        runner.run([str(Path(tempfile.gettempdir()) / "绝对不存在的解释器.exe"), "-c", "pass"])
    except Exception as err:  # noqa: BLE001 - 这里就是要它抛 ✓
        check("运行器: 命令**根本起不来** ⇒ 抛 RunError ✓（与「返回码非 0」分得开 ✓）",
              type(err).__name__ == "LoraTrainRunError", type(err).__name__)
    else:
        check("运行器: 起不来的命令居然静默成功了（该红 ✗）", False, "no raise")

    slow = "import time\nprint('开始', flush=True)\ntime.sleep(60)\n"
    box: dict[str, object] = {}
    thread = threading.Thread(
        target=lambda: box.__setitem__("result", runner.run([sys.executable, "-c", slow],
                                                           env=child_env(_fake_tool()))),
        daemon=True)
    started = time.monotonic()
    thread.start()
    time.sleep(1.5)
    cancelled = runner.cancel(grace=10)
    thread.join(timeout=30)
    outcome = box.get("result")
    check("运行器: 取消**真能打断**长睡进程 ✓（子线程及时回来了 ✓）",
          cancelled and not thread.is_alive(),
          f"cancelled={cancelled} alive={thread.is_alive()} 用时={time.monotonic() - started:.1f}s")
    check("运行器: 被取消 ⇒ 如实标记 `cancelled=True` ✓（取消是正常路径 ✓，不是失败 ✗）",
          outcome is not None and getattr(outcome, "cancelled", False) is True,
          None if outcome is None else outcome.as_dict())


# ---------------------------------------------------------------------------
# ⑥ 素材工具
# ---------------------------------------------------------------------------


def _make_images(directory: Path, count: int, *, suffix: str = ".jpg",
                 real: bool = False) -> None:
    """造假图 ✓（``real=True`` 时用 PIL 写一张**真能打开**的小图 ✓ —— 转格式那条要真图 ✓）。"""
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        target = directory / f"img_{index}{suffix}"
        if not real:
            target.write_bytes(b"not-a-real-image")
            continue
        from PIL import Image  # noqa: PLC0415 - 只在这条用例里用 ✓

        Image.new("RGB", (8, 8), (10, 20, 30)).save(target)


def case_dataset_files(root: Path) -> None:
    folder = root / "rename"
    _make_images(folder, 3)
    (folder / "img_0.txt").write_text("旧标签", encoding="utf-8")
    dataset_files.run_rename(folder, suffix="jpg", prefix="huluwa_", digits=4, rename_caption=True)
    names = sorted(p.name for p in folder.glob("*.jpg"))
    check("素材/改名: 按 `前缀 + 位数` 重命名 ✓（**从 0001 起** ✓ —— 照抄参考实现的 enumerate ✓）",
          names == ["huluwa_0001.jpg", "huluwa_0002.jpg", "huluwa_0003.jpg"], names)
    check("素材/改名: 同名 `.txt` **跟着走** ✓（参考实现只改图、不管 txt ⇒ 标签全对不上 ✗）",
          (folder / "huluwa_0001.txt").read_text(encoding="utf-8") == "旧标签",
          sorted(p.name for p in folder.iterdir()))
    check("素材/改名: 幂等 ✓（再跑一次改名 0 张 ✓）",
          dataset_files.run_rename(folder, suffix="jpg", prefix="huluwa_",
                                   digits=4)["renamed"] == 0)

    # ⚠️⭐ **先腾位**那一步的回归 ✓（这条是本仓自己补的 ✓，写它是因为**真会丢数据** ✓✗）：
    #    ``Path.replace`` 是覆盖语义 ✓ ⇒ ``img_0.jpg → other_0001.jpg`` 会把**原本的**
    #    ``other_0001.jpg`` 直接盖掉 ✓✗（而那张自己也在这批里要挪到 0003 ✓✗ ⇒ 它的内容就没了 ✓✗✗）。
    swap = root / "swap"
    _make_images(swap, 2)
    (swap / "img_0.jpg").write_bytes(b"AAA")          # 内容各不相同 ✓ 才看得出谁被盖了 ✓
    (swap / "img_1.jpg").write_bytes(b"BBB")
    (swap / "other_0001.jpg").write_bytes(b"CCC")
    (swap / "other_0001.txt").write_text("标签C", encoding="utf-8")
    (swap / "img_1.txt").write_text("标签B", encoding="utf-8")
    dataset_files.run_rename(swap, suffix="jpg", prefix="other_", digits=4, rename_caption=True)
    kept = {p.name: p.read_bytes() for p in sorted(swap.glob("*.jpg"))}
    check("素材/改名: **谁也不许被盖掉** ✓✓（三张图的内容都在 ⇒ 先腾位那步没白写 ✓）",
          kept == {"other_0001.jpg": b"AAA", "other_0002.jpg": b"BBB",
                   "other_0003.jpg": b"CCC"}, kept)
    check("素材/改名: 被腾过位的那张，它的 `.txt` 也跟到了新名字 ✓",
          (swap / "other_0003.txt").read_text(encoding="utf-8") == "标签C"
          and (swap / "other_0002.txt").read_text(encoding="utf-8") == "标签B",
          sorted(p.name for p in swap.iterdir()))

    # 真冲突：目标位子被**不会挪窝**的东西占着（这里放一个同名目录 ✓）⇒ 一个文件都不许动 ✓
    clash = root / "clash"
    _make_images(clash, 2)
    (clash / "other_0002.jpg").mkdir()  # ⚠️ 是**目录** ✓（编号从 0001 起 ⇒ 要占 0002 才撞得上 ✓）
    try:
        dataset_files.run_rename(clash, suffix="jpg", prefix="other_", digits=4)
    except Exception as err:  # noqa: BLE001 - 这里就是要它抛 ✓
        check("素材/改名: 位子被占 ⇒ 报错点名 ✓ 且**一张都没动** ✓（改了一半更难收拾 ✗）",
              "一个都没改" in str(err) and sorted(p.name for p in clash.iterdir())
              == ["img_0.jpg", "img_1.jpg", "other_0002.jpg"], str(err)[:160])
    else:
        check("素材/改名: 位子被占居然静默返回了（该红 ✗ —— 会伪装成「没有需要改名的文件」✓✗）",
              False, "no raise")

    convert = root / "convert"
    _make_images(convert, 2, suffix=".png", real=True)
    facts = dataset_files.run_convert(convert, target_suffix="jpg", backup=True, quality=90)
    backup = dataset_files.backup_directory(convert)
    check("素材/转格式: 备份目录**在素材目录之外** ✓✓（在里面会把备份自己再抄一遍 ⇒ 无终止递归 ✗）",
          not str(backup).startswith(str(convert) + os.sep) and backup.is_dir()
          and len(list(backup.rglob("*.png"))) == 2,
          {"backup": str(backup), "source": str(convert), "facts": facts})
    check("素材/转格式: 目标格式真产出 ✓",
          len(list(convert.glob("*.jpg"))) == 2, sorted(p.name for p in convert.iterdir()))

    # —— 标签读写 + 计数 ✓（用**独立的**目录 ✓：转格式那处两种格式叠在一起 ✓，数不准 ✓）
    labels = root / "labels"
    _make_images(labels, 2, real=True)
    check("素材/标签: 写 / 读同名 `.txt` ✓",
          dataset_files.write_caption(labels / "img_0.jpg", "一个人") is not None
          and dataset_files.read_caption(labels / "img_0.jpg") == "一个人")
    stats = dataset_files.caption_stats(labels)
    check("素材/标签: 缺标签数得对 ✓（打标前必须先知道缺几张 ✓）",
          stats["images"] == 2 and stats["labeled"] == 1 and stats["missing"] == 1, stats)
    check("素材/标签: 图不在 ⇒ 报错 ✗（不静默写一个孤儿 txt ✗）",
          raises(lambda: dataset_files.write_caption(labels / "没这张.jpg", "x")))

    check("素材/加工: `trigger_word` 前置 ✓ + `filter_word` 删掉 ✓（照抄参考口径 ✓）",
          caption_mod.compose_caption(" 一个女孩 ", trigger_word="huluwa ", filter_word="女孩")
          == "huluwa 一个", caption_mod.compose_caption(" 一个女孩 ", trigger_word="huluwa ",
                                                       filter_word="女孩"))

    check("素材/打标: 不认识量化档 ⇒ 报错 ✓ 且 `bf16` 不加量化配置 ✓",
          raises(lambda: caption_mod.quantization_config("int3"))
          and caption_mod.quantization_config("bf16") == {})
    check("素材/打标: `cpu` + 量化 ⇒ **当场报错** ✓"
          "（bitsandbytes 只能在 CUDA 上 ⇒ 提前说清，别抛一句看不懂的 CUDA 错 ✗）",
          raises(lambda: caption_mod.CaptionRequest.from_values(
              {"dataset_path": str(convert), "quantization": "nf4", "device": "cpu"})))
    check("素材/打标: `cuda:1` ⇒ `device_map` 指到那张卡 ✓"
          "（参考实现固定 auto ⇒ 多卡机器上可能与输入设备对不上 ✓✗）",
          caption_mod._device_map("cuda:1") == {"": "cuda:1"}
          and caption_mod._device_map("cuda") == "auto")


# ---------------------------------------------------------------------------
# ⑦ 素材运行时
# ---------------------------------------------------------------------------


def case_dataset_runtime(root: Path) -> None:
    runtime = DatasetRuntime()  # ⚠️ 用**独立**实例 ✓（别污染全局单例的队列 ✓）
    folder = root / "runtime"
    _make_images(folder, 2)

    check("素材运行器: 三个工具的动作表齐 ✓（打标四个动作 ✓）",
          sorted(ACTIONS) == ["caption", "convert", "rename"]
          and ACTIONS["caption"] == ("caption", "trigger", "filter", "scan"),
          {key: list(value) for key, value in ACTIONS.items()})
    check("素材运行器: 不认识的动作 ⇒ **报错点名** ✓（不静默当默认动作 ✗）",
          raises(lambda: runtime.submit("rename", "nope", {"dataset_path": str(folder)})))
    check("素材运行器: 目录不存在 ⇒ **提交时就报错** ✓（不是任务跑起来才失败 ✗）",
          raises(lambda: runtime.submit("rename", None, {"dataset_path": str(root / "没这个目录")})))

    scanned = runtime.wait(runtime.submit("caption", "scan", {"dataset_path": str(folder)}),
                           timeout=60)
    check("素材运行器: `scan` 跑通 ✓（只数不改 ✓）",
          scanned["status"] == "succeeded" and scanned["result"]["images"] == 2,
          scanned.get("result") or scanned.get("error"))
    check("素材运行器: `scan` 之后磁盘**一个字没变** ✓（只读动作要真只读 ✓）",
          sorted(p.name for p in folder.iterdir()) == ["img_0.jpg", "img_1.jpg"],
          sorted(p.name for p in folder.iterdir()))

    triggered = runtime.wait(runtime.submit("caption", "trigger",
                                            {"dataset_path": str(folder),
                                             "trigger_word": "huluwa"}), timeout=60)
    check("素材运行器: `trigger` 把触发词写进每个同名 `.txt` ✓",
          triggered["status"] == "succeeded" and all(
              dataset_files.read_caption(image) == "huluwa"
              for image in sorted(folder.glob("*.jpg"))), triggered.get("result"))

    (folder / "img_0.txt").write_text("huluwa 一个女孩", encoding="utf-8")
    filtered = runtime.wait(runtime.submit("caption", "filter",
                                           {"dataset_path": str(folder),
                                            "filter_word": "女孩"}), timeout=60)
    check("素材运行器: `filter` 把过滤词删掉 ✓",
          filtered["status"] == "succeeded"
          and dataset_files.read_caption(folder / "img_0.jpg") == "huluwa 一个",
          filtered.get("result"))

    check("素材运行器: 任务号不认识 ⇒ **404 形状**的错 ✓（「没跑过」和「跑完了」分得开 ✓）",
          _is_not_found(runtime, "data-根本不存在"))


def _is_not_found(runtime: DatasetRuntime, task_id: str) -> bool:
    try:
        runtime.get_task(task_id)
    except LoraTrainNotFoundError:
        return True
    return False


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="lora_train_case_"))
    case_options()
    case_iter_output()
    case_log_buffer()
    case_dataset_files(root)
    case_dataset_runtime(root)

    # ⚠️ 命令层要求两套工具**真在本机** ✓（reference/ 是开发期脚手架 ✓ 不在就跳过 ✓，
    #    而不是让它以"报了个文件找不到"的假象红掉 ✓）
    missing = [toolkit for toolkit in (paths.TOOLKIT_MUSUBI, paths.TOOLKIT_SD_SCRIPTS)
               if not paths.tool_available(
                   options.spec("wan" if toolkit == paths.TOOLKIT_MUSUBI
                                else "flux").train_script, toolkit=toolkit)]
    if missing:
        skip("缺工具 ⇒ 命令层 argv 结构那组跳过 ✓（缺：" + " / ".join(missing)
             + " ✓；装到 reference/lora/ 或设 MUSUBI_TUNER_ROOT / SD_SCRIPTS_ROOT ✓）")
    else:
        case_commands(root)
        case_runner()

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name
              + ("" if passed else f"   <<< {detail!r}"))
    for reason in _SKIPS:
        print("SKIP  " + reason)
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed"
          + (f"（skip {len(_SKIPS)}）" if _SKIPS else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
