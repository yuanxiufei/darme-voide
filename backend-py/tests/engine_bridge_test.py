"""S? 自检：**业务链路 ↔ 自研引擎的桥** ✓（``provider=engine`` 那条路 —— 2026-09-26 ✓）。

这一套钉的是「**"不依赖外部"这句承诺到底成不成立**」✓，不是"函数返回了个 dict"✗：

1. **装配键集逐键对账** ✓ —— 桥产出什么、运行时认什么，两边**同一个来源**核对 ✓
   （多一个键运行时当场 ``TypeError`` 炸 ✗ ⇒ 判据**不许靠手抄**两边清单 ✓✗）；
2. **缺 DiT 就报** ✗ —— ``settings.ditPath`` 没给、清单里也没有在盘上的权重 ⇒ 报错，
   且报错里必须带**怎么修** ✓（不说"引擎不可用"一句糊过去 ✗，也**绝不猜一个"差不多"的路径** ✓✗）；
3. **取整口径必填** ✓ —— ``audioLatentMode`` 非法 ⇒ 报错 ✗；不给 ⇒ 装配里**就没有**它 ✓
   ⇒ 双流不开 ✓（**不是**静默给个默认 ✓✗）；
4. **产物口径** ✓ —— 落点、相对数据根的 ``static/...``、参考素材（data URL → 真文件 ✓、
   远程 URL ⇒ **明确不认** ✓ —— 引擎不联网 ✓）；
5. **真跑一次** ✓ —— 缩小版 H3 形态权重 ⇒ 真 mp4 ⇒ **首帧真 PNG** ✓（图片链路那条 ✓）；
6. **反向证明** ✓ —— 跑的时候把 httpx **打死** ✗ ⇒ 照样出片 ✓（"进程内 + 零 HTTP"不是口号 ✓）；
7. **权重/词表怎么被找到** ✓（2026-09-26 补 ✓）—— 换根不换名 ✓、清单 ``file_path`` 是**远端**路径
   ⇒ 本地落点只跟**安装器**一个口径 ✓（跨模块对账 ✓）、词表按上游约定找 ✓（两样缺谁当场报 ✗）。

运行::

    ./.venv/Scripts/python.exe tests/engine_bridge_test.py
"""
from __future__ import annotations

import asyncio
import inspect
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="enginebridge_"))
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.core.config import get_data_root, get_storage_root  # noqa: E402
from app.services.engine import bridge  # noqa: E402
from app.services.engine import geometry  # noqa: E402
from app.services.engine import h3_form  # noqa: E402
from app.services.engine import inventory as inventory_mod  # noqa: E402
from app.services.engine import media as media_mod  # noqa: E402
from app.services.engine import weights as weights_mod  # noqa: E402
from app.services.engine.runtime import EngineRuntime, engine_runtime  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(reason: str) -> None:
    _SKIPS.append(reason)


def _have_torch() -> bool:
    try:
        import torch  # noqa: F401,PLC0415
    except ImportError:
        return False
    return True


#: **缩小版** H3 主干 ✓（只改尺寸 ✓ —— 模块名 / 键名与真权重一致 ✓ ⇒ 形态判别照样认它 ✓）
TRUNK: dict[str, Any] = {
    "hidden": 32, "layers": 2, "heads": 4, "head_dim": 16, "ffn": 64,
    "text_dim": 16, "latents_dim": 4, "audio_latents_dim": 8, "patch_size": (1, 2, 2),
    "time_input_dim": 16, "time_hidden": 32, "time_dim": 32, "inv_freq_len": 2,
    "refiner_layers": 1,
}


# ══════════════════════════════════════════════════════════════════════════
# ① provider 判定 + settings 归一化
# ══════════════════════════════════════════════════════════════════════════
def case_provider() -> None:
    check("provider: ``engine`` 认（大小写 / 空格都认）",
          all(bridge.is_engine_provider(v) for v in ("engine", "Engine", " ENGINE ")))
    check("provider: 别的都不认（ollama / local-sd / minimax / 空 / None）",
          not any(bridge.is_engine_provider(v)
                  for v in ("ollama", "local-sd", "minimax", "", None, "engine-x")))


def case_settings() -> None:
    normalized = bridge.normalize_settings({
        "settings": {"ditPath": "D:/x.safetensors", "audioLatentMode": "round",
                     "megapixels": 0.05, "tokenizerPath": "t.json"},
    })
    check("settings: camelCase ⇒ snake_case（契约是 camel ✓ 两种都收 ✓）",
          set(normalized) == {"dit_path", "audio_latent_mode", "megapixels", "tokenizer_path"},
          sorted(normalized))
    check("settings: JSON 字符串也收（跨语言配置里出现过 ✓）",
          bridge.normalize_settings({"settings": '{"ditPath": "a"}'}) == {"dit_path": "a"})
    check("settings: 读不动 ⇒ 当**没配**（不是抛错、也不是编一个默认值）",
          bridge.normalize_settings({"settings": "不是JSON"}) == {}
          and bridge.normalize_settings({"settings": 7}) == {}
          and bridge.normalize_settings({}) == {}
          and bridge.normalize_settings(None) == {})


# ══════════════════════════════════════════════════════════════════════════
# ② DiT 主权重：settings 优先 / 清单兜底 / **换根不换名** / 都没有就报
#    ⚠️ 2026-09-26 补第 ③ 级（换根 ✓）：真机实测本机 DiT 只长在 ``…/models/diffusion_models/`` ✓，
#    而那时只认 ``models_dir`` ✓✗ ⇒ **扫描看得见、视频看不见** ✓✗（与出图那条同一个病 ✓）。
# ══════════════════════════════════════════════════════════════════════════
def case_dit(tmp: Path) -> None:
    path, source = bridge.resolve_dit_path({"dit_path": "D:/真权重.safetensors"})
    check("DiT: ``settings.ditPath`` 永远优先，且**来源如实标注**",
          path == "D:/真权重.safetensors" and source == "settings.ditPath", (path, source))

    saved_catalog, saved_dir = inventory_mod.load_catalog, inventory_mod.models_dir
    saved_roots = inventory_mod.candidate_roots
    try:
        # ⓐ 清单空 ⇒ 必须**报**（不是给个默认、"引擎不可用"一句糊过去）
        inventory_mod.load_catalog = lambda: {"models": []}
        try:
            bridge.resolve_dit_path({})
        except ValueError as err:
            text = str(err)
            check("⭐ DiT: 清单也没有 ⇒ 报错，且**带怎么修**（可行动 ✓）",
                  "ditPath" in text and "怎么修" in text, text[:100])
        else:
            check("DiT: 居然没报（这条该红 ⇒ 检查是不是偷偷猜了个路径 ✗）", False, "no raise")

        # ⓑ 清单里有登记、但**文件不在盘上** ⇒ 还是必须报（"登记了" ≠ "能用" ✓）
        root = tmp / "清单根"
        entry = {"key": "dit_fl2va_int8", "kind": "diffusion_models", "filename": "h3.safetensors"}
        inventory_mod.load_catalog = lambda: {"models": [entry]}
        inventory_mod.models_dir = lambda: root
        # ⚠️ **动态探测**的根也要一起钉死 ✗✗：只钉 ``models_dir`` ⇒ 第 ③ 级照样会去翻**真实磁盘** ✓✗
        #    ⇒ 这条自检的结论随「本机恰好装了什么」而变 ✓✗（本仓铁律：任何机器上同一个结论 ✓）
        inventory_mod.candidate_roots = lambda: [(root, "自检临时根")]
        try:
            bridge.resolve_dit_path({})
        except ValueError:
            check("⭐ DiT: 清单登记了但**盘上没有** ⇒ 照样报 ✓（「登记了」 ≠ 「能用」 ✓）", True)
        else:
            check("DiT: 盘上没有却挑中了（这条该红）", False, "no raise")

        # ⓒ 文件真在盘上 ⇒ 挑它 ✓，来源里能看见**是哪个登记项**
        landed = inventory_mod.component_path(entry)
        landed.parent.mkdir(parents=True, exist_ok=True)
        landed.write_bytes(b"not-a-real-checkpoint")
        path, source = bridge.resolve_dit_path({})
        check("DiT: 文件真在盘上 ⇒ 按清单挑中，来源写出**登记项的 key**（可回看 ✓）",
              Path(path) == landed and "dit_fl2va_int8" in source, (path, source))

        # ⓓ **本机的装法**（只长在 ``<根>/<kind>/`` 里 ✓）也得认得下来 ✓ —— 2026-09-26 真机实测：
        #    探测根里的 DiT 就在 ``…/models/diffusion_models/`` ✓，而以前只认 ``models_dir`` ✓✗
        #    ⇒ **扫描看得见、视频看不见** ✓✗（同一台机器上）
        elsewhere = tmp / "别处根"
        moved = elsewhere / "diffusion_models" / "h3.safetensors"
        moved.parent.mkdir(parents=True)
        moved.write_bytes(b"x")
        landed.unlink()
        inventory_mod.candidate_roots = lambda: [(elsewhere, "自检临时根")]
        path, source = bridge.resolve_dit_path({})
        check("⭐ DiT: 只长在 ``<根>/<diffusion_models>/`` 里的那一份也认得下来 ✓"
              "（换根不换名 ✓ ⇒ 与出图那条**同一套**口径 ✓）",
              Path(path) == moved and "自检临时根" in source, (path, source))

        # ⚠️ 换根也**不换名** ✗：同一个目录里摆着**别的**权重 ⇒ 一个都不认 ✓（不猜权重 ✓）
        moved.rename(elsewhere / "diffusion_models" / "h3-fl2va-pruned-int8.safetensors")
        try:
            bridge.resolve_dit_path({})
        except ValueError:
            check("⭐ DiT: 换根**不换名** ✗（``diffusion_models/`` 里摆的是别的权重 ⇒ 不认 ✓）", True)
        else:
            check("DiT: 名字对不上却挑中了（这条该红 ⇒ 是不是偷偷「挑了个像样的」 ✗）", False, "no raise")
    finally:
        inventory_mod.load_catalog, inventory_mod.models_dir = saved_catalog, saved_dir
        inventory_mod.candidate_roots = saved_roots


# ══════════════════════════════════════════════════════════════════════════
# ③ 装配：键集与运行时**逐键对账** + 不猜取整口径
# ══════════════════════════════════════════════════════════════════════════
def case_assembly() -> None:
    allowed = set(inspect.signature(EngineRuntime._ensure_loaded_internal).parameters) - {"self"}
    assembly = bridge.assembly_from_config({"settings": {"ditPath": "D:/x.safetensors"}})
    check("⭐ 装配: 每个键运行时都**真的认**（多一个键运行时当场 TypeError ✗ ⇒ 判据不许手抄清单 ✓）",
          set(assembly) <= allowed, (sorted(assembly), sorted(allowed - set(assembly))))
    check("装配: 必需项 ``dit_path`` 在（缺它就没得装 ✓）", "dit_path" in assembly, sorted(assembly))
    check("⭐ 装配: 没给 ``audioLatentMode`` ⇒ 装配里**就没有**它（⇒ 双流不开 ✓，不是静默给默认 ✓✗）",
          "audio_latent_mode" not in assembly, sorted(assembly))

    try:
        bridge.assembly_from_config({"settings": {"ditPath": "x", "audioLatentMode": "大概"}})
    except ValueError as err:
        check("装配: 取整口径非法 ⇒ **报**（不许猜一个看着对的 ✗）",
              "round" in str(err) and "ceil" in str(err), str(err)[:80])
    else:
        check("装配: 非法取整口径居然过了（这条该红）", False, "no raise")

    ok = bridge.assembly_from_config({"settings": {"ditPath": "x", "audioLatentMode": "CEIL"}})
    check("装配: 合法取整口径归一成小写并带上",
          ok.get("audio_latent_mode") == "ceil", ok)


# ══════════════════════════════════════════════════════════════════════════
# ④ 落点 / 相对路径 / 参考素材
# ══════════════════════════════════════════════════════════════════════════
def case_paths(tmp: Path) -> None:
    out = bridge.outputs_dir_for("video", 7)
    check("落点: ``<storageRoot>/engine/<kind>-<id>/``（在数据目录里 ✓ ⇒ 好被 /static 服务 ✓）",
          out == Path(get_storage_root()) / "engine" / "video-7", str(out))

    inside = Path(get_storage_root()) / "images" / "a.png"
    check("相对路径: 数据根**里面** ⇒ ``static/...``（与 download_file 同一口径 ✓）",
          bridge.static_relative(inside) == "static/images/a.png", bridge.static_relative(inside))
    outside = tmp / "别处" / "b.mp4"
    check("⭐ 相对路径: 数据根**外面** ⇒ 原样返回（**不假装**它是个 static URL ✓✗）",
          bridge.static_relative(outside) == str(outside), bridge.static_relative(outside))

    data_url = ("data:image/png;base64,"
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
    landed = bridge.materialize_reference(data_url)
    check("参考素材: data URL ⇒ **真落成文件**（引擎读的是文件，不是 URL ✓）",
          bool(landed) and Path(str(landed)).exists() and Path(str(landed)).read_bytes()[:4] == b"\x89PNG",
          landed)
    check("参考素材: 远程 URL ⇒ **明确不认**（引擎不联网 ✓ ⇒ 不许把 URL 当路径丢进去 ✓✗）",
          bridge.materialize_reference("https://example.com/a.png") is None)
    check("参考素材: 空 / 垃圾 ⇒ None（不是空字符串路径 ✓）",
          bridge.materialize_reference("") is None and bridge.materialize_reference(None) is None)

    local = Path(get_storage_root()) / "engine" / "refs" / "c.png"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(b"x")
    check("参考素材: 数据根里的相对路径 ⇒ 能解成**本机绝对路径**",
          bridge.materialize_reference("static/engine/refs/c.png") == str(local),
          bridge.materialize_reference("static/engine/refs/c.png"))

    check("参考素材: 一串里**解不出的那条只 warn 不静默丢**（长度如实变短 ✓）",
          bridge.materialize_references(["https://x/a.png", "static/engine/refs/c.png"])
          == (str(local),),
          bridge.materialize_references(["https://x/a.png", "static/engine/refs/c.png"]))


# ══════════════════════════════════════════════════════════════════════════
# ⑤ 记录 ⇒ 请求（图片 / 视频）
# ══════════════════════════════════════════════════════════════════════════
def case_requests(tmp: Path) -> None:
    out = tmp / "out"
    image = bridge.request_from_image_record(
        {"prompt": "雨夜霓虹街头", "negativePrompt": "模糊", "size": "224x224"},
        outputs_dir=out, settings={})
    check("图片请求: ``1920x1080`` 这类 size ⇒ ratio **原样解析**（不是四舍五入到 16:9 ✓）",
          tuple(image.ratio) == (224, 224), image.ratio)
    check("图片请求: 像素预算由**引擎自己的** ``megapixels_for_size`` 算"
          "（口径只该有一个来源 ✓ —— 桥**不许自己发明**一条公式 ✗）",
          image.megapixels == geometry.megapixels_for_size(224, 224)
          and 0.04 < image.megapixels < 0.06, image.megapixels)
    check("图片请求: 时长走**最短那段**（H3 是视频模型 ✓ ⇒ 图片是首帧派生的 ✓）",
          image.seconds == 0.2, image.seconds)
    check("图片请求: 落点就是调用方给的（不许偷偷落临时目录 ✓）",
          str(image.outputs_dir) == str(out), image.outputs_dir)
    check("图片请求: 双流口径**没给就不带**（⇒ 不出音轨 ✓ 如实 ✓，不是塞个默认值 ✗）",
          image.temporal_compression is None, image.temporal_compression)

    video = bridge.request_from_video_record(
        {"prompt": "p", "duration": 5, "aspectRatio": "9:16"}, outputs_dir=out, settings={})
    check("视频请求: ``duration`` ⇒ ``seconds``", video.seconds == 5.0, video.seconds)
    check("视频请求: ``aspectRatio`` ⇒ ``ratio``", video.ratio == "9:16", video.ratio)

    explicit = bridge.request_from_image_record(
        {"prompt": "p", "size": "不解析的写法"}, outputs_dir=out,
        settings={"megapixels": 0.25, "steps": 8, "sampler": "euler", "seed": 3})
    check("请求: size 解不动时**不编 ratio**（交给引擎自己的默认 ✓）",
          explicit.megapixels == 0.25 and explicit.steps == 8 and explicit.seed == 3,
          (explicit.megapixels, explicit.steps, explicit.seed))

    artifacts = bridge.artifact_paths({"result": {"outputs": {"videoPath": "v.mp4", "audioPath": None}}})
    check("产物: 从任务事实里**如实**取（没有的音轨就是 None ✓ 不编一个 ✓）",
          artifacts["videoPath"] == "v.mp4" and artifacts["audioPath"] is None, artifacts)


# ══════════════════════════════════════════════════════════════════════════
# ⑥ 跑：干跑（快 ✓ 验事件搬运 + 如实报"没音轨"）+ 真跑（慢 ✓ 验真产物）
# ══════════════════════════════════════════════════════════════════════════
class _LogSpy:
    """拦下 ``task_logger`` ✓ —— 钉的是"引擎的事件**真的搬出去了**"✓（不是"搬了个空列表"✗）。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def __call__(self, scope: str, action: str, meta: dict[str, Any] | None = None) -> None:
        self.calls.append((scope, action, dict(meta or {})))

    def actions(self) -> list[str]:
        return [action for _scope, action, _meta in self.calls]

    def find(self, action: str) -> list[dict[str, Any]]:
        return [meta for _scope, act, meta in self.calls if act == action]


def case_dry_job(tmp: Path) -> None:
    """干跑 ✓：**快**且**不装权重** ⇒ 正好用来钉「事件搬运」与「没音轨就如实说」两条 ✓。"""
    progress, warn, error = _LogSpy(), _LogSpy(), _LogSpy()
    saved = (bridge.log_task_progress, bridge.log_task_warn, bridge.log_task_error)
    bridge.log_task_progress, bridge.log_task_warn, bridge.log_task_error = progress, warn, error  # type: ignore[assignment]
    try:
        request = bridge.request_from_image_record(
            {"prompt": "干跑", "size": "224x224"}, outputs_dir=tmp / "dry", settings={})
        task = asyncio.run(bridge.run_job(request, {"dryRun": True},
                                          task_type="ImageTask", record_id=1, label="图片"))
    finally:
        bridge.log_task_progress, bridge.log_task_warn, bridge.log_task_error = saved

    check("干跑: 任务 done（不是靠等待超时「看着像完成」✓）",
          task["status"] == "done", task.get("status"))
    actions = progress.actions()
    check("⭐ 干跑: 事件**逐条搬出去了**（engine-submit + 真阶段的 engine-* + engine-done）",
          "engine-submit" in actions and "engine-done" in actions
          and any(a.startswith("engine-") and a not in ("engine-submit", "engine-done")
                  for a in actions),
          actions)
    check("干跑: 完成那条带**引擎任务号 + 记录 id**（前端按它归位 ✓）",
          all("engineTaskId" in meta and meta.get("id") == 1
              for meta in progress.find("engine-done")),
          progress.find("engine-done")[:1])
    check("干跑: 没失败就不该有 error 日志（失败才记 ✗）", not error.calls, error.calls)
    # ⚠️ 干跑后端不落音轨 ⇒ 「如实说没有」这条正好在这里验 ✓（真跑那边反而验不到 ✓）
    check("⭐ 干跑: 没音轨时**如实记一条 warn**（不是失败 ✓、更不是假装有音轨 ✓✗）",
          "engine-no-audio" in warn.actions(), warn.actions())


def case_real_run(tmp: Path) -> None:
    checkpoint = weights_mod.save_module_weights(h3_form.H3FormTrunk(**TRUNK),
                                                 tmp / "h3_bridge.safetensors")
    config = {
        "provider": "engine",
        "model": "h3",
        "settings": {"ditPath": str(checkpoint), "audioLatentMode": "round",
                     "temporalCompression": 1, "steps": 3},
    }
    outputs_dir = tmp / "real"
    settings = bridge.normalize_settings(config)
    request = bridge.request_from_image_record(
        {"prompt": "雨夜霓虹街头", "size": "224x224"},
        outputs_dir=outputs_dir, settings=settings)
    assembly = bridge.assembly_from_config(config)

    try:
        task = asyncio.run(bridge.run_job(request, assembly, task_type="ImageTask",
                                          record_id=2, label="图片"))
    except Exception as err:  # noqa: BLE001 —— 抛错就如实记一条失败 ✓（别让整套自检中断 ✓）
        check("真跑: 跑完不抛错", False, f"{type(err).__name__}: {err}")
        return
    check("真跑: 任务 done", task["status"] == "done", (task.get("status"), task.get("error")))
    check("⭐ 真跑: synthetic **仍为 True**（参考 TE / VAE 未训练 ⇒ 不许改口说是真画面 ✗）",
          (task.get("result") or {}).get("synthetic") is True,
          (task.get("result") or {}).get("synthetic"))

    paths = bridge.artifact_paths(task)
    video = paths.get("videoPath")
    check("真跑: **真 mp4** 落盘且落在**指定目录**里",
          bool(video) and Path(str(video)).exists() and Path(str(video)).stat().st_size > 0
          and Path(str(video)).parent == outputs_dir,
          (video, str(outputs_dir)))
    check("真跑: 双流开了 ⇒ **真 wav** 也在（``audioLatentMode`` 显式给了 ✓）",
          bool(paths.get("audioPath")) and Path(str(paths["audioPath"])).exists(),
          paths.get("audioPath"))

    still = asyncio.run(bridge.extract_still_to_storage(str(video), record_id=2))
    still_path = Path(str(still["absolutePath"]))
    check("⭐ 真跑: 视频 ⇒ **首帧真 PNG**（图片链路那条 ✓ —— 不是「引擎直接画了张图」✗）",
          still_path.exists() and still_path.read_bytes()[:4] == b"\x89PNG",
          (still.get("localPath"), still_path.exists()))
    check("真跑: 首帧落在**数据目录里**、报的是相对数据根的 ``static/images/...``",
          str(still["localPath"]).startswith("static/images/")
          and still_path.parent == Path(get_storage_root()) / "images",
          still.get("localPath"))
    check("真跑: 抽帧事实里带**源视频帧数**（可回看「从几步的视频里抽的」 ✓）",
          int(still.get("sourceFrames") or 0) > 0, still.get("sourceFrames"))


#: 桥里**绝对不许**出现的门名 ✓（HTTP 客户端 ✓ / 外部推理服务 ✓）—— 比对口径见 :func:`_code_words` ✓
_BANNED_GATES = ("httpx", "requests", "aiohttp", "websocket", "urlopen", "comfyui")


def _code_words(path: Path) -> set[str]:
    """源码里**真正当名字/字面量用**的词 ✓ ⇒ 小写集合 ✓（``tokenize`` ✓ 纯标准库 ✓）。

    ⚠️ 为什么**不是** ``token in source`` ✗✗（这条自检**曾经的真 bug** ✓ 2026-09-26 抓到 ✓）：
    整份源码做子串匹配 ⇒ **说明文字**与**更长的标识符**照样算命中 ✓✗。本仓的桥里就有一句
    "不 HTTP 调 ComfyUI ✗" 的 docstring、一个叫 ``get_comfyui_roots`` 的**本地扫盘**助手 ✓
    （它只是**读目录** ✗ 根本不是门 ✓）⇒ 守卫被点着 ✓✗，而报出来的 ``['comfyui']`` 跟真正
    要防的"外部门"**完全不是一回事** ✓✗ —— 看日志的人只会被支去查一堆不存在的 HTTP 调用 ✓。

    ⇒ 这里只取两类**代码**事实 ✓：
    ① ``NAME`` 记号 ✓（**整词**比对 ✓ ⇒ ``get_comfyui_roots`` **不等于** ``comfyui`` ✓）；
    ② **单个**字符串字面量的内容 ✓（堵住 ``import_module("httpx")`` 这种藏起来的门 ✓；
    ⚠️ 一整句 docstring 不会等于任何一个门名 ✓）。注释里的字**一律不算** ✓。
    """
    import tokenize  # noqa: PLC0415

    words: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for token in tokenize.generate_tokens(handle.readline):
            if token.type == tokenize.NAME:
                words.add(token.string.lower())
            elif token.type == tokenize.STRING:
                body = token.string.lstrip("rbufRBUF")  # 前缀（``f""`` / ``rb""`` 之类 ✓）
                for quote in ('"""', "'''", '"', "'"):
                    if body.startswith(quote) and body.endswith(quote) and len(body) > 2 * len(quote):
                        words.add(body[len(quote):-len(quote)].strip().lower())
                        break
    return words


def case_no_http_during_run(tmp: Path) -> None:
    """⭐⭐ **反向证明** ✓：把 httpx 打死 ⇒ 这条路照样跑完 ✓（"零 HTTP"不是口号 ✓）。"""
    import httpx  # noqa: PLC0415

    async def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("这条路上**不许**发 HTTP ✗（不依赖外部 ✓）")

    saved = httpx.AsyncClient.send
    httpx.AsyncClient.send = _boom  # type: ignore[method-assign]
    try:
        request = bridge.request_from_image_record(
            {"prompt": "零HTTP", "size": "224x224"}, outputs_dir=tmp / "nohttp", settings={})
        task = asyncio.run(bridge.run_job(request, {"dryRun": True},
                                         task_type="ImageTask", record_id=3, label="图片"))
    finally:
        httpx.AsyncClient.send = saved  # type: ignore[method-assign]
    check("⭐ 零 HTTP: 把 httpx 打死之后任务**照样 done**",
          task["status"] == "done", task.get("status"))

    words = _code_words(Path(bridge.__file__))
    hits = sorted(set(_BANNED_GATES) & words)
    check("零 HTTP: 桥的**代码**里没有 httpx / requests / comfyui 这类外部门"
          "（源码级反证 ✓ —— 比的是**名字/字面量**，说明文字不算 ✓✗）", not hits, hits)

    # ⚠️ **守卫自己**也得能证伪 ✗✗（否则它只是一行**永远绿**的字 ✓✗ —— 本仓最忌那种"看着安心" ✓）：
    #    真门（``import httpx`` ✓、``"requests"`` 这种藏起来的 ✓）必须抓得住 ✓；
    #    而"提一句 ComfyUI 的说明文字 ✓"与"``get_comfyui_roots`` 这种**本地扫盘**名字 ✓"**不许**误伤 ✓✗。
    probe = tmp / "假外部门.txt"
    probe.write_text(
        '"""这条路上**不许**发 HTTP ✗（ComfyUI / SD WebUI 一概不调 ✓）"""\n'
        "from app.services.local_model_scan import get_comfyui_roots\n"
        "import httpx\n"
        'MODULE = "requests"\n',
        encoding="utf-8")
    probe_hits = sorted(set(_BANNED_GATES) & _code_words(probe))
    check("⭐ 零 HTTP: 守卫**能证伪**（真门抓得住 ✓、说明文字与本地扫盘名字不误伤 ✓✗）",
          probe_hits == ["httpx", "requests"], probe_hits)


def case_stages(tmp: Path) -> None:
    """⭐ 两条**阶段**名的三处口径必须**同一份** ✓ + 请求/装配的默认必须**落到同一条路** ✓。

    ``stage`` 一个词串起三层 ✓：``bridge``（字面量 ✓，刻意只做懒加载）✓、``pipeline``（σ 调度与
    帧数的分流依据 ✓）、``SdxlBackend.name``（运行时的分流依据 ✓）。⚠️ 分叉的后果不是"报错" ✗
    而是「装得进去、跑得出来、结果全错」✓✗ ⇒ 这里三处**逐个钉住** ✓。
    """
    from app.services.engine import pipeline  # noqa: PLC0415
    from app.services.engine.sdxl_backend import SdxlBackend  # noqa: PLC0415

    check("⭐ 阶段名三处一致：``bridge`` == ``pipeline`` == ``sdxl_backend`` ✓"
          "（桥刻意用字面量 ✗ 免得拖进引擎层 ✓ ⇒ 必须有这条钉住 ✓）",
          bridge.VIDEO_STAGE == pipeline.VIDEO_STAGE == "h3"
          and bridge.IMAGE_STAGE == pipeline.IMAGE_STAGE == SdxlBackend.name == "sdxl",
          (bridge.VIDEO_STAGE, pipeline.VIDEO_STAGE, bridge.IMAGE_STAGE, SdxlBackend.name))

    # ⚠️ 两个都不显式传 ⇒ 必须落到**同一条**路 ✓（两处默认各写各的 ⇒ 请求按图建 + 权重按视频装 ✓✗）
    request = bridge.request_from_image_record({"prompt": "p", "size": "224x224"},
                                               outputs_dir=tmp / "stage", settings={})
    assembly = bridge.assembly_from_config({"settings": {"ditPath": "D:/x.safetensors"}})
    check("⭐ 请求与装配的**默认 stage 一致** ✓（异口同声 ⇒ 不传也不会错配 ✓）",
          request.stage == assembly.get("stage") == bridge.VIDEO_STAGE,
          (request.stage, assembly.get("stage")))

    # ⚠️ 真错配（请求按图 ✓ + 装配按视频 ✓）⇒ **当场报** ✗，别等到"出来一张看不懂的图" ✓✗
    image_request = bridge.request_from_image_record(
        {"prompt": "p", "size": "224x224"}, outputs_dir=tmp / "stage", settings={},
        stage=bridge.IMAGE_STAGE)
    mismatch: object = None
    try:
        asyncio.run(bridge.run_job(image_request, assembly, task_type="ImageTask",
                                   record_id=99, label="图片"))
    except Exception as err:  # noqa: BLE001
        mismatch = err
    check("⭐ 请求 ⇄ 装配 stage 错配 ⇒ **当场报错并说清怎么修** ✗"
          "（不报 ⇒ H3 的 DiT 去解 SDXL 的潜变量 ✓✗ —— 最坏情况还不报错 ✓）",
          isinstance(mismatch, ValueError) and "stage" in str(mismatch)
          and "怎么修" in str(mismatch), mismatch)


def case_business_wiring() -> None:
    """⭐ 业务侧**真的接上了**没有 ✓ —— 而且是**接在正确的那个位置** ✓。

    ``provider=engine`` 在 registry 里**没有适配器** ✗ ⇒ 如果分叉写在取适配器**之后** ✓✗，
    这条路会在第一步就撞 ``KeyError`` ✓ —— **功能是死的** ✓✗，而单测照样全绿 ✓（因为它从不走那条分支 ✓）。
    所以这里不比字符串、直接比**源码里的先后顺序** ✓（唯一能钉死"顺序"的机械判据 ✓）。
    """
    from app.services import image_generation, video_generation  # noqa: PLC0415
    from app.services.ai_configs import LOCAL_PROVIDERS  # noqa: PLC0415

    for label, module, adapter_call in (
        ("图片流", image_generation, "get_image_adapter("),
        ("视频流", video_generation, "get_video_adapter("),
    ):
        source = Path(module.__file__).read_text(encoding="utf-8")
        fork = source.index("is_engine_provider")
        adapter = source.index(adapter_call)
        check(f"⭐ {label}: 引擎分叉**在取适配器之前**（分叉写在后面 ⇒ 这条路第一步就死 ✗）",
              fork < adapter, (fork, adapter))
        check(f"{label}: 恢复分支也认 ``engine``（重启后**判失败**而不是「当成外部任务去轮询」 ✗）",
              source.count("is_engine_provider") >= 2, source.count("is_engine_provider"))

    check("配置层: ``engine`` 在白名单里 ⇒ 按**本机**处理（不会当云服务去要 URL / 走计费 ✓）",
          "engine" in LOCAL_PROVIDERS, sorted(LOCAL_PROVIDERS))


# ══════════════════════════════════════════════════════════════════════════
# ⑨ 出图要的那**两样在盘上的东西**：SDXL 主权重（**换根不换名** ✓）+ CLIP 词表 ✓
#    ⚠️ 2026-09-26 补 ✓：以前这两样各缺一半 —— 权重只认 ``models_dir`` ✓（**动态探测**出来的
#    ComfyUI 目录**看不见** ✗）、词表在**生产侧根本没有解析** ✗（只能靠人工配 ``settings`` ✓✗）
#    ⇒ 症状正是「扫描器看得见这台机器的 SDXL ✓、出图却找不到它 ✗」＋「缺词表要等到**装载**才炸 ✓✗」。
# ══════════════════════════════════════════════════════════════════════════
def case_candidates(tmp: Path) -> None:
    """清单条目 ⇒ **候选落点** ✓：``<根>/<kind>/<文件>`` ✓、根可以有很多个 ✓、**文件名只认清单** ✗。"""
    entry = {"key": "sdxl_base", "kind": "checkpoints",
             "filename": "sd_xl_base_1.0.safetensors",
             "file_path": "sd_xl_base_1.0.safetensors"}

    # ⓐ 「按类别分子目录」那种根（ComfyUI 惯例 ✓，也正是**安装器**的落点 ✓）⇒ 只试这一层 ✓
    nested_root = tmp / "假ComfyUI-models"
    (nested_root / "checkpoints").mkdir(parents=True)
    weights = nested_root / "checkpoints" / "sd_xl_base_1.0.safetensors"
    weights.write_bytes(b"x")
    roots = [(nested_root, "临时根")]
    listed = [str(item) for item, _ in inventory_mod.component_candidates(entry, roots)]
    check("⭐ 候选: 只在 ``<根>/<kind>/<文件>`` 这一层找 ✓（本机那份 SDXL 与**安装器**的落点都长这样 ✓）",
          listed == [str(weights)], listed)
    # ⚠️ 清单里的 ``file_path`` 是**远端**仓库内的路径 ✓（``model_manager.build_download_url`` 用的就是它 ✓）
    #    ⇒ 拿它当**本地**落点会去找 ``<根>/sd_xl_base_1.0.safetensors`` ✓✗ —— SDXL 那个 HF 仓库**是平的** ✓，
    #    本地却在 ``checkpoints/`` 里 ✓（两个事实都真 ✓，混用就错 ✓✗）
    spot = inventory_mod.component_path(entry, nested_root)
    check("⭐ 落点: ``file_path`` **不作**本地落点 ✓（HF 仓库平 ⟂ 本地平 ✗ —— 2026-09-26 修的就是这条 ✓）",
          spot == weights and spot != nested_root / "sd_xl_base_1.0.safetensors", spot)
    # ⚠️ 传 ``roots``（**根表** ✓）而不是只传根 ✗：候选表**本身**就是给 ``resolve_component`` 的
    #    根表口径 ✓ ⇒ 这里**圈定了域** ⇒ 结论与本机 ``models_dir`` 里恰好有没有那份权重**无关** ✓
    found, source = inventory_mod.resolve_component(entry, roots=roots)
    check("⭐ 候选: **只**长在 ``<kind>/`` 里的那一份也认得下来（那台机器的装法就是这样 ✓）",
          found == weights and source == "临时根", (found, source))

    # ⓑ 根**本身**就是那个类别目录（本机根表里，每个类别目录**各是一条根** ✓）⇒ 别再拼一层 ✗
    kind_root = tmp / "假ComfyUI" / "checkpoints"
    kind_root.mkdir(parents=True)
    (kind_root / "sd_xl_base_1.0.safetensors").write_bytes(b"x")
    here = [str(item) for item, _ in inventory_mod.component_candidates(entry, [(kind_root, "临时根")])]
    check("⭐ 候选: 根已是 ``<kind>`` 目录 ⇒ **只试平的那一种**"
          "（否则造出 ``…/checkpoints/checkpoints/…`` 这种**不存在**的形状 ✗）",
          here == [str(kind_root / "sd_xl_base_1.0.safetensors")], here)

    # ⓒ 名字对不上 ⇒ 一个都不认 ✓（"挑个像样的权重"是**绝对禁止**的 ✓✗：猜中的代价是结果全错 ✓）
    other = tmp / "别的权重"
    (other / "checkpoints").mkdir(parents=True)
    (other / "checkpoints" / "sd_xl_refiner_1.0.safetensors").write_bytes(b"x")
    missed, missed_source = inventory_mod.resolve_component(entry, roots=[(other, "临时根")])
    check("⭐ 候选: 名字对不上就**一个都不认**（宁缺勿错 ✓ —— 本仓不猜权重 ✗）",
          missed is None and missed_source == "", (missed, missed_source))

    # ⓓ 根被调用方**钉死**（自检注入临时目录 ✓）⇒ **绝不**再翻真实磁盘 ✓
    empty = tmp / "空根"
    empty.mkdir()
    pinned, pinned_source = inventory_mod.resolve_component(entry, empty)
    check("⭐ 候选: 根被钉住 ⇒ 只认它（否则自检结论**随机器变** ✓✗ —— 装了 SDXL 的那台机器上，"
          "这条会\"找到\"那份真实权重 ✓✗，于是同一份自检在两台机器上给两个答案 ✓）",
          pinned is None and pinned_source == "", (pinned, pinned_source))


def case_component_spots(tmp: Path) -> None:
    """⭐⭐ **落点只有一份口径** ✗：体检算出来的（:func:`inventory.component_path` ✓），必须**逐字就是**
    安装器（``app/scripts/model_manager.py`` ✓）装 / 查 / 删**三处**算的那个位置 ✓。

    ⚠️ 为什么要跨模块对账 ✗✗：这两边各写各的 ⇒ 分歧时**两边都是"对的"** ✓✗，而症状是
    「同一台机器上『装好了』与『体检说没装』同时成立」✓✗ —— 2026-09-26 就是这么发生的（体检读 ``file_path`` ✓
    ⇒ 算成 ``<models_dir>/sd_xl_base_1.0.safetensors`` ✓；安装器写的是 ``<models_dir>/checkpoints/…`` ✓）。
    ⇒ 判据不许手抄公式 ✗（手抄的那份**自己也会漂** ✓✗）：直接调安装器自己的 ``model_status`` ✓。
    """
    import importlib.util  # noqa: PLC0415

    # ⚠️ ``app/scripts/`` **不是包** ✗（见 ``engine_readiness_script_test.py`` 的说明 ✓）⇒ 只能按**路径**装 ✓
    manager_path = BACKEND_PY / "app" / "scripts" / "model_manager.py"
    spec = importlib.util.spec_from_file_location("model_manager_spots", manager_path)
    manager = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(manager)  # type: ignore[union-attr]

    root = tmp / "清单根"
    root.mkdir(parents=True, exist_ok=True)  # 空目录 ✓（别的用例也用这个名字 ✓）
    paths = manager.Paths("", str(root), "", "")  # 空根 ⇒ ``model_status`` 只做 isfile ✓ 不下潜 ✓
    keys: list[str] = []
    mismatched: list[tuple[str, str, str]] = []
    for entry in inventory_mod.load_catalog()["models"]:
        if entry.get("runtime") != "comfyui" or not entry.get("filename"):
            continue
        mine = inventory_mod.component_path(entry, root)
        _state, theirs = manager.model_status(entry, paths)
        keys.append(str(entry.get("key")))
        if str(mine) != theirs:
            mismatched.append((str(entry.get("key")), str(mine), str(theirs)))
    check("⭐⭐ 落点: 体检算的 == 安装器算的（逐字 ✓）—— 不等 ⇒ 「装好了」与「体检说没装」能同时成立 ✗",
          not mismatched
          and {"sdxl_base", "dit_fl2va_int8", "vae_video_fp8mix"} <= set(keys),
          (len(keys), mismatched[:3]))


def case_tokenizer(tmp: Path) -> None:
    """**CLIP 词表**怎么被定下来 ✓：``settings`` 优先 ✓ → 上游约定的**安装内**落点 ✓ → 都没有 ⇒ 报 ✗。"""
    from app.services import local_model_scan as scan_mod  # noqa: PLC0415

    given, given_source = bridge.resolve_tokenizer_path({"tokenizer_path": "D:/词表目录"})
    check("词表: ``settings.tokenizerPath`` **永远优先**（人不该被探测结果盖掉 ✓）",
          given == "D:/词表目录" and given_source == "settings.tokenizerPath", (given, given_source))

    install = tmp / "假ComfyUI安装"
    vocab = install / "comfy" / "sd1_tokenizer"
    vocab.mkdir(parents=True)
    (vocab / "vocab.json").write_text("{}", encoding="utf-8")
    (vocab / "merges.txt").write_text("", encoding="utf-8")

    saved_roots = scan_mod.get_comfyui_roots
    try:
        scan_mod.get_comfyui_roots = lambda: [(str(install), "COMFYUI_PATH")]
        probed, probed_source = bridge.resolve_tokenizer_path({})
        check("⭐ 词表: 不给 ``settings`` ⇒ 按**上游约定**在 ``<ComfyUI 根>/comfy/sd1_tokenizer/`` 找到 ✓",
              Path(probed) == vocab and "COMFYUI_PATH" in probed_source, (probed, probed_source))

        # ⚠️ 判据是**文件真在** ✗ 不是"目录存在" ✓ —— 只有 ``vocab.json``（缺 ``merges.txt``）⇒ 不认 ✗
        (vocab / "merges.txt").unlink()
        half: object = None
        try:
            bridge.resolve_tokenizer_path({})
        except ValueError as err:
            half = err
        check("⭐ 词表: **半个词表不算词表** ✓（缺 ``merges.txt`` ⇒ 报错并说清怎么修 ✓）",
              isinstance(half, ValueError) and "词表" in str(half) and "怎么修" in str(half), half)

        # 单文件那种形态（``tokenizer.json`` ✓）也是完整的 ✓ —— 口径必须与 ``tokenizer_bpe`` 的**嗅探一致** ✓
        (vocab / "vocab.json").unlink()
        (vocab / "tokenizer.json").write_text("{}", encoding="utf-8")
        single, _single_source = bridge.resolve_tokenizer_path({})
        check("词表: 单文件 ``tokenizer.json`` 那种形态也认（与 ``tokenizer_bpe`` 的嗅探同口径 ✓）",
              Path(single) == vocab, single)

        scan_mod.get_comfyui_roots = lambda: []
        nowhere: object = None
        try:
            bridge.resolve_tokenizer_path({})
        except ValueError as err:
            nowhere = err
        check("⭐ 词表: 两处都没有 ⇒ **当场报**且给两条修法"
              "（视频缺词表只是挂桩 ✓，出图缺词表是**真跑不了** ✗ ⇒ 不能等到装载才说 ✓）",
              isinstance(nowhere, ValueError) and "tokenizerPath" in str(nowhere)
              and "COMFYUI_PATH" in str(nowhere), nowhere)
    finally:
        scan_mod.get_comfyui_roots = saved_roots


def case_image_assembly(tmp: Path) -> None:
    """``stage="sdxl"`` 的装配：**权重 + 词表两样都得能定** ✓（缺一样 ⇒ 当场报 ✗，不等到装载 ✓✗）。"""
    from app.services import local_model_scan as scan_mod  # noqa: PLC0415

    install = tmp / "假ComfyUI装配"
    vocab = install / "comfy" / "sd1_tokenizer"
    vocab.mkdir(parents=True)
    (vocab / "vocab.json").write_text("{}", encoding="utf-8")
    (vocab / "merges.txt").write_text("", encoding="utf-8")
    weights = tmp / "假SDXL主权重.safetensors"
    weights.write_bytes(b"x")

    saved_roots = scan_mod.get_comfyui_roots
    try:
        scan_mod.get_comfyui_roots = lambda: [(str(install), "临时安装")]
        assembly = bridge.assembly_from_config({"settings": {"sdxlPath": str(weights)}},
                                               stage=bridge.IMAGE_STAGE)
        check("⭐ 出图装配: 权重 ✓ 与 CLIP 词表 ✓ **两样都在装配里**（运行时不必自己再找一遍 ✓）",
              assembly.get("stage") == bridge.IMAGE_STAGE
              and assembly.get("dit_path") == str(weights)
              and Path(str(assembly.get("tokenizer_path"))) == vocab, assembly)

        # ⚠️ 视频那条路**不受这条规矩影响** ✗：词表仍是"给了才带" ✓（缺了挂桩 ✓、运行时如实标注 ✓）
        video = bridge.assembly_from_config({"settings": {"ditPath": str(weights)}})
        check("视频装配: 缺词表**照样能装**（挂桩 ✓ —— 这条要求只加在出图上 ✗ 别越界 ✓）",
              video.get("stage") == bridge.VIDEO_STAGE and not video.get("tokenizer_path"), video)

        scan_mod.get_comfyui_roots = lambda: []
        missing: object = None
        try:
            bridge.assembly_from_config({"settings": {"sdxlPath": str(weights)}},
                                        stage=bridge.IMAGE_STAGE)
        except ValueError as err:
            missing = err
        check("⭐ 出图装配: 缺词表 ⇒ **报**（不许「装作能跑」✗ —— 那条路一装载就炸 ✓✗）",
              isinstance(missing, ValueError) and "词表" in str(missing), missing)
    finally:
        scan_mod.get_comfyui_roots = saved_roots


def case_job_stage(tmp: Path) -> None:
    """⭐ ``run_job`` 必须把**刚刚对齐过的 stage** 交给 ``submit`` ✓（不交 ⇒ 任务事实里标成 ``h3`` ✓✗）。

    ⚠️ 干跑用的后端**不是** SDXL 后端 ✓（``DryRunBackend.name = "dryrun"`` ✓）⇒ 这条**干跑跑成什么样
    都不重要** ✗：钉的是「**交给运行时的那份事实**」✓ —— 摘要 / 进度 / 排障全靠它认路 ✓。
    """
    from app.services.engine.runtime import engine_runtime as runtime_singleton  # noqa: PLC0415

    seen: list[dict[str, Any]] = []
    saved = runtime_singleton.submit

    def _spy(request: Any, *, assembly: dict[str, Any] | None = None, stage: str = "h3") -> str:
        seen.append({"stage": stage, "assemblyStage": (assembly or {}).get("stage")})
        return saved(request, assembly=assembly, stage=stage)

    runtime_singleton.submit = _spy  # type: ignore[method-assign]
    try:
        image_request = bridge.request_from_image_record(
            {"prompt": "标签", "size": "224x224"}, outputs_dir=tmp / "标签", settings={},
            stage=bridge.IMAGE_STAGE)
        image_task = asyncio.run(bridge.run_job(
            image_request, {"dryRun": True, "stage": bridge.IMAGE_STAGE},
            task_type="ImageTask", record_id=5, label="图片"))
        video_request = bridge.request_from_image_record(
            {"prompt": "标签", "size": "224x224"}, outputs_dir=tmp / "标签", settings={})
        video_task = asyncio.run(bridge.run_job(video_request, {"dryRun": True},
                                                task_type="ImageTask", record_id=6, label="图片"))
    finally:
        runtime_singleton.submit = saved

    check("⭐ 任务事实: 出图任务的 ``stage`` == ``sdxl`` ✓（**没**被 ``submit`` 的默认 ``h3`` 盖掉 ✓✗）"
          "—— 摘要 / 进度 / 排障都要靠它认路 ✓",
          seen and seen[0]["stage"] == bridge.IMAGE_STAGE and image_task.get("stage") == bridge.IMAGE_STAGE,
          (seen[:1], image_task.get("stage")))
    check("任务事实: 默认（视频）任务仍是 ``h3`` ✓（修的是**漏传** ❗不是改默认 ✗）",
          len(seen) > 1 and seen[1]["stage"] == bridge.VIDEO_STAGE
          and video_task.get("stage") == bridge.VIDEO_STAGE, (seen[1:], video_task.get("stage")))


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="engine_bridge_case_"))
    case_provider()
    case_settings()
    case_dit(tmp)
    case_assembly()
    case_paths(tmp)
    case_requests(tmp)
    case_stages(tmp)
    case_business_wiring()
    case_candidates(tmp)
    case_component_spots(tmp)
    case_tokenizer(tmp)
    case_image_assembly(tmp)
    case_dry_job(tmp)
    case_job_stage(tmp)
    case_no_http_during_run(tmp)

    if not _have_torch():
        skip("缺 torch ⇒ 真装载 / 真 mp4 / 真首帧 PNG 跳过 ✓")
    elif not media_mod.have_ffmpeg():
        skip("缺 ffmpeg ⇒ 真 mp4 / 首帧 PNG 跳过 ✓")
    else:
        case_real_run(tmp)
    engine_runtime.unload(force=True)

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
    # ⚠️ **必须重配 stdout** ✗（与同目录其它套件一致 ✓）：Windows 控制台默认 **GBK** ✓，
    #    而引擎的 plan 告警**通篇**带 ``✓``/``✗``/``⇒``（U+2713 这些**不在 GBK 里** ✗）⇒
    #    ``print`` 直接 ``UnicodeEncodeError`` 把自检炸掉 ✓✗（实测撞到过 ✓ ⇒ 那时会**误读成**
    #    "桥坏了" ✓ 其实是**编码**问题 ✓）。``errors="replace"`` 兜住**任何**漏网的字符 ✓。
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
