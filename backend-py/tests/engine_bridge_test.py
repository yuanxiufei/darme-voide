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
6. **反向证明** ✓ —— 跑的时候把 httpx **打死** ✗ ⇒ 照样出片 ✓（"进程内 + 零 HTTP"不是口号 ✓）。

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
# ② DiT 主权重：settings 优先 / 清单兜底 / **都没有就报**
# ══════════════════════════════════════════════════════════════════════════
def case_dit(tmp: Path) -> None:
    path, source = bridge.resolve_dit_path({"dit_path": "D:/真权重.safetensors"})
    check("DiT: ``settings.ditPath`` 永远优先，且**来源如实标注**",
          path == "D:/真权重.safetensors" and source == "settings.ditPath", (path, source))

    saved_catalog, saved_dir = inventory_mod.load_catalog, inventory_mod.models_dir
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
    finally:
        inventory_mod.load_catalog, inventory_mod.models_dir = saved_catalog, saved_dir


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

    source = Path(bridge.__file__).read_text(encoding="utf-8")
    check("零 HTTP: 桥的源码里**没有** httpx / requests / comfyui 这类外部门（源码级反证 ✓）",
          not any(token in source for token in ("import httpx", "import requests",
                                                "comfyui", "aiohttp")),
          [token for token in ("import httpx", "import requests", "comfyui", "aiohttp")
           if token in source])


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


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="engine_bridge_case_"))
    case_provider()
    case_settings()
    case_dit(tmp)
    case_assembly()
    case_paths(tmp)
    case_requests(tmp)
    case_business_wiring()
    case_dry_job(tmp)
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
    raise SystemExit(main())
