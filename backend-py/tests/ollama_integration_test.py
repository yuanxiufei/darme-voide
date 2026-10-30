"""自检：**Ollama 完整集成**（能力面 / 目录 / 客户端 / 视频识别 / 路由）。

守的是「结论从哪来」这件事本身：

1. ``capabilities`` —— 六路信号里可离线复现的那五路，逐条对着上游 ``server/images.go``；
   ✅ 有 ``projector`` 层才算**硬**视觉证据；config 自述次之；模板/parser/族是推测；
   ❌ ``safetensors + nemotron`` 必须**抑制** vision（上游专门为它写的 ``suppressVisionCapability``）；
2. ``ollama_catalog`` —— 盘上清单 → 统一形状：能力面、插件清单、哪些能读视频；
   ⚠️ 重点守「config 描述符在**顶层** ``config`` 而不是 ``layers``」这个坑（读错 ⇒ 能力全空）；
3. ``ollama_client`` —— NDJSON 坏行跳过、错误体透传、``/api/show`` 形状归一、
   **图片必须挂消息上**（放请求顶层会被静默忽略）；
4. ``ollama_video`` —— 挑视觉模型（名字不作数，``vision`` 才算）、base64 前缀剥离、
   无视觉模型时给**人话**失败（不是异常栈）；
5. 路由 —— 6 个新端点真的注册上了（防「写了函数没挂路由」）。

全程不打网络、不碰真机模型库（临时目录 + 纯函数 + 打桩）。
"""
from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="ollamaintegration_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.services import ollama_catalog, ollama_video  # noqa: E402
from app.services.ollama_client import (  # noqa: E402
    OllamaClient,
    OllamaError,
    iter_ndjson,
    normalize_show,
    user_message,
)
from app.services.ollama_native import capabilities as C  # noqa: E402
from app.services.ollama_native import manifest as M  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def test_capabilities() -> None:
    # ── 1. projector 层是**硬证据**：有它就有 vision，没有就不能靠猜 ✓ ──
    facts = C.capabilities_from(media_types=[M.MEDIA_TYPE_MODEL, M.MEDIA_TYPE_PROJECTOR])
    check("capabilities: projector 层 ⇒ vision（硬证据）",
          facts["capabilities"] == ["completion", "vision"], facts["capabilities"])
    check("capabilities: videoReader 跟 vision 走（Ollama 没有 video 能力名）",
          facts["video_reader"] is True, facts)
    check("capabilities: 来源标注为 projector（前端要能解释「凭什么」）",
          facts["sources"]["vision"] == ["projector"], facts["sources"])

    # ── 2. config 自述优先，且顺序归一化（上游是插入序，会让同一模型排序抖动 ✗）──
    facts = C.capabilities_from(
        config={"capabilities": ["tools", "completion", "vision", "completion"]},
    )
    check("capabilities: config 自述去重 + 按固定顺序归一",
          facts["capabilities"] == ["completion", "vision", "tools"], facts["capabilities"])

    # ── 3. Go 模板变量：tools ⇒ 工具调用；suffix ⇒ 中间填充 ──
    facts = C.capabilities_from(template="{{ if .Tools }}{{ .Tools }}{{ end }}{{ .Suffix }}")
    check("capabilities: 模板 .Tools ⇒ tools", "tools" in facts["capabilities"], facts["capabilities"])
    check("capabilities: 模板 .Suffix ⇒ insert", "insert" in facts["capabilities"], facts["capabilities"])

    # ── 4. parser 名 ⇒ 能力（表外名字一个都不加，与上游返回 nil 等价）──
    check("capabilities: parser=qwen3-thinking ⇒ tools+thinking",
          C.parser_capabilities("qwen3-thinking") == ["tools", "thinking"],
          C.parser_capabilities("qwen3-thinking"))
    check("capabilities: parser=qwen3（非 thinking 档）⇒ 只有 tools",
          C.parser_capabilities("qwen3") == ["tools"], C.parser_capabilities("qwen3"))
    check("capabilities: 表外 parser 不加任何能力（宁可漏报）",
          C.parser_capabilities("some-future-parser") == [], C.parser_capabilities("some-future-parser"))

    # ── 5. 模型族：只有 gpt-oss ⇒ thinking（上游就这一条）──
    facts = C.capabilities_from(config={"model_family": "gpt-oss"})
    check("capabilities: family=gpt-oss ⇒ thinking", "thinking" in facts["capabilities"],
          facts["capabilities"])
    check("capabilities: 其他族不凭空加 thinking",
          "thinking" not in C.capabilities_from(config={"model_family": "llama"})["capabilities"],
          C.capabilities_from(config={"model_family": "llama"})["capabilities"])

    # ── 6. 抑制：safetensors + nemotron 不许自称 vision（上游注释原文是 text-only）──
    facts = C.capabilities_from(
        config={"model_format": "safetensors", "model_family": "nemotron-3-nano"},
        media_types=[M.MEDIA_TYPE_PROJECTOR],
    )
    check("capabilities: safetensors+nemotron 抑制 vision（上游 filterUnsupportedCapabilities）",
          "vision" not in facts["capabilities"], facts["capabilities"])
    check("capabilities: 抑制后仍保留 projector 事实（hasProjector≠能不能读图）",
          [p["kind"] for p in facts["plugins"]] == ["projector"], facts["plugins"])

    # ── 7. 什么都没判出来时给 completion（上游 ggufCapabilities 的 else 分支）──
    facts = C.capabilities_from()
    check("capabilities: 无任何信号 ⇒ 默认 completion",
          facts["capabilities"] == ["completion"] and facts["sources"]["completion"] == ["default"],
          facts)

    # ── 8. 插件清单：权重之外的附加层，顺序照清单 ──
    plugins = C.plugin_layers([
        M.MEDIA_TYPE_MODEL, M.MEDIA_TYPE_ADAPTER, M.MEDIA_TYPE_PROJECTOR,
        M.MEDIA_TYPE_DRAFT, M.MEDIA_TYPE_LICENSE,
    ])
    check("capabilities: 插件清单 = adapter/projector/draft/license（model 层不算插件）",
          [p["kind"] for p in plugins] == ["adapter", "projector", "draft", "license"], plugins)

    # ── 9. 能力差集：调用前就能给人话错误 ──
    check("capabilities: check_capabilities 只报缺的",
          C.check_capabilities(["completion", "tools"], ["completion", "vision"]) == ["vision"],
          C.check_capabilities(["completion", "tools"], ["completion", "vision"]))
    check("capabilities: describe 出中文摘要（前端直接用）",
          C.describe(["completion", "vision"]) == "对话/补全 · 读图（可读视频帧）",
          C.describe(["completion", "vision"]))


# ── 造一个假的 Ollama 模型库（清单 + blob，真形状）──────────────────────────
def _blob(root: str, payload: bytes) -> str:
    digest = "sha256:" + __import__("hashlib").sha256(payload).hexdigest()
    blob_dir = os.path.join(root, "blobs")
    os.makedirs(blob_dir, exist_ok=True)
    with open(os.path.join(blob_dir, digest.replace(":", "-")), "wb") as handle:
        handle.write(payload)
    return digest


def _manifest(root: str, name: str, *, config: dict, layers: list[tuple[str, bytes]],
              weight_bytes: int) -> None:
    ns, _, tag = name.partition(":")
    tag = tag or "latest"
    target = os.path.join(root, "manifests", "registry.ollama.ai", "library", ns)
    os.makedirs(target, exist_ok=True)
    config_digest = _blob(root, json.dumps(config).encode("utf-8"))
    entries = [{
        "mediaType": M.CONFIG_MEDIA_TYPE,
        "digest": config_digest,
        "size": len(json.dumps(config).encode("utf-8")),
    }]
    payloads: list[tuple[str, str, int]] = []
    for media_type, raw in layers:
        payloads.append((media_type, _blob(root, raw), len(raw)))
    weight = b"\x00" * weight_bytes
    entries.append({"mediaType": M.MEDIA_TYPE_MODEL, "digest": _blob(root, weight), "size": weight_bytes})
    for media_type, digest, size in payloads:
        entries.append({"mediaType": media_type, "digest": digest, "size": size})
    manifest = {
        "schemaVersion": 2,
        "mediaType": M.MANIFEST_MEDIA_TYPE,
        "config": entries[0],
        "layers": entries[1:],
    }
    with open(os.path.join(target, tag), "wb") as handle:
        handle.write(json.dumps(manifest).encode("utf-8"))


def test_catalog_disk() -> None:
    root = tempfile.mkdtemp(prefix="ollamalib_")
    _manifest(root, "qwen3vl:7b", config={
        "model_family": "qwen3vl", "model_format": "gguf",
        "capabilities": ["completion", "vision"],
    }, layers=[(M.MEDIA_TYPE_TEMPLATE, b"{{ if .Tools }}{{ .Tools }}{{ end }}")], weight_bytes=4096)
    _manifest(root, "llava:13b", config={"model_family": "llama", "model_format": "gguf"},
              layers=[(M.MEDIA_TYPE_PROJECTOR, b"mmproj-bytes")], weight_bytes=8192)
    _manifest(root, "qwen3:8b", config={"model_family": "qwen3", "model_format": "gguf"},
              layers=[(M.MEDIA_TYPE_ADAPTER, b"lora-bytes")], weight_bytes=6144)
    _manifest(root, "nemotron:1b", config={
        "model_family": "nemotron-3-nano", "model_format": "safetensors",
        "capabilities": ["completion", "vision"],
    }, layers=[(M.MEDIA_TYPE_PROJECTOR, b"mmproj2")], weight_bytes=2048)

    report = ollama_catalog.disk_inventory(root)
    models = {m["name"]: m for m in report["models"]}
    check("catalog: 盘上 4 个模型都列出来了", len(models) == 4, sorted(models))
    check("catalog: 名字按 registry/library/<名>:<tag> 归一",
          "qwen3vl:7b" in models, sorted(models))

    qwen_vl = models.get("qwen3vl:7b", {})
    check("catalog: config 顶层描述符的能力自述**真的读到了**（读错这里会是空）",
          "vision" in (qwen_vl.get("capabilities") or []), qwen_vl.get("capabilities"))
    check("catalog: 模板层 ⇒ tools 也算进能力面",
          "tools" in (qwen_vl.get("capabilities") or []), qwen_vl.get("capabilities"))
    check("catalog: 能力面来源标成 manifest（离线不是瞎猜，是清单判的）",
          qwen_vl.get("capabilitiesSource") == "manifest", qwen_vl.get("capabilitiesSource"))
    check("catalog: videoReader 由 vision 推出",
          qwen_vl.get("videoReader") is True, qwen_vl.get("videoReader"))

    nemotron = models.get("nemotron:1b", {})
    check("catalog: safetensors+nemotron 离线也抑制 vision",
          nemotron.get("videoReader") is False, nemotron.get("capabilities"))
    check("catalog: 但 hasProjector 仍为真（层确实在，只是不当作能力）",
          nemotron.get("hasProjector") is True, nemotron)

    readers = ollama_catalog.video_readers(report["models"])
    check("catalog: 可读视频的模型 = llava + qwen3vl（各自路径不同，都算）",
          [m["name"] for m in readers] == ["llava:13b", "qwen3vl:7b"], [m["name"] for m in readers])
    check("catalog: 纯文本模型不会被算成可读视频",
          all(m["name"] != "qwen3:8b" for m in readers), [m["name"] for m in readers])

    summary = ollama_catalog.plugin_summary(report["models"])
    check("catalog: 插件计数含 adapter/projector/template",
          all(k in summary["counts"] for k in ("adapter", "projector", "template")), summary["counts"])
    adapter_owner = [m for m in summary["models"] if m["kinds"] == ["adapter"]]
    check("catalog: adapter 被标成「会改能力面的插件」",
          adapter_owner and adapter_owner[0]["changesCapabilities"] == ["adapter"], adapter_owner)


def test_client() -> None:
    # ── NDJSON：坏行跳过，好行解析 ──
    check("client: 正常行解析成 dict", iter_ndjson('{"status":"pulling"}') == {"status": "pulling"})
    check("client: 空行 ⇒ None", iter_ndjson("   ") is None)
    check("client: 坏行 ⇒ None（不抛，整条流不断）", iter_ndjson("not json") is None)
    check("client: JSON 数组不算对象 ⇒ None", iter_ndjson("[1,2]") is None)

    # ── 错误对象：状态/原句/URL 都要带出去（前端要显示人话）──
    error = OllamaError("boom", status=400, detail="model requires more system memory", url="http://x/api/chat")
    check("client: OllamaError 带上游原句与状态",
          error.status == 400 and "system memory" in error.detail, error.as_dict())

    # ── /api/show 形状归一 ──
    detail = normalize_show({
        "details": {"family": "qwen3vl", "parameter_size": "7B", "quantization_level": "Q4_K_M"},
        "capabilities": ["completion", "vision"],
        "template": "{{ .Prompt }}",
        "model_info": {"qwen3vl.vision.block_count": 24},
        "projector_info": {"clip.has_vision_encoder": True},
    })
    check("client: normalize_show 摊平 details",
          detail["family"] == "qwen3vl" and detail["parameterSize"] == "7B", detail["family"])
    check("client: normalize_show 认 projector ⇒ hasProjector",
          detail["hasProjector"] is True, detail["hasProjector"])

    legacy = normalize_show({"details": {"family": "llava"}, "projector_info": {"x": 1}})
    check("client: 老版无 capabilities 时按 projector 补 vision",
          legacy["capabilities"] == ["vision"], legacy["capabilities"])
    bare = normalize_show({"details": {}})
    check("client: 空响应不炸（返回空结构而不是 KeyError）",
          bare["capabilities"] == [] and bare["hasProjector"] is False, bare)

    # ── 图片位置：必须挂消息上（放顶层会被静默忽略）──
    message = user_message("看这几帧", ["QUJD", "REVG"])
    check("client: user_message 把帧挂在消息的 images 上",
          message["images"] == ["QUJD", "REVG"] and message["role"] == "user", message)
    check("client: 不带图时**不生成** images 键（空列表会被上游当坏输入）",
          "images" not in user_message("纯文本"), user_message("纯文本"))
    check("client: 客户端默认地址走 resolve_ollama_url（唯一真源）",
          OllamaClient().base_url == __import__(
              "app.services.ollama", fromlist=["resolve_ollama_url"]
          ).resolve_ollama_url(), OllamaClient().base_url)


def test_video() -> None:
    # ── 挑模型：名字不作数，vision 才算 ──
    models = [
        {"name": "llava:34b", "size": 34, "videoReader": False, "capabilities": ["completion"]},
        {"name": "qwen3vl:7b", "size": 7, "videoReader": True, "capabilities": ["completion", "vision"]},
        {"name": "llava:13b", "size": 13, "videoReader": True, "capabilities": ["completion", "vision"]},
    ]
    picked = ollama_video.pick_reader(models)
    check("video: 默认挑最大的视觉模型（不是名字里带 vl 的那个）",
          picked and picked["name"] == "llava:13b", picked)
    prefer = ollama_video.pick_reader(models, prefer="qwen3vl:7b")
    check("video: 显式指定优先（真具备 vision 才认）",
          prefer and prefer["name"] == "qwen3vl:7b", prefer)
    fallback = ollama_video.pick_reader(models, prefer="ghost:1b")
    check("video: 指定了不存在的模型 ⇒ 回落默认而不是报错",
          fallback and fallback["name"] == "llava:13b", fallback)
    check("video: 没有视觉模型 ⇒ None",
          ollama_video.pick_reader([{"name": "qwen3:8b", "videoReader": False}]) is None)

    # ── base64 前缀必须剥：带前缀解出来是坏图，且不报错（静默降级）──
    raw = base64.b64encode(b"jpeg-bytes").decode("ascii")
    frames = ollama_video.decode_frames([f"data:image/jpeg;base64,{raw}", raw, ""])
    check("video: decode_frames 剥掉 data: 前缀且丢掉空串",
          len(frames) == 2 and frames[0]["base64"] == raw, frames)
    check("video: decode_frames 的 index 连续（空串被跳过也不留洞）",
          [f["index"] for f in frames] == [0, 1], frames)

    # ── 没有视觉模型时：**人话失败**，不是异常栈 ──
    import asyncio

    original = ollama_catalog.inventory

    async def fake_inventory(*args, **kwargs):
        return {"models": [{"name": "qwen3:8b", "size": 8, "videoReader": False,
                            "capabilities": ["completion"]}]}

    ollama_catalog.inventory = fake_inventory          # type: ignore[assignment]
    try:
        result = asyncio.run(ollama_video.analyze_video("whatever.mp4"))
    finally:
        ollama_catalog.inventory = original            # type: ignore[assignment]
    check("video: 无视觉模型时 ok=False 且给可执行的建议",
          result["ok"] is False and "视觉模型" in result["reason"], result)
    check("video: 失败结果也带 baseUrl（否则用户不知道该改哪个端点）",
          bool(result.get("baseUrl")), result)


def test_routes() -> None:
    from app.routers import ai_configs

    # ⚠️ 路由表里带着 ``/api/v1/ai-configs`` 前缀 ⇒ 只能按**后缀**比（按全路径比必假红 ✗）
    paths = {getattr(route, "path", "") for route in ai_configs.router.routes}
    ollama_paths = {p.split("/ai-configs", 1)[-1] for p in paths if "/ollama/" in p}
    expected = {
        "/ollama/catalog", "/ollama/show", "/ollama/ps",
        "/ollama/copy", "/ollama/embed", "/ollama/video/analyze",
    }
    missing = sorted(expected - ollama_paths)
    check("routes: 6 个新端点都已注册（写了函数没挂路由 = 白写）", not missing, missing)
    check("routes: 老的 ollama/status|start|pull|delete 仍在（不能被新端点顶掉）",
          {"/ollama/status", "/ollama/start", "/ollama/pull", "/ollama/delete"} <= ollama_paths,
          sorted(ollama_paths))


def main() -> int:
    started = time.time()
    for suite in (test_capabilities, test_catalog_disk, test_client, test_video, test_routes):
        try:
            suite()
        except Exception as exc:  # noqa: BLE001 - 套件崩了要报，不能中断全程
            check(f"{suite.__name__} 未抛异常地跑完", False, f"{type(exc).__name__}: {exc}")

    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed（{time.time() - started:.2f}s）")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
