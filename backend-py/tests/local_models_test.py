"""S6 自检：本地模型扫描服务 + 11 端点（``local-model-scan.ts`` 709 行 + ``localModels.ts`` 688 行）。

这个域的错都**不报错**、只表现为「扫不到 / 识错类 / 删错目录」，四类重点：

1. **规则顺序**：H3 的文本编码器（qwen3vl）属**视频**、必须排在文本 LLM 之前；
   Wan/Hunyuan 的 GGUF 必须排在通用 GGUF LLM 之前；组件目录（vae/clip…）要排在
   「目录=LLM」与「扩散主权重」之前 ⇒ 顺序错就整片误判；
2. **组件不等于跳过**：VAE/CLIP/ControlNet 都扫出来，只是 ``role='component'``；
   ComfyUI 则是 ``callable=false``（只读来源，不注册为可调用后端）；
3. **上限与取消**：同步 ``5/8000``、异步 ``8/50000``（两个默认值**不同**，容易抄错）；
   异步版随时可取消，取消即抛 ``ScanCancelledError``；
4. **路径安全**：下载/删除都必须限制在「模型存储目录」内（防路径穿越），且不许删根目录。

扫描全部在**临时目录树**上做（不碰真实磁盘），下载/网络全部打桩（**不打网络**）。

运行::

    ./.venv/Scripts/python.exe tests/local_models_test.py
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="localmodels_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.core.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.core.models import ai_service_configs  # noqa: E402
from app.routers import local_models as r  # noqa: E402
from app.services import local_model_scan as ls  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


# ============================================================
# 临时模型目录树
# ============================================================
TREE = tempfile.mkdtemp(prefix="modeltree_")
MODELS = os.path.join(TREE, "models")


def touch(rel: str, size: int = 1024) -> str:
    path = os.path.join(MODELS, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(b"\0" * size)
    return path


touch("checkpoints/flux-dev.safetensors", 12 * 1024 * 1024)
touch("diffusion_models/wan2.1_i2v.gguf", 5 * 1024 * 1024)
touch("vae/ae.safetensors", 1024)
touch("text_encoders/qwen3vl_32b_minimax_h3.safetensors", 2 * 1024 * 1024)
touch("llm/qwen3-4b.gguf", 3 * 1024 * 1024)
touch("checkpoints/README.txt", 10)                 # 非模型扩展名
touch("node_modules/skip-me.safetensors", 1024)     # SKIP_DIRS
touch(".git/skip-too.safetensors", 1024)            # SKIP_DIRS
touch("Windows/system.safetensors", 1024)           # SYSTEM_DIRS
touch("deep/a/b/c/d/e/f/deep.safetensors", 1024)    # 深度 7（同步默认 5 会剪掉，异步默认 8 会扫到）


def main() -> int:  # noqa: C901
    # ================= 规则引擎 =================
    check("规则: 12 条，顺序即优先级", len(ls._RULES) == 12, len(ls._RULES))
    check("规则: H3 文本编码器（text_encoders 下的 qwen3vl）判为**视频组件**，不落文本 LLM",
          ls.classify("/x/models/text_encoders/qwen3vl_32b_minimax_h3.safetensors")
          == {"kind": "video", "runtime": "h3", "confidence": "high",
              "role": "component", "matchedBy": ["h3-text-encoder"]})
    check("规则: H3 视频 DiT（standalone）/ VAE（component）分得开",
          ls.classify("/x/models/diffusion_models/minimax_h3_dit.safetensors")["role"] == "standalone"
          and ls.classify("/x/models/diffusion_models/minimax_h3_vae.safetensors")["role"] == "component"
          and ls.classify("/x/models/vae/minimax_h3_vae.safetensors")["matchedBy"] == ["h3-vae"])
    check("规则: Wan/Hunyuan 的 GGUF 判**视频**（不会被通用 GGUF LLM 抢走）",
          ls.classify("/x/models/unet/wan2.1_i2v.gguf")["kind"] == "video"
          and ls.classify("/x/models/unet/wan2.1_i2v.gguf")["matchedBy"] == ["wan-video-gguf"])
    check("规则: Wan safetensors 主权重（diffusion_models 目录）判视频",
          ls.classify("/x/models/diffusion_models/wan2.2_t2v.safetensors")["kind"] == "video")
    check("规则: 普通 GGUF 判文本/ollama",
          ls.classify("/x/models/llm/qwen3-4b.gguf")
          == {"kind": "text", "runtime": "ollama", "confidence": "high",
              "role": "standalone", "matchedBy": ["gguf-llm"]})
    check("规则: TTS 关键词（cosyvoice/tts/vits…）判音频",
          ls.classify("/x/models/checkpoints/cosyvoice.safetensors")["kind"] == "audio"
          and ls.classify("/x/models/checkpoints/gpt-sovits-v2.pth")["kind"] == "audio")
    check("规则: 扩散主权重目录（checkpoints/unet/loras）判图像 standalone",
          ls.classify("/x/models/checkpoints/flux-dev.safetensors")
          == {"kind": "image", "runtime": "comfyui", "confidence": "high",
              "role": "standalone", "matchedBy": ["diffusion-weights"]})
    check("规则: **组件目录**（vae/clip/controlnet…）判图像 component —— 不跳过，照样扫出来",
          all(ls.classify(f"/x/models/{d}/some.safetensors")
              == {"kind": "image", "runtime": "comfyui", "confidence": "high",
                  "role": "component", "matchedBy": ["image-component"]}
              for d in ("vae", "clip", "controlnet", "upscale_models", "embeddings")))
    check("规则: 目录名=llm 判文本（ComfyUI LLM_party 挂载）",
          ls.classify("/x/models/llm/whatever.bin")["kind"] == "text"
          and ls.classify("/x/models/llm/whatever.bin")["runtime"] == "comfyui")
    check("规则: 兜底 —— 认不出名字的 safetensors 判图像/低置信度",
          ls.classify("/x/models/misc/mystery.safetensors")["matchedBy"] == ["default-weights"]
          and ls.classify("/x/models/misc/mystery.safetensors")["confidence"] == "low")
    check("规则: 非模型扩展名判 unknown（且不会被扫描收录）",
          ls.classify("/x/models/misc/readme.txt")["kind"] == "unknown"
          and ".txt" not in ls.MODEL_EXTS)

    # ================= 注册建议 =================
    comfy = ls.build_suggestion(ls.classify("/x/models/vae/ae.safetensors"), "ae.safetensors", ".safetensors")
    check("建议: comfyui 一律 **callable=false + 空 provider/baseUrl**（只读来源）",
          comfy["runtime"] == "comfyui" and comfy["callable"] is False
          and comfy["provider"] == "" and comfy["baseUrl"] == "" and comfy["role"] == "component",
          comfy)
    ollama = ls.build_suggestion(ls.classify("/x/models/llm/qwen3-4b.gguf"), "qwen3-4b.gguf", ".gguf")
    check("建议: GGUF 文本 -> openai 兼容 + 11434 + model **去掉扩展名**",
          ollama["serviceType"] == "text" and ollama["provider"] == "openai"
          and ollama["baseUrl"] == "http://localhost:11434" and ollama["model"] == "qwen3-4b"
          and ollama["callable"] is True, ollama)
    h3 = ls.build_suggestion(ls.classify("/x/models/diffusion_models/minimax_h3_dit.safetensors"),
                             "minimax_h3_dit.safetensors", ".safetensors")
    h3c = ls.build_suggestion(ls.classify("/x/models/vae/minimax_h3_vae.safetensors"),
                              "minimax_h3_vae.safetensors", ".safetensors")
    check("建议: H3 视频 -> minimax/hailuo-02/8765，组件与主权重**文案不同**",
          h3["serviceType"] == "video" and h3["provider"] == "minimax"
          and h3["model"] == "hailuo-02" and h3["baseUrl"] == "http://localhost:8765"
          and "checkpoint_map" in h3["note"] and "不单独注册" in h3c["note"], (h3["note"], h3c["note"]))
    small = ls.build_suggestion(ls.classify("/x/models/image/cosy.safetensors"), "x.safetensors", ".safetensors")
    check("建议: runtime=unknown/kind=unknown -> None（不生成建议）",
          ls.build_suggestion({"kind": "unknown", "runtime": "unknown", "role": "standalone"},
                              "x.bin", ".bin") is None and small is not None)

    # ================= human_size =================
    check("体积: <1024 用 B；否则 KB/MB/GB/TB",
          ls.human_size(0) == "0 B" and ls.human_size(1023) == "1023 B"
          and ls.human_size(1024) == "1.0 KB" and ls.human_size(1536) == "1.5 KB"
          and ls.human_size(1024 ** 3 * 2) == "2.0 GB"
          and ls.human_size(1024 ** 4) == "1.0 TB", ls.human_size(1536))
    check("体积: ≥100 不留小数位（150 KB 而不是 150.0 KB）",
          ls.human_size(1024 * 150) == "150 KB", ls.human_size(1024 * 150))
    check("体积: **四舍五入而非银行家舍入**（102.5 -> 103 KB，Python 默认会给 102）",
          ls.human_size(1024 * 102.5) == "103 KB", ls.human_size(1024 * 102.5))

    # ================= 同步扫描 =================
    result = ls.scan_local_models({"roots": [MODELS]})
    names = {m["filename"] for m in result["models"]}
    check("扫描: 模型文件都扫到（含组件），非模型扩展名不收",
          {"flux-dev.safetensors", "wan2.1_i2v.gguf", "ae.safetensors",
           "qwen3vl_32b_minimax_h3.safetensors", "qwen3-4b.gguf"} <= names
          and "README.txt" not in names, sorted(names))
    check("扫描: SKIP_DIRS / SYSTEM_DIRS 里的模型**跳过**",
          "skip-me.safetensors" not in names and "skip-too.safetensors" not in names
          and "system.safetensors" not in names, sorted(names))
    check("扫描: byKind 计数正确且 total 一致",
          result["byKind"]["video"] == 2 and result["byKind"]["image"] == 2
          and result["byKind"]["text"] == 1 and result["byKind"]["audio"] == 0
          and result["total"] == len(result["models"]) == 5, result["byKind"])
    check("扫描: 排序 = video > image > text，同类按体积降序",
          [m["kind"] for m in result["models"]] == ["video", "video", "image", "image", "text"]
          and result["models"][0]["filename"] == "wan2.1_i2v.gguf",  # 5MB > 2MB
          [(m["kind"], m["filename"]) for m in result["models"]])
    check("扫描: 未截断、耗时字段存在、roots 是绝对路径",
          result["truncated"] is False and isinstance(result["elapsedMs"], float)
          and result["roots"] == [os.path.abspath(MODELS)], result["roots"])
    check("扫描: 组件仍被收录（role=component 而不是被过滤掉）",
          any(m["role"] == "component" and m["kind"] == "image" for m in result["models"]))

    shallow = ls.scan_local_models({"roots": [MODELS], "maxDepth": 0})
    check("扫描: maxDepth=0 只扫根目录本级（子目录全剪）", shallow["total"] == 0, shallow["total"])
    check("扫描: maxDepth=1 扫到一级子目录（deep 里那个仍在范围外）",
          "ae.safetensors" in {m["filename"] for m in
                               ls.scan_local_models({"roots": [MODELS], "maxDepth": 1})["models"]}
          and "deep.safetensors" not in {m["filename"] for m in
                                         ls.scan_local_models({"roots": [MODELS],
                                                               "maxDepth": 1})["models"]})
    check("扫描: 默认深度 5 —— 深度 7 的模型**扫不到**（同步版默认值就是这么窄）",
          "deep.safetensors" not in names)
    limited = ls.scan_local_models({"roots": [MODELS], "maxFiles": 2})
    check("扫描: 达到 maxFiles 上限 -> truncated=True 且不再收录",
          limited["truncated"] is True and limited["total"] <= 2, limited["total"])
    only_video = ls.scan_local_models({"roots": [MODELS], "kinds": ["video"]})
    check("扫描: kinds 过滤只留指定大类（但**仍会 stat/计数**全部文件）",
          {m["kind"] for m in only_video["models"]} == {"video"}, only_video["byKind"])
    twice = ls.scan_local_models({"roots": [MODELS, MODELS]})
    check("扫描: 同一目录重复作为 root 不产生重复条目（seen 去重）",
          twice["total"] == result["total"], twice["total"])
    check("扫描: roots 指向不存在的目录 -> 空结果不报错",
          ls.scan_local_models({"roots": [os.path.join(TREE, "nope")]})["total"] == 0)

    # ================= 异步扫描 =================
    async_same = asyncio.run(ls.scan_local_models_async({"roots": [MODELS], "maxDepth": 5}))
    check("异步: 同参（显式 maxDepth=5）与同步结果一致",
          {m["filename"] for m in async_same["models"]} == names, sorted(
              m["filename"] for m in async_same["models"]))
    async_result = asyncio.run(ls.scan_local_models_async({"roots": [MODELS]}))
    check("异步: 默认深度 **8** —— 深度 7 的模型这次扫得到（两个默认值不同！）",
          "deep.safetensors" in {m["filename"] for m in async_result["models"]})


    async def _cancelled() -> str:
        try:
            await ls.scan_local_models_async({"roots": [MODELS], "shouldCancel": lambda: True})
        except ls.ScanCancelledError as err:
            return str(err)
        return ""

    check("异步: shouldCancel 一上来就为真 -> 抛 ScanCancelledError('scan cancelled')",
          asyncio.run(_cancelled()) == "scan cancelled", asyncio.run(_cancelled()))
    progress: list[dict] = []
    asyncio.run(ls.scan_local_models_async({"roots": [MODELS],
                                            "onProgress": progress.append}))
    check("异步: onProgress 不报错（每 1000 文件一次；小树可以不触发）",
          isinstance(progress, list))

    # ================= 参数解析 =================
    check("参数: parse_roots 区分 None 与**空数组**（空数组要在路由里回 400）",
          r.parse_roots(None) is None and r.parse_roots("  ") is None
          and r.parse_roots("[]") == [] and r.parse_roots('[1,2]') == []
          and r.parse_roots('["a","b"]') == ["a", "b"]
          and r.parse_roots(" a , b ") == ["a", "b"],
          (r.parse_roots("[]"), r.parse_roots('[1,2]')))
    # ⚠️ 原 TS 的怪癖：`[,]` 虽以 `[` 开头但 JSON 解析失败 ⇒ **降级为逗号分隔**，
    #    于是得到 `['[', ']']`（不是空数组）。这里**照抄**，不改。
    check("参数: `[,]` 走 JSON 失败降级 -> `['[', ']']`（原 TS 行为，照抄）",
          r.parse_roots("[,]") == ["[", "]"], r.parse_roots("[,]"))
    check("参数: parse_kinds 只留合法值，全非法 -> None",
          r.parse_kinds(None) is None and r.parse_kinds("foo") is None
          and r.parse_kinds("text, IMAGE ,video") == ["text", "image", "video"])
    check("参数: `Number(x) || 默认值` —— 非数字/0/空串都回退（不是报错）",
          [r._js_number_or(v, 5) for v in ("abc", "0", "", None, "3", "2.5")] == [5, 5, 5, 5, 3, 2.5])
    check("参数: normalize_repo 防穿越；parse_source 非法回退 hf；默认分支按来源",
          r.normalize_repo("Qwen/Qwen3-4B") == "Qwen/Qwen3-4B"
          and r.normalize_repo("/Qwen/Qwen3-4B/") == "Qwen/Qwen3-4B"
          and r.parse_source("HF_Mirror") == "hf_mirror" and r.parse_source("nope") == "hf"
          and r.default_revision("modelscope", "") == "master"
          and r.default_revision("hf", "") == "main"
          and r.default_revision("hf", " dev ") == "dev")
    for bad in ("Qwen", "../x/y", "a\\b/c", ""):
        try:
            r.normalize_repo(bad)
            check(f"参数: normalize_repo 拒绝 {bad!r}", False)
        except ValueError:
            check(f"参数: normalize_repo 拒绝 {bad!r}", True)
    check("参数: h3_checkpoint_key 只认 fl2va/ref2va",
          r.h3_checkpoint_key("minimax_h3_fl2va.safetensors") == "fl2va"
          and r.h3_checkpoint_key("minimax_h3_ref2va.safetensors") == "ref2va"
          and r.h3_checkpoint_key("minimax_h3_dit.safetensors") is None)
    check("参数: build_download_url —— HF 直链 vs ModelScope /resolve，空格编成 %20",
          r.build_download_url("hf", "Q/R", "main", "sub dir/a b.bin")
          == "https://huggingface.co/Q/R/resolve/main/sub%20dir/a%20b.bin"
          and r.build_download_url("modelscope", "Q/R", "master", "a.bin")
          == "https://modelscope.cn/models/Q/R/resolve/master/a.bin"
          and r.build_download_url("hf_mirror", "Q/R", "main", "a.bin")
          == "https://hf-mirror.com/Q/R/resolve/main/a.bin",
          r.build_download_url("hf", "Q/R", "main", "sub dir/a b.bin"))

    # ================= 端点 =================
    client = TestClient(app)
    scan_ok = client.get("/api/v1/local-models/scan", params={"roots": MODELS})
    check("端点: GET /scan 正常 -> 200 且带上模型",
          scan_ok.status_code == 200 and scan_ok.json()["data"]["total"] > 0)
    check("端点: GET /scan `maxDepth=abc` -> **200 + 默认值**（不是 422）",
          client.get("/api/v1/local-models/scan",
                     params={"roots": MODELS, "maxDepth": "abc"}).status_code == 200)
    check("端点: GET /scan `roots=[]` -> 400 `roots 参数为空或格式不正确`；`roots=` -> 200",
          client.get("/api/v1/local-models/scan", params={"roots": "[]"}).json()
          == {"code": 400, "message": "roots 参数为空或格式不正确"}
          and client.get("/api/v1/local-models/scan", params={"roots": ""}).status_code == 200)
    roots_ok = client.get("/api/v1/local-models/roots").json()["data"]
    check("端点: GET /roots -> roots/detail/extra/paths 四件套",
          set(roots_ok) == {"roots", "detail", "extra", "paths"}
          and all(set(item) == {"path", "name"} for item in roots_ok["detail"]), sorted(roots_ok))
    check("端点: GET /drives -> 返回盘符数组",
          isinstance(client.get("/api/v1/local-models/drives").json()["data"]["drives"], list))

    # PUT /roots —— ⚠️ 改的是 configs/model-paths.json，自检里换成临时文件，别污染仓库
    config_file = os.path.join(TREE, "model-paths.json")
    original_paths_file = ls._paths_config_file
    ls._paths_config_file = lambda: config_file
    try:
        put_ok = client.put("/api/v1/local-models/roots",
                            json={"roots": [TREE, TREE], "models_dir": TREE}).json()["data"]
        check("端点: PUT /roots -> extra **去重且绝对化**，并回 roots/paths",
              put_ok["extra"] == [os.path.abspath(TREE)] and set(put_ok) == {"extra", "roots", "paths"},
              put_ok)
        check("端点: PUT /roots 落盘的 extra_roots 与 paths.models_dir 都对",
              ls.get_extra_roots() == [os.path.abspath(TREE)]
              and ls.get_model_paths()["models_dir"] == TREE)
        client.put("/api/v1/local-models/roots", json={"models_dir": ""})
        check("端点: PUT /roots 传**空串** = 清空该字段（partial 更新 ⇒ 回落到默认 <data_root>/models ✓）",
              ls.get_model_paths()["models_dir"] == ls.default_models_dir())
        client.put("/api/v1/local-models/roots", json={"roots": [TREE]})
        check("端点: PUT /roots 不传 roots 时保留现状（不会清空）",
              ls.get_extra_roots() == [os.path.abspath(TREE)])
    finally:
        ls._paths_config_file = original_paths_file

    # ---- POST /register ----
    check("端点: POST /register 空 models -> 400 `models 不能为空`",
          client.post("/api/v1/local-models/register", json={}).json()
          == {"code": 400, "message": "models 不能为空"})
    register_body = {"models": [
        {"filename": "ae.safetensors", "path": "/x/vae/ae.safetensors", "role": "component",
         "suggested": {"serviceType": "image", "provider": "", "baseUrl": "",
                       "model": "ae", "runtime": "comfyui", "role": "component",
                       "callable": False, "note": "组件说明"}},
        {"filename": "qwen3-4b.gguf", "path": "/x/llm/qwen3-4b.gguf",
         "suggested": {"serviceType": "text", "provider": "openai",
                       "baseUrl": "http://localhost:11434", "model": "qwen3-4b",
                       "runtime": "ollama", "role": "standalone", "callable": True, "note": "n"}},
        {"filename": "sd_xl_base.safetensors", "path": "/x/checkpoints/sd_xl_base.safetensors",
         "suggested": {"serviceType": "image", "provider": "", "baseUrl": "",
                       "model": "sd_xl_base", "runtime": "comfyui", "role": "standalone",
                       "callable": False,
                       "note": "ComfyUI 仅作只读扫描来源，不直接调用；"
                               "如需使用请部署为项目自有服务（H3/local-sd/Ollama/CosyVoice）"}},
        {"filename": "whatever.safetensors", "path": "/x/misc/whatever.safetensors"},
        {"filename": "h3_fl2va.safetensors", "path": "/x/dm/h3_fl2va.safetensors",
         "suggested": {"serviceType": "video", "provider": "minimax",
                       "baseUrl": "http://localhost:8765", "model": "hailuo-02",
                       "runtime": "h3", "role": "standalone", "callable": True, "note": "n"}},
        {"filename": "h3_ref2va.safetensors", "path": "/x/dm/h3_ref2va.safetensors",
         "suggested": {"serviceType": "video", "provider": "minimax",
                       "baseUrl": "http://localhost:8765", "model": "hailuo-02",
                       "runtime": "h3", "role": "standalone", "callable": True, "note": "n"}},
    ]}
    registered = client.post("/api/v1/local-models/register", json=register_body).json()["data"]
    check("端点: register —— 组件 / comfyui 只读 / 无法识别 三类都进 skipped（原因各不同）",
          [s["filename"] for s in registered["skipped"]]
          == ["ae.safetensors", "sd_xl_base.safetensors", "whatever.safetensors"]
          and registered["skipped"][0]["reason"] == "组件说明"
          and registered["skipped"][1]["reason"].startswith("ComfyUI 仅作只读扫描来源")
          and registered["skipped"][2]["reason"] == "无法识别模型类型，未生成注册建议",
          registered["skipped"])
    check("端点: register —— 文本（ollama）与视频（H3）各建一条配置",
          sorted(registered["created"]) == ["本地文本服务", "本地视频服务"], registered["created"])
    with engine.begin() as conn:
        rows = {row.service_type: row for row in conn.execute(select(ai_service_configs)).all()}
    text_row = rows["text"]
    video_row = rows["video"]
    check("端点: 新建配置的 api_key='local'、priority=80、is_active=1、model 是 JSON 数组",
          text_row.api_key == "local" and text_row.priority == 80 and text_row.is_active is True
          and json.loads(text_row.model) == ["qwen3-4b"], (text_row.api_key, text_row.model))
    check("端点: H3 的两个权重写进 settings.**checkpoint_map**，键去掉扩展名",
          json.loads(video_row.settings).get("checkpoint_map")
          == {"fl2va": "h3_fl2va", "ref2va": "h3_ref2va"},
          video_row.settings)
    check("端点: register 回执 configs 是**全量**配置（model 已反序列化成数组）",
          {c["service_type"] for c in registered["configs"]} == {"text", "video"}
          and all(isinstance(c["model"], list) for c in registered["configs"]))
    again = client.post("/api/v1/local-models/register", json={"models": [
        {"filename": "qwen3-8b.gguf", "path": "/x/llm/qwen3-8b.gguf",
         "suggested": {"serviceType": "text", "provider": "openai",
                       "baseUrl": "http://localhost:11434", "model": "qwen3-8b",
                       "runtime": "ollama", "role": "standalone", "callable": True, "note": "n"}},
    ], "name": "我的本地服务"}).json()["data"]
    check("端点: 二次注册命中同一 (type|provider|baseUrl) -> **updated 而非 created**，且覆盖 name",
          again["created"] == [] and again["updated"] == ["我的本地服务"], again)
    with engine.begin() as conn:
        text_row2 = conn.execute(select(ai_service_configs)
                                 .where(ai_service_configs.c.service_type == "text")).first()
    check("端点: model 数组**合并去重**（旧的 qwen3-4b 保留）",
          json.loads(text_row2.model) == ["qwen3-4b", "qwen3-8b"], text_row2.model)

    # ---- scan 任务（异步）----
    original_async_scan = r.scan_local_models_async

    async def _fast_scan(_opts: dict) -> dict:
        return {"roots": [MODELS], "models": [], "total": 0,
                "truncated": False, "elapsedMs": 1.0,
                "byKind": {"text": 0, "image": 0, "video": 0, "audio": 0, "unknown": 0}}

    r.scan_local_models_async = _fast_scan  # type: ignore[assignment]
    try:
        started = client.post("/api/v1/local-models/scan", json={"roots": [MODELS]}).json()["data"]
        check("端点: POST /scan -> 返回 taskId（UUID）", len(started["taskId"]) == 36, started)
        check("端点: POST /scan `roots=[]` -> 400",
              client.post("/api/v1/local-models/scan", json={"roots": []}).json()
              == {"code": 400, "message": "roots 参数为空或格式不正确"})
        import time

        for _ in range(60):
            status = client.get("/api/v1/local-models/scan/status",
                                params={"taskId": started["taskId"]}).json()["data"]
            if status["progress"]["done"]:
                break
            time.sleep(0.02)
        check("端点: GET /scan/status -> done 且带 result",
              status["progress"]["done"] is True and status["progress"].get("result") is not None,
              status["progress"])
        check("端点: /scan/status 缺 taskId / 未知 taskId 都是 400 且文案不同",
              client.get("/api/v1/local-models/scan/status").json()
              == {"code": 400, "message": "taskId 不能为空"}
              and client.get("/api/v1/local-models/scan/status",
                             params={"taskId": "nope"}).json()
              == {"code": 400, "message": "任务不存在或已过期"})
        check("端点: POST /scan/cancel 未知任务 -> 400",
              client.post("/api/v1/local-models/scan/cancel", params={"taskId": "nope"}).status_code == 400)

        async def _slow_scan(_opts: dict) -> dict:
            while not _opts["shouldCancel"]():
                await asyncio.sleep(0.01)
            raise ls.ScanCancelledError()

        r.scan_local_models_async = _slow_scan  # type: ignore[assignment]
        started2 = client.post("/api/v1/local-models/scan", json={"roots": [MODELS]}).json()["data"]
        cancelled = client.post("/api/v1/local-models/scan/cancel",
                                params={"taskId": started2["taskId"]}).json()["data"]
        check("端点: POST /scan/cancel -> {taskId, cancelled: true}",
              cancelled == {"taskId": started2["taskId"], "cancelled": True}, cancelled)
        # ⚠️ 任务收尾**不通过 HTTP 轮询**验证：TestClient 的 portal 只在请求期间给事件循环
        #    切片，后台任务拿不到足够的调度（实测一直 done=False）⇒ 改成**直接驱动任务体**，
        #    确定性断言「取消 -> done+cancelled（不是 error）」。
        cancel_state: dict = {"id": "t", "cancelled_flag": False,
                              "progress": {"scannedFiles": 0, "foundModels": 0,
                                           "currentDir": "", "done": False,
                                           "cancelled": False}}
        cancel_state["cancel"] = lambda: cancel_state.update(cancelled_flag=True)

        async def _drive() -> None:
            r._scan_tasks["t"] = cancel_state  # 路由里是这么登记的（同一步骤漏了这条断言就没意义）
            task = asyncio.get_running_loop().create_task(
                r._run_scan_task("t", cancel_state, {"roots": [MODELS]}))
            for _ in range(20):
                if cancel_state["progress"]["done"]:
                    break
                cancel_state["cancel"]()
                await asyncio.sleep(0.01)
            await task

        asyncio.run(_drive())
        check("端点: 取消后的任务收尾为 **done+cancelled**（不是 error）",
              cancel_state["progress"]["done"] is True
              and cancel_state["progress"]["cancelled"] is True
              and "error" not in cancel_state["progress"], cancel_state["progress"])
        check("端点: 任务收尾后仍在内存表里（保留 10 分钟供前端拉结果）",
              "t" in r._scan_tasks)
    finally:
        r.scan_local_models_async = original_async_scan  # type: ignore[assignment]

    # ---- HF / ModelScope ----
    original_list = r.list_remote_files
    r.list_remote_files = lambda repo, source, revision: _fake_list(repo, source, revision)  # type: ignore[assignment]
    try:
        files_ok = client.post("/api/v1/local-models/hf/files",
                               json={"repo": "Qwen/Qwen3-4B", "source": "hf"}).json()["data"]
        check("端点: POST /hf/files -> 归一化 repo/source/revision + 文件列表",
              files_ok["repo"] == "Qwen/Qwen3-4B" and files_ok["source"] == "hf"
              and files_ok["revision"] == "main" and len(files_ok["files"]) == 2, files_ok)
        check("端点: POST /hf/files 非法 repo -> 400（带原因）",
              client.post("/api/v1/local-models/hf/files", json={"repo": "Qwen"}).json()["code"] == 400)
    finally:
        r.list_remote_files = original_list  # type: ignore[assignment]

    original_get_paths = r.get_model_paths
    r.get_model_paths = lambda: {"models_dir": ""}  # type: ignore[assignment]
    check("端点: POST /hf/download 未设存储目录 -> 400 提示先设置",
          client.post("/api/v1/local-models/hf/download", json={"repo": "Q/R", "source": "hf"}).json()
          == {"code": 400, "message": "请先在「模型存储目录」中设置下载目录"})
    r.get_model_paths = original_get_paths  # type: ignore[assignment]

    download_root = tempfile.mkdtemp(prefix="hfdl_")

    class _FakeResponse:
        def __init__(self, status: int, chunks: list[bytes], content_length: int | None = None):
            self.status_code = status
            self._chunks = chunks
            self.headers = ({} if content_length is None
                            else {"content-length": str(content_length)})

        @property
        def is_success(self) -> bool:
            return 200 <= self.status_code < 300

        async def aiter_bytes(self):
            for chunk in self._chunks:
                yield chunk

        async def aclose(self) -> None:
            return None

    class _FakeClient:
        async def aclose(self) -> None:
            return None

    def _patch_download(status: int, payload: bytes, content_length: int | None = None,
                        record: list | None = None):
        def _fake_open(url: str, headers: dict[str, str]):
            if record is not None:
                record.append({"url": url, "headers": dict(headers)})

            async def _open():
                return _FakeClient(), _FakeResponse(status, [payload], content_length)
            return _open()

        r.open_download = _fake_open  # type: ignore[assignment]

    r.list_remote_files = lambda repo, source, revision: _fake_list(repo, source, revision)  # type: ignore[assignment]
    r.get_model_paths = lambda: {"models_dir": download_root}  # type: ignore[assignment]
    try:
        # 1) 正常下载
        _patch_download(200, b"x" * 8, content_length=8)
        frames = _read_frames(client.post("/api/v1/local-models/hf/download",
                                          json={"repo": "Q/R", "source": "hf",
                                                "files": ["model.bin"]}))
        check("下载: 帧序 start -> file_start -> progress -> file_done -> done",
              [f["status"] for f in frames]
              == ["start", "file_start", "progress", "file_done", "done"], [f["status"] for f in frames])
        check("下载: 内容真的落盘，且 **.part 已改名**（不留残件）",
              open(os.path.join(download_root, "Q__R", "model.bin"), "rb").read() == b"x" * 8
              and not os.path.exists(os.path.join(download_root, "Q__R", "model.bin.part")))
        check("下载: 目录名把仓库的 `/` 换成 `__`",
              os.path.isdir(os.path.join(download_root, "Q__R")))
        check("下载: done 帧带 overall/total/failed",
              frames[-1]["overall"] == 8 and frames[-1]["total"] == 8 and frames[-1]["failed"] == [])

        # 2) 已存在且够大 -> 跳过
        frames2 = _read_frames(client.post("/api/v1/local-models/hf/download",
                                           json={"repo": "Q/R", "source": "hf",
                                                 "files": ["model.bin"]}))
        check("下载: 已存在且大小达标 -> file_done/skipped，**不再开流**",
              [f["status"] for f in frames2] == ["start", "file_done", "done"]
              and frames2[1]["skipped"] is True, frames2)

        # 3) 断点续传（206）
        part_dir = os.path.join(download_root, "Q__R2")
        os.makedirs(part_dir, exist_ok=True)
        with open(os.path.join(part_dir, "big.bin.part"), "wb") as handle:
            handle.write(b"abc")
        record: list = []
        _patch_download(206, b"defg", content_length=4, record=record)
        frames3 = _read_frames(client.post("/api/v1/local-models/hf/download",
                                           json={"repo": "Q/R2", "source": "hf",
                                                 "files": ["big.bin"]}))
        check("下载: 206 -> resumed=true、**带 Range 头**、最终文件=已有+新增",
              frames3[1]["resumed"] is True
              and record[0]["headers"].get("Range") == "bytes=3-"
              and open(os.path.join(part_dir, "big.bin"), "rb").read() == b"abcdefg",
              (frames3[1], record))

        # 4) 服务器 4xx（⚠️ 换个仓库名：上一个用例已把 Q__R/model.bin 落盘，会被「已完成」跳过）
        _patch_download(404, b"", content_length=None)
        frames4 = _read_frames(client.post("/api/v1/local-models/hf/download",
                                           json={"repo": "Q/R4", "source": "hf",
                                                 "files": ["model.bin"]}))
        check("下载: 非 2xx -> error 帧（`HTTP 404`）+ 进 failed 列表（其余文件继续下）",
              frames4[1]["status"] == "error" and frames4[1]["error"] == "HTTP 404"
              and frames4[-1]["failed"] == ["model.bin"]
              and frames4[-1]["status"] == "done", frames4)

        # 5) 请求了仓库里没有的文件
        bad = client.post("/api/v1/local-models/hf/download",
                          json={"repo": "Q/R", "source": "hf", "files": ["nope.bin"]})
        check("下载: 请求不存在的文件 -> **400**（不是流里报错）",
              bad.status_code == 400 and "仓库中不存在文件" in bad.json()["message"], bad.json())
    finally:
        r.list_remote_files = original_list  # type: ignore[assignment]
        r.get_model_paths = original_get_paths  # type: ignore[assignment]

    # ---- POST /delete ----
    delete_root = tempfile.mkdtemp(prefix="delroot_")
    os.makedirs(os.path.join(delete_root, "sub"), exist_ok=True)
    with open(os.path.join(delete_root, "a.bin"), "wb") as handle:
        handle.write(b"a")
    with open(os.path.join(delete_root, "sub", "b.bin"), "wb") as handle:
        handle.write(b"b")
    with open(os.path.join(TREE, "outside.bin"), "wb") as handle:
        handle.write(b"o")

    check("端点: POST /delete 无 path -> 400 `请指定要删除的文件或目录`",
          client.post("/api/v1/local-models/delete", json={}).json()
          == {"code": 400, "message": "请指定要删除的文件或目录"})
    r.get_model_paths = lambda: {"models_dir": ""}  # type: ignore[assignment]
    no_dir = client.post("/api/v1/local-models/delete", json={"paths": ["a.bin"]}).json()
    # ⚠️ 未设存储目录**不是 400**：错误落在**逐条 failed** 里，整体仍是 200（原 TS 如此）
    check("端点: POST /delete 未设存储目录 -> 200 + failed[0] 报「先设置目录」",
          no_dir["code"] == 200
          and no_dir["data"]["failed"][0]["error"] == "请先在「模型存储目录」中设置目录",
          no_dir)
    r.get_model_paths = lambda: {"models_dir": delete_root}  # type: ignore[assignment]
    deleted = client.post("/api/v1/local-models/delete", json={"paths": [
        "a.bin", "sub", "nope.bin", os.path.join(TREE, "outside.bin"), ""]}).json()["data"]
    check("删除: 文件与目录都能删（目录递归）",
          sorted(deleted["deleted"]) == ["a.bin", "sub"]
          and not os.path.exists(os.path.join(delete_root, "a.bin"))
          and not os.path.exists(os.path.join(delete_root, "sub")), deleted)
    check("删除: 不存在 -> `路径不存在`；越界 -> `目标路径不在模型存储目录内`",
          [f["error"] for f in deleted["failed"]] == ["路径不存在", "目标路径不在模型存储目录内"],
          deleted["failed"])
    check("删除: 目录外的文件**真的还在**（越界被拦下）",
          os.path.exists(os.path.join(TREE, "outside.bin")))
    root_itself = client.post("/api/v1/local-models/delete", json={"path": "."}).json()["data"]
    check("删除: 不许删模型存储目录**本身**",
          root_itself["deleted"] == []
          and root_itself["failed"][0]["error"] == "不能删除整个模型存储目录", root_itself)
    r.get_model_paths = original_get_paths  # type: ignore[assignment]

    # ================= 汇总 =================
    failed_items = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed_items)}/{len(_RESULTS)} passed")
    return 1 if failed_items else 0


async def _fake_list(repo: str, source: str, revision: str) -> list[dict]:
    """假的仓库文件列表（避免打网络）。"""
    return [{"name": "model.bin", "size": 8}, {"name": "big.bin", "size": 4096}]


def _read_frames(response) -> list[dict]:
    """把 NDJSON 响应体切成帧。"""
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        shutil.rmtree(TREE, ignore_errors=True)
