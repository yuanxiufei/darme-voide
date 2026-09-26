"""ComfyUI HTTP 客户端（本地服务驱动层，**零第三方依赖** = 只用标准库 ``urllib``）。

## 为什么有这个文件（2026-09-16 移植）

本项目的本地推理运行时是 **ComfyUI**（引擎本体是**机器相关的外部服务**，由
``app/scripts/model_manager.py --runtime git/comfyui`` 安装到 ``configs/model-paths.json`` 指的目录 ✓；
引擎**不 vendor 进仓库** ✗）。缺的一直是「**怎么跟它说话**」这一层 ✗ —— 在 2026-09-16 之前，
全仓产品代码里 ``/prompt`` / ``/history`` / ``/view`` / ``/free`` **零命中** ✗。

本文件按 ``ComfyUI/server.py`` 的**实际实现**逐条对齐（不是照文档猜 ✓），是本项目与
ComfyUI 之间**唯一的接口层**：

===========================  ==========================================================
``GET  /system_stats``        体检（ComfyUI 版本 / 设备 / 显存）
``GET  /object_info[/{cls}]`` **节点能力表** —— 用来①提前发现「自定义节点没装」✓②把 UI 图的
                              ``widgets_values`` 映射回字段名（见 ``workflow.py``）✓
``POST /prompt``              提交 **API 格式**工作流 → ``{prompt_id, number, node_errors}``；
                              校验不过时 **400 + {error, node_errors}** ✓
``GET  /history/{prompt_id}`` 轮询任务：未完成返回 ``{}``；完成返回
                              ``{prompt_id: {prompt, outputs, status}}`` ✓
``GET  /view``                取产物字节（``?filename=&subfolder=&type=output``）✓
``POST /free``                卸载模型 / 释放显存（``{unload_models, free_memory}``）✓
``POST /interrupt``           打断当前执行 ✓
``POST /upload/image``        上传首帧 / 参考图（multipart）✓
===========================  ==========================================================

⚠️ 三个**真机上最容易踩**的点，这里都做了处理：

1. **``/prompt`` 要的是 API 格式**（``{"3": {"class_type": ..., "inputs": {...}}}``），
   而 ComfyUI 界面导出的、以及参考 workflow 里的都是 **UI 格式**（``nodes[]/links[]``）✗
   ⇒ 转换在 :mod:`workflow` ✓，别把 UI 图直接 POST 上来（会 400）✓。
2. **缺自定义节点时报错发生在很深的地方**（400 的 ``node_errors`` 里，或执行期抛异常 ✗）
   ⇒ :meth:`ComfyUIClient.missing_nodes` 用 ``/object_info`` **先探一遍**，把
   「到底缺哪几个节点」直接说出来 ✓（H3 要 4 个节点：Easy / Multishot / GGUF / KJNodes ✓）。
3. **``/history`` 未完成时是空 dict**（不是 404、也不是 status=pending ✗）⇒ 轮询要判「空 = 还没好」✓，
   并且**必须有超时**（H3 冷启动 50s+、标准 5 秒片 85s+ ✓ 见参考 README 实测）。
"""
from __future__ import annotations

import json
import mimetypes
import random
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

__all__ = [
    "ComfyUIClient",
    "ComfyUIError",
    "NodeMissingError",
    "WorkflowValidationError",
    "extract_output_items",
]

#: 参考 README 实测：冷启动 53s、5 秒片 85s+ ⇒ 默认给足，别用「HTTP 默认超时」的直觉 ✗
DEFAULT_TIMEOUT_SECONDS = 1800.0
DEFAULT_POLL_SECONDS = 1.0


class ComfyUIError(RuntimeError):
    """与 ComfyUI 通信/执行失败（带 HTTP 状态与响应摘要，便于排障 ✓）。"""

    def __init__(self, message: str, status: int | None = None, detail: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.detail = detail


class NodeMissingError(ComfyUIError):
    """工作流需要的节点类在 ComfyUI 里不存在（自定义节点没装 ✓）。"""

    def __init__(self, missing: list[str]) -> None:
        self.missing = sorted(missing)
        super().__init__("ComfyUI 缺少节点类：" + "、".join(self.missing))


class WorkflowValidationError(ComfyUIError):
    """``POST /prompt`` 被 ComfyUI 校验拒绝（400）：带上 ``node_errors`` 原文 ✓。"""

    def __init__(self, error: Any, node_errors: Any) -> None:
        self.node_errors = node_errors
        super().__init__(f"工作流校验失败：{error}")


def extract_output_items(history_entry: dict[str, Any]) -> list[dict[str, Any]]:
    """从 ``/history`` 记录里取出**产物清单**（``[{filename, subfolder, type, kind}]``）。

    ⚠️ 键名随节点实现而变（``SaveVideo`` 走 ``PreviewVideo`` ⇒ ``videos`` ✓；
    ``PreviewImage`` ⇒ ``images`` ✓；老节点还有 ``gifs`` ✓）⇒ **不写死**，三类都收 ✓。
    """
    items: list[dict[str, Any]] = []
    for node_output in (history_entry.get("outputs") or {}).values():
        if not isinstance(node_output, dict):
            continue
        for kind in ("videos", "gifs", "images", "audio"):
            for entry in node_output.get(kind) or []:
                if not isinstance(entry, dict) or not entry.get("filename"):
                    continue
                items.append({
                    "filename": entry.get("filename"),
                    "subfolder": entry.get("subfolder") or "",
                    "type": entry.get("type") or "output",
                    "kind": kind,
                })
    return items


class ComfyUIClient:
    """ComfyUI(8188) 的极简客户端。同步阻塞（调用方在独立线程里跑 ✓）。"""

    def __init__(self, base_url: str = "http://127.0.0.1:8188",
                 timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.timeout = timeout
        #: 每台客户端一个 id：ComfyUI 用它做执行上下文隔离（前端同款做法 ✓）
        self.client_id = uuid.uuid4().hex

    # ── 底层 ───────────────────────────────────────────────────────────
    def _url(self, path: str, query: dict[str, Any] | None = None) -> str:
        url = f"{self.base_url}{path}"
        if query:
            clean = {k: v for k, v in query.items() if v is not None}
            url = f"{url}?{urllib.parse.urlencode(clean)}"
        return url

    def _request(self, method: str, path: str, *, query: dict[str, Any] | None = None,
                 body: Any = None, raw: bytes | None = None,
                 content_type: str | None = None) -> tuple[int, bytes]:
        data = raw
        headers: dict[str, str] = {}
        if body is not None:
            # ⚠️ 紧凑分隔符 + 不转义非 ASCII：本仓约定「请求体与 JS 的 ``JSON.stringify`` 字节一致」✓
            #    （守卫 `tests/route_parity_test.py` 会查这个 —— 它当场就抓到了本文件的初版 ✗）。
            data = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"
        if content_type:
            headers["Content-Type"] = content_type
        request = urllib.request.Request(self._url(path, query), data=data,
                                         headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                return response.status, response.read()
        except urllib.error.HTTPError as err:
            detail = ""
            try:
                detail = err.read().decode("utf-8", "replace")[:600]
            except Exception:  # noqa: BLE001
                pass
            raise ComfyUIError(f"ComfyUI {method} {path} -> HTTP {err.code}", err.code, detail) from err
        except urllib.error.URLError as err:
            raise ComfyUIError(f"ComfyUI 不可达（{self.base_url}）：{err.reason}") from err

    def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        _status, payload = self._request(method, path, **kwargs)
        if not payload:
            return None
        return json.loads(payload.decode("utf-8"))

    # ── 体检 / 能力表 ──────────────────────────────────────────────────
    def system_stats(self) -> dict[str, Any]:
        """``GET /system_stats``：ComfyUI 版本 + 设备/显存（排障第一站 ✓）。"""
        return self._json("GET", "/system_stats") or {}

    def object_info(self, node_class: str | None = None) -> dict[str, Any]:
        """``GET /object_info[/{cls}]``：节点能力表（字段顺序**就是** widgets 的顺序 ✓）。"""
        path = "/object_info" if not node_class else f"/object_info/{urllib.parse.quote(node_class)}"
        return self._json("GET", path) or {}

    def missing_nodes(self, class_types: set[str] | list[str]) -> list[str]:
        """列出**装不上/没装**的节点类（用 ``/object_info`` 逐类探 ✓）。

        ⚠️ 每次调用只探需要的类（``/object_info/{cls}`` ✓）而不是拉全表（全表可能几 MB ✗）。
        """
        missing: list[str] = []
        for name in sorted(set(class_types)):
            if not self.object_info(name):
                missing.append(name)
        return missing

    # ── 提交 / 轮询 / 取产物 ───────────────────────────────────────────
    def queue_prompt(self, api_prompt: dict[str, Any], *,
                     extra_data: dict[str, Any] | None = None) -> str:
        """``POST /prompt``（**API 格式**）→ ``prompt_id``；被拒时抛 :class:`WorkflowValidationError` ✓。"""
        body: dict[str, Any] = {"prompt": api_prompt, "client_id": self.client_id}
        if extra_data:
            body["extra_data"] = extra_data
        try:
            result = self._json("POST", "/prompt", body=body)
        except ComfyUIError as err:
            if err.status == 400 and err.detail:
                try:
                    parsed = json.loads(err.detail)
                except ValueError:
                    raise
                raise WorkflowValidationError(parsed.get("error"), parsed.get("node_errors")) from err
            raise
        prompt_id = (result or {}).get("prompt_id")
        if not prompt_id:
            raise ComfyUIError(f"ComfyUI 未返回 prompt_id：{result!r}")
        return str(prompt_id)

    def history(self, prompt_id: str) -> dict[str, Any]:
        """``GET /history/{id}``：**未完成时返回空 dict** ✓（不是 404 ✗）。"""
        all_history = self._json("GET", f"/history/{urllib.parse.quote(prompt_id)}") or {}
        entry = all_history.get(prompt_id)
        return entry if isinstance(entry, dict) else {}

    def wait(self, prompt_id: str, *, timeout: float = DEFAULT_TIMEOUT_SECONDS,
             poll_interval: float = DEFAULT_POLL_SECONDS,
             on_tick: Any = None) -> dict[str, Any]:
        """轮询到**完成/失败/超时**，返回 history 记录。

        完成判据（与 ComfyUI 一致 ✓）：history 里出现该 ``prompt_id``；
        成败看 ``status.status_str``（``success`` / ``error`` ✓）。
        """
        deadline = time.monotonic() + timeout
        while True:
            entry = self.history(prompt_id)
            if entry:
                status = (entry.get("status") or {})
                if status.get("completed") is True or status.get("status_str") in ("success", "error"):
                    return entry
                # 特殊：有些版本 completed 缺失但已有 outputs ⇒ 视为已完成 ✓
                if entry.get("outputs"):
                    return entry
            if time.monotonic() > deadline:
                raise ComfyUIError(
                    f"ComfyUI 任务超时（{timeout:.0f}s，prompt_id={prompt_id}）—— "
                    f"H3 冷启动/大分辨率会很久，必要时调大超时 ✓")
            if on_tick:
                on_tick(prompt_id)
            time.sleep(poll_interval)

    def fetch_output(self, item: dict[str, Any]) -> bytes:
        """``GET /view`` 取产物字节（视频/图/音频都用这条 ✓）。"""
        _status, payload = self._request("GET", "/view", query={
            "filename": item.get("filename"),
            "subfolder": item.get("subfolder") or "",
            "type": item.get("type") or "output",
        })
        return payload

    def save_output(self, item: dict[str, Any], dest_dir: str | Path,
                    filename: str | None = None) -> Path:
        """取产物并落盘（保存**原始文件名**便于追溯 ✓）；返回落盘路径 ✓。"""
        payload = self.fetch_output(item)
        target_dir = Path(dest_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        name = filename or Path(str(item.get("filename"))).name
        path = target_dir / name
        path.write_bytes(payload)
        return path

    # ── 运维 ───────────────────────────────────────────────────────────
    def free(self, *, unload_models: bool = True, free_memory: bool = True) -> bool:
        """``POST /free`` 卸载模型/释放显存（H3 跑完必须调 ✓ 否则 24G 卡留不住下一镜 ✗）。"""
        try:
            self._request("POST", "/free",
                          body={"unload_models": bool(unload_models),
                                "free_memory": bool(free_memory)})
            return True
        except ComfyUIError:
            return False

    def interrupt(self, prompt_id: str | None = None) -> bool:
        """``POST /interrupt`` 打断（可指定 ``prompt_id`` ✓）。"""
        try:
            self._request("POST", "/interrupt", body={"prompt_id": prompt_id} if prompt_id else {})
            return True
        except ComfyUIError:
            return False

    def _upload_multipart(self, path: str, field: str, filename: str, data: bytes, *,
                          subfolder: str = "", extra: dict[str, str] | None = None,
                          content_type: str | None = None) -> dict[str, Any]:
        """手搓 ``multipart/form-data`` 上传（零依赖 ✓）；``/upload/image`` 与 ``/upload/mask`` 共用 ✓。"""
        boundary = "----voide" + uuid.uuid4().hex
        body = bytearray()

        def _field(name: str, value: str) -> None:
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
            body.extend(f"{value}\r\n".encode())

        for key, value in (extra or {}).items():
            _field(key, str(value))
        if subfolder:
            _field("subfolder", subfolder)
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'.encode())
        mime = content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        body.extend(f"Content-Type: {mime}\r\n\r\n".encode())
        body.extend(data)
        body.extend(f"\r\n--{boundary}--\r\n".encode())

        result = self._json("POST", path, raw=bytes(body),
                            content_type=f"multipart/form-data; boundary={boundary}")
        return result if isinstance(result, dict) else {}

    def upload_image(self, filename: str, data: bytes, *, subfolder: str = "",
                     overwrite: bool = True, content_type: str | None = None) -> dict[str, Any]:
        """``POST /upload/image`` 上传首帧/参考图（multipart/form-data 手搓，零依赖 ✓）。

        返回 ``{name, subfolder, type}`` ✓ —— 这三个字段正是 ``LoadImage`` 节点的
        ``image``（widget）需要的形状 ✓。
        """
        return self._upload_multipart("/upload/image", "image", filename, data,
                                      subfolder=subfolder,
                                      extra={"overwrite": "true" if overwrite else "false"},
                                      content_type=content_type)

    # ══════════════════════════════════════════════════════════════════
    # 目录 / 清单类（对齐 ComfyUI 的读端点 ✓ —— 2026-09-17 补齐到全覆盖）
    # ══════════════════════════════════════════════════════════════════
    def features(self) -> dict[str, Any]:
        """``GET /features``：能力开关（服务端告诉前端「哪些特性可用」✓）。"""
        return self._json("GET", "/features") or {}

    def embeddings(self) -> list[str]:
        """``GET /embeddings``：embeddings 清单（**服务端已经去扩展名** ✓）。"""
        return list(self._json("GET", "/embeddings") or [])

    def model_types(self) -> list[str]:
        """``GET /models``：模型**类别**清单（checkpoints / diffusion_models / vae / text_encoders… ✓）。

        ⚠️ 这就是 ComfyUI 的**权威目录表**来源 ✓ —— 比我们自己按目录名猜更准 ✓
        （例如 `text_encoders` 同时也涵盖旧目录 `clip` ✓、`diffusion_models` 涵盖 `unet` ✓）。
        """
        return list(self._json("GET", "/models") or [])

    def models(self, folder: str) -> list[str]:
        """``GET /models/{folder}``：某一类下的**文件清单**（相对路径 ✓，可含子目录 ✓）。"""
        return list(self._json("GET", f"/models/{urllib.parse.quote(str(folder))}") or [])

    def folder_paths(self) -> dict[str, Any]:
        """``GET /internal/folder_paths``：各类模型的**实际磁盘目录**（桌面包内部端点 ✓）。

        比 `/models/{folder}` 更进一步：它给出**路径** ⇒ 我们能在自己的「本地模型」页里直接展示
        「ComfyUI 从哪儿读模型」✓（`model_manager.detect_comfyui()` 只能猜根目录 ✗）。
        """
        return self._json("GET", "/internal/folder_paths") or {}

    def internal_files(self, directory_type: str) -> list[dict[str, Any]]:
        """``GET /internal/files/{type}``：输入/输出目录的文件清单（含大小/时间 ✓）。"""
        return list(self._json("GET", f"/internal/files/{urllib.parse.quote(str(directory_type))}") or [])

    def object_info_all(self) -> dict[str, Any]:
        """``GET /object_info``：**全部**节点类的能力表。

        ⚠️ 在装了很多自定义节点的机器上，这份表可能有**几 MB** ✗ ⇒ 只要几类时请用
        :meth:`object_info`（逐类探 ✓）或 :meth:`missing_nodes`（只判存在性 ✓）。
        """
        return self._json("GET", "/object_info") or {}

    def view_metadata(self, folder_name: str) -> dict[str, Any]:
        """``GET /view_metadata/{folder_name}``：输出目录资产元数据（按文件夹 ✓）。"""
        return self._json("GET", f"/view_metadata/{urllib.parse.quote(str(folder_name))}") or {}

    def upload_mask(self, filename: str, data: bytes, *, subfolder: str = "",
                    original_ref: str | None = None) -> dict[str, Any]:
        """``POST /upload/mask`` 上传遮罩（走和图片同一套 multipart ✓，可带 ``original_ref`` ✓）。"""
        extra = {"original_ref": original_ref} if original_ref is not None else None
        return self._upload_multipart("/upload/mask", "image", filename, data,
                                      subfolder=subfolder, extra=extra)

    # ══════════════════════════════════════════════════════════════════
    # 队列 / 历史 / 作业（控制面 ✓）
    # ══════════════════════════════════════════════════════════════════
    def queue(self) -> dict[str, Any]:
        """``GET /queue``：``{queue_running, queue_pending}`` 快照 ✓。"""
        return self._json("GET", "/queue") or {}

    def queue_remaining(self) -> dict[str, Any]:
        """``GET /prompt``：剩余的 prompt 数与执行信息（官方给前端轮询用 ✓）。"""
        return self._json("GET", "/prompt") or {}

    def clear_queue(self) -> bool:
        """``POST /queue {"clear": true}``：**清空**待执行队列 ✓。"""
        try:
            self._request("POST", "/queue", body={"clear": True})
            return True
        except ComfyUIError:
            return False

    def delete_queue_item(self, prompt_id: str) -> bool:
        """``POST /queue {"delete": [id]}``：从队列里删掉某个待执行任务 ✓。"""
        try:
            self._request("POST", "/queue", body={"delete": [prompt_id]})
            return True
        except ComfyUIError:
            return False

    def history_all(self, *, max_items: int | None = None, offset: int = -1) -> dict[str, Any]:
        """``GET /history``：历史（可限量/偏移 ✓）。"""
        query: dict[str, Any] = {"offset": offset}
        if max_items is not None:
            query["max_items"] = max_items
        return self._json("GET", "/history", query=query) or {}

    def clear_history(self) -> bool:
        """``POST /history {"clear": true}``：清空历史 ✓。"""
        try:
            self._request("POST", "/history", body={"clear": True})
            return True
        except ComfyUIError:
            return False

    def jobs(self) -> Any:
        """``GET /api/jobs``：作业列表（**新版**作业 API；老版本可能 404 ⇒ 返回 None ✓）。"""
        try:
            return self._json("GET", "/api/jobs")
        except ComfyUIError as err:
            if err.status == 404:
                return None
            raise

    def job(self, job_id: str) -> Any:
        """``GET /api/jobs/{id}``：单个作业详情（老版本 404 ⇒ None ✓）。"""
        try:
            return self._json("GET", f"/api/jobs/{urllib.parse.quote(str(job_id))}")
        except ComfyUIError as err:
            if err.status == 404:
                return None
            raise

    def cancel_job(self, job_id: str) -> bool:
        """``POST /api/jobs/{id}/cancel``：取消作业 ✓。"""
        try:
            self._request("POST", f"/api/jobs/{urllib.parse.quote(str(job_id))}/cancel")
            return True
        except ComfyUIError:
            return False

    def cancel_all_jobs(self) -> bool:
        """``POST /api/jobs/cancel``：取消**全部**作业 ✓。"""
        try:
            self._request("POST", "/api/jobs/cancel")
            return True
        except ComfyUIError:
            return False

    # ══════════════════════════════════════════════════════════════════
    # 提交前预检（**我们自己加的**：ComfyUI 只在 POST /prompt 时才校验 ✗）
    # ══════════════════════════════════════════════════════════════════
    def validate(self, api_prompt: dict[str, Any], *, known_classes: set[str] | None = None) -> list[str]:
        """**不提交**就把问题找出来；返回问题清单（空 = 看起来可提交 ✓）。

        检的三件事（都是真机上最容易白跑一轮的原因 ✓）：
        ① 结构：每个节点必须有 ``class_type`` 与 ``inputs``；
        ② 连线：``[id, slot]`` 形态的输入，其**源节点必须在同一份 prompt 里**（悬空连线 ⇒ 必失败 ✗）；
        ③ 节点类：给了 ``known_classes`` 就比对（通常来自 ``/object_info`` ✓），不在其中的逐个点名 ✓。
        """
        problems: list[str] = []
        if not isinstance(api_prompt, dict) or not api_prompt:
            return ["prompt 为空或不是「API 格式」的对象 ✗（UI 图要先过 workflow.ui_to_api ✓）"]
        for node_id, node in api_prompt.items():
            if not isinstance(node, dict) or "class_type" not in node or "inputs" not in node:
                problems.append(f"节点 {node_id} 缺 class_type/inputs ✗")
        for node_id, node in api_prompt.items():
            if not isinstance(node, dict):
                continue
            for field, value in (node.get("inputs") or {}).items():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    if value[0] not in api_prompt:
                        problems.append(
                            f"节点 {node_id}.{field} 的连线指向不存在的节点 {value[0]!r} ✗")
        if known_classes is not None:
            for node_id, node in api_prompt.items():
                if isinstance(node, dict) and node.get("class_type") not in known_classes:
                    problems.append(f"节点 {node_id} 的类 {node.get('class_type')!r} 不在已知节点表里 ✗")
        return problems

    # ── 便捷：一次完整闭环 ─────────────────────────────────────────────
    def run(self, api_prompt: dict[str, Any], *, required_nodes: set[str] | None = None,
            timeout: float = DEFAULT_TIMEOUT_SECONDS,
            wait_for_outputs: bool = True,
            on_tick: Any = None) -> list[dict[str, Any]]:
        """提交 → 轮询 → 取产物清单（**不开火不管**：失败抛 :class:`ComfyUIError` ✓）。

        ``required_nodes`` 给了就先探节点（把「缺哪个节点」提前说清 ✓，而不是等 400/执行期异常 ✗）。
        """
        if required_nodes:
            missing = self.missing_nodes(required_nodes)
            if missing:
                raise NodeMissingError(missing)
        prompt_id = self.queue_prompt(api_prompt)
        entry = self.wait(prompt_id, timeout=timeout, on_tick=on_tick) if wait_for_outputs else {}
        if entry:
            status = (entry.get("status") or {})
            if status.get("status_str") == "error":
                messages = status.get("messages") or []
                raise ComfyUIError(f"ComfyUI 执行失败（prompt_id={prompt_id}）：{messages!r}"[:600])
        return extract_output_items(entry)
