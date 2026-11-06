"""本地 VLM 重识别通道：把「本地 OCR 认不出的页」交给进程外的视觉语言模型重跑一遍。

两种后端，决策规则一份
----------------------
``OCR_VLM_BACKEND`` 选传输方式，**上面的判断规则与下面的决策函数完全不变**：

* ``local``（推荐）：``ocr_vlm_local`` 在**本进程内**用 torch + transformers 直接跑
  ``PaddleOCR-VL``。transformers 5.x 已原生收录 ``paddleocr_vl``，本仓现成的 CUDA
  环境即可，不需要 paddlepaddle / vLLM / Docker。代价是每个真正用到它的 worker
  各占一份显存，因此那边做了懒加载与显存门槛。
* ``http``（默认，接口兼容）：只做**一个 HTTP 客户端**，接口用 OpenAI 兼容的
  ``/v1/chat/completions`` —— vLLM、SGLang、``paddleocr genai_server`` 都是这个形状，
  换模型/换机器不用改代码。适合"VLM 跑在另一台机器或容器里"的部署。

两条后端产出的都是「一段纯文本」，所以 ``decide`` 不必知道文本是怎么来的。

三条硬约束决定了本模块的形状
----------------------------
1. **只许环回**：端点必须解析到 127.0.0.1 / ::1 / localhost，其它地址**直接拒绝构造**。
   语料是采购库，出网就是版权与保密事故；这件事不能只靠「记得别配错」。
2. **只许逐字转录**：prompt 里禁止改写、纠错、翻译、合并；返回后还要做长度体检
   （比本地引擎少 15% 以上 ⇒ 大概率漏行，不予采用）。VLM 最危险的错误是
   「通顺的错误」—— 结构完全正常，人读都不一定发现，只能靠这两道闸。
3. **零新增依赖**：stdlib ``urllib`` 足够。不引入 requests/httpx/openai，少一个能
   拖垮 Windows venv 的东西，也少一条不受控的出网路径。

默认**不启用**（``OCR_VLM_ENDPOINT`` 为空 ⇒ 整条通道不存在），因此不改变现有行为。
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

__all__ = ["VlmConfig", "VlmClient", "Escalator", "divergence", "decide",
           "is_loopback", "is_blank_page", "TRANSCRIBE_PROMPT"]

#: 逐字转录指令。**不要**往这里加「请修正错别字/补全标点」之类的句子：
#: 剧本与史料的价值就在原文，模型改一个字就等于篡改语料，而且改完看不出痕迹。
TRANSCRIBE_PROMPT = (
    "请逐字转录这张图片中的全部文字，保持原有的阅读顺序与分行。"
    "只输出转录结果：不要翻译、不要改写、不要纠正错别字、不要补全省略的标点、"
    "不要合并或拆分段落、不要添加任何说明或标题。无法辨认的字用 □ 代替。"
)

#: 档位（越大越好用），决策时按档位取优
_RANK = {"ok": 2, "suspect": 1, "reject": 0}

#: 与本地引擎分歧超过这个比例 ⇒ 两遍都不算数，标 escalated 交人看
DIVERGENCE_ALERT = 0.02
#: VLM 结果比本地结果短这么多 ⇒ 判为漏行（VLM 漏行比多字更常见，也更致命）
SHRINK_REJECT = 0.15
#: 连续失败这么多次就熔断本次进程内的 VLM（否则每页都白等一次超时）
MAX_CONSECUTIVE_FAILURES = 3

#: 一页「暗像素」占比低于此数 ⇒ 判为空白页。210 万像素（150 dpi A4）对应约 420 个
#: 暗像素，即**不到一个字**；纯白页实测为 0，带噪点的扫描空白页通常几十个。
BLANK_INK_RATIO = 0.0002

_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def is_blank_page(image, ink_ratio: float = BLANK_INK_RATIO) -> bool:
    """页图是否几乎空白（没有任何可读墨迹）。

    实测动因：一张**纯白**页（``mean=255, std=0``）送进 PaddleOCR-VL，它吐出了
    「100%」—— 一个字的证据都没有，纯属编造；而本地引擎正确地读出 0 字。空白页是
    VLM 最危险的一类输入，因为它找不到任何视觉证据，只能靠先验补全。所以升级前先
    判空白：这时"读不出字"本身就是正确答案，不送 VLM、不烧预算、不给幻觉留缝。

    判据用**暗像素占比**而不是灰度方差：扫描噪点会把方差抬得很高，但噪点不产生墨迹。

    取不到像素时（不是数组也不是 PIL）一律返回 ``False`` —— 宁可多试一次，也不能
    把有字的页误判成空白（那等于静默丢正文）。
    """
    try:
        import numpy as np  # noqa: PLC0415

        array = np.asarray(image.convert("L")) if hasattr(image, "convert") else np.asarray(image)
        if array.ndim == 3:
            array = array[..., :3].mean(axis=2)
        if array.size == 0:
            return False
        return float((array < 200).mean()) < ink_ratio
    except Exception:  # noqa: BLE001 —— 判不了就不判
        return False
_IPV4_LOOPBACK = re.compile(r"^127(?:\.\d{1,3}){3}$")


def is_loopback(url: str) -> bool:
    """端点是否只在本机。非 http(s)、无主机名、或解析到外部地址 ⇒ False。"""
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").strip().lower()
    return host in _LOOPBACK_HOSTS or bool(_IPV4_LOOPBACK.match(host))


def _finite(raw: str | None, default: float) -> float:
    try:
        value = float(raw or "")
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


class VlmConfig:
    """VLM 通道配置。全部来自环境变量，不改配置文件、不落盘。"""

    __slots__ = ("backend", "endpoint", "model", "model_path", "timeout",
                 "max_pages", "probe", "prompt")

    def __init__(self, endpoint: str = "", model: str = "", timeout: float = 120.0,
                 max_pages: int = 6, probe: bool = False, prompt: str = TRANSCRIBE_PROMPT,
                 backend: str = "http", model_path: str = "") -> None:
        self.backend = (backend or "http").strip().lower()
        if self.backend not in ("http", "local"):
            #: 写错（如 ``inproc``）一律退回 http：静默变成一个不认识的第三态，
            #: 表现是"配了但一页都没升级"，比直接退回更难查
            self.backend = "http"
        self.endpoint = (endpoint or "").strip().rstrip("/")
        self.model = (model or "").strip() or "PaddleOCR-VL-0.9B"
        self.model_path = (model_path or "").strip()
        self.timeout = timeout
        self.max_pages = max(0, int(max_pages))
        self.probe = bool(probe)
        self.prompt = (prompt or TRANSCRIBE_PROMPT).strip()

    @classmethod
    def from_env(cls) -> "VlmConfig":
        try:
            max_pages = int(os.environ.get("OCR_VLM_MAX") or 6)
        except ValueError:
            max_pages = 6
        return cls(
            endpoint=os.environ.get("OCR_VLM_ENDPOINT") or "",
            model=os.environ.get("OCR_VLM_MODEL") or "",
            timeout=_finite(os.environ.get("OCR_VLM_TIMEOUT"), 120.0),
            max_pages=max_pages,
            probe=bool(os.environ.get("OCR_VLM_PROBE")),
            prompt=os.environ.get("OCR_VLM_PROMPT") or TRANSCRIBE_PROMPT,
            backend=os.environ.get("OCR_VLM_BACKEND") or "http",
            model_path=os.environ.get("OCR_VLM_MODEL_PATH") or "",
        )

    @property
    def local(self) -> bool:
        """是否进程内 transformers 后端。"""
        return self.backend == "local"

    def weights_dir(self) -> Path:
        """权重目录（local 后端用）。延迟 import，避免 http 模式白白拉起路径推导。"""
        if self.model_path:
            return Path(self.model_path)
        from app.scripts.corpus import ocr_vlm_local  # noqa: PLC0415

        return ocr_vlm_local.default_model_dir()

    @property
    def usable(self) -> bool:
        """允许升级（``max_pages>0`` 或探针模式）**且**后端真能连上/真有权重。"""
        if not (self.max_pages > 0 or self.probe):
            return False
        if self.local:
            #: 权重不在盘上 ⇒ 整条通道不存在。这样 ``--stats`` 会如实说"未就绪"，
            #: 而不是每份文档第一个可疑页都去撞一次加载失败
            return (self.weights_dir() / "config.json").is_file()
        return bool(self.endpoint) and is_loopback(self.endpoint)

    def describe(self) -> str:
        if self.local:
            path = self.weights_dir()
            if not (path / "config.json").is_file():
                return (f"VLM：进程内未就绪（{path.as_posix()} 缺 config.json ⇒ "
                        f"先跑 ocr_vlm_fetch.py --fetch）")
            return f"VLM：进程内 PaddleOCR-VL @ {path.as_posix()}（每份最多 {self.max_pages} 页）"
        if not self.endpoint:
            #: 「没装权重」与「装了但没启用」是两种状态，混成一句「未配置」的后果是
            #: 实测踩过的：``data/models/PaddleOCR-VL-1.6`` 齐备，``--stats`` 照样写
            #: 「未配置」，于是没人知道加 ``--vlm`` 就能用，整条通道白装。
            #: ⚠️ 这里只**报告**就绪，绝不自动启用 —— VLM 每页一次推理，默认必须关。
            try:
                path = self.weights_dir()
                ready = (path / "config.json").is_file()
            except Exception:  # noqa: BLE001 —— 目录推导/依赖缺失一律当"没有"
                path, ready = None, False
            if ready:
                return (f"VLM：未启用（本地权重已就绪 @ {path.as_posix()}"
                        f" ⇒ 加 --vlm 启用）")
            return "VLM：未配置"
        if not is_loopback(self.endpoint):
            return f"VLM：拒绝（{self.endpoint} 不是本机地址）"
        return f"VLM：{self.model} @ {self.endpoint}（每份最多 {self.max_pages} 页）"


def _encode_png(image) -> bytes:
    """numpy RGB 图 → PNG bytes。只在实际要送 VLM 时调用（编码本身约 10ms/页）。"""
    from PIL import Image  # noqa: PLC0415 —— 只有真跑时才需要

    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def divergence(left: str, right: str) -> float:
    """两段文本的分歧度：``0``＝逐字相同，``1``＝毫无重叠。

    用 ``difflib`` 的比值（2M/(T)）：对「漏了一行」「顺序颠倒」都会明显抬高，
    而单纯标点差异抬得很少 —— 正好是我们要的灵敏度方向。
    """
    import difflib  # noqa: PLC0415

    a = re.sub(r"\s+", "", left or "")
    b = re.sub(r"\s+", "", right or "")
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    return round(1.0 - difflib.SequenceMatcher(None, a, b, autojunk=False).ratio(), 4)


def decide(local_text: str, local_verdict: str, vlm_text: str, vlm_verdict: str) -> dict:
    """两遍结果取谁 —— **纯函数**，全部规则在这里，便于逐条钉住。

    返回 ``{"text","source","verdict","note"}``，``note`` 为空表示无需留痕。

    规则（按优先级）：

    1. VLM 没给出内容 ⇒ 用本地（不许因为"对面没说话"就丢内容）；
    2. VLM 明显漏行（字数缩水 ≥15%）⇒ 用本地，留痕（**不能**采纳一份缺段的文本）；
    3. VLM 档位更高 ⇒ 用 VLM（这是整条通道的收益：把 reject 页救回来）；
    4. 档位相同、都是 ok ⇒ 用本地（本地不会改写原文；VLM 一致时没有收益），
       但分歧过大时降级为 suspect 并留痕 —— 两遍读法不同，人得看一眼；
    5. 其余 ⇒ 用本地。
    """
    local_text = local_text or ""
    vlm_text = vlm_text or ""
    if not vlm_text.strip():
        return {"text": local_text, "source": "local", "verdict": local_verdict, "note": ""}

    local_size = len(re.sub(r"\s+", "", local_text))
    vlm_size = len(re.sub(r"\s+", "", vlm_text))
    gap = divergence(local_text, vlm_text)
    #: ⚠️ 缩水判定只在本地文本**还算像话**时才有基准。本地已经是 reject（比如整页
    #: 被认成 ``|||||``）时字数天然偏多，拿它当尺子量 VLM 会把好结果全判成"漏行" ——
    #: 等于在这条通道最该生效的场景上永久失效。
    if local_verdict != "reject" and local_size >= 40 and vlm_size < local_size * (1.0 - SHRINK_REJECT):
        return {"text": local_text, "source": "local", "verdict": local_verdict,
                "note": f"VLM 结果偏短（{vlm_size}/{local_size} 字，疑漏行），未采用"}

    if _RANK.get(vlm_verdict, 0) > _RANK.get(local_verdict, 0):
        return {"text": vlm_text, "source": "vlm", "verdict": vlm_verdict,
                "note": f"VLM 升级：本地 {local_verdict} → VLM {vlm_verdict}（分歧 {gap:.0%}）"}

    if vlm_verdict == local_verdict == "ok" and gap > DIVERGENCE_ALERT:
        return {"text": local_text, "source": "local", "verdict": "suspect",
                "note": f"双引擎分歧 {gap:.0%}（本地与 VLM 判 ok 但文本不同，待人工复核）"}

    return {"text": local_text, "source": "local", "verdict": local_verdict, "note": ""}


class VlmClient:
    """双后端（进程内 transformers / OpenAI 兼容 HTTP）的最小客户端。

    ``http`` 后端**不发任何非环回请求**（``VlmConfig.usable`` 已挡）。
    """

    def __init__(self, config: VlmConfig) -> None:
        self.config = config
        self._failures = 0
        self._local = None
        if config.local and config.usable:
            from app.scripts.corpus import ocr_vlm_local  # noqa: PLC0415

            #: 构造**不加载权重**（懒加载）：没有可疑页的 worker 一分显存都不花
            self._local = ocr_vlm_local.LocalEngine(config.model_path or None)

    @property
    def available(self) -> bool:
        return self.config.usable and self._failures < MAX_CONSECUTIVE_FAILURES

    def describe(self) -> str:
        """人话描述当前后端（``--stats`` 用）。进程内模式要连加载状态一起说。"""
        if self._local is not None:
            return self._local.describe()
        return self.config.describe()

    def recognize(self, image) -> str:
        """一张页图 → 文本。失败抛 ``RuntimeError``（调用方负责降级与熔断）。"""
        if not self.available:
            raise RuntimeError("VLM 通道不可用")
        if self._local is not None:
            try:
                text = self._local.recognize(image, prompt=self.config.prompt)
            except RuntimeError:
                #: 计入熔断：显存不够时不该每页都去撞一次加载，白等又白占
                self._failures += 1
                raise
            self._failures = 0
            return text
        png = _encode_png(image)
        payload = {
            "model": self.config.model,
            "temperature": 0,
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode("ascii")}},
                {"type": "text", "text": self.config.prompt},
            ]}],
        }
        request = urllib.request.Request(
            f"{self.config.endpoint}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                body = json.loads(response.read().decode("utf-8", "replace"))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            self._failures += 1
            raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc
        self._failures = 0
        try:
            return str(body["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"回执结构不认识：{list(body)[:5]}") from exc


class Escalator:
    """可疑页升级器：一页一页地问「本地结果够不够用，要不要请 VLM 重跑」。

    预算按**文档**计（``max_pages``），不是按批 —— 一本 500 页的书里坏页再多，
    也只升级最前面的几页，防止一份烂书把整批时间吃光。
    """

    def __init__(self, config: VlmConfig | None = None) -> None:
        self.config = config or VlmConfig.from_env()
        self.client = VlmClient(self.config) if self.config.usable else None
        self._budget = 0
        self._used = 0
        self.attempted = 0
        self.adopted = 0

    @property
    def enabled(self) -> bool:
        return self.client is not None and self.client.available

    def reset(self) -> None:
        """每份文档开始时调用（预算复位）。"""
        self._budget = self.config.max_pages if self.config.usable else 0
        self._used = 0

    def consider(self, page_no: int, image, local_text: str) -> tuple[str, str | None]:
        """返回 ``(最终文本, 留痕说明)``。留痕为空 ⇒ 无需写审计。"""
        if not self.enabled or self._budget <= 0:
            return local_text, None
        from app.scripts.corpus import text_clean  # noqa: PLC0415 —— 避免顶层循环导入

        verdict = text_clean.judge_page(local_text)
        if verdict["verdict"] == "ok" and not self.config.probe:
            return local_text, None
        if is_blank_page(image):
            #: 空白页不升级（**含** probe 模式）：这类页"读不出字"就是正确答案，送进去
            #: 只会换来幻觉（实测纯白页 ⇒「100%」），标定时拿它做对拍也没有意义。
            return local_text, None
        self._budget -= 1
        self._used += 1
        self.attempted += 1
        try:
            vlm_text = self.client.recognize(image)
        except RuntimeError as exc:
            #: VLM 挂了不能拖垮整批：原样返回本地结果，并把原因写进审计（可查）
            return local_text, f"VLM 不可用（第{page_no}页）：{exc}"
        if not self.client.available:
            return local_text, "VLM 连续失败，本进程内已熔断"
        vlm_verdict = text_clean.judge_page(vlm_text)["verdict"]
        result = decide(local_text, verdict["verdict"], vlm_text, vlm_verdict)
        if result["source"] == "vlm":
            self.adopted += 1
        return result["text"], (result["note"] or None)
