"""**registry 签名认证** —— 上游 ``auth/auth.go`` 的对等实现。

用途：向私有 registry ``push``/``pull`` 时证明「这个请求是我发的」。协议三步：

1. 服务端下发 nonce（随机串）；
2. 客户端把 ``<公钥base64>:<签名base64>`` 作为凭据回传；
3. 服务端用公钥验签名。

签名对象是**服务端给的原始字节**，签名算法是 ed25519，私钥来自 ``~/.ollama/id_ed25519``
（OpenSSH 格式）。

⚠️ 与上游唯一的实现差异：Go 用 ``golang.org/x/crypto/ssh`` 解析 OpenSSH 私钥，Python 侧
需要 ``cryptography``。本仓 ``requirements.txt`` 里**没有**它（它属于「推私有 registry 才用」
的可选能力，不该拖累普通后端安装）。所以这里做**延迟导入**：没装时抛 :class:`AuthUnavailable`
并给出确切的安装命令，而不是 ``ImportError`` 冒到接口层变成 500。
"""

from __future__ import annotations

import base64
import secrets
from pathlib import Path
from typing import Any

__all__ = [
    "AuthUnavailable",
    "default_key_path",
    "get_public_key",
    "key_exists",
    "new_nonce",
    "sign",
]

DEFAULT_PRIVATE_KEY = "id_ed25519"


class AuthUnavailable(RuntimeError):
    """缺少 ``cryptography`` 或私钥不可用 —— 调用方应降级为「匿名拉公共模型」。"""


def default_key_path() -> Path:
    """``~/.ollama/id_ed25519``（上游写死 ``defaultPrivateKey``）。"""
    return Path.home() / ".ollama" / DEFAULT_PRIVATE_KEY


def _load_private_key(path: str | Path | None = None) -> Any:
    target = Path(path) if path is not None else default_key_path()
    try:
        data = target.read_bytes()
    except OSError as exc:
        raise AuthUnavailable(f"读不到私钥 {target}：{exc}") from exc

    try:
        from cryptography.hazmat.primitives.serialization import load_ssh_private_key
    except ImportError as exc:  # pragma: no cover - 依赖缺失路径
        raise AuthUnavailable(
            "使用 registry 签名需要 cryptography："
            ".venv/Scripts/python.exe -m pip install -i "
            "https://pypi.tuna.tsinghua.edu.cn/simple cryptography"
        ) from exc

    try:
        return load_ssh_private_key(data, password=None)
    except (ValueError, TypeError) as exc:
        raise AuthUnavailable(f"私钥 {target} 不是可用的 ed25519/OpenSSH 私钥：{exc}") from exc


def get_public_key(path: str | Path | None = None) -> str:
    """``ssh-ed25519 AAAA…``（对应 ``GetPublicKey``，已去掉末尾空白）。"""
    key = _load_private_key(path)
    return _openssh_public(key).decode("ascii").strip()


def _openssh_public(key: Any) -> bytes:
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    return key.public_key().public_bytes(Encoding.OpenSSH, PublicFormat.OpenSSH)


def new_nonce(length: int = 32, source: Any = None) -> str:
    """生成 nonce（对应 ``NewNonce``）—— **无填充** 的 base64url，即 JWT 那种形式。

    ⚠️ ``base64.urlsafe_b64encode`` 会补 ``=``，而上游用的是 ``RawURLEncoding``；
    带 ``=`` 的串在 URL/查询参数里会被转义，服务端校验必然失败。
    """
    if source is not None:
        raw = source.read(length)
    else:
        raw = secrets.token_bytes(length)
    if len(raw) != length:
        raise AuthUnavailable(f"随机源只给出 {len(raw)} 字节，需要 {length}")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def sign(data: bytes, path: str | Path | None = None) -> str:
    """返回 ``<公钥base64>:<签名base64>``（对应 ``Sign``）。

    ⚠️ 只编码 **64 字节裸签名**，不带 OpenSSH 的 ``ssh-ed25519`` 长度前缀 ——
    Go 侧取的是 ``ssh.Signature.Blob``，多打前缀服务端就验不过。
    """
    key = _load_private_key(path)
    public = _openssh_public(key)
    parts = public.split(b" ")
    if len(parts) < 2:
        raise AuthUnavailable("公钥格式非法，取不到 base64 段")

    signature = key.sign(data)
    return f"{parts[1].decode('ascii')}:{base64.b64encode(signature).decode('ascii')}"


def key_exists(path: str | Path | None = None) -> bool:
    """私钥在不在（调用方用它决定「要不要提示用户先 ``ollama key generate``」）。"""
    return (Path(path) if path is not None else default_key_path()).is_file()
