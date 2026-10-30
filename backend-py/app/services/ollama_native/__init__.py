"""ollama 的原生能力层（纯 Python 复刻 Go 侧可对等的部分）。

分工：

================================  ================================  ==================================
模块                              对应上游                          说明
================================  ================================  ==================================
``thinking``                      ``thinking/parser.go``            ``<think>`` 流式切分
``toolparse``                     ``tools/tools.go``                文本流里的工具调用
``modelfile``                     ``parser/parser.go``              Modelfile 状态机
``params``                        ``api/types.go::FormatParams``    参数字符串 → 正确类型
``names``                         ``types/model/name.go``           模型名解析/规范化
``manifest``                      ``manifest/*.go``                 blobs + manifests 读写
``gotemplate``                    Go ``text/template`` 子集         模板引擎本体
``template``                      ``template/*.go``                 内置模板库 + 渲染分支
``harmony``                       ``harmony/harmonyparser.go``      gpt-oss harmony 流解析
``envconfig``                     ``envconfig/config.go``           ``OLLAMA_*`` 真源
``format``                        ``format/*.go``                   人类可读数字/时间
``progress``                      ``progress/*.go``                 进度、速率、ETA（去终端渲染）
``auth``                          ``auth/auth.go``                  registry ed25519 签名
``version``                       ``version/version.go``            版本号与语义化比较
================================  ================================  ==================================

**不具备**的部分（Go 里靠 CGO 绑 llama.cpp 实现的推理引擎 ``llm/`` ``runner/``
``model/`` ``mlx/`` ``llama/``）不在这里重写 —— 那需要编译器和 GPU 运行时，Python 侧
一律通过 HTTP 走外部 ``ollama serve``（"正常实现"）。同理不搬 ``server/``（HTTP 服务端与
调度）与 ``fs/gguf``（本仓 ``app/services/engine/gguf.py`` 已有自研实现）。

客户端能力（chat/generate/embed/pull/push/create/show/ps …）直接使用 vendor 进来的
官方 SDK：``from app.vendor.ollama import AsyncClient``。
"""

from __future__ import annotations

from . import (
    auth,
    envconfig,
    format,
    gotemplate,
    harmony,
    manifest,
    modelfile,
    names,
    params,
    progress,
    template,
    thinking,
    toolparse,
    version,
)
from .envconfig import Models as ollama_models_dir
from .envconfig import Var as env
from .format import HumanBytes, HumanBytes2, HumanNumber, HumanTime
from .gotemplate import GoTemplateError
from .harmony import HarmonyMessageHandler, HarmonyParser
from .manifest import (
    Layer,
    Manifest,
    all_manifests,
    blobs_path,
    new_layer,
    new_layer_from_file,
    read_manifest,
    write_manifest,
)
from .modelfile import Modelfile, ParserError, parse, parse_file
from .names import Name, parse_name
from .params import ParamError, format_params
from .progress import Bar, ProgressTracker
from .template import Template, Values
from .thinking import ThinkingParser, ThinkingState
from .toolparse import ToolCallParser, parse_tag_from_template
from .version import Version

__all__ = [
    "Bar",
    "GoTemplateError",
    "HarmonyMessageHandler",
    "HarmonyParser",
    "HumanBytes",
    "HumanBytes2",
    "HumanNumber",
    "HumanTime",
    "Layer",
    "Manifest",
    "Modelfile",
    "Name",
    "ParamError",
    "ParserError",
    "ProgressTracker",
    "Template",
    "ThinkingParser",
    "ThinkingState",
    "ToolCallParser",
    "Values",
    "Version",
    "all_manifests",
    "auth",
    "blobs_path",
    "env",
    "envconfig",
    "format",
    "format_params",
    "gotemplate",
    "harmony",
    "manifest",
    "modelfile",
    "names",
    "new_layer",
    "new_layer_from_file",
    "ollama_models_dir",
    "params",
    "parse",
    "parse_file",
    "parse_name",
    "parse_tag_from_template",
    "progress",
    "read_manifest",
    "template",
    "thinking",
    "toolparse",
    "version",
    "write_manifest",
]
