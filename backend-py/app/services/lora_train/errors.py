"""LoRA 训练侧的异常类型 —— 全包**只抛这几种** ✓，routers 按类型映射 HTTP 状态 ✓。

为什么单独一层 ✗：训练链路的失败原因是**分类**的 ✓（用户填错 / 环境没装 / 正在跑 / 被取消 ✓），
而 ``bad_request`` 与 ``server_error`` 的**分界线**要一眼可读 ✓。

⚠️ 关键约定：**没有"静默兜底"这一种** ✗ —— 见本仓 ``services/engine/__init__.py`` 的硬约束：
不认识的档位/不认识的任务一律**报错并列出合法值** ✗✗，不许回落到默认 ✓。
"""
from __future__ import annotations


class LoraTrainError(Exception):
    """本包所有异常的基类 ✓。routers 里只要 ``except LoraTrainError`` 就能兜住全部 ✓。"""

    #: routers 映射用的 HTTP 状态码 ✓
    status_code = 500


class LoraTrainConfigError(LoraTrainError):
    """**用户/调用方传参不对** ✓ —— 缺必填、取值不在域内、路径不是绝对路径…… ⇒ HTTP 400 ✓。

    与 :class:`LoraTrainEnvironmentError` 的区别是**责任方**：这个错改请求就能过 ✓；
    那个错得先装工具/下权重 ✓。
    """

    status_code = 400


class LoraTrainEnvironmentError(LoraTrainError):
    """**机器上缺东西** ✓ —— 找不到 musubi-tuner、缺脚本、python 解释器不可用 ⇒ HTTP 409 ✓。

    为什么是 409 而不是 500 ✗：这不是「后端崩了」✓，而是「当前环境跑不了这个任务」✓；
    409 让前端能把文案落在「去环境检查页修」而不是「重试」✓。
    """

    status_code = 409


class LoraTrainBusyError(LoraTrainError):
    """**已有任务在跑** ✓ —— 本仓是**单任务**模型（一张 A5000 ✓，同机跑两份训练只会互相抢显存 ✗）⇒ HTTP 409 ✓。

    ⚠️ 单任务是**实测结论**不是偷懒 ✓：参考实现的 GUI 也是全局单进程变量 ✓（``train_process`` ✗ 一份 ✓），
    它同样没有并发语义 ✓。
    """

    status_code = 409


class LoraTrainCancelled(LoraTrainError):
    """任务**被取消**（用户点了停止 ✓）—— 取消是**正常路径** ✓，不是错误 ✗。

    运行器内部用它把「等待中的那一步」直接打断 ✓；routers 一般不把它当错误回 ✓。
    """

    status_code = 409


class LoraTrainNotFoundError(LoraTrainError):
    """**任务号不认识** ✓（查一个不存在/已被清掉的 task ✓）⇒ HTTP 404 ✓。

    ⚠️ 为什么不复用一个 ``None`` ✗：``None`` 到了路由里会变成「查不到就是没在跑」✓✗，
    而这两件事完全不同 ✓ —— 「跑完了」和「这个号我就没发过」必须分得开 ✓。
    """

    status_code = 404


class LoraTrainRunError(LoraTrainError):
    """**命令根本没起来** ✓ —— 解释器不在、脚本不存在、cwd 不对 ⇒ HTTP 500 ✓。

    ⚠️ 与「返回码非 0」的区别 ✓✗：返回码非 0 是**训练自己失败** ✓（那是任务状态 ``failed`` ✓，
    不是异常 ✗）；这个错是**连启动都没成功** ✓（根本没产生训练过程 ✓）。
    """

    status_code = 500
