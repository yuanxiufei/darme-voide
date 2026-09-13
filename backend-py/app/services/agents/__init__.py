"""Agent 运行时（S5）—— 对应 Node 侧 ``backend/src/agents/``。

⚠️ **S5 不是"搬代码"而是"造运行时"**：Node 用 **Mastra**（``@mastra/core/agent`` +
``@ai-sdk/openai``）建 agent，并在每次请求里动态把 ``episodeId`` / ``dramaId`` **注入工具闭包**；
Python 侧没有对应框架 ⇒ 这层要手写：

* ``protocol``    —— Agent 输出协议（YAML 收尾块）的契约文本与解析；
* ``tool``        —— 工具基座（id / description / **JSON Schema** / execute）；
* ``tools/*``     —— 六组工具（script / extract / storyboard / voice / grid-prompt / corpus），
  它们只是**薄封装**，底层全部调用已迁服务；
* ``runtime``     —— agent 循环（system prompt + 工具注册表 + 步数上限 + 流式事件）。

工具入参用 **JSON Schema**（不是 zod）：LLM 的 function-calling 契约本来就吃 JSON Schema，
写死在这一层可以同时喂给 OpenAI 兼容与 Gemini 两条 text 适配器。
"""
