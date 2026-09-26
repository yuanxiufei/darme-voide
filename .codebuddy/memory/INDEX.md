# 记忆索引 —— 会话开始读这里，不要整读日志

> **三层读法**：`MEMORY.md`（不变量与约定，**必读**）→ `TOPICS.md`（低频长专题，按需）→ 本索引（日志定位）。
> 日志是**过程存档**，其结论/红线**已提炼**进 `MEMORY.md` / `TOPICS.md` / `docs/` / `backend-py/app/skills/README.md`，**通常不需要读原文**。
> 确需过程细节/证据/命令时：按下方 `@行号` 用 `read_file(offset, limit)` **只读那一节**。
> ⚠️ **下方行号是 2026-09-24 那篇的快照**（当日日志只追加 ⇒ 锚点不会失效）；更早的日志已清理，见本节末尾说明。

## 日志清单（按需跳读）

**`2026-09-24.md`**（消费方核查 + 网格规则逐值钉住）
① 把昨天「两条掩码与 `segments` 不在同一坐标空间」的警告**往消费方查实** ✓：`app/` 里只有
`H3FormTrunk.forward` 用 ✓，且只取 `numel()` 当行数 ✓ ⇒ **只是文档问题不是隐藏 bug** ✓
（⭐ 教训：发现"两个坐标空间"要先找全消费方再下结论 ✓）；② ⭐⭐ 三条「混了不报错」的**网格规则**
从摘要式断言升级为**逐值** ✓（`engine_h3_form_test` 74 → **76 项**）@㉓′~㉓″：`cond` 用**目标**网格 ✓
而 `ref_img` 用**它自己的** ✓（两者**必须不相等** ✓）；两段 `ref_audio` **同名不同源** ✓（`audio` 类 ⇒
**目标**两端 ✓、`video` 类 ⇒ **它自己**两端 ✓）+ 钉住 video 类参考块「**音频行在前、视频行在后**」✓；
③ ⭐⭐ 混合形态下的**模态标签**逐段钉住 ✓（`engine_h3_form_test` 76 → **77 项**）@㉔ ——
`cond`/`cond_audio`/`ref_img`/`ref_audio` 这**四档只在混合形态出现** ✓✗，此前**只有文档没有断言** ✗
（标签只是 adaLN 的**行内偏移** ⇒ 错了**完全不报错** ✓，只是模态接到别的模态那组参数 ✓✗）；
④ ⭐⭐ 对着**事实表**逐条核覆盖 ✓：`dit.H3_PACK_FACTS` 第 3/4 条此前只有摘要式覆盖 ✗ ⇒ 补逐值判据 ✓
（`engine_h3_form_test` 77 → **79 项**）—— 跨度表 **(1,4,4,4,4) 循环** × 5/3 ✓（给错照样单调 ✓✗，
只有回绕那格会露 ✓）+ 立体声 **channel-major 的落点是 `w` 两端** ✓（两种声道顺序的行数与 t **完全一样** ✓✗，
只有 w 分得开 ✓）；⑤ ⭐⭐ 修掉**产品侧终态竞态** ✓（这才是那两次「假红」的真因 ✗✗）：`h3` 服务此前**先写 `succeeded`** ✓
再 `finally` 里 `/free` ✗ ⇒ 中间态「成功了但 `freed_vram=null`」✓✗（API 自相矛盾 ✓；真机同样中招 ✗）
⇒ 改成**先卸载、再落终态** ✓（两条执行体 ✓）+ 补 ⑭′「失败路径也 `/free`」✓（`h3_stage2_test` 20 → **21 项**）@⑥；
⚠️ **更正昨天的判断** ✗✗：昨天判成「并发资源竞争、不是代码」 ✓✗ —— 真因是竞态 ✓，并发只是让它更容易露 ✓
⇒ ⭐ **「单独复跑就好了」≠「不是代码问题」** ✗（概率低而已 ✓）：有红要**单独连跑 3 次** ✓，还绿再查时序 ✓；
⑥ 我自己的错 ✗：期望段表漏掉目标两条流 ✓✗、把 video 类参考块顺序写反 ✓✗、**第三次**在双引号串里写半角引号 ⇒
语法错 ✓；⑦ **干净全量：104 套 / 3629 项 / 0 失败** ✓（只跑一次 ✓；⚠️ `h3_stage2_test` 自带轮询 ⇒
会**慢一阵** ✓ 不是卡死 ✓，判据是「有没有 `结论：` 行」✓）；⑦ ⭐⭐ **记忆层第四次腾预算** ✓（起因：顺手
发现 `MEMORY.md` **13906 字符** ✗，而它的守卫 `app/scripts/check_memory.py` **不在 `run_all.py` 里** ✗
⇒ 一直在悄悄烂 ✓✗；跑守卫 ⇒ **7 处致命** ✓：预算超限 ✓ + **5 篇日志的末节锚点早漂** ✗（⚠️ 末节/步骤小节
锚点**会随日志增长漂移** ✗，写 `@末节` **不算登记** ✓✗）+ 一处步骤小节未登记 ✓）⇒ 细节**整体下移**
`TOPICS.md` §自 MEMORY.md 下移（2026-09-24）五节 ✓（§A 分词器/§B 跨来源校验/§C 前端类型检查/§D 上机自检/
§E 记忆守卫自身/§F 环境与自检纪律 ✓）⇒ **13906 → 7907** ✓、守卫 **exit 0 ✓ 致命 0 ✓**；⑧ 逆向
**SLM 无限漫剧创造台** ✓（`D:\app\SLM` 只是**壳** ✓，本体是 `H3EasyDirector` 工作台 ✓ **在同一模型家族
MiniMax H3** ✓ 上 ✓ —— ⭐ 最值三件：**三档引擎表** ✓（官方 25 步/标准 8 步/极速 4 步 + 每档必备模型与
显存建议 ✓）/ **已知冲突台账** ✓（DaSila ⇒ WinError 433 ✓ 等 7 条 ✓）/ **契约同步+离线兜底** ✓
（权威在工作台 `/api/workflow_tiers` ✓）⇒ 细节见 `TOPICS.md` §SLM 逆向笔记 ✓；⚠️ 它的提示词/工作流在
74.7 MB 的 `v3.5.40` 包里 ✗ 未拉 ✓）@108

⭐ **⑨~⑳ 两轮「接线」**（用户「没有实现的继续实现」✓）：**判据层立了却只被自检调用** ✗✗ ⇒ 一轮补
**提交前那几道关**（配音契约 / H3 形态决策 / prompt 契约 + 加速链 / 分段与超清计划端点 ✓ ③④⑤⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱），
二轮补**加载层与张量层**（混合加载**加载层** ✓ / 超清**二采阶段** ✓ / 段级音频合成接进单镜合成 ✓ ⑲⑳）
⇒ 落点表见 `TOPICS.md` §接线进度 ✓、判据见 `MEMORY.md` §接线 ✓；⚠️ 早先「仍未接 `cache_key`」已过时 ✗ —— 二轮已接上 ✓
（`find_reusable_video` ✓ 先补「标量按值进指纹」✓✗）。⭐ **㉑ 第三轮收尾**（2026-09-25 ✓，用户「请继续」✓）：「能力已有、没接出去」最后两处 ✓
① 多集节奏相位注入 ✓（`runtime.py` 早先「`rhythm-phase.ts` 未迁」warn 占位 ✗，`rhythm_phase.py` 早已迁好 ⇒ `storyboard_breaker`
少一段跨集节奏引导 ✓✗ ⇒ 真调 ✓）；② `refineNote` **自述与实现相反** ✓✗（`denoise:False` ✗ 而 `_second_pass` 已实现 ✓ ⇒ 对齐 ✓）。
全量 **125 套 / 4099 项 / 0 失败** ✓（+3 项 ✓；顺带校正 `engine_refine_test` 记 14 条、实测 23 条 ✓）。
⭐ **㉒ 逆向收口 + 自述对齐**（2026-09-25 ✓，用户「有用的就逆向，没用的就不做」+「继续完善」✓）：① 逆向待做项判定收口 ✓ ——
TTS 三件套**有用部分早已落地**（`voice_contract.EMOTION_ORDER` 8 维 = 照抄 IndexTTS ✓，另两条情绪入口 emo_audio/文本情绪是
IndexTTS 专属、本仓 CosyVoice 用不上 ✗）；工作台 HTTP 面**核心已落地**（三档表→`tiers.py` ✓）后端 PyInstaller 编译读不到 ✗ ⇒ 不做 ✓
（详见 `TOPICS.md` §逆向待做项收口 ✓）；顺手纠正「IndexTTS 情绪 9 维」✗ → 实为 **8 维** ✓。② 清掉 3 处**自述与实现相反**的 stale ✓
（`refineNote.denoise` ✓ / `torch_backend` 模块头「架构装载还没写」✗ / `engine/__init__`「本机无 torch」✗ —— 实已装 CPU 版 torch ✓）。
⚠️ 功能层面**已无「不需要真权重」的可做** ✓，余下全卡真权重/真机 ✓。
⭐ **㉓ 模型目录自主化**（2026-09-25 ✓，用户「完全自主不依赖第三方」+「也要能检测电脑内模型」✓）：① **下载/存储自主** ✓ ——
models_dir 默认从 ComfyUI 目录改为 **`<data_root>/models`** ✓（`local_model_scan.default_models_dir()` ✓ + `get_model_paths` 与
`model_manager.resolve_paths` 默认回落 ✓；`configs/model-paths.json` 清空 ComfyUI 硬编码 ✓）；② ⚠️ **「检测/扫描」不排斥第三方** ✗ ——
`get_default_roots` 仍覆盖电脑内模型（本仓目录 + 本地服务 + ComfyUI 目录探测 + extra_roots ✓）：自主的是「从哪下载、存到哪」，
**不是**「不许看见别人已装的模型」✓（先误删了 ComfyUI 扫描、用户一句话纠正 ✓）；③ 全量 **125 套 / 4099 项 / 0 失败** ✓（项数不变 ✓
只改 1 条「清空 models_dir 回落到默认」的断言 ✓）。
⭐ **㉔ 自研 LLM 运行时起步**（2026-09-25 ✓，用户「要自研实现」✓）：文本生成此前走 ollama（HTTP 11434）✗ ⇒ 新建
`engine/llm.py` ✓ —— 自研 decoder-only transformer（RMSNorm / GQA 分组查询 / RoPE / SwiGLU / 因果掩码 / KV cache /
采样 greedy+top-p+top-k ✓）；架构参数**全显式** ✓（`LlmConfig` 不写死 Qwen3 ✗ 本仓纪律 ✓）；自检 `engine_llm_test` **13 条** ✓
（⭐⭐ KV cache 不变量：整段 vs 逐 token 逐位相同 ✓ / 因果掩码「改后不动前」✓ / RoPE 保范数 ✓ / GQA ✓ / tie embeddings ✓）。
⚠️ **尚未接线** ✗：GGUF 权重加载 + k-quant 反量化**未做** ✗（quant.py 现只 cover fp8/int8 ✗）⇒ 下一轮 ✓；
全量 **126 套 / 4112 项 / 0 失败** ✓（+1 套 / +13 项 ✓）。
⭐ **㉕ GGUF 反量化**（2026-09-25 ✓，用户「继续」✓）：新建 `engine/gguf_dequant.py` ✓ —— 把 GGUF 数据区的量化字节 ⇒
fp32 张量 ✓（F32/F16/BF16/Q8_0/Q4_K ✓；公式照 llama.cpp `ggml-quants.c` ✓ **MIT** ✓ web_fetch 取权威 ✓）；⭐⭐ 判据 = 手造
Q4_K block（d=1/dmin=0.5/scales 全 0xFF/qs 全 0x11 ⇒ 反量化**每元素精确 = 31.5** ✓✗ —— 字节布局或 6-bit 解包错一处就对不上 ✓）；
⚠️ 其余 k-quant（Q2_K/Q3_K/Q5_K/Q6_K）**具名拒绝** ✗；`load_weights` 能整文件提权重 ✓ 但大文件（14B≈17GiB fp32）要**逐张量**读 ✓。
⚠️ **下一步** ✗：GGUF 张量名 → `llm.LlmModel` 参数的**映射**（`model.layers.N.self_attn.q_proj.weight` → `blocks.N.attn.q_proj.weight` ✓）未做 ✗；
全量 **127 套 / 4121 项 / 0 失败** ✓（+1 套 / +9 项 ✓）。
⭐ **㉖ GGUF → LlmModel 装载**（2026-09-25 ✓，用户「继续」✓）：新建 `engine/gguf_to_llm.py` ✓ —— ggml 命名
（`blk.N.attn_q.weight` ✓）→ LlmModel 命名（`blocks.N.attn.q_proj.weight` ✓）+ 6 类线性层**转置** ✓（GGUF (in,out)→PyTorch (out,in) ✓）
+ tie embeddings 时 `output.weight` 跳过 ✓；⭐⭐ 判据 = **往返恒等**（LlmModel state_dict → 反向转 GGUF 命名 → 正向映射回 ⇒ 逐张量相等 ✓✗）；
`load_llm_from_gguf` 逐张量读 + `load_state_dict(strict=True)` ✓（漏/形状错当场报 ✓）。⚠️ **下一步** ✗：自研文本后端
（describe+generate）替换 `OllamaTextAdapter` ✗（这才是「不调 ollama」的最后一跳 ✓）；全量 **128 套 / 4128 项 / 0 失败** ✓（+1 套 / +7 项 ✓）。
⭐ **㉗ 自研文本后端**（2026-09-25 ✓，用户「继续」✓）：新建 `engine/llm_backend.py` ✓ —— 把「架构+反量化+装载」收成**一个后端**
（像 `TorchBackend` ✓）：`describe` 如实报缺（权重/词表/模型缺啥报啥 ✓ 不冒充 ✗）/ `load` 幂等装载 / `generate` 走 encode→生成→decode ✓
（⚠️ **同步** ✗ CPU 推理会阻塞 ✗ 接线要 `asyncio.to_thread` ✓；⚠️ chat 模板未做 ✗ 裸拼接 ✓）。⭐⭐ 判据 = 未装载就 generate ⇒ 明确拒 ✗
（不静默返回空串 ✗）。⚠️ **下一步** ✗：`text_generation` 的 engine 分支 + `infer_llm_config`（从 GGUF 元数据读架构参数 ✓）⇒ 这才是「接线」闭环 ✗；
全量 **129 套 / 4134 项 / 0 失败** ✓（+1 套 / +6 项 ✓）。
⭐ **㉘ 接线闭环**（2026-09-25 ✓，用户「继续」✓）：① `gguf_to_llm.infer_llm_config` ✓ —— 从 GGUF 元数据读架构参数
（`{arch}.embedding_length/block_count/attention.head_count[_kv]/feed_forward_length/rope.freq_base/...` ✓，Qwen3 GGUF 沿用 `llama.` 前缀 ⇒ 回退读 ✓）；
缺必需字段 ⇒ **具名拒绝** ✗（不猜默认 ✗ —— 猜了装出「名字对、形状全错」的模型 ✗）；head_dim 缺省 = hidden/heads 推导 ✓。② **接线** ✓ ——
`text_generation.generate_text` 加 engine 分支 ✓：provider=engine ⇒ 走自研后端（`asyncio.to_thread` 包同步推理 ✓）而非 HTTP 调 ollama ✗；
后端缓存（大权重只 load 一次 ✓）。⭐⭐ **文本不依赖 ollama 已闭环** ✓（剩「下载真权重 + chat 模板」这两件数据/细节 ✓）；
全量 **129 套 / 4141 项 / 0 失败** ✓（+7 项 ✓：infer 5 + 接线 2 ✓）。
⭐ **㉙ 对话骨架（chat 模板）**（2026-09-25 ✓，用户「继续实现功能」✓）：㉘ 自述「剩『下载真权重 + chat 模板』」⇒ 补 chat 模板 ✓ ——
新建 `engine/chat_template.py` ✓（**骨架从权重元数据取** ✓ 不内置 ✗✗；`StrictUndefined` ✓✗ 缺变量**报错**而非静默空串 ✓；
注入 `raise_exception` ✓）；接线 `llm_backend` ✓（load 读骨架 / generate 走骨架 / describe 如实报 ✓，**无骨架时老行为一字未动** ✓）；
⭐⭐ **「没有模板」与「模板坏了」是两回事** ✗✗（坏了**当场拒** ✗ 不静默回落裸拼接 ✗）；自检 `engine_chat_template_test` **18 项** ✓
（+1 套 / +18 项 ✓；既有 `engine_llm_backend_test` 8/8 ✓ 未破）；⚠️ 如实报「未核」✗：用骨架时 `add_special_tokens`
仍 `True` ✓（待真词表定 ✓）。⭐ 同轮**核查用户点的方向** ✓ ⇒ 配音契约**早已闭环** ✓（那条「下一步」过时 ✗）、
音色克隆**样本面无「静默错误」级真缺口** ✓（详见日志 ⑨ ✓）；⚠️ 清理 `.codebuddy` 踩坑 ✗✗：连带删了 `memory/` 10 篇日志
⇒ `git restore` 完整恢复 ✓ 守卫复绿 ✓ 致命 0 ✓（教训：删除逐条 `-LiteralPath` ✓）。@486
⭐ **㉚ 裸跑编码收口（GBK 控制台）**（2026-09-25 ✓，用户「全收口」✓）：起因 = 照 `PENDING_PARTS` 的「上机前先跑一次」跑
`h3_readiness.py`（本机真权重 ✓）⇒ **裸跑第一条 `print` 就崩** ✗（满屏 ✓/✗，而 py3.14 的 stdout 仍是 **locale/GBK** ✗ —— PEP 686 要 3.15 ✓）；
⭐⭐ **四层掩盖**：① 脚本崩 ⇒ ② `engine_readiness_script_test` ⑩ FAIL（stdout 非 JSON）⇒ ③ **该测试自己也崩在打印失败原因上** ✗✗
（`⇒`=U+21D2 GBK 编不了 ⇒ 只剩 `EXIT=1`、原因一个字看不到 ✓✗）⇒ ④ **全量 133 套全绿** ✗（`run_all.py` 给子进程灌了 `PYTHONIOENCODING=utf-8` ✓）
⇒ **文档推荐的「单跑」用法在本机是坏的** ✗✗。口径（三缺一不报 ✓）：**含 locale 外字符 + 真有 `print` + 入口无 `reconfigure`** ⇒ 判会崩 ✗；
收口 ✓：新守卫 `check_cli_encoding.py` ✓（扫 152 文件；探针命中 **65** ✗ ⇒ 63 个 AST 批量补 ✓ + 2 个模块级脚本手改 ✓：`smoke_test.py` ✓ /
`corpus/analyze3.py` ✓ 老写法 `sys.stdout = io.TextIOWrapper(...)` **丢原 wrapper** ✗ ⇒ 统一 `reconfigure` ✓）+ 接进 `check_all.py` 第 3 道 ✓ +
`test_guards.py` 负向 3 例 ✓（17/17 ✓）；**硬证据**：四道自检全绿 ✓ + 抽样**裸跑** 4 套 `exit=0` ✓（20/20 ✓、18/18 ✓、29/29 ✓、45/45 ✓）。@517

> ⚠️ **历史日志已于 2026-09-26 清理 —— 现只保留最新一天**。此前 10 篇（09-13 / 09-14 / 09-15 / 09-16 / 09-17 / 09-18 / 09-20 / 09-21 / 09-22 / 09-23）已从工作区删除，**原文仍在 git 历史里**：`git show HEAD:.codebuddy/memory/2026-09-20.md` 或 `git checkout HEAD -- .codebuddy/memory/2026-09-20.md` 取回（把日期换成要的那篇）。本节下面的「已出栈的落点」与 `MEMORY.md` / `TOPICS.md` 不受影响。

> 动**视频首尾帧 / 尾帧 / 角色合并**时**先读本文件**。





## 已出栈的落点（优先看这些，别翻日志）

| 内容 | 权威落点 |
|---|---|
| 全部 API 契约 | `docs/api-contract.md` |
| H3 本地视频链路 | `docs/local-h3-video-system.md` |
| 本地模型评估/下载 | `docs/local-model-evaluation.md`、`TOPICS.md` |
| 语料源与合规 | `docs/video-prompt-data-sources.md`、`docs/seedance2-corpus-analysis.md` |
| preset skill 模板 | `docs/preset-skill-template.md` |
| **Skill 体系（改前必读）** | `backend-py/app/skills/README.md` + `MEMORY.md §Skill` + `TOPICS.md §Skill 体系细节` |
| **Python 后端（绞杀者迁移）** | `backend-py/README.md`（运行/环境变量/迁移 SOP）+ `TOPICS.md §backend-py` + `backend-py/tests/smoke_test.py` |
| 引用完整性守卫（skills 路径） | `backend-py/app/scripts/check_skill_refs.py`（基线见脚本头） |
| 记忆层自检（8k 预算 + 锚点/登记/落点路径）+ 守卫自检 | `backend-py/app/scripts/check_memory.py`、`backend-py/app/scripts/test_guards.py`（规则见脚本头） |
| CLI 裸跑编码（GBK 控制台别崩 ✗；口径 + 四层掩盖链） | `backend-py/app/scripts/check_cli_encoding.py`（规则见脚本头）+ `TOPICS.md §CLI 裸跑编码` |

## 日志写入规范（防再膨胀 —— `09-12` 单日已 780 行）

- 每轮 **≤ 8 行**：只写 **结论 / 落点(文件:行) / 红线 / 未决**。
- **不写**过程叙述与命令回放；需长期留证的内容 → `docs/` 或 `TOPICS.md`。
- 结论属「日后会导致 bug 的不变量」→ 同步进 `MEMORY.md`；属低频长专题 → `TOPICS.md`。
- 写时假定「将来只会按 `@行号` 跳读」，**不要依赖前后文**。
