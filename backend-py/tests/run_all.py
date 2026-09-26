"""跑完全部后端自检（每迁完一块请跑这个）。

**本文件的 ``TESTS`` 就是套件权威清单**（README 里那棵树只是摘录，别去数它）。

规模：**以本文件下面的 `TESTS` 为唯一权威** ✓（⚠️ 不再在这里写死总数 ✗ —— 逐项罗列会随
增删而腐烂 ✓，本文件自己就被它咬过：下面那行曾是 **82 套 / 2911 项** ✗，早过期好几轮 ✓）。
最近一次**全量实测**（2026-09-25 深夜 ✓ **在 A5000 目标机**上跑 ✓、耗时 **201s** ✓、
**连跑两次结论一致** ✓（208s / 201s 两次都是 `133 套 / 4265 项 / 红 0` ✓ ⇒ 那套不是"时有时无" ✓✗）；
**与上一跑同一台机器** ✓
⇒ 数字**可直接比** ✓）：登记 **133 套** ✓ ⇒ **OK 133 / FAIL 0** ✓；
其中 **132 套按 `SUMMARY:` 收敛出 4265 项 / 4265 过（红 0 ✓）** ✓、
1 套是常量守卫型（`OK: 镜像常量漂移 0 条` ✓ 无项数 ✓）、**崩的 0 套** ✓。
⚠️ **账要这么对**：`132(有摘要) + 1(常量守卫) = 133(登记)` ✓ —— **少一套就是"凭空消失"** ✗✗
（本轮之前正是 129+1+2 崩 ✓ ⇒ 补完 ffmpeg 那两套才重新"出现"在全量表里 ✓；
⚠️⚠️ **这条刚刚又应验一次** ✓✗：上一跑（即下面"改前"那个 `132/4236` ✓）里
`engine_chat_template_test`（对话骨架那套 ✓）**根本没出现在日志里** ✗ —— 而它现在在 `TESTS` 里 ✓、
单独跑也 **18/18** ✓ ⇒ **已查清** ✓（**不是**收尾漏计 ✗）：**它是上一跑之后才登记进来的** ✓✗ ——
`git log -S` 指到提交 `83743d6`（**17:33:24** ✓），而上一跑日志的落盘时间是 **17:28:24** ✓
⇒ **那一跑比登记早 5 分钟** ✓，当时清单里还没有它 ✓（账本本身变了 ✓，两跑之间还夹着一次提交 ✓）。
⇒ 教训（比原来那句更值钱 ✓）：**账会因中途提交而变** ✗ ⇒ 每轮都要**重新对账**
（`有摘要 + 常量守卫 = 登记` ✓）、**别信上轮的账** ✗；也别拿**跨提交**的两跑直接比项数 ✗——
要说明"这一跑之后多了什么" ✓。
⚠️ **更早那一跑**（同 2026-09-25 ✓ 但**是那台没显卡的机器**）：129 套 / 4141 项 / 0 失败 ✓；
跨机还差着 `model_ecosystems_test` ✓ + `comfyui_runs_test` ✓ + `ollama_store_test` ✓（以 `TESTS` 为准 ✓）。
项数差里**混着"换了机器 + 补了工具"**这两个变量 ✗ ⇒ 只能看"**同一台机器上**改前改后"的对比 ✓✗，跨机比大小是自欺 ✗。
⭐ **同机改前改后**（这才是能比的 ✓；**同一个会话里连着三跑** ✓）：
`131/4222(红 3，崩 0)` → `132/4236(红 0，崩 0)` → **`133/4265(红 0，崩 0)`** ✓
⇒ 上一段 `+1 套 / +14 项`（全出在 `ollama_store_test`：28 → 42 ✓；那 3 条**环境**红因活体那套把
`qwen3:14b` 拉下来而转绿 ✓ ⇒ 9/9 ✓）；
⇒ 本段 `+1 套 / +29 项` ＝ `engine_chat_template_test` **+18**（对话骨架 ✓；⚠️ **上一跑之后才登记** ✓
见上 ✓ —— 不是漏计 ✗，也**不是本轮的活** ✗）
＋ `engine_io_test` **+11**（48 → 59 ✓ —— 本轮新增的**设备口径逐入口**判据 ✓ 见下 ✓）✓。

⭐⭐⭐ **2026-09-25 深夜「A5000 目标机第一次全量」实测抓到一条产品缺陷** ✗✗（**记在这里，因为它是"判据"最该长成样的例子** ✓）：
现象：`engine_dual_stream_test` **47/54** ✓✗（红 7 条全在 `pipe.run_sync` 的 `stage: 'sample'` ✓），
`engine_io_test` / `engine_refine_test` **直接崩、无 `SUMMARY:`** ✓✗ ⇒ 报的话是
`RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!` ✓
⇒ **不是测试的锅** ✗：真因在产品 ✓ —— `h3_form.rope_angles` 把 `packed_layout` 用**纯常量**拼的
坐标表（**天生在 CPU** ✓）与**模型缓冲区** `inv_freq`（跟着 `.to("cuda")` ✓）相乘 ✗ ⇒ **CUDA 上必炸** ✓；
同族第二处：`forward` 把 `t_vals_for` 拼的时间戳表（同样在 CPU ✓）直接喂给 `time_embedder`（权重在 CUDA ✓）✗。
⚠️⚠️ **在没显卡的机器上一切正常** ✓✗ ⇒ 这套判据**此前永远抓不到它** ✗（docstring 里那句「到 A5000 换 CUDA 轮子
即可 ✓（`device` 自动探测 ✓）」当时是**没验过的话** ✓）⇒ 两处都改成**跟随模型/计算设备** ✓（`inv_freq.device` /
`h.device` ✓，都写了"为什么"的注释 ✓），`engine_dual_stream_test` 当场 **54/54** ✓（真 CUDA ✓）。
⭐ 另两套（`engine_io` **当时 48/48** ✓ ⇒ 后续补到 **59/59** ✓ 见下 ✓、`engine_refine` 23/23 ✓）红的是**测试自己造了 CPU 张量** ✓✗
（`dual_latents()` / `torch.randn(...)` 没给 device ✓）而 `TorchBackend()` 默认**自动探测** ⇒ 本机选 CUDA ✓ ⇒
混用当场炸 ✓，报的还是 `aten::slow_conv3d_forward` 这种**指不到原因**的文案 ✗ ⇒ 这两套**显式钉 `device="cpu"`** ✓
（验的是张量层数学 ✓、与设备无关 ✓；**真 CUDA 的整条管线**由 `engine_dual_stream_test` 覆盖 ✓）。
⚠️ 产品侧**没动**那两处语义 ✗：调用方给 CPU 张量仍是错 ✓（真输入只有 `static/…`、`frames/…`、`C:\\…` 三类 ✓，
管线内部张量都在 `self._device` 上 ✓）—— 若要**替调用方搬张量**，得先想清 `condition_first_frame` 的**原地改写**语义 ✗
（拷一份搬走 ⇒ 调用方的张量就不再被改了 ✓✗）⇒ 当时**留作待议** ✓，不静默改 ✗。
⭐⭐ **同日晚已收口** ✓（判据：`engine_io_test::case_device_guard` ✓ 48 → 59 项 ✓）：结论是**「不搬」** ✗✗
（搬就破坏"就地改写"语义 ✓✗，且白占一份显存 ✓，而"该在哪个设备"只有调用方知道 ✓）⇒ 改成**五个入口逐个**
当场**明确报错** ✓（说清**哪个参数** / **哪个设备** / **该怎么办** ✓，替掉 `aten::slow_conv3d_forward`
那种指不到原因的文案 ✗）；`write` 那条**查过后判定不校验** ✓（落盘边界自己 `.to("cpu")` ✓）；
逐入口清点表钉在 `TorchBackend._require_own_device` 的 docstring 里 ✓（连**不校验的**入口也写明结论 ✓）。
⚠️⚠️ 收口过程本身又踩一个坑 ✓（已钉成判据 ✓）：**判据不许比字符串** ✗✗ —— 后端写 `"cuda"` ✓ 而张量的
`.device` 永远是 `cuda:0` ✓ ⇒ 初版 `str()` 一比就把 `engine_pipeline_test` 的**真链路**误判成混用 ✗
（好在 `cuda:0` vs `cuda` 并排摆在报错里 ✓ 一眼看出是判据错而非链路错 ✓）⇒ 改比**规范化后的具名设备** ✓；
⚠️ 规范化**不是放水** ✗：多卡上 `cuda:1` ≠ `cuda:0` 照旧报错 ✓。
⚠️ 同轮还发现 5 条**判据自己过期**（不是产品坏 ✓，都改了 ✓、都留了"为什么"✓）：
① `smoke_test` 断言 `ollama status` 文案含「无法连接」✗ —— 而那**正是** 2026-09-25 要修掉的写法 ✗
（服务没起时改说「服务未运行 + 盘上读到 N 个」✓，`source="disk"` ✓）⇒ 改成钉**两件事都说清** ✓（477/477 ✓）；
② `http_logger_test` 用 `client.post(json=…)` ✗ —— httpx 的 `json.dumps` 默认 `ensure_ascii=True` ✓
⇒ 线上字节是 `\\u65b0\\u5267` ✓ ⇒ 中间件**如实回显**就**不含「新剧」** ✗（这条问的和想验的不是一回事 ✗）
⇒ 改成**自己发 UTF-8 字节** ✓（顺带钉住"按 UTF-8 解、不是 locale 乱码" ✓；13/13 ✓）；
③④ `video_generation_test` / `sse_hub_frames_test` 拿 `/abs/x.mp4` 当"绝对路径" ✗ —— **无盘符根路径在
Windows 上 `os.path.isabs` 是 False** ✓（Python ≥3.13 收紧的 `ntpath` ✓；**3.12 那会儿是 True** ⇒ 这条
**只在 3.13+/Windows 上才翻** ✓✗）⇒ 判据改成**本平台原生绝对路径** ✓（"仍是绝对 ✓ + 没被拼到 storage 下 ✓"，
平台无关 ✓；59/59 ✓、24/24 ✓）；
⑤ `cosyvoice_seam_test` 在**包装源码没就位**时**整套崩**（`spec_from_file_location(None)` ✗）——
而那份源码是 `.gitignore` 掉的**本地部署产物** ✓（干净检出里必不在 ✓）⇒ 改成**显式 SKIP + 打摘要** ✓
（`0/0` ✓，比"崩"诚实 ✓：崩了连"这次没验"都看不见 ✗）。⚠️⚠️ **教训**：`skip`/红/崩三者的区别就是记账的全部 ✓ ——
**崩 = 没摘要 = 在全量表里凭空消失** ✗（本轮**一开始正是这样** ✓：2 套 ffmpeg 系崩着 ⇒ 账是 `129+1+2` ✓、
那 2 套的 40 项**根本没进统计** ✗；补完 ffmpeg 后变 `131+1` ✓、40 项回来了 ✓）。
✅ **「像素级校色 / 参考图压缩」那两套的红已解除** ✓（2026-09-25 深夜 ✓ **环境补齐、产品代码一行没动** ✓）：
本机 PATH 上**原本根本没有 ffmpeg/ffprobe** ✗（之前看到的 7.1 是我自己临时注入诊断目录的 ✓✗）⇒
`winget install --id Gyan.FFmpeg -e` ⇒ 装到 **9.0.2-full_build** ✓（含真 `ffprobe` ✓，`-show_entries stream=pix_fmt`
读 PNG 得 `rgb24` ✓）；别名落在 `%LOCALAPPDATA%\\Microsoft\\WinGet\\Links` ✓ ⇒ **重启 shell/IDE 后 PATH 自然生效** ✓
（本轮是显式注入 `…\\ffmpeg-9.0.2-full_build\\bin` 跑的 ✓）。结果：像素级校色 **24/24** ✓、参考图压缩 **16/16** ✓。
⚠️⚠️ **诊断过程留下的教训（值得记 ✓）**：**诊断目录里的工具也要验真伪** ✗ —— 这轮我差点把假货当既成事实 ✓✗：
`D:\\老李skill\\_diag\\bin\\ffmpeg.exe` **是真的** ✓（7.1、87MB、能生成 PNG ✓），但同目录 `ffprobe.exe`（6.1.1、仅 7MB）
**是残缺二进制** ✗ —— 连**它自己 ffmpeg 产出的** PNG/JPEG 都读不了（`Invalid data found when processing input` ✗）、
`-demuxers` 输出**为空** ✗；另一个候选 `D:\\老李skill\\_ffprobe_test\\ffprobe.exe`（83.6MB ✓ 看着最像真的 ✓）
**其实是改名的 `ffmpeg`** ✗ —— `-version` 自报 `ffmpeg version 7.1-…` ✓、`select_streams` 当场 `Unrecognized option` ✗。
⇒ **判据就一句**：`-version` **加上真读一次目标格式** ✓（只看版本号/文件大小都会被骗 ✗）。
✅ **那 1 套红已解除** ✓（2026-09-25 夜）：曾 **6/9**（3 条红**同根**：本机**没装产品内置的默认本地文本模型**
`qwen3:14b` ✗ —— `app/services/ai_configs.py` 的内置预设就是它 ✓ + `gpu_manager` 按 9GB 登记 ✓；
本机那会儿 14 个模型里**没有 14b** ✗（有 `qwen3:8b` / `qwen3:32b` / `gemma3:27b` / `deepseek-r1:32b` … ✓））。
⭐⭐ **当时先证明「接缝本身是好的」** ✓（**拿本机真有的 `qwen3:8b` 打** ✓，不改判据 ✓）：`provider=ollama` ⇒ 原生
`/api/chat` + `think:false` ⇒ **`200 '收到'`** ✓；同模型走 OpenAI 兼容 ⇒ `content=''` / `reasoning='好的，用户让我…'` ✓
⇒ 印证「三条红 = 环境缺模型 ✓，不是代码坏 ✗」✓。⇒ 两条出路**当时都没擅自做** ✓（**待用户定** ✓）；
用户选 **①**：`ollama pull qwen3:14b`（**9.3GB** ✓，本机库 14 → **15 个** ✓）⇒ 复跑 **9/9** ✓
（`PASS ollama: 本地文本默认模型 qwen3:14b 已就位` ✓、`PASS ollama: **真推理**一次 → 原生适配器拿到非空文本` ✓、
`PASS ollama: 记录成因 —— OpenAI 兼容端点下 content 为空而 reasoning 非空` ✓），skip 仍是 **5**
（local-sd / comfyui / h3-8765 / cosyvoice 未起 ✓ 按设计 SKIP ✓ **不是通过** ✓）。
⚠️⚠️ **记账须与实测对得上** ✗✗：上一轮 docstring 把 `engine_refine_test` 记成 **14 条** ✓✗，实测是 **23 条** ✓
（多出的 `case_second_pass`/`case_keyframes`/`case_gate` 是上一轮就写好的，记账时没跟上 ✓）⇒ 已校正 ✓；
早前还把 `engine_cache_guard` 写成 **12/12** ✓✗（实测 20/20 ✓）⇒ ⭐ **判据一次写齐再跑全量** ✗，跑完再报数 ✓。

⭐⭐ **2026-09-25 夜「本机模型扫描不许依赖外部服务」入账** ✓：新增 `ollama_store_test.py` ✓，
`TESTS` 登记 **129 → 130** ✓（**登记数是数出来的** ✓；本套实测 **28/28** ⇒ ⭐ 真机打脸后补到 **42/42** ✓，见下 ✓）。
✅ **全量总数已实测补齐** ✓（不再是「待实测」✗）：见本文件开头「最近一次全量实测」✓
（**133 套 / 133 绿 / 0 红** ✓；本套在其中 ✓ 42/42 ✓）。
用户口径：**「以后关于扫描电脑内的模型，只要后端服务启动就可以扫描，不要依赖外部服务」** ✓ ——
原来 `/ollama/status` 只认 `/api/tags` ✗（**唯一**的官方列模型接口 ✓，但要求 `ollama serve` 在跑 ✗）
⇒ 「服务没起 ⇒ 本机模型一个都看不见」✗✗（盘上明明有 ✓）。现补 `services/ollama_store.py` ✓：
把 Ollama 模型库当**文件系统**读 ✓（`OLLAMA_MODELS` > **桌面端声明的库根** > `%LOCALAPPDATA%\\Ollama\\models`
> `~/.ollama/models` ✓；
清单 `manifests/<registry>/[<ns>/]<model>/<tag>` ✓ + 内容寻址的 `blobs/sha256-*` ✓），**只读** ✓
（删除仍走服务端 `/api/delete` ✓ —— 那是模型库一致性的边界 ✓）。⚠️ 与 `/api/tags` 的口径差异**是有意的** ✓：
体积取**盘上真实字节** ✓、blob 缺失计 `missingBlobs` ✓ ⇒「清单在、权重没了 ≠ 能推理」✓（**没查 ≠ 通过** ✓）。
⭐⭐⭐ **2026-09-25 夜「离线扫描」被真机打脸一次并修掉** ✗✗→✓（**这条是「看着有库、却扫出 0 个」的活标本** ✓）：
现象：本机离线路径给出 **0 个模型** ✗，而 `ollama list` / `/api/tags` 明明有 **14 个** ✓；更坏的是
`is_available()` 还报 **True** ✓✗ ⇒ 前端只会显示「0 个模型」，**连「库找错地方」都看不出来** ✗✗。
真因（两条，都在挑根逻辑上 ✗）：① 本机库在 `D:\\app\\LLM\\models\\ollama\\models` ✓，而那个 `OLLAMA_MODELS`
**只被桌面端注入给 `ollama serve` 子进程** ✗（`HKCU`/`HKLM` 环境里都没有 ✗）⇒ 后端的候选里**根本没这个根** ✗；
② 候选里最靠前的**存在**目录是空壳 `~/.ollama/models` ✗（存在、但既没 `manifests/` 也没 `blobs/` ✓✗）⇒
按「第一个存在的候选」挑就挑中空壳 ✗ ⇒ `is_available()=True` + `list_models()=[]` ✗✗。
修法（都**只读** ✓ 仍**不依赖外部服务** ✓）：候选根新增**桌面端配置库里的声明根** ✓ —— 桌面端把用户选的库根
持久化在 `%LOCALAPPDATA%\\Ollama\\db.sqlite` 的 `settings.models` 列 ✓（真机读出正是
`D:\\app\\LLM\\models\\ollama\\models` ✓）；挑根条件从「存在」改成「**先挑真能读（有 `manifests/`）的**」✓。
⚠️ 只读的硬边界：sqlite 一律 `mode=ro` ✓（桌面端此刻也在写这个库 ⇒ 我们绝不能建日志/改文件 ✗）、
`timeout=0.5s` ✓（被独占也不能把后端卡住 ✗）、读不到／不是 sqlite／没那一列／值不合格 ⇒ **静默 None** ✓ 不抛错 ✓；
缓存按 `(mtime, 大小)` ✓（请求链路上会反复调用 ⇒ 不能每次开一次 sqlite ✗，桌面端改了就立刻跟上 ✓）。
实测（本机 ✓）：`models_root()` → `D:\\app\\LLM\\models\\ollama\\models` ✓、离线模型 **14 → 15 个** ✓
（`qwen3:14b` 拉完 ✓）、生态扫描里 ollama 命中 **15 条** ✓（改前 **0 条** ✗）；`ollama_store_test` **42/42** ✓
（新增 14 条：真机声明根与**独立读出**的一致性 ✓、空壳 vs 真库的**回归** ✓、库不存在/垃圾文件/缺列/值不合格
（JSON、相对路径、空）✓、**真抢一次独占锁**（桌面端正写着 ⇒ 静默 None ✓ 也不卡住 ✓）、缓存失效重读 ✓）；
`model_ecosystems_test` **49/49** ✓、`local_models_test` **89/89** ✓、`smoke_test` **477/477** ✓。
✅ **另记一笔已结** ✓（2026-09-25 深夜）：`comfyui_runs_test.py` 曾「在磁盘上但**未登记** ✗」⇒ **查完并登记** ✓。
它不是活体 ✓：自起**真 HTTP stub 上游**（8765 形状 ✓）+ **真 DB**（临时数据根 ✓）⇒ 本机直接 **17/17** ✓
（唯一一条 ERROR 行是**故意**造的上游失败 ✓：`⑩ 上游失败 ⇒ DB 里 failed 且保留**上游真原因**` ✓）⇒
登记 **131 → 132** ✓。⚠️ 之所以必须登记而不是"搁着"：**磁盘上有、全量表里没有 = 它会静默腐烂** ✗✗。

⭐⭐ **2026-09-25 夜「扫描要覆盖世界各大模型，不只 Ollama」入账** ✓：新增 `model_ecosystems_test.py` ✓，
`TESTS` 登记 **130 → 131** ✓（本套实测 **49/49** ✓；同轮 `ollama_store_test` 当时 **28/28** ✓（后续补到 42/42 ✓）、
`local_models_test` 89/89 ✓）。
✅ **全量总数已实测补齐** ✓（**登记数 133 = 数出来的** ✓、**项数以刚跑出来的汇总为准** ✓）：
**132 套按 `SUMMARY:` 收敛出 4265 项 / 4265 过（红 0 ✓）** ✓ + 1 套常量守卫（0 项 ✓）
+ **崩的 0 套** ✓ —— 详见本文件开头 ✓。
口径：默认扫描根目录原来只有本仓 `models/`、`local_services/` 与 ComfyUI ✗ ⇒ HuggingFace /
ModelScope / LM Studio / GPT4All / Jan / llama.cpp 里躺着的模型**扫不到也不报错** ✗✗。
现补 `services/model_ecosystems.py` ✓（12 个生态的落点表 + 归属判定 + 「怎么才真能用」的实话 ✓）：
`~`/`%VAR%`/`$VAR` 展开 ✓、变量没设**不编路径** ✓、`roots()` 只返回**真实存在**的目录 ✓；
`detect()` **最长匹配** ✓（`hub-other` ≭ `hub` ✓）；**弱命中**（comfyui/unknown/ollama）才按生态改判 ✓，
`h3`/`local-sd`/`cosyvoice` 等**强命中原样保留** ✓（HF 缓存里的 H3 DiT 依然是 H3 权重 ✓）。
⚠️ 连带修掉一个**老误判** ✓：`tts-service` 规则原来把**整条路径**拼进正则 ✗ ⇒
`…\ecosystems_tmp\…\flux1-dev.safetensors` 里的 “**ecosys**” 含 “cosy” ⇒ 图像权重被判成 TTS 模型 ✗
（还会配一个 CosyVoice:9880 的**错建议** ✗）；现整条路径只认「以关键词开头的路径段」✓（89/89 未回退 ✓）。
⚠️ Ollama 的权重是无扩展名 blob ⇒ 文件遍历**看不见** ✗，改由 manifests 清单并入 ✓，
且**只在**「默认扫描」或「扫的目录覆盖了 Ollama 库」时并 ✓（免得「只扫 C: 盘」跨盘串味 ✗）。


⭐⭐ **2026-09-25 深夜这一轮起「自研运行时」** ✓✗（用户「要自研实现」✓）：文本生成此前走 ollama HTTP 服务 ✗ ⇒
① 新建 `engine/llm.py` ✓ —— 自研 decoder-only transformer（RMSNorm / GQA / RoPE / SwiGLU / 因果掩码 / KV cache / 采样 ✓），
Qwen3/Llama 这类 LLM 的架构 ✓；架构参数**全显式** ✓（`LlmConfig` 不写死 Qwen3 ✗）；缩小版走同一条前向 ✓。
② 新建 `engine/gguf_dequant.py` ✓ —— GGUF 权重**反量化**（F32/F16/BF16/Q8_0/Q4_K ✓ 公式照 llama.cpp ggml-quants.c ✓ MIT ✓；
Q4_K 的 6-bit 解包逐字节核过 ✓）；⚠️ 其余 k-quant（Q2_K/Q3_K/Q5_K/Q6_K）**具名拒绝** ✗ 未实现 ✓。
③ 新建 `engine/gguf_to_llm.py` ✓ —— GGUF 权重 → LlmModel 的**装载**（ggml 命名映射 ✓ + 线性层转置 ✓ + tie embeddings ✓ +
`infer_llm_config` 从 GGUF 元数据读架构 ✓）；⭐⭐ 往返恒等验证：state_dict → GGUF → 映射回 ⇒ 逐张量相等 ✓。
④ 新建 `engine/llm_backend.py` ✓ —— 自研文本后端（`describe` 如实报缺 / `load` 装载 / `generate` encode→生成→decode ✓；
未装载就 generate ⇒ 明确拒 ✓）。
⑤ **接线** ✓ —— `text_generation.generate_text` 加 engine 分支 ✓：provider=engine ⇒ 走自研后端（`asyncio.to_thread` 包同步推理 ✓）
而非 HTTP 调 ollama ✗；后端缓存（大权重只 load 一次 ✓）。⇒ **文本不依赖 ollama 已闭环** ✓（剩「下载真权重」这一步 ✓）。

⭐⭐ **2026-09-25 晚这一轮补的是「能力已有、没接出去」的收尾** ✓✗：
① **多集节奏相位注入** ✓ —— `runtime.py::append_style_profile` 里早先是「`rhythm-phase.ts` 未迁」的 warn 占位 ✗，
而 `services/rhythm_phase.py` 的 `rhythm_guidance_for_episode` 早就迁好了 ⇒ `storyboard_breaker` 一直少一段跨集节奏引导 ✓✗
（与 `assign_rhythm_phases` 那半「光收不读」同族 ✓）⇒ 现在真调它 ✓（读库失败只 warn 不阻断 ✓ 与视觉图谱分支同口径 ✓）；
② `TorchBackend.refineNote` **自述与实现相反** ✓✗ —— 它写着 `denoise: False` +「带掩码二采**未实现**」✗，但
`refine_latents` 的 `_second_pass` 早就实现了带掩码二采 ✓（调用方读到 `denoise=False` 会以为二采没做 ✗）⇒ 对齐 ✓。

⭐ **这一轮补的是「加载层/张量层」的接线** ✓✗（上一轮补的是提交前那几道关 ✓）：
① 混合加载从「计划层」补到**真加载层**（搬字节 / 原子落盘 / 缓存两道门 / 磁盘余量 ✓）
—— ⭐⭐ 两条会静默的判据：**合并不改键集** ✗✗、**必须继承基底的 ``__metadata__``** ✗✗
（``load_weights`` 靠它读结构 ⇒ 丢了它产物报「读不出结构」，真因却在合并这步 ✓）；
② **超清二采接进管线** ✓（计划由调用方给 ✓ 管线不自己猜倍率/分块 ✗）—— ⭐⭐ **音频流锁定必须是
后端自述的** ✓✗（没声明就拒跑 ✓：重采音频会把音轨弄坏而画面看着正常 ✓）；
③ ``audio_mix`` 终于有落点 ✓✗（单镜合成原是 ``-map 1:a`` **顶替** ⇒ 生成视频自带的音轨被**丢掉** ✓）。

⭐⭐ 2026-09-24 晚**这一轮是「接线」**✓✗ —— 上一轮的判据层（契约/硬约束/规划层）**只被自检调用** ✗✗
⇒ 生产面上一堆「静默改用户意图 / 能力接不出去」：① 配音**未知情绪静默变 happy** ✓、CosyVoice 两条路径
把 `emotion`/`speed` **静默丢掉** ✓；② H3 **有参考素材却挑到 FL2VA 权重** ✓（引擎里那条硬约束没人执行 ✓）、
8765 薄封装**从来没读过 `body.model`** ✗（后端算出的权重静默落空 ✓）；③ 提交前那道关（编号校验 / 声明追加 ✓）
与加速链**都没接** ✗；④ 导演稿分段 / 超清计划**只有自检在调** ✗。
⇒ 现在都接上了 ✓（判据一律写成「**能触发** + **拦在提交前** + **没查≠通过**」✓）。
⭐⭐ 修掉一条**产品侧竞态** ✓ ——
H3 服务此前是「**先写 `succeeded`** ✓、再在 `finally` 里 `/free` + 写 `freed_vram`」✗✗ ⇒ 中间那一瞬
调用方读到的是「成功了但 `freed_vram=null`」✓✗（API **自相矛盾** ✓）。⇒ 现在**先卸载、再落终态** ✓
（终态与 `freed_vram` 在**同一次 `_update`** 里落地 ✓，两条执行体都改了 ✓）；同轮补判据：失败路径也
`/free` ✓（`h3_stage2_test` 20 → 21 项 ✓）+ `dit.H3_PACK_FACTS` 两条「给错也不报错」的事实的**逐值**判据 ✓
（**跨度表 (1,4,4,4,4) 循环** ✓ / 立体声 **channel-major 的落点是 `w` 两端** ✓）+
**混合形态的模态标签** ✓ 与三条「混了不报错」的**网格规则** ✓ —— 104 套按 `SUMMARY:` 收敛出 3629 项 ✓，
另 1 套是常量守卫型（打印 `OK: 镜像常量漂移 0 条` ✓ 无项数 ✓））。
⭐ 2026-09-24 新增**六套**（都是「逆向出来的可用项」落地 ✓）：加速链 **30 条** ✓ + H3 prompt 契约
**37 条** ✓ + 权重守卫（内嵌元数据契约 ✓ + 未实现布局族**具名**拒绝 ✓）**18 条** ✓ + 导演稿分段
**26 条** ✓ + 联合 AV 潜变量容器（鸭子类型 ✓ 零依赖 ✓）**16 条** ✓ + 超清模式规划（1.5× 先 2× ✓ /
4× 小分块 ✓）**16 条** ✓ + 混合加载计划（fl2va 基底 + ref2va 的 adaLN 覆盖 ✓）**11 条** ✓ +
档位表（步数 / 分辨率 / 加速件 / 显存建议 / **必备模型** ✓）**16 条** ✓ +
两条**接缝**（加速链 ↔ 真 `/object_info` ✓ 8 条 / H3 prompt 提交前那道关 ✓ 10 条 ✓）✓
⇒ `TESTS` 登记 105 → **115** ✓（**登记数是数出来的** ✓；项数以「最近一次全量实测」为准 ✓ **不推算** ✗）。
⚠️ 上一轮：105 套 / 3623 项 ✓（H3 混合打包形态 ✓ + 掩码坐标空间补注 ✓）
—— 按规则**不推算** ✗，数字取自刚跑出来的汇总 ✓。
⚠️⚠️ **并发跑全量会让个别套件假红** ✗（2026-09-22 实测 ✓✗）：两个回归同时在跑时
`h3_stage2_test` 报 `19/20` ✓✗（单独复跑**两次都 20/20** ✓ ⇒ 当时判成「资源竞争不是代码」✗）。
⚠️ **2026-09-24 更正** ✗：那**不是**并发的问题 ✗✗ —— 是**产品侧的终态竞态** ✓（`h3` 服务先写
`succeeded` ✓ 再 `/free` ✓ ⇒ 中间态「成功但 `freed_vram=null`」✓✗）；并发/负载只是**让它更容易露** ✓。
⇒ 已修（先卸载再落终态 ✓）并加了判据 ✓；⭐ 教训：**「单独复跑就好了」≠「不是代码问题」** ✗✗，
它可能只是**概率**问题 ✓ —— 该做的是**问"什么条件下会露"** ✓，而不是收工 ✓。
⇒ 判据：① 看有没有 `结论：` 行 ✓；② 有红先数进程 ✓（>1 就清 ✓）③ **单独连跑 3 次** ✓
（一次就绿说明不了什么 ✗）—— 仍绿才去看代码里"什么时候写终态"这类**时序** ✗。
⚠️⚠️ **跑之前先把 ffmpeg 的真实 bin 目录「前置」到 PATH** ✗（2026-09-22 实测踩到 ✓✗）：
Windows 上 WinGet 装的 ``…\\Microsoft\\WinGet\\Links\\ffmpeg.exe`` 是**应用别名（重解析点）** ✗ ——
在本进程里可能 `lexists=True` 但 `exists=False` ✓✗，于是两条路一起坏：
① `shutil.which("ffmpeg")` 返回 `None` ✓（引擎侧会说「找不到 ffmpeg ✗」✗ —— 其实装了 ✓）；
② **裸名** spawn（`subprocess.run(["ffmpeg", …])` ✓ 产品代码那种写法 ✓）直接
`OSError: [WinError 448] 无法遍历该路径，因为它包含不受信任的装入点` ✗✗
⇒ `image_generation` / `consistency_qc` / `technical_qc` / `color_grade` / `compressed_data_url`
**5 套会红** ✓，看着像代码坏了 ✗✗。⇒ 对策：把
``…\\WinGet\\Packages\\Gyan.FFmpeg_…\\ffmpeg-*-full_build\\bin`` **前置**到 `PATH` ✓
（**前置**才行 ✗ —— 追加只修好 ① ✗，② 仍会命中坏别名 ✓；判据是 `where.exe ffmpeg` 指到真实 bin ✓）。
⚠️ 前端两条自检与**构建命令**耦合 ✗（2026-09-22 踩过 ✓）：`frontend/.output/public/index.html` 是
`npm run **generate**` 的产物 ✓（Dockerfile 用的也是它 ✓）⇒ 只跑 `npm run **build**` 会把它**覆盖掉** ✓✗
（`dockerfile_contract_test` 当场红 ✓）⇒ 改完前端**该跑的清单**：`typecheck` ✓ + `generate` ✓。
⚠️⚠️ **别同时开多个全量回归** ✗：并发会互相抢 CPU（表现为"卡在某套很久" ✓✗）；
判据是日志里有没有 `结论：` 行 ✓ —— **半截日志不算跑过** ✗（2026-09-21 实测踩到 ✓）。
⚠️ 自检**汇总行格式有硬要求** ✗：必须是 `SUMMARY: n/m passed` ✓（`run_all.py` 按这个前缀收敛项数 ✓）
—— 写成「n/m 项通过」会在总表里显示**空摘要** ✓✗（2026-09-20 实测踩过 ✓ 已修 4 个套件 ✓）。
⚠️ 与上一轮（78 套 / 2806 项）**独立互证** ✓：2806 + 11（`safetensors_crosscheck_test` ✓）
+ 8（`engine_pipeline_test` 90 → 98 ✓）+ 26（`engine_dit_test` ✓）+ 39（`engine_io_test` ✓）
+ 21（`engine_text_test` ✓）= **2911** ✓✓（两条路径同一个数 ✓）。
⚠️ **批次日志标签不可复用** ✗：`runbatch.py` 按标签写 `tmp/<label>.txt` ✓ ⇒ 复用旧标签会读到
**上一次会话的陈旧结果** ✓（2026-09-17 实测踩到过一次 ✓）⇒ 每次换新标签 ✓。
⚠️ 其后又**新增** `engine_segments_test`（长视频分段 ✓ 54 项 ✓）⇒ 现为 **83 套**、总数**待实测** ✗
（按规则**不推算** ✗ —— 跑一次全量即可补齐 ✓）。
⚠️ 再**新增** `engine_runtime_test`（在机运行时 ✓）与 `engine_bridge_test`（业务桥 ✓）⇒ 现为 **85 套** ✓；
**这两个此前没登记** ✗（磁盘上有、清单里没有 ✓✗ —— 正是本节开头说的"静默腐烂"✓）
⇒ 2026-09-26 已补登记 ✓；总数照规则**不推算** ✗ 待全量实测 ✓。
那一轮的完整复测**没跑成** ✗（环境连跳多次 ✗）⇒ 按规则**不推算总数** ✗：
本轮改为逐个实测**受影响的**套件（管线 98/98 ✓、交叉验证 11/11 ✓、加载计划 28/28 ✓、
体检 34/34 ✓、算法 26/26 ✓）✓。**下次跑一次全量即可补齐** ✓。
⚠️ 每轮都与上一轮**独立互证** ✓（两条路径得出同一个数 ✓）：2762 − 46 + 65（引导 ✓）− 65 + 82
（首帧条件 ✓）− 82 + 90（torch 依赖闸门 ✓）= **2806** ✓✓。
⚠️ **活体类套件的项数随本地服务在否浮动** ✗：全停时 `local_services_live_test.py` 记 9 项、
`h3_backend_live_test.py` 记 0 项（都显式 SKIP ✓）；把 H3 薄封装（8765）起来后变 14 项与 8 项 ✓。
⇒ **别把数字当硬约束，自己跑一遍看汇总** ✓。
⚠️ **不做算术推算** ✗（09-16 按「2526+8」算错过 13 项 ✓ 教训在案）⇒ 一律以**刚跑出来的汇总**为准 ✓。
⚠️ **活体类套件的项数随本地服务在否浮动** ✗：全停时 `local_services_live_test.py` 记 9 项、
`h3_backend_live_test.py` 记 0 项（都显式 SKIP ✓）；把 H3 薄封装（8765）起来后变 14 项与 8 项
⇒ 合计 **+13**（这个差也已实测过 ✓）⇒ **别把数字当硬约束，自己跑一遍看汇总** ✓。
跳过条数一律显式打印（`（skip n：服务未启动）`）⇒ **不要把这几个数字当硬约束，自己跑一遍看汇总** ✓）（删 `backend/` 后实测；比删库前少 2 项
**显式 `[skip]`**，见 README；另**新增 2 个套件**，都是「删库后 Python 成为唯一后端」才存在的不变量：
``frontend_api_coverage_test.py``（前端每个调用点都在后端路由表里）与 ``contract_mirror_test.py``
（``frontend/app/types/contracts.ts`` 的字段在后端源码里都能找到 —— 共享契约只许单向同步：后端改 ⇒ 镜像跟）。各套件**各自独立进程**、都不碰真实库：

* 契约冒烟 ``smoke_test.py``（跑在**数据库副本**上，覆盖每一条已注册端点）
* 纯逻辑/适配层：``adapters_test.py``、``vendor_errors_test.py``（``MockTransport``，零真实网络）、
  ``text_generation_test.py``、``prompt_storyboard_test.py``、``grid_prompt_test.py`` 等
* 迁移域回归：各 ``*_route_test.py`` / ``*_generate_test.py``（事件与落盘都打桩）
* 机械守卫：``route_parity_test.py``（路径 0 遮蔽 + 九道镜像常量漂移）、
  ``freeze_snapshot_test.py``（TS 快照反漂移）、``parity_diff_test.py``（Node↔Python 差分比较器）
* ⚠️ 少数套件会用 **ffmpeg 现场造真实媒体**（``consistency_qc_test.py`` / ``color_grade_test.py`` /
  ``compressed_data_url_test.py``）⇒ 机器上没有 ffmpeg 时这几套会失败，属环境依赖（见 README）。

用法::

    ./.venv/Scripts/python.exe tests/run_all.py          # 全量
    ./.venv/Scripts/python.exe tests/<某个>_test.py      # 单跑（改哪儿跑哪儿）

⚠️ 单跑很快，全量偏慢（分钟级）；若被上层环境截断，可按 ``run_all.TESTS[a:b]`` 切片分批跑，
   结论等价（本项目实测过：三批 = 61 套件一次全绿）。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

TESTS = [
    ("契约冒烟", "smoke_test.py"),
    ("适配器层", "adapters_test.py"),
    ("厂商错误归因", "vendor_errors_test.py"),
    ("文本生成", "text_generation_test.py"),
    ("图片生成链路", "image_generation_test.py"),
    ("视频生成链路", "video_generation_test.py"),
    ("TTS / 音色复刻", "tts_generation_test.py"),
    ("分镜 prompt + 图谱", "prompt_storyboard_test.py"),
    ("宫格 prompt + 运镜", "grid_prompt_test.py"),
    ("逐镜路由 + videos", "videos_route_test.py"),
    ("单镜合成 compose", "compose_test.py"),
    ("整集拼接 merge", "merge_test.py"),
    ("宫格 prompt/切分 grid", "grid_route_test.py"),
    ("图谱/图片/回调 misc", "misc_routes_test.py"),
    ("AI 音色 ai-voices", "ai_voices_test.py"),
    ("预设框架 preset-framework", "preset_framework_test.py"),
    ("Agent 协议/工具基座", "agent_protocol_test.py"),
    ("Agent 工具集", "agent_tools_test.py"),
    ("Agent 音色工具", "agent_voice_tools_test.py"),
    ("Agent 剧本/语料工具", "agent_script_corpus_test.py"),
    ("Agent 提取工具", "agent_extract_tools_test.py"),
    ("Agent 分镜工具", "agent_storyboard_tools_test.py"),
    ("Agent 运行时", "agent_runtime_test.py"),
    ("Agent 聊天域", "agent_route_test.py"),
    ("MCP 接入层 + 路由", "mcp_test.py"),
    ("Agent 出厂提示词", "agent_prompts_test.py"),
    ("评测打分器 + 基准目录", "evaluation_scorer_test.py"),
    ("Skill 解析 + 加载", "skills_test.py"),
    ("评测执行器 + 端点", "evaluation_route_test.py"),
    ("基准资产搬迁 + 评测 CLI", "eval_cli_test.py"),
    ("GPU 显存管理器 + 2 端点", "gpu_manager_test.py"),
    ("GPU 租约接线（text / tts + image/video 长租约）", "gpu_lease_wiring_test.py"),
    ("差分对拍比较器（Node↔Python 工具）", "parity_diff_test.py"),
    ("多集节奏相位 + 端点", "rhythm_phase_test.py"),
    ("镜头 QC 打分 + 端点", "qc_scoring_test.py"),
    ("审片重跑闭环 + 端点", "qc_retry_test.py"),
    ("设置分镜首尾帧 + 抽帧泛化", "set_frame_test.py"),
    ("重生成镜头帧 + 端点", "regenerate_frame_test.py"),
    ("图像连续性 QC + 端点", "consistency_qc_test.py"),
    ("技术维度 QC + 接线", "technical_qc_test.py"),
    ("宫格 Agent 提示词 + 端点", "grid_agent_prompt_test.py"),
    ("像素级校色（ffmpeg）", "color_grade_test.py"),
    ("子 Agent 调度工具", "subagent_test.py"),
    ("请求日志中间件", "http_logger_test.py"),
    ("参考图压缩（ffmpeg）", "compressed_data_url_test.py"),
    ("TS 源码快照反漂移", "freeze_snapshot_test.py"),
    ("Agent 创建器 + generate", "creator_test.py"),
    ("提示词优化器 + optimize", "optimizer_test.py"),
    ("评测调度器 + 端点", "evaluation_scheduler_test.py"),
    ("SSE 总线 + 尾帧提取", "sse_hub_frames_test.py"),
    ("全自动管线 + SSE 端点", "auto_pipeline_test.py"),
    ("本地模型扫描 + 11 端点", "local_models_test.py"),
    ("本机 Ollama 模型库**离线**读取（服务没起也列得出；零外部依赖）", "ollama_store_test.py"),
    ("**世界各大模型**生态落点（HF/ModelScope/LM Studio/GPT4All/Jan/llama.cpp…；离线 + 缓存≠能推理）",
     "model_ecosystems_test.py"),
    ("角色生成链路 8 端点", "characters_generate_test.py"),
    ("物品/场景出图 2 端点", "props_scenes_generate_test.py"),
    ("分镜 TTS/出图/LLM 5 端点", "storyboards_generate_test.py"),
    ("剧集续写端点", "episodes_continue_script_test.py"),
    ("导出服务 + EDL/ZIP 2 端点", "export_service_test.py"),
    ("剪映草稿导出", "jianying_draft_test.py"),
    ("QC 报告 + 联系表 2 端点", "qc_report_test.py"),
    ("时代背景提炼 + 风格提炼 2 端点", "era_style_distill_test.py"),
    ("数据根切换 + 存储 2 端点", "storage_change_test.py"),
    ("生产镜像布局一致性", "dockerfile_contract_test.py"),
    ("前端调用点 ↔ 后端路由覆盖", "frontend_api_coverage_test.py"),
    ("共享契约镜像（contracts.ts ↔ 后端）", "contract_mirror_test.py"),
    ("层级守卫（app/ 内单向依赖）", "layering_test.py"),
    ("资产布局守卫（三类资产在 app/ 内 + 常量权威）", "assets_layout_test.py"),
    ("本地 H3 链接缝契约（适配器 ↔ 8765 薄封装）", "h3_chain_test.py"),
    ("本地服务活体接缝（ollama 真推理；其余按设计 SKIP）", "local_services_live_test.py"),
    ("H3 全栈活体（配置 → 服务层 → 真 8765 → 落库 + task_id 交叉证明）", "h3_backend_live_test.py"),
    ("CosyVoice 接缝契约（JSON↔form/multipart、裸 PCM→WAV；真 HTTP stub 上游）", "cosyvoice_seam_test.py"),
    ("ComfyUI 能力（客户端 + UI→API 工作流转换；真 HTTP stub ComfyUI）", "h3_comfyui_test.py"),
    ("H3 阶段2 闭环（body → 组装/注入 → 提交 → 轮询 → 取片 → /files → /free）", "h3_stage2_test.py"),
    ("ComfyUI 能力门面（系统/目录/队列/历史/作业/媒体 + 通用工作流）", "comfyui_capability_test.py"),
    ("ComfyUI 工作流运行（持久化/崩溃恢复/串行租约/产物落点/用量/归因；真 stub 上游 + 真 DB）",
     "comfyui_runs_test.py"),
    ("自研引擎·加速链（配置解析 + 提交前校验 + 接线；零依赖）", "engine_accel_chain_test.py"),
    ("自研引擎·H3 prompt 契约（声明 / 跳过判据 / 标签 / 任务选择；零依赖）",
     "engine_conditioning_test.py"),
    ("自研引擎·权重守卫（内嵌元数据契约 + 未实现布局族具名拒绝；零依赖）",
     "engine_checkpoint_guards_test.py"),
    ("自研引擎·导演稿分段（5 种标记 / 4.5 字每秒 / 断点优先级 / ≤15s 合并；零依赖）",
     "engine_script_parse_test.py"),
    ("自研引擎·联合 AV 潜变量容器（认出视频流 / 保类型换流；鸭子类型、零依赖）",
     "engine_latent_container_test.py"),
    ("自研引擎·超清模式规划（契约校验 / 1.5×先 2× / 4× 小分块；零依赖）",
     "engine_upscale_test.py"),
    ("自研引擎·超清放大器网络（V2 主干 + V3 因子化注意力；T 不变 / 残差恒等 / 严格装载）",
     "engine_upscale_net_test.py"),
    ("自研引擎·段级音频合成（长度守恒 / 偏移 / 裁剪 / 三支削波口径；纯样本、零依赖）",
     "engine_audio_mix_test.py"),
    ("自研引擎·缓存完整性守卫（头自洽 / 偏移吃满 / 头哈希指纹 / 源数对齐；零依赖）",
     "engine_cache_guard_test.py"),
    ("自研引擎·参考素材指纹（顺序无关 / 两种形态都认 / 采样率入指纹 / 出错退化成不相等）",
     "engine_cache_key_test.py"),
    ("配音契约（8 维定序 / 单值↔预设 ↔ 向量 / 未知情绪拒 / 不归一化 / 语速不钳位；零依赖）",
     "voice_contract_test.py"),
    ("配音契约**接线**（非法拦在提交前 / 引擎不认则响亮报告 / 不替用户默认 / 「没查」如实报告）",
     "voice_contract_wiring_test.py"),
    ("H3 **形态决策接线**（有参考素材 ⇒ Ref2VA / 互斥诉求响亮报告 / 薄封装尊重调用方权重）",
     "h3_form_routing_test.py"),
    ("混合加载**加载层**（合并不改键集 / 继承内嵌元数据 / 坏缓存与源变了都重合并）",
     "engine_hybrid_load_test.py"),
    ("段级音频合成（模型声×0.6 + 配音 / 长度守恒 / 不静默重采样；标准库读写 wav）",
     "segment_audio_test.py"),
    ("超清二采的**张量层**（真装载放大器 / 只换视频流 / 时间维不变；CPU 可跑）",
     "engine_refine_test.py"),
    ("自研引擎·混合加载计划（fl2va 基底 + ref2va 的 adaLN 覆盖；零依赖）",
     "engine_hybrid_merge_test.py"),
    ("自研引擎·档位表（步数 / 分辨率 / 加速件 / 显存建议 / 必备模型；零依赖）",
     "engine_tiers_test.py"),
    ("H3 加速链接缝（链校验 + 接线 → 真 /object_info；假客户端、零网络）",
     "h3_accel_wiring_test.py"),
    ("H3 prompt 接缝（全局提示词 + 标签转换 + 编号校验 + 声明追加）",
     "h3_prompt_wiring_test.py"),
    ("自研引擎·算法层（σ 调度 / 帧网格与像素预算 / 采样循环；零依赖）", "engine_core_test.py"),
    ("自研引擎·权重体检（纯 Python 读 safetensors + 就绪报告；零依赖）", "engine_inventory_test.py"),
    ("自研引擎·管线编排（阶段/事件/取消/错误归因 + 干跑后端；零依赖）", "engine_pipeline_test.py"),
    ("自研引擎·加载计划（量化配套/层号连续性/显存排班；零依赖）", "engine_loader_test.py"),
    ("自研引擎·GGUF 读取器（头格式/张量表/截断检测/量化方案名；零依赖）", "engine_gguf_test.py"),
    ("自研引擎·H3 键名核对器（键全集/形状关系/PDD 头库/curve 变体；零依赖 torch-free）", "engine_h3_keys_test.py"),
    ("自研引擎·BPE 分词器（真词表 tokenizer.json/vocab+merges；往返恒等；零依赖离线）", "engine_tokenizer_test.py"),
    ("自研引擎·反量化（fp8/int8 × 四种布局 × 两种 scale 方向；判不出来就拒绝；接进装载）", "engine_quant_test.py"),
    ("自研引擎·分词器总入口（形态嗅探/自研优先/参考回退/批量+LRU/离线开关；互校）", "engine_tokenizer_hub_test.py"),
    ("自研引擎·自研 Unigram/WordPiece/Metaspace（Viterbi/fuse_unk/整词 UNK；与参考逐例同 id）",
     "engine_tokenizer_own_test.py"),
    ("自研引擎·transformers 运行时改造（离线兜底含 Auto 工厂/缓存目录/降噪/计数/可撤；幂等）",
     "engine_tokenizers_tuning_test.py"),
    ("自研引擎·上机前自检入口（依赖/权重就绪/加载计划/真权重预检 + 词表；JSON 与退出码）",
     "engine_readiness_script_test.py"),
    ("safetensors 交叉验证（纯 Python 读取器 vs 官方库；缺库则显式 SKIP）", "safetensors_crosscheck_test.py"),
    ("自研引擎·真模型层（DiT 前向/条件生效/权重往返/差异报告；CPU 可验）", "engine_dit_test.py"),
    ("自研引擎·H3 形态积木（RMSNorm / SwiGLU / 18 路 adaLN / 正弦时间嵌入 / RoPE 旋转不变量 / "
     "双 fp32 输出头 / PDD 头库 —— 已接进 `TorchBackend` 双流路径 ✓ `H3_FORM_TODO` 已清零 ✓）",
     "engine_h3_form_test.py"),
    ("自研引擎·音频 VAE（32 kHz 立体声 ⇄ 潜变量；**真写 wav + 标准库读回核对** ✓）",
     "engine_audio_vae_test.py"),
    ("自研引擎·decoder-only LLM（RMSNorm / GQA / RoPE / SwiGLU / 因果掩码 / KV cache / 采样；缩小版 CPU 可验）",
     "engine_llm_test.py"),
    ("自研引擎·GGUF 权重反量化（F32/F16/BF16/Q8_0/Q4_K 精确字节布局 + 6-bit 解包；未实现具名拒绝）",
     "engine_gguf_dequant_test.py"),
    ("自研引擎·GGUF → LlmModel 装载（ggml 命名映射 + 线性层转置；往返恒等；tie embeddings）",
     "engine_gguf_to_llm_test.py"),
    ("自研引擎·文本后端（describe 如实报缺 / 未装载拒生成 / encode→生成→decode 链路；缩小版 CPU 可验）",
     "engine_llm_backend_test.py"),
    ("自研引擎·对话骨架（**骨架从权重元数据取** ✓ 不内置 ✗ / 逐字渲染 / 「没有」与「坏了」两回事 / "
     "坏了当场拒；缩小版 CPU 可验）", "engine_chat_template_test.py"),
    ("自研引擎·H3 双流接进管线（**真 mp4 + 真 wav**；参考块四类入口 ✓ / 能力自述逐条报缺 / "
     "取整口径必填 / 当场拒绝 / 单流默认路径一字未动 ✓）", "engine_dual_stream_test.py"),
    ("自研引擎·解码与落盘（VAE 编解码 + **真 mp4/wav 用 ffprobe/标准库复核** + 整链出片 + "
     "**设备口径逐入口**：错设备当场说清 / 判据不许比字符串 / 不许误伤真链路）", "engine_io_test.py"),
    ("自研引擎·文本编码（真 TE + 注入式 tokenizer + 截断回报 + 整链 TE→DiT→VAE→mp4）", "engine_text_test.py"),
    ("自研引擎·长视频分段（网格长度/重叠接缝/保留帧守恒 + 首帧落 PNG 传递）", "engine_segments_test.py"),
    ("自研引擎·在机运行时（**装/排队/取消/卸** + 真出 mp4/wav；串行有证据 + 设备口径逐入口）",
     "engine_runtime_test.py"),
    ("自研引擎·业务桥（`provider=engine` 那条路：装配键集与运行时**逐键对账** / 缺 DiT 就报 / "
     "取整口径必填 / 产物口径 / **真跑出真 mp4+首帧真 PNG** / 把 httpx 打死照样跑完）",
     "engine_bridge_test.py"),
    ("自研引擎·命名映射（预设改名/合并顺序/**干跑报告**缺哪些键 + 往返装载逐位一致）", "engine_mappings_test.py"),
    ("短剧提示词生产契约（占位符**解析成具体内容** + Mx-Shell 五段质感层；零依赖）",
     "prompt_contract_test.py"),
    ("资产清单 + 验收门（依赖拓扑/成环点名/逐镜阻断/**付费生成前的门**；零依赖）",
     "asset_gate_test.py"),
    ("连续性表（§6-§11：道具时间线/越轴/线索提前暴露/转场动机/可删动作；零依赖；**事前**）",
     "continuity_test.py"),
    ("Agent 上下文预算（估算/机械压缩/**tool 往返成对**/摘要交给调用方；零依赖）",
     "agent_context_test.py"),
    ("开跑前体检（四块串成一次调用：占位符→质感→连续性→验收门 + 按成本排序的行动项）",
     "preflight_test.py"),
    ("请求体守卫（`read_json` 已保证 dict ⇒ 路由里的 isinstance 判断是**死代码**；AST 检出）",
     "read_json_guard_test.py"),
    ("分镜→连续性适配器（只映射真实列、**不补默认值** + 契约字段的 schema 缺口清单）",
     "storyboard_continuity_test.py"),
    ("体检取数（按 episodeId 组装；逐镜 asset_status→逐资产验收 + 空集**不给绿灯**）",
     "preflight_source_test.py"),
    ("连续性写入（词汇**拒收而非忽略** / 幂等替换 / **全有或全无** / 归属校验；临时库 ✓）",
     "continuity_store_test.py"),
    ("生产链端到端（真工具写状态 → 取数 → 适配 → 判定 → 门；**填齐就当绿** ✓ 临时库 ✓）",
     "chain_e2e_test.py"),
    ("路径 + 常量守卫", "route_parity_test.py"),
]


def syntax_problems(root: Path) -> list[str]:
    """**AST 预检**整棵源码树 ✓ ⇒ ``["相对路径:行 说明", …]`` ✓（空 = 全过 ✓）。

    ⚠️⚠️ 为什么要它 ✗（2026-09-20 加 ✓）：那一天里「中文文案里嵌了半角双引号 ``"``」把
    `SyntaxError` 带进来**四次** ✓✗（本仓 MEMORY 早写着这条 ✓ 我还是犯了 ✓）。那种错的**代价很阴** ✗：
    套件**连一行都不输出** ✓ ⇒ 表现只是"没有 SUMMARY"✓，看起来像"被判后台 / 输出被吞"✓✗
    —— 每次都得另外跑一次才看出来 ✓。⇒ 开跑前统一 parse ✓，一有问题**当场点名 `文件:行`** ✓
    （"响亮"而不是静默 ✓ —— 本仓那条判据 ✓）。
    """
    import ast  # noqa: PLC0415 —— 只在预检用 ✓

    bad: list[str] = []
    for path in sorted(Path(root).rglob("*.py")):
        if {".venv", "__pycache__"} & set(path.parts):
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"), str(path))
        except SyntaxError as err:
            bad.append(f"{path.relative_to(root)}:{err.lineno} {err.msg}")
    return bad


def main() -> int:
    here = Path(__file__).resolve().parent
    failures: list[str] = []

    syntax_bad = syntax_problems(here.parent)
    if syntax_bad:
        for item in syntax_bad:
            print("SYNTAX " + item)
        print(f"\nSYNTAX 预检失败：{len(syntax_bad)} 个文件有语法错 ✗（先修它们再跑套件 ✓）")
        return 1

    # ⚠️ Windows 上的双坑（都踩过）：
    # 1. 子进程 stdout 是**管道**时，Python 按**本地代码页（GBK）**输出 ⇒ 若按 UTF-8 解码
    #    会得到 U+FFFD，再打印回 GBK 控制台直接 UnicodeEncodeError 崩掉。
    #    这里给子进程强制 PYTHONIOENCODING=utf-8，两边编码就对齐了。
    # 2. 自身 stdout 也加 errors="replace"，保证任何字符都不会让汇总步骤崩。
    child_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    try:
        sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]
    except (AttributeError, OSError):  # pragma: no cover
        pass

    for label, filename in TESTS:
        completed = subprocess.run(
            [sys.executable, str(here / filename)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(here.parent),
            env=child_env,
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        summary = ""
        for line in output.splitlines():
            if line.startswith("SUMMARY:") or line.startswith("OK:") or line.startswith("FAIL:"):
                summary = line.strip()
        status = "OK  " if completed.returncode == 0 else "FAIL"
        print(f"{status}  {label:<16} {summary}")
        if completed.returncode != 0:
            failures.append(label)
            # 失败时把 FAIL 行打出来，方便直接定位
            for line in output.splitlines():
                if line.startswith("FAIL") or line.startswith("  DRIFT") or line.startswith("  SHADOW"):
                    print(f"        {line.strip()}")

    print()
    if failures:
        print(f"结论：{len(failures)} 项未通过 -> {'、'.join(failures)}")
        return 1
    print(f"结论：{len(TESTS)} 项自检全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
