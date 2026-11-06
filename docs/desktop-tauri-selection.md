# 桌面端技术选型与体积/兼容性指标

> 目标：在不重写现有 Nuxt 3 前端的前提下，交付一个**跨平台、安装包尽可能小**的桌面应用。
> 本文回答四件事：为什么这么选、覆盖哪些平台、体积上限是多少、怎么验证兼容性。
> 落地代码在 `desktop/`（Tauri 2 外壳），前端改造点见 §6。

---

## 1. 结论速览

| 项 | 决策 |
|---|---|
| 桌面框架 | **Tauri 2**（Rust 外壳 + 系统 WebView） |
| 前端复用 | 直接复用现有 Nuxt 3 SPA，`nuxt generate` 产静态资源，**零重写** |
| 后端 | **不打进安装包**（见 §2.3），由壳探测/拉起/托管本地 Python 服务 |
| 进程通信 | 前端 → 后端走 HTTP（跨源，壳注入 origin + 注入 `CORS_ORIGINS`）；前端 → 壳走 6 个 IPC 命令 |
| 支持平台 | Windows 10 1803+ / 11（x64、ARM64）；macOS 11+（Intel、Apple Silicon）；Linux 发行版（glibc 2.31+，webkit2gtk-4.1） |
| 安装包上限 | Windows NSIS ≤ **4.5 MB**（实测 3.57 MB，含后端源码）；macOS DMG ≤ **14 MB**、Linux AppImage ≤ **18 MB**、deb ≤ **10 MB**（三项待真机实测后收紧） |
| 体积闸门 | `node desktop/scripts/size-budget.mjs --strict`，超限退出码 1（可进 CI） |

---

## 2. 为什么是 Tauri 2

### 2.1 候选对比（只比与本项目相关的维度）

| 方案 | 安装包量级 | 运行时依赖 | 复用 Vue/Nuxt | 与本项目后端（Python）协同 | 结论 |
|---|---|---|---|---|---|
| **Electron** | 90–150 MB | 自带 Chromium + Node | 好 | 需自己 fork python 进程，杀进程/端口冲突要手写 | 用户明确不建议；体积超出量级 |
| **Tauri 2** | **3–10 MB** | 系统 WebView（Win 内置 WebView2、macOS WKWebView、Linux WebKitGTK） | 好（任意静态产物） | Rust 侧管进程最稳（job/kill/端口探测），无需额外运行时 | **采用** |
| Wails v3 | 6–15 MB | 同上（系统 WebView） | 好 | Go 侧进程管理同样可行 | 备选；生态与文档弱于 Tauri，Linux 打包更依赖发行版 |
| pywebview + Nuitka | 30–80 MB | 需要 Python 运行时 | 好 | 天然同语言 | 体积不可控（Nuitka 打 Python + 依赖），且要再打一份 torch 才"自包含" |
| CEF / Qt WebEngine | 100 MB+ | 自带 Chromium | 好 | 无关 | 直接排除（体积目标冲突） |
| Flutter desktop | 20–40 MB | 自带 Skia | **差**（要重写 UI） | 无关 | 排除（等于放弃现有前端） |

**决定性因素不是"哪个框架好"，而是三条硬约束**：

1. **不能重写前端** —— 现有 17 个页面 + 画布组件是主要资产；
2. **安装包必须小** —— 排除一切自带浏览器内核的方案；
3. **后端是 Python + torch** —— 不可能"打成单文件"（见 §2.3），所以壳必须会管外部进程。

Tauri 2 是唯一同时满足三条的方案。Rust 侧只写进程/路径/IPC，业务逻辑一行不写（壳越薄，平台差异越少）。

### 2.2 代价与应对（先认下来，别掩盖）

| 代价 | 具体表现 | 应对 |
|---|---|---|
| 依赖系统 WebView | Win10 1803 以下没有 WebView2；Linux 缺 webkit2gtk 就起不来 | Windows 用 `downloadBootstrapper` 首次静默安装；Linux deb 声明 `Depends: libwebkit2gtk-4.1-0`；AppImage 文档注明需宿主自备 |
| WebView 版本碎片 | 老 Win10 上的 WebView2 可能是 90 版，缺新 API | 前端**不用**前沿 API；构建目标对齐 esbuild `target: chrome100` 级别；启动时探测版本并在设置页显示 |
| Linux 桌面环境差异 | 目录选择器各发行版不一 | 依次尝试 `zenity` → `kdialog`，都缺时**明确报错**（不假装用户取消） |
| Rust 构建链 | 首次编译要拉 crate、要 MSVC/MinGW | 只在**发布**机器上需要；日常只跑前端 |

### 2.3 后端源码进包、运行时**不**进包（体积指标仍成立）

2026-11-02 起改成「**源码进包、运行时留在系统**」：`bundle.resources` 把 `dist-backend/backend-py`
装到安装目录的 `backend-py/`，由 `scripts/sync-backend.mjs` 生成（排除 `.venv`(3.2 GB) / `output` /
`tmp` / `tests` / `_bench` / `__pycache__`，实测 **614 文件 / 10.51 MB** 源，压进 NSIS 约 +2.2 MB）。
（2026-11-04 复测：`_bench`（引擎基准 2.3 MB）此前漏排除，已加进 `EXCLUDE_DIRS`，否则后端源码 12.51 MB 会顶破 12 MB 预算。）

仍然**不**打进包的是运行时与模型，理由没变：

- 依赖 `torch` + CUDA，装完 5–7 GB；模型池是 **789 GB** 量级（`D:\Comfy-Desktop\ComfyUI-Shared`）；
- 就算只打最小子集，Python 运行时 + numpy + onnx 也要 150 MB 以上。

所以定位仍是**"窗口 + 前端资源 + 后端源码 + 后端进程托管器"**：后端目录的解析顺序是
`desktop.json` → `DARME_BACKEND_DIR` → **包内 `resources/backend-py`** → exe 祖先目录 →
（仅 debug）仓库路径 → cwd；解释器同理（`desktop.json` → `DARME_PYTHON` → 后端内 `.venv` →
（仅 debug）仓库 `.venv` → 包内 → PATH）。release 二进制里**不允许**残留构建机绝对路径，
所以「仓库路径」这一档用 `#[cfg(debug_assertions)]` 隔离。

| 分发形态 | 内容 | 体积 | 适用 |
|---|---|---|---|
| **A. 托管便携版（默认）** | 壳 + 内嵌前端资源 + 后端源码 | 4–12 MB | 开发者/创作者本机，后端跑在系统 Python 或仓库 `.venv` |
| **B. 一体化版（可选）** | 壳 + 前端 + 后端源码 + 嵌入式 Python + 依赖（不含模型） | 180–260 MB | 给完全不懂命令行的用户；**明确不作为默认**，是 A 之上再加一层 |
| **C. 开发模式** | 壳只开窗口，`devUrl` 指向 `nuxt dev` | n/a | 前端热更新 |

**A 形态的边界（别误读成"干净机器一键能跑"）**：源码进包只解决「没有仓库也能找到后端」，
不解决「没有 Python」。干净机器缺解释器/缺依赖时，`backend_start` 会由 `explain()`
把日志尾部翻成三条可执行建议（`ModuleNotFoundError` ⇒ 装依赖；`address already in use`
⇒ 端口被占；`PermissionError` ⇒ 目录不可写），其余情况**原样贴日志、不编原因**。

首启动若找不到后端目录/解释器，`backend_start` 返回 `failed` 并在界面上给出**可执行的下一步**（`desktop.json` 里填 `backendDir` / `pythonPath`，或设 `DARME_BACKEND_DIR` / `DARME_PYTHON`）——不静默降级。

---

### 2.4 子进程所有权：壳怎么死，后端都得跟着死

`RunEvent::Exit` 只在**正常退出**路径上跑。任务管理器结束进程、`taskkill /F`、壳自己 panic
都不走它，uvicorn 就成了孤儿 —— 它继续占着 5790，下次启动 `reuse_external` 会把它当
「外部后端」复用，于是**关窗不确认、退出不清扫**，现象看起来像"关窗确认坏了"。
本项目在冒烟里真踩过一次（两次强杀留下两个孤儿，排查花了半天），所以这层不是洁癖：

| 层 | 机制 | 覆盖 | 平台 |
|---|---|---|---|
| 内核兜底 | Windows Job Object + `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` | 壳**任何**死法（句柄由内核关闭 ⇒ 整棵子进程树连坐）。句柄故意泄漏到进程结束，Drop 掉反而会在壳活着时杀死后端 | **仅 Windows** |
| 落盘凭据 | `backend.pid`（`{pid, created, port, backendDir, python}`，落 `<appConfigDir>`） | 老版本遗留 / Job 未生效（父进程已在不可 breakaway 的 job 里）。下次启动**认领**并清掉 | 全平台（macOS 只能提示，见下） |

**认领必须认身份，不许认 pid**：重启后 pid 会复用，只比 pid 等于随机杀人。判据是
Windows 比 `GetProcessTimes` 的创建时间（100ns，逐位相等）、Unix 比 `/proc/<pid>/cmdline`
同时含 `app.main:app` 与 `--port <端口>`；`created == 0`（凭据残缺/被手改）⇒ **一律不认领**，
宁可留一个孤儿，也不误杀用户自己在终端起的服务。

#### 平台差异（诚实的降级面）

三个平台都在 §3.1 的支持范围内，但这层的强度差得很远，文档不能只写最好那个：

| 平台 | 内核兜底 | 身份识别手段 | 强杀壳之后 |
|---|---|---|---|
| Windows 10 1803+ | Job Object | `GetProcessTimes` FILETIME 逐位相等 | 后端随壳一起被内核回收，**无残留** |
| Linux（glibc 2.31+） | 无 | `/proc/<pid>/cmdline` | 后端成孤儿；下次启动认领并杀掉（先 `pkill -P` 收拾 uvicorn worker，再杀父） |
| macOS 11+ | 无 | **无 procfs ⇒ 认不出来** | 后端成孤儿且**无法识别**：只提示、不杀、不删凭据 |

无 procfs 平台只做三件事：不杀、不删凭据、把话说给用户看
（`status.message` 里写「上次可能有残留服务（记录 pid N、端口 P），但本平台无法确认它是否还在，未作处理」）。
Linux 侧的内核级等价物是有的（`prctl(PR_SET_PDEATHSIG)` / cgroup v2），macOS 侧是 kqueue 监听父进程，
但都要引入 libc 或额外依赖，且本仓这两个平台尚无真机跑过 —— 与其写一套没验证过、看起来对的实现，
不如让这层如实降级到凭据认领。

**`Identity` 必须三态（`Match` / `Mismatch` / `Unknown`），不能是 bool**：把 `Unknown` 当
`Mismatch` 处理就是"认不出来 ⇒ 把唯一的证据删掉"，孤儿从此再也清不掉 —— 本仓改之前正是
这样，macOS 上每次启动都会静默把凭据作废（凭据还在、后端还在，但下次连提示都没有）。
`Match` 才杀并删凭据；`Mismatch`（进程已退 / pid 被复用 / 凭据缺时间戳）把凭据当废纸删掉；
`Unknown` 一个字节都不动。

非 Windows 的 `supervise` **返回 `Err` 而不是 `Ok(())`**。假绿灯比没有保护更糟：上层会把警告
吞掉、界面显示一切正常，出问题时没人往这方向查。现在降级会写进状态栏
「内核级兜底未启用（…），仅靠退出时 kill + 下次启动认领」。

凭据 struct 带 `#[serde(default)]`：跨版本读写都不至于解析失败 —— 否则新版本加一个字段，
老凭据就整份读不出来、被当垃圾删掉，那次升级后的残留后端永远清不掉（表现为"端口莫名被占"）。
缺字段退化为 `created == 0` ⇒ 该条不认领，是安全的默认。

另一个顺序陷阱：`start()` 一进来先 `kill_child()` 做清场，如果它**无条件**删凭据，
就会把上次崩溃留下的证据抹掉 ⇒ 认领永远看不到东西（实测踩过）。所以只有"确实杀掉了
自己的子进程"才删。同理认领必须排在 `reuse_external` 探测**之前**，否则先被复用就再也清不掉。

## 3. 目标平台范围与体积上限

### 3.1 平台矩阵

| 平台 | 最低版本 | 架构 | WebView | 产物 |
|---|---|---|---|---|
| Windows | 10 1803（17134） | x64、ARM64 | WebView2 Runtime（Evergreen） | `.exe`（NSIS，currentUser 安装） |
| macOS | 11.0 Big Sur | x86_64、aarch64 | WKWebView（系统自带） | `.dmg`（按架构分别出，不打 universal 以省一半体积） |
| Linux | glibc 2.31（Ubuntu 20.04+/Debian 11+） | x86_64 | WebKitGTK 4.1 | `.AppImage`、`.deb` |

不支持：Win7/8（无 WebView2）、macOS 10.15 以下（Tauri 2 要求 11+）、Linux musl/Alpine（webkit2gtk 不可用）。

### 3.2 体积上限指标

上限 = 实测基线 × 约 1.6 的余量，写死进 `desktop/scripts/size-budget.mjs`。

| 检查项 | 上限 | 实测（2026-11-04 / Windows 10-11 x64） | 余量 |
|---|---|---|---|
| 壳二进制（strip + LTO + opt-level=s + panic=abort） | **6 MB** | `DramaStudio.exe` **3.35 MB** | 1.8× |
| 前端资源（未压缩） | **2 MB** | Vite SPA 产物 **1.02 MB** | 2.0× |
| 前端资源（gzip 传输量估算） | **0.75 MB** | 同上压缩后 **0.31 MB** | 2.4× |
| 后端源码（`dist-backend/backend-py`，进包源） | **12 MB** | 614 文件 **10.51 MB** | 1.1× |
| Windows NSIS 安装包 | **4.5 MB** | `DramaStudio_0.1.0_x64-setup.exe` **3.70 MB** | 1.2× |
| macOS DMG | **14 MB** | 未实测（需 macOS 主机 / CI runner） | — |
| Linux AppImage | **18 MB** | 未实测（需 webkit2gtk-4.1 环境） | — |
| Linux deb | **10 MB** | 未实测 | — |

实测口径：`npm --prefix desktop run build` 后跑 `node desktop/scripts/size-budget.mjs --strict`，五项全 OK、退出码 0。
首次编译耗时 1 分 23 秒（Rust release），NSIS 由 Tauri 自动下载。

**校准纪律**：第一次真机出包后，把实测值写回本节并在脚本里把上限收紧到 `实测 × 约 2`（既留成长余量、又不放跑体积漂移）；**只许收紧不许放宽**，放宽必须在本文留下理由。

**2026-11-02 的一次放宽（留档）**：NSIS 上限 `3 MB → 4.5 MB`。理由是功能面扩张（后端源码进包
`+2.2 MB`，实测 1.39 → 3.56 MB），按 `实测 × 约 1.25` 取整到 4.5 —— 不是"放宽到实测×2"，
而是把新增的真实负载记进预算；同时新增「后端源码 12 MB」一项，避免后端体积悄悄膨胀。macOS/Linux 三项在拿到真机产物前保持预估上限 —— 脚本按**实际存在的产物**逐项核对，某个平台没出包就不产生该行（`--strict` 只在「一项产物都没有」时才退 1）。

### 3.3 瘦身手段（按收益排序）

1. **不引插件**：`capabilities/default.json` 只有 `core:default`；文件/进程/对话框能力全用 `std::process` + 系统命令实现（`reqwest` 一项就 +2 MB）。
2. **`strip = true` + `lto = true` + `opt-level = "s"` + `panic = "abort"`**（已写进 `Cargo.toml`）：通常砍掉 40%–55%。
3. **前端按需**：`lucide-vue-next` 已是 per-icon 引入；不要引入整套 UI 框架。
4. **不打 universal 包**：macOS 分 x64/arm64 两次出包，体积减半。
5. **图标程序化生成**（`scripts/gen-icons.mjs`）：全套 6 个文件共 ~26 KB，不引入 sharp/CLI。

---

## 4. 架构与运行时契约

```
┌──────────────────────── Tauri 窗口 (WebView2 / WKWebView / WebKitGTK) ────────────────────────┐
│  Nuxt SPA（dist-frontend，hash 路由）                                                          │
│    utils/apiBase.ts      ← 唯一基址真源：Web 走 '/api/v1'，桌面走 http://127.0.0.1:<port>       │
│    composables/useDesktop.ts ← 6 个 IPC 命令 + bootstrap；Web 形态全部降级为"不可用"            │
└───────────────┬───────────────────────────────────────────────┬───────────────────────────────┘
                │ fetch / EventSource (HTTP, 跨源)               │ window.__TAURI_INTERNALS__.invoke
                ▼                                               ▼
        http://127.0.0.1:<port>                        Rust 外壳（desktop/src-tauri）
        FastAPI + 引擎                                   app_bootstrap / backend_start /
        （由壳 spawn，cwd=backend-py）                    backend_stop / backend_status /
                ▲                                        open_path / pick_directory
                └──────── spawn + 健康检查(GET /api/v1/health) ────────┘
```

**四条契约**（改任一条都要同步改另一端）：

1. **origin 注入**：壳在 `app_bootstrap` 里返回 `apiOrigin`，前端 `setApiOrigin()`；漏了 ⇒ 界面全空、控制台 404。
2. **CORS 注入**：壳 spawn 后端时设 `CORS_ORIGINS=tauri://localhost,http://tauri.localhost,...`；漏了 ⇒ 请求全被浏览器拦。
3. **命令名逐字对应**：`commands.rs` 的 6 个函数名 = `useDesktop.ts` 的 `DesktopCommand` 联合类型。
4. **路由模式**：桌面构建必须 `NUXT_PUBLIC_HASH_ROUTER=1`（`nuxt.config.ts` 的 `router.options.hashMode`）；否则 Ctrl+R/深链可能 404。

**启动时序**（这条最容易做错）：

```
窗口创建 → 插件 desktop-bridge（enforce:'pre', async）→ invoke app_bootstrap
        → 壳: 先探 5790..5809 有无活后端 → 有则 external（秒开）
                                        → 无则 pick 空闲端口 + spawn uvicorn → 立即返回 starting
        → 前端拿到 origin，正常渲染；DesktopStatusBanner 每 800ms 轮询 backend_status
        → 转 running 后横幅消失（首屏请求若已失败，用户点一次"重新启动"或自动重试）
```

**刻意不做**的事：首屏等待后端就绪（会让冷启动白屏 30–90 秒）；把后端打进包（体积失控）；在壳里实现业务逻辑（跨平台差异爆炸）。

---

## 5. 兼容性验证方式

### 5.1 三层验证

| 层 | 手段 | 通过标准 |
|---|---|---|
| 静态 | `npm run check`（= `gen-icons` + `cargo check`）、`npm --prefix frontend run typecheck` | 0 error |
| 体积 | `node desktop/scripts/size-budget.mjs --strict` | 全部 OK，退出码 0 |
| 真机冒烟 | §5.3 清单，每个平台至少跑一次 | 全部通过 |

### 5.2 CI 矩阵（建议）

```yaml
# 关键点：三平台都跑 cargo check（Linux 要装 webkit2gtk-4.1-dev + libgtk-3-dev）
matrix:
  os: [windows-latest, macos-latest, ubuntu-24.04]
steps:
  - npm ci && npm --prefix frontend ci
  - npm --prefix frontend run generate        # 桌面形态
  - node desktop/scripts/gen-icons.mjs
  - node desktop/scripts/sync-frontend.mjs
  - npm --prefix desktop run check
  - npm --prefix desktop run build            # 三平台出包
  - node desktop/scripts/size-budget.mjs --strict
```

### 5.3 真机冒烟清单（每平台逐项勾）

**启动与进程**
1. 冷启动：双击 → 窗口出现 ≤ 3 秒 → 后端 `starting` → 横幅消失、页面有数据（≤ 90 秒）。
2. 二次启动：**已经开着**再双击 → 不新增窗口、不新起 uvicorn，而是把已有窗口提到最前（单实例互斥体 `Local\com.darme.voide.desktop.single` + 激活事件）；只有前一个实例已完全退出时才重新 spawn 后端。
   注意「复用已有后端」（`state=external`）是**另一回事**：端口 5790 上已有别人起的 uvicorn 时，壳只连不管（`owned=false`），**按设计不拦关窗**——测「关窗确认」前必须先把外部后端停掉，否则会误判成确认框失效。
3. 后端崩溃：任务管理器杀掉 python → 横幅出现"已退出"+ 日志尾部 → 点"重新启动"能恢复。
4. 退出清理：关窗口 → `python` 进程消失（任务管理器确认），端口释放。

**WebView 与渲染**
5. 中文路径：把仓库放在 `D:\中文 目录\` 下再启动，页面与图片正常（测 URL 编码）。
6. DPI 缩放 200%：文字不糊、画布不错位；多显示器拖到不同 DPI 屏不炸。
7. 刷新：`Ctrl+R` 在 `/dramas/<id>` 深链上刷新 → 仍停留在原页面（验证 hash 路由）。
8. 大图渲染：打开含 2048×1536 PNG 的产出页，滚动不卡死（内存 < 1.5 GB）。

**系统集成**
9. 目录选择器：设置页选后端目录/输出目录 → 弹的是系统原生对话框；取消不报错。
10. 打开日志：点"查看日志" → 系统文件管理器定位到 `%APPDATA%\<identifier>\logs\backend-*.log`。
11. 离线：断网启动 → 不卡在"正在检查更新"（本壳无自动更新，验证无隐藏网络请求）。

**平台专项**
12. Windows：无 WebView2 的干净机器上，安装包能自动拉起 bootstrapper（或给出明确指引）。
13. macOS：Intel 与 Apple Silicon 各跑一次；首次打开（未签名/未公证）给出 Gatekeeper 说明。
14. Linux：Ubuntu 22.04 与 Fedora 40 各一次；确认 `zenity` 缺失时报错文案可读。

**已实测记录（2026-11-02 / Windows 10-11 x64 / WebView2）**

| 清单项 | 结论 | 证据 |
|---|---|---|
| 1 冷启动 | ✅ | 双击 `DramaStudio.exe` → 窗口标题「短剧工坊」8 秒内出现；后端 spawn 后 20 秒内 `/api/v1/health` 返回 `{"status":"ok"}` |
| 2 二次启动（单实例） | ✅ | 第一个实例在跑时再启动一次 → 系统只剩 **1 个** `DramaStudio.exe`、**1 个** uvicorn；第二实例自行退出 |
| 3 后端崩溃恢复 | ⬜ 未测 | 需在运行中杀 python 观察横幅与"重新启动" |
| 4 退出清理 | ✅ | 「是」退出后窗口进程与全部 `python` 子进程消失、5790 端口释放（无孤儿 uvicorn）；`backend.pid` 凭据同步删除 |
| 4b 强杀兜底 | ✅（仅 Windows） | `taskkill /F` 结束壳（不走任何退出路径）→ 4 秒内 uvicorn 消失、5790 释放（Job Object 连坐回收）；macOS/Linux 无此层，靠 4c |
| 4c 残留认领 | ✅（Windows 实测 / 逻辑层全平台） | 手工造一个占 5791 的 uvicorn 并伪造 `backend.pid` → 壳启动时认领并杀掉它，随后在 5790 起自己的，凭据改写为新 pid；身份不符（创建时间差 1）时**不杀**。单测覆盖：正向认领、反向不动手、`created==0` 不动手、`Match`/`Mismatch` 两态映射、凭据跨版本缺字段/多字段可解析 |
| 9 目录选择器 | ✅（逻辑层） | 进程外探针直接引用 `folder_picker.rs` 真调 `IFileOpenDialog`：对话框弹出、`WM_CLOSE` 取消 ⇒ `Ok(None)`、退出码 0；旧槽位（10/18/21）已被编译期断言拦下 |
| 5–14 其余 | ⬜ 未测 | 需中文路径 / 多 DPI / 无 WebView2 干净机 / macOS / Linux 环境 |

**另有两项「清单外但已实测」**（属于 6 项收尾的补丁，记录在此以免只存在于提交信息里）：

| 项 | 结论 | 证据 |
|---|---|---|
| 窗口状态记忆 | ✅ | `tauri.conf.json` 默认 1440×900，上次关窗存 `1200×760 @ (120,90)`；重启后客户区实测正好 **1200×760**、位置一致（`inner_size` 存 + `PhysicalSize` 还原；用 `outer_size` 会每关一次少一圈边框，`LogicalSize` 会在 125%/150% 缩放下被放大） |
| 关窗确认 | ✅ | 后端 `owned && running` 时送 `WM_CLOSE` → 出现 `#32770` 确认框；点「否」窗口留存、进程不退；点「是」才退出。后端不是自己起的（`external`）时**不拦** |

**本机能拿到的最强证据**（三条运行时契约同时成立）：WebView 首屏请求确实打到 `127.0.0.1:5790` —— 日志里 `visual-graph`、`visual-graph/guidance`、`storage/info`、`generations`、`app-settings` 全部 `200`，说明①前端资源加载成功、②origin 注入生效、③CORS 注入生效。

**路由模式的双向验证**：桌面形态产物里能搜到 `+= "#"`（`createWebHashHistory` 特征），Web 形态 `npm run generate` 产物里搜不到 ⇒ 两种形态确实按 `NUXT_PUBLIC_HASH_ROUTER` 分岔，不是"配了但没生效"。

### 5.4 版本探测（建议加进"关于"页）

- Windows：读 `HKLM\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}\pv`
- macOS：`WKWebView` 随系统，显示 `sw_vers -productVersion`
- Linux：`pkg-config --modversion webkit2gtk-4.1`

探测失败时**显示"未知"并允许继续**，不要因为它挡住启动。

---

## 6. 代码改动清单（Web 与桌面共用一套 UI）

### 6.1 新增

| 文件 | 职责 |
|---|---|
| `desktop/src-tauri/src/lib.rs` | 窗口、命令注册、退出清理、`CloseRequested` 勾子、单实例与状态恢复的接入点 |
| `desktop/src-tauri/src/backend.rs` | 端口探测、spawn、健康检查、watch、kill、`busy()`、`explain()`（日志尾部翻建议）、认领残留的接线 |
| `desktop/src-tauri/src/child_job.rs` | 子进程所有权：Job Object（`KILL_ON_JOB_CLOSE`，仅 Windows）+ `backend.pid` 凭据认领（认身份不认 pid，`Match`/`Mismatch`/`Unknown` 三态） |
| `desktop/src-tauri/src/commands.rs` | 6 个 IPC 命令 + 跨平台 `open_path` / `pick_directory` |
| `desktop/src-tauri/src/config.rs` | `desktop.json`、后端/Python 路径推断（含包内 `resources` 候选）、CORS origin 常量 |
| `desktop/src-tauri/src/single_instance.rs` | 命名互斥体 + 激活事件：第二个实例只负责把已有窗口提到最前，然后自己退出 |
| `desktop/src-tauri/src/window_state.rs` | 窗口尺寸/位置/最大化记忆（`config_dir/window.json`，存 `inner_size` 还原 `PhysicalSize`） |
| `desktop/src-tauri/src/dialog.rs` | 关窗确认（`MessageBoxW`，默认按钮是「否」） |
| `desktop/src-tauri/src/folder_picker.rs` | Windows 原生目录选择器（COM 直调 `IFileOpenDialog`，vtable 槽位带编译期断言） |
| `desktop/src-tauri/tauri.conf.json` | 窗口、CSP、三平台打包、`bundle.resources`（后端源码进包）与体积相关配置 |
| `desktop/src-tauri/capabilities/default.json` | 最小权限（只有 `core:default`） |
| `desktop/scripts/gen-icons.mjs` | 程序化生成 PNG/ICO/ICNS（无依赖） |
| `desktop/scripts/sync-frontend.mjs` | Nuxt 产物 → `dist-frontend`，报告体积 |
| `desktop/scripts/sync-backend.mjs` | 后端源码 → `dist-backend/backend-py`（增量拷贝 + `.manifest.json` 差集清理，报告文件数/字节） |
| `desktop/scripts/size-budget.mjs` | 体积预算闸门（5 项：壳/前端/gz/后端源码/NSIS） |
| `frontend/app/utils/apiBase.ts` | API 基址唯一真源（可注入） |
| `frontend/app/composables/useDesktop.ts` | 平台桥：`isDesktop` / 6 命令 / bootstrap |
| `frontend/app/plugins/desktop-bridge.client.ts` | 首屏渲染前完成 origin 注入 |
| `frontend/app/components/DesktopStatusBanner.vue` | 后端未就绪提示 + 重启入口 |

### 6.2 重构（都只动一处口径，不改行为）

| 文件 | 改动 | 为什么必须改 |
|---|---|---|
| `frontend/app/composables/useApi.ts` | 删除 `const BASE = '/api/v1'`，7 处 `${BASE}` → `${apiBase()}` | 桌面端跨源，相对路径必 404 |
| `frontend/app/composables/useComfyRun.ts` | 同上（1 处 `EventSource`） | SSE 也会打到壳的协议栈上 |
| `frontend/nuxt.config.ts` | 新增 `router.options.hashMode`（由 `NUXT_PUBLIC_HASH_ROUTER` 控制） | 桌面端刷新/深链 404 |
| `frontend/app/app.vue` | 挂 `<DesktopStatusBanner />` | 后端失败要可见可恢复 |

### 6.3 Web 与桌面差异（逐项）

| 维度 | Web | 桌面 |
|---|---|---|
| API 基址 | `/api/v1`（同源，dev 由 vite proxy） | `http://127.0.0.1:<port>/api/v1`（壳注入） |
| 路由 | history | hash（`#/dramas/123`） |
| 后端来源 | 部署侧/用户自起 | 壳托管（复用或 spawn） |
| 系统能力 | 浏览器下载/上传 | 原生目录选择、资源管理器定位、日志直达 |
| 失败可见性 | 网络错误提示 | 横幅 + 日志路径 + 一键重启 |
| 更新 | 部署即更新 | 无自动更新（本次不引入 updater 插件，体积优先） |

### 6.4 完成标准（DoD）

- [x] `npm --prefix frontend run generate` 通过（`typeCheck: true` 下 0 error）—— 2026-11-02 实测（并顺带修好仓库既有的 `CanvasCropDialog.vue` 缺 `<script setup>` 与 utils 相对路径两处构建期错误）
- [x] `npm --prefix desktop run check` 通过（`cargo check` 0 error）
- [ ] 三平台均能出包 —— **Windows 已出包**（`DramaStudio_0.1.0_x64-setup.exe` 3.70 MB），macOS/Linux 待真机
- [x] Windows 的 `size-budget --strict` 全绿（壳 3.35 / NSIS 3.70 / 前端 1.02 / gz 0.31 / 后端源码 10.51 MB，上限已按实测收紧；2026-11-04 复测，含把 `_bench` 2.3 MB 从进包源排除）
- [ ] §5.3 冒烟清单每平台逐项通过并留记录 —— Windows 的「冷启动」「单实例」「退出清理」「强杀兜底」「残留认领」「关窗确认」「窗口记忆」「目录选择器」已过，其余见 §5.3 实测表
- [x] 壳被强杀不残留后端（Windows）：Job Object 兜底 + `backend.pid` 认领，`cargo test --lib child_job` **9** 条（正向认领 / 身份不符不动手 / 缺时间戳不动手 / 两态映射 / 凭据跨版本缺字段与多字段）。macOS 无 procfs 只能提示，Linux 有 procfs 可认领 —— 差异与降级文案见 §2.4「平台差异」
- [x] Web 形态回归：`npm run generate`（不带 `NUXT_PUBLIC_HASH_ROUTER`）通过，产物无 hash 路由特征 ⇒ `/api/v1` 仍走同源相对路径（`npm run dev` 交互走查未做）
- [ ] 首次运行无后端时，界面给出**可执行**的修复指引（`explain()` 已按四类日志翻建议；「删掉 backend-py」场景尚未实测）
- [ ] release 二进制无构建机绝对路径（已实测：`D:\code` / `CARGO_MANIFEST_DIR` 均搜不到，仅剩 crate 名与产品描述）

---

## 7. 快速开始

```bash
# 1) 构建前端（桌面形态：hash 路由）并同步进壳
cd desktop
node scripts/gen-icons.mjs
node scripts/sync-frontend.mjs --build
node scripts/sync-backend.mjs        # 后端源码 → dist-backend/backend-py（进包源）
                                     # 注：predev / prebuild 已自动串上，手动跑只为看报告

# 2) 编译/运行（需 Rust 工具链；首次会拉 crate）
npm run dev            # 开发：另开一个终端跑 npm --prefix ../frontend run dev，然后 tauri dev 指向 3013
npm run build          # 出包：产物在 src-tauri/target/release/bundle/

# 3) 体积闸门
node scripts/size-budget.mjs --strict
```

**开发模式注意**：`tauri dev` 用 `devUrl = http://localhost:3013`，需要先起 Nuxt dev server（`npm run dev:web`）。此时前端是 history 路由，由 vite 提供 SPA fallback，行为与 Web 一致。

**后端怎么来的**：壳按 `backendDir`/`pythonPath`（`desktop.json`）→ `DARME_BACKEND_DIR`/`DARME_PYTHON` → 包内 `resources/backend-py` → exe 祖先目录的 `backend-py` →（仅 debug）仓库 `backend-py` → cwd 的顺序找目录；解释器同理（最后回落 PATH）。找不到会在界面报明确原因。**直接从 `target/release` 裸跑 release exe 时**，靠 exe 祖先目录兜到仓库后端 —— 这是唯一"开发便利"，装到别处就只能靠包内 `resources`。
