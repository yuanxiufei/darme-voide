//! 本地 Python 后端的生命周期管理（探测 / 启动 / 健康检查 / 停机）。
//!
//! ## 为什么不用 `tauri-plugin-shell` 或 `reqwest`
//!
//! 体积是硬指标：这几件事用 `std::process` + `std::net` 就够，引一整套 HTTP 客户端与插件
//! 会给安装包加几 MB，不值得。健康检查只需要读 `HTTP/1.1 200` 这一行。
//!
//! ## 三条必须守住的行为
//!
//! 1. **先探活再拉起**：二次启动（或用户已用 `npm run dev` 起着后端）时必须复用，
//!    否则端口冲突、两套 SQLite 同时写。
//! 2. **绝不阻塞首屏**：`start()` 立刻返回 `starting`，真正就绪由后台线程轮询后改状态；
//!    前端 `DesktopStatusBanner` 轮询到 `running` 再刷新一次页面。
//! 3. **退出必杀**：壳关闭时 kill 子进程，否则下次启动会看到一堆孤儿 uvicorn。

use crate::child_job;
use crate::config::{self, DesktopConfig};
use serde::Serialize;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BackendStatus {
    /// `stopped` / `starting` / `running` / `failed` / `external`
    pub state: String,
    pub origin: Option<String>,
    pub port: Option<u16>,
    pub pid: Option<u32>,
    /// 是否由本壳拉起（`false` = 复用外部已在跑的后端）
    pub owned: bool,
    pub message: String,
    pub log_path: Option<String>,
}

impl BackendStatus {
    pub fn stopped(message: impl Into<String>) -> Self {
        Self {
            state: "stopped".into(),
            origin: None,
            port: None,
            pid: None,
            owned: false,
            message: message.into(),
            log_path: None,
        }
    }
}

impl Default for BackendStatus {
    fn default() -> Self {
        Self::stopped("尚未启动")
    }
}

pub struct AppState {
    pub status: Mutex<BackendStatus>,
    pub child: Mutex<Option<Child>>,
    pub config: Mutex<DesktopConfig>,
    pub log_dir: Mutex<PathBuf>,
    /// `<appConfigDir>`：`backend.pid` 凭据落在这里（`kill_child` 要靠它删凭据）
    pub config_dir: Mutex<PathBuf>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            status: Mutex::new(BackendStatus::default()),
            child: Mutex::new(None),
            config: Mutex::new(DesktopConfig::default()),
            log_dir: Mutex::new(std::env::temp_dir().join("drama-studio-logs")),
            config_dir: Mutex::new(std::env::temp_dir().join("drama-studio-config")),
        }
    }
}

fn origin_of(port: u16) -> String {
    format!("http://127.0.0.1:{port}")
}

/// 只有**安装后**才存在的后端候选位置（开发期这些目录不存在，命中不了就跳过）。
///
/// Windows NSIS 把 `resources` 铺在 exe 同级，macOS 在 `Contents/Resources`，
/// Linux 在 `/usr/lib/<app>` —— 三种布局都由 `resource_dir()` 统一回答，所以不自己拼。
fn packaged_dirs(app: &tauri::AppHandle) -> Vec<PathBuf> {
    use tauri::Manager;
    let mut out: Vec<PathBuf> = Vec::new();
    let mut push = |p: PathBuf| {
        if !out.contains(&p) {
            out.push(p);
        }
    };
    if let Ok(dir) = app.path().resource_dir() {
        push(dir.join("backend-py"));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            push(parent.join("backend-py"));
        }
    }
    out
}

/// 单次健康探活：连得上且首行是 200 才算活着（只有 TCP 连通不算 —— uvicorn 起来了但
/// `app.main` 导入失败时会立刻退出，端口不会留在监听状态，但代理/半开连接仍可能连上）。
pub fn health_ok(port: u16, timeout: Duration) -> bool {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let Ok(mut stream) = TcpStream::connect_timeout(&addr, timeout) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(900)));
    let _ = stream.set_write_timeout(Some(Duration::from_millis(900)));
    let req = b"GET /api/v1/health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
    if stream.write_all(req).is_err() {
        return false;
    }
    let mut buf = [0u8; 64];
    match stream.read(&mut buf) {
        Ok(n) if n > 12 => String::from_utf8_lossy(&buf[..n]).contains(" 200"),
        _ => false,
    }
}

/// 在 `prefer..prefer+20` 里挑一个能 bind 的端口（探测完立即释放，交给 uvicorn）。
fn pick_port(prefer: u16) -> Option<u16> {
    for offset in 0..20u16 {
        let port = prefer.saturating_add(offset);
        if TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return Some(port);
        }
    }
    None
}

fn tail(path: &Path, max_lines: usize) -> String {
    let Ok(bytes) = std::fs::read(path) else {
        return String::new();
    };
    let start = bytes.len().saturating_sub(4096);
    let text = String::from_utf8_lossy(&bytes[start..]).to_string();
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let from = lines.len().saturating_sub(max_lines);
    lines[from..].join("\n")
}

/// 把日志尾部翻成一句可执行的建议。
///
/// 装到别人机器上，「本地服务已退出」这句话本身没有信息量 —— 用户不知道该装什么。
/// 这里只覆盖实测出现过的四种死法，认不出来就返回 `None`（不编原因）。
fn explain(log_tail: &str) -> Option<&'static str> {
    let t = log_tail.to_ascii_lowercase();
    if t.contains("modulenotfounderror") || t.contains("cannot import name") {
        Some(
            "缺少 Python 依赖。开发机：在 backend-py 下执行 \
             `.venv\\Scripts\\pip install -r requirements.txt`；\
             安装版：检查打包是否带上了 resources/runtime（可重定位解释器）。",
        )
    } else if t.contains("address already in use") || t.contains("errno 10048") {
        Some("端口被占用。关掉占用该端口的进程，或在 desktop.json 里调高 preferPort。")
    } else if t.contains("unicodeencodeerror") || t.contains("codec can't encode") {
        Some("控制台编码问题。本壳已注入 PYTHONUTF8=1，请检查 backend-py/.env 是否把它覆盖了。")
    } else if t.contains("permissionerror") || t.contains("permission denied") {
        Some(
            "目录无写权限。装到 Program Files 下时数据库与日志写不进去，\
             请改装到当前用户目录（NSIS 默认 installMode=currentUser）。",
        )
    } else {
        None
    }
}

/// 起后端。**立刻返回**（`starting`），就绪由 `watch()` 在后台确认。
pub fn start(app: &tauri::AppHandle, state: &AppState) -> BackendStatus {
    {
        let st = state.status.lock().unwrap();
        if matches!(st.state.as_str(), "running" | "starting" | "external") {
            return st.clone();
        }
    }
    kill_child(state);

    // 0) 认领上次**被强杀**留下的自家后端（Job Object 生效时本该为空；这层兜旧版本/兜 Job 失效）。
    //    顺序不能挪到外部探测之后：那样它会先被当成"外部后端"复用，等于永远清不掉。
    let config_dir = state.config_dir.lock().unwrap().clone();
    // 攒"这次启动做了什么"的说明，最后拼进 status.message —— 否则平台降级
    // （内核级兜底未启用、残留无法识别）只在 eprintln 里，界面上看不见。
    let mut note = String::new();
    match child_job::reclaim(&config_dir) {
        child_job::Reclaim::Killed(old) => {
            // 杀掉 ≠ 端口立刻松开：不等就会在下面第 1 步探到它还活着
            if !child_job::wait_port_free(old.port, Duration::from_secs(5)) {
                eprintln!(
                    "[desktop] 已结束残留服务 pid {}，但端口 {} 仍被占用",
                    old.pid, old.port
                );
            }
            note.push_str(&format!("已清理上次残留的服务（pid {}）。", old.pid));
        }
        child_job::Reclaim::Clean => {}
        child_job::Reclaim::Unverified(msg) => note.push_str(&format!("{msg} ")),
    }

    let cfg = state.config.lock().unwrap().clone();

    let packaged = packaged_dirs(app);
    let backend_dir = match config::resolve_backend_dir(&cfg, &packaged) {
        Ok(dir) => dir,
        Err(err) => return fail(state, err),
    };
    let python = match config::resolve_python(&cfg, &backend_dir, &packaged) {
        Ok(p) => p,
        Err(err) => return fail(state, err),
    };

    // 1) 复用已在跑的后端（二次启动秒开；也避免两个进程写同一个 SQLite）
    if cfg.reuse_external {
        for offset in 0..20u16 {
            let port = cfg.prefer_port.saturating_add(offset);
            if health_ok(port, Duration::from_millis(250)) {
                let st = BackendStatus {
                    state: "external".into(),
                    origin: Some(origin_of(port)),
                    port: Some(port),
                    pid: None,
                    owned: false,
                    // 「外部启动」要说出来：那种情况下退出不会关它、关窗也不弹确认，
                    // 用户看到"关窗没提示"时得有据可查（本仓实测踩过一次这个坑）
                    message: format!("复用已在运行的本地服务（127.0.0.1:{port}，外部启动，退出时不关闭）"),
                    log_path: None,
                };
                *state.status.lock().unwrap() = st.clone();
                return st;
            }
        }
    }

    // 2) 挑端口
    let Some(port) = pick_port(cfg.prefer_port) else {
        return fail(
            state,
            format!(
                "端口 {}~{} 全被占用，请关闭占用进程或在 desktop.json 里改 preferPort。",
                cfg.prefer_port,
                cfg.prefer_port.saturating_add(19)
            ),
        );
    };

    // 3) 准备日志文件
    let log_dir = state.log_dir.lock().unwrap().clone();
    let _ = std::fs::create_dir_all(&log_dir);
    let log_path = log_dir.join(format!("backend-{}.log", config::stamp()));
    let Ok(log_file) = std::fs::File::create(&log_path) else {
        return fail(state, format!("无法创建日志文件：{}", log_path.display()));
    };
    let log_err = match log_file.try_clone() {
        Ok(f) => f,
        Err(err) => return fail(state, format!("日志句柄复制失败：{err}")),
    };

    // 4) 组装命令
    let mut cmd = Command::new(&python);
    cmd.arg("-m")
        .arg("uvicorn")
        .arg("app.main:app")
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .arg("--no-access-log") // 访问日志量太大，排障看异常即可
        .current_dir(&backend_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(log_err))
        // ⚠️ CORS_ORIGINS 必须注入：后端默认白名单不含 tauri://localhost ⇒ 漏了界面全空
        .env("CORS_ORIGINS", config::DESKTOP_ORIGINS)
        .env("PY_PORT", port.to_string())
        .env("HOST", "127.0.0.1")
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONUTF8", "1")
        .env("PYTHONIOENCODING", "utf-8");
    for (k, v) in &cfg.extra_env {
        cmd.env(k, v);
    }

    // Windows：不要弹出黑窗口
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    let child = match cmd.spawn() {
        Ok(c) => c,
        Err(err) => {
            return fail(
                state,
                format!(
                    "启动失败：{}（解释器 {}）。请检查 Python 与依赖是否已安装。",
                    err,
                    python.display()
                ),
            )
        }
    };

    let pid = child.id();
    *state.child.lock().unwrap() = Some(child);

    // 纳入 Job Object（壳被强杀时由内核回收整棵子进程树）并落凭据（下次启动能认领）
    if let Some(warn) = child_job::adopt(&config_dir, pid, port, &backend_dir, &python) {
        eprintln!("[desktop] {warn}");
        note.push_str(&format!("{warn}。"));
    }

    let st = BackendStatus {
        state: "starting".into(),
        origin: Some(origin_of(port)),
        port: Some(port),
        pid: Some(pid),
        owned: true,
        message: format!("{note}正在启动本地服务（127.0.0.1:{port}，pid {pid}）"),
        log_path: Some(log_path.to_string_lossy().to_string()),
    };
    *state.status.lock().unwrap() = st.clone();

    watch(app.clone(), port, cfg.startup_timeout_ms.max(15_000), log_path, 0);
    st
}

fn fail(state: &AppState, message: String) -> BackendStatus {
    let st = BackendStatus {
        state: "failed".into(),
        origin: None,
        port: None,
        pid: None,
        owned: false,
        message,
        log_path: None,
    };
    *state.status.lock().unwrap() = st.clone();
    st
}

/// 后台确认就绪。`round` 是**已等待轮次**，上限 3 轮（默认 45s×3 = 135s）——
/// 故意有上限：不设上限就会在"进程卡死不退出"时无限轮询（症状是状态永远 `starting`，
/// 用户也看不出到底在等什么）。
fn watch(app: tauri::AppHandle, port: u16, timeout_ms: u64, log_path: PathBuf, round: u8) {
    use tauri::Manager;
    std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_millis(timeout_ms);
        loop {
            if health_ok(port, Duration::from_millis(400)) {
                let state = app.state::<AppState>();
                let mut st = state.status.lock().unwrap();
                if st.port == Some(port) && st.state == "starting" {
                    st.state = "running".into();
                    st.message = format!("本地服务已就绪（127.0.0.1:{port}）");
                }
                return;
            }
            // 子进程自己退出了就别再等满超时（Python 缺依赖时是这种死法）
            {
                let state = app.state::<AppState>();
                let mut guard = state.child.lock().unwrap();
                if let Some(child) = guard.as_mut() {
                    if let Ok(Some(exit)) = child.try_wait() {
                        let detail = tail(&log_path, 12);
                        let mut st = state.status.lock().unwrap();
                        if st.port == Some(port) {
                            st.state = "failed".into();
                            st.message = format!(
                                "本地服务已退出（{exit}）。{}日志尾部：\n{}",
                                explain(&detail)
                                    .map(|h| format!("建议：{h}\n"))
                                    .unwrap_or_default(),
                                if detail.is_empty() { "（无输出）".to_string() } else { detail }
                            );
                        }
                        return;
                    }
                }
            }
            if Instant::now() >= deadline {
                let state = app.state::<AppState>();
                let detail = tail(&log_path, 12);
                let mut st = state.status.lock().unwrap();
                if st.port == Some(port) && st.state == "starting" {
                    // 超时不算"失败"：慢机器上 torch 导入可能要一两分钟，继续等下一次
                    st.message = format!(
                        "启动较慢，仍在等待…（日志尾部）\n{}",
                        if detail.is_empty() { "（无输出）".to_string() } else { detail.clone() }
                    );
                    drop(st);
                    if round < 2 {
                        // 再给一轮完整超时（慢机器的 torch 冷导入可能一分钟以上）
                        watch(app.clone(), port, timeout_ms.max(30_000), log_path.clone(), round + 1);
                    } else {
                        let mut st = state.status.lock().unwrap();
                        if st.port == Some(port) && st.state == "starting" {
                            st.state = "failed".into();
                            st.message = format!(
                                "等待超时（已等 {} 秒）。日志尾部：\n{}",
                                (u64::from(round) + 1) * timeout_ms / 1000,
                                if detail.is_empty() { "（无输出）".to_string() } else { detail }
                            );
                        }
                    }
                }
                return;
            }
            std::thread::sleep(Duration::from_millis(400));
        }
    });
}

pub fn kill_child(state: &AppState) {
    let killed = {
        let mut guard = state.child.lock().unwrap();
        match guard.take() {
            Some(mut child) => {
                let _ = child.kill();
                let _ = child.wait();
                true
            }
            None => false,
        }
    };
    // 只有"确实杀掉了自己的子进程"才删凭据。
    // `start()` 一进来就调 kill_child 做清场，此时手上并没有子进程；无条件删会把
    // **上次崩溃留下的凭据**抹掉，等于认领逻辑永远看不到证据（实测踩过）。
    if killed {
        child_job::release(&state.config_dir.lock().unwrap());
    }
}

/// 停止后端（只停自己起的；外部复用的一律不动 —— 那可能是用户终端里的 dev server）。
pub fn stop(state: &AppState) -> BackendStatus {
    let owned = state.status.lock().unwrap().owned;
    if !owned {
        kill_child(state);
        let st = BackendStatus::stopped("外部后端不受桌面端管理，未停止");
        *state.status.lock().unwrap() = st.clone();
        return st;
    }
    kill_child(state);
    let st = BackendStatus::stopped("本地服务已停止");
    *state.status.lock().unwrap() = st.clone();
    st
}

/// 是否有**本壳拉起**的后端正在跑（关窗确认用它决定要不要拦）。
///
/// `external` 不算：那是用户自己在终端里起的服务，关掉桌面端不该拦他。
pub fn busy(app: &tauri::AppHandle) -> bool {
    use tauri::Manager;
    let state = app.state::<AppState>();
    let st = current(&state);
    st.owned && matches!(st.state.as_str(), "running" | "starting")
}

/// 读当前状态并顺手校正（子进程偷偷退出时不能一直显示 `running`）。
pub fn current(state: &AppState) -> BackendStatus {
    let mut st = state.status.lock().unwrap().clone();
    if st.owned {
        if let Some(child) = state.child.lock().unwrap().as_mut() {
            match child.try_wait() {
                Ok(Some(exit)) => {
                    st.state = "failed".into();
                    st.message = format!("本地服务已退出（{exit}）");
                }
                Ok(None) => {}
                Err(err) => {
                    st.state = "failed".into();
                    st.message = format!("无法读取本地服务状态：{err}");
                }
            }
        } else if st.state == "running" {
            st.state = "failed".into();
            st.message = "本地服务进程句柄丢失（可能已被外部结束）".into();
        }
        if st.state == "failed" {
            *state.status.lock().unwrap() = st.clone();
        }
    }
    st
}
