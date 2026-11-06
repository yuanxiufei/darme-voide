//! 壳配置与路径推断。
//!
//! 设计原则：**一切都能被外部覆盖，且失败时给出可执行的下一步**。
//! 用户可见配置只有一个小 JSON（`<appConfigDir>/desktop.json`），不需要进图形界面就能排障。

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// 允许访问后端的 WebView origin 白名单（写进后端 `CORS_ORIGINS`）。
///
/// - `tauri://localhost`：macOS / Linux 的自定义协议 origin
/// - `http(s)://tauri.localhost`：Windows 是 `http://`，显式开 `useHttpsScheme` 时是 `https://`
/// - `localhost:3013`：Web 端 `nuxt dev`，浏览器直连后端调试时用到
/// - `localhost:3014`：桌面端 `vite dev` ⇒ **`tauri dev` 时 WebView 的真实 origin**
/// - `127.0.0.1:5790`：同一个后端被浏览器直连的调试场景
///
/// ⚠️ 后端 `core/config.py::_cors_origins()` 的**默认值不含**这几个 origin，
///    所以本壳启动后端时**必须显式注入** `CORS_ORIGINS`；漏了的表现是界面全空、
///    控制台一片 CORS 报错，但 `curl 127.0.0.1:5790/api/v1/health` 是通的。
///
/// ⚠️ **3014 必须与 `tauri.conf.json` 的 `devUrl` 同步**：`tauri dev` 下 WebView
///    加载的就是 `http://localhost:3014`，它不在白名单 ⇒ 每个请求都被 CORS 挡掉。
///    这条由 `tests::dev_origin_is_allowed` 守着（读 `tauri.conf.json` 逐字比对）。
pub const DESKTOP_ORIGINS: &str =
    "tauri://localhost,http://tauri.localhost,https://tauri.localhost,http://localhost:3013,http://localhost:3014,http://127.0.0.1:5790";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct DesktopConfig {
    /// 后端目录（含 `app/main.py`）；留空 = 自动推断
    pub backend_dir: Option<String>,
    /// Python 解释器绝对路径；留空 = 自动探测（优先后端的 `.venv`）
    pub python_path: Option<String>,
    /// 首选端口；被占用时向后顺延（最多 20 个）
    pub prefer_port: u16,
    /// 先探测外部已在跑的后端并复用（二次启动秒开的关键）
    pub reuse_external: bool,
    /// 健康检查最长等待（毫秒）。超时不算失败——转 `starting` 继续在后台等
    pub startup_timeout_ms: u64,
    /// 交给后端的额外环境变量（如 `HTTP_LOG=0` 降噪）
    pub extra_env: BTreeMap<String, String>,
}

impl Default for DesktopConfig {
    fn default() -> Self {
        Self {
            backend_dir: None,
            python_path: None,
            prefer_port: 5790,
            reuse_external: true,
            startup_timeout_ms: 45_000,
            extra_env: BTreeMap::new(),
        }
    }
}

impl DesktopConfig {
    pub fn load(config_dir: &Path) -> Self {
        let path = config_dir.join("desktop.json");
        let Ok(raw) = std::fs::read_to_string(&path) else {
            // 首次运行：落一份默认配置，用户照着改就行（比"什么都看不到"友好）
            let cfg = Self::default();
            let _ = std::fs::create_dir_all(config_dir);
            if let Ok(text) = serde_json::to_string_pretty(&cfg) {
                let _ = std::fs::write(&path, text);
            }
            return cfg;
        };
        // 配置写坏时**不 panic**：退回默认值并在 stderr 留下线索
        match serde_json::from_str::<Self>(&raw) {
            Ok(cfg) => cfg,
            Err(err) => {
                eprintln!("[desktop] desktop.json 解析失败，已退回默认配置：{err}");
                Self::default()
            }
        }
    }
}

/// 从可执行文件所在目录向上找工作区根（开发期是 `target/debug/` ⇒ 上溯 4 层到仓库根）。
fn ancestors_from_exe() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        let mut cur = exe.parent().map(|p| p.to_path_buf());
        while let Some(dir) = cur {
            out.push(dir.clone());
            cur = dir.parent().map(|p| p.to_path_buf());
        }
    }
    out
}

/// 定位后端目录（含 `app/main.py` 的那一层）。
///
/// `packaged` 是**安装后**才存在的候选（由调用方用 `resource_dir()` 拼出来）。
/// 顺序上它排在「向上找仓库」之前：装到别人机器上，只有这一条能命中 ——
/// 这也是「装完打不开」的第一根因（旧版只有仓库布局，必然报找不到后端）。
pub fn resolve_backend_dir(cfg: &DesktopConfig, packaged: &[PathBuf]) -> Result<PathBuf, String> {
    let mut candidates: Vec<PathBuf> = Vec::new();

    if let Some(dir) = cfg.backend_dir.as_deref().filter(|s| !s.trim().is_empty()) {
        candidates.push(PathBuf::from(dir));
    }
    if let Ok(dir) = std::env::var("DARME_BACKEND_DIR") {
        if !dir.trim().is_empty() {
            candidates.push(PathBuf::from(dir));
        }
    }
    candidates.extend(packaged.iter().cloned());
    for dir in ancestors_from_exe() {
        candidates.push(dir.join("backend-py"));
        candidates.push(dir.join("resources").join("backend-py"));
    }
    // 开发期：编译期写死仓库路径。
    // ⚠️ 必须只在 debug 生效 —— `env!` 在 release 里同样被求值，会把构建机的绝对路径
    //    （如 `D:\code\...\backend-py`）写进二进制，既泄露本机路径，又给安装版留一条死候选。
    #[cfg(debug_assertions)]
    {
        candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../backend-py"));
        candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../backend-py"));
    }
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join("backend-py"));
    }

    for candidate in &candidates {
        let probe = candidate.join("app").join("main.py");
        if probe.is_file() {
            return Ok(candidate.canonicalize().unwrap_or_else(|_| candidate.clone()));
        }
    }
    let tried = candidates
        .iter()
        .take(6)
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join("；");
    Err(format!(
        "找不到后端目录（需含 app/main.py）。已尝试：{tried}。\
         若这是安装版，说明打包时漏了 resources/backend-py（重新跑 sync-backend.mjs 再打包）；\
         否则请在 desktop.json 里设置 backendDir，或设置环境变量 DARME_BACKEND_DIR。"
    ))
}

/// 定位 Python 解释器：显式配置 → 环境变量 → 后端自带 venv → 打包布局 → PATH。
pub fn resolve_python(
    cfg: &DesktopConfig,
    backend_dir: &Path,
    packaged: &[PathBuf],
) -> Result<PathBuf, String> {
    let mut candidates: Vec<PathBuf> = Vec::new();

    if let Some(p) = cfg.python_path.as_deref().filter(|s| !s.trim().is_empty()) {
        candidates.push(PathBuf::from(p));
    }
    if let Ok(p) = std::env::var("DARME_PYTHON") {
        if !p.trim().is_empty() {
            candidates.push(PathBuf::from(p));
        }
    }
    // 后端自带的 venv（本仓开发期：backend-py/.venv/Scripts/python.exe）
    candidates.push(backend_dir.join(".venv").join("Scripts").join("python.exe"));
    candidates.push(backend_dir.join(".venv").join("bin").join("python"));
    // 打包布局：将来把可重定位解释器放进 resources/runtime 时走这两条
    for base in packaged {
        candidates.push(base.join(".venv").join("Scripts").join("python.exe"));
        candidates.push(base.join(".venv").join("bin").join("python"));
        if let Some(parent) = base.parent() {
            candidates.push(parent.join("runtime").join("python.exe"));
            candidates.push(parent.join("runtime").join("bin").join("python3"));
        }
    }
    // 开发期仓库根 venv。同样只在 debug 生效（理由见 resolve_backend_dir）
    #[cfg(debug_assertions)]
    candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.venv/Scripts/python.exe"));
    candidates.push(PathBuf::from("python"));
    candidates.push(PathBuf::from("python3"));

    for candidate in candidates {
        // 裸命令名（python / python3）交给 PATH 解析，file 不存在也算候选
        if candidate.components().count() == 1 || candidate.is_file() {
            return Ok(candidate);
        }
    }
    Err("找不到可用的 Python 解释器。安装版应自带运行时（resources/runtime）；\
         若用系统 Python，请安装 3.11+ 并在 desktop.json 里设置 pythonPath，\
         或设置环境变量 DARME_PYTHON。"
        .to_string())
}

/// 日志目录（与配置目录分离，避免用户误删配置）。
pub fn log_dir(app: &tauri::AppHandle) -> PathBuf {
    use tauri::Manager;
    app.path()
        .app_log_dir()
        .unwrap_or_else(|_| std::env::temp_dir().join("drama-studio-logs"))
}

pub fn config_dir(app: &tauri::AppHandle) -> PathBuf {
    use tauri::Manager;
    app.path()
        .app_config_dir()
        .unwrap_or_else(|_| std::env::temp_dir().join("drama-studio-config"))
}

/// Unix 秒 → `YYYYMMDD-HHMMSS`（不引 chrono：只为给日志文件起个能排序的名字）。
pub fn stamp() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (y, m, d) = civil_from_days(days);
    format!(
        "{y:04}{m:02}{d:02}-{:02}{:02}{:02}",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

/// Howard Hinnant 的 days → civil date 算法（公历，含闰年）。
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `tauri dev` 的窗口加载 `devUrl`，它的 origin 必须出现在 CORS 白名单里。
    ///
    /// 2026-11-04 改端口时真漏过一次：`devUrl` 从 3013 改成 3014，白名单还停在
    /// 3013 ⇒ 壳自己拉后端时注入的 `CORS_ORIGINS` 不含 3014，WebView 每个请求
    /// 都被挡掉、界面全空，而 `curl 127.0.0.1:5790/api/v1/health` 却是通的
    /// （最容易误判成"后端没起来"）。
    ///
    /// 判据读 `tauri.conf.json` 的**实际值**而不是再抄一遍端口号：两处只要漂移就报。
    #[test]
    fn dev_origin_is_allowed() {
        let conf: serde_json::Value =
            serde_json::from_str(include_str!("../tauri.conf.json"))
                .expect("tauri.conf.json 不是合法 JSON");
        let dev_url = conf["build"]["devUrl"]
            .as_str()
            .expect("tauri.conf.json 缺 build.devUrl");

        let host_port = dev_url
            .split("//")
            .nth(1)
            .expect("devUrl 不是 http(s) URL")
            .trim_end_matches('/');
        let origin = format!("http://{host_port}");

        assert!(
            DESKTOP_ORIGINS.split(',').any(|o| o == origin),
            "tauri.conf.json 的 devUrl 是 {dev_url}，但 DESKTOP_ORIGINS 里没有 {origin}。\n\
             `tauri dev` 下 WebView 的 origin 就是它，漏了会让所有 API 请求被 CORS 挡掉。\n\
             现白名单：{DESKTOP_ORIGINS}"
        );
    }
}
