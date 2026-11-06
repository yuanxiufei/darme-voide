//! 暴露给渲染层的 6 个命令（与前端 `composables/useDesktop.ts` 的 `DesktopCommand` 逐字对应）。
//!
//! 刻意**只开 6 个**：每一个都会扩大攻击面，而且 `capabilities/default.json` 里除了
//! `core:default` 什么都没给。需要文件系统能力时请在这里加**窄接口**（带路径校验），
//! 不要引入 `tauri-plugin-fs` 然后开 `fs:allow-*`。

use crate::backend::{self, AppState, BackendStatus};
use serde::Serialize;
use std::path::Path;
use std::process::Command;
use tauri::State;

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Bootstrap {
    /// 后端 origin，如 `http://127.0.0.1:5790`；`null` 表示启动失败
    pub api_origin: Option<String>,
    pub platform: String,
    pub app_version: String,
    pub data_dir: String,
    pub backend: BackendStatus,
}

/// 首屏调用：确保后端在启动中，并把绝对 origin 交给前端。
///
/// ⚠️ **不等待就绪**（`start` 立即返回 `starting`）—— 首屏不该被 Python 冷启动卡住；
/// 就绪信号由 `backend_status` 轮询拿到，前端在转 `running` 后刷新一次。
#[tauri::command]
pub fn app_bootstrap(app: tauri::AppHandle, state: State<'_, AppState>) -> Bootstrap {
    let st = backend::current(&state);
    let st = if matches!(st.state.as_str(), "running" | "starting" | "external") {
        st
    } else {
        backend::start(&app, &state)
    };
    Bootstrap {
        api_origin: st.origin.clone(),
        platform: std::env::consts::OS.to_string(),
        app_version: app.package_info().version.to_string(),
        data_dir: crate::config::config_dir(&app).to_string_lossy().to_string(),
        backend: st,
    }
}

#[tauri::command]
pub fn backend_status(state: State<'_, AppState>) -> BackendStatus {
    backend::current(&state)
}

#[tauri::command]
pub fn backend_start(app: tauri::AppHandle, state: State<'_, AppState>) -> BackendStatus {
    backend::stop(&state);
    backend::start(&app, &state)
}

#[tauri::command]
pub fn backend_stop(state: State<'_, AppState>) -> BackendStatus {
    backend::stop(&state)
}

/// 用系统文件管理器打开文件/目录（日志、输出目录、模型目录）。
#[tauri::command]
pub fn open_path(path: String) -> Result<(), String> {
    if path.trim().is_empty() {
        return Err("路径为空".into());
    }
    if !Path::new(&path).exists() {
        return Err(format!("路径不存在：{path}"));
    }
    open_impl(&path).map_err(|err| format!("打开失败：{err}"))
}

#[cfg(target_os = "windows")]
fn open_impl(path: &str) -> std::io::Result<()> {
    // explorer 对存在的文件用 `/select,` 打开父目录并选中它；目录直接开
    let mut cmd = Command::new("explorer");
    if Path::new(path).is_file() {
        cmd.arg(format!("/select,{path}"));
    } else {
        cmd.arg(path);
    }
    // explorer 常常以退出码 1 表示"成功打开了"，所以只判 spawn 是否成功
    cmd.spawn().map(|_| ())
}

#[cfg(target_os = "macos")]
fn open_impl(path: &str) -> std::io::Result<()> {
    let mut cmd = Command::new("open");
    if Path::new(path).is_file() {
        cmd.arg("-R");
    }
    cmd.arg(path).spawn().map(|_| ())
}

#[cfg(all(unix, not(target_os = "macos")))]
fn open_impl(path: &str) -> std::io::Result<()> {
    Command::new("xdg-open").arg(path).spawn().map(|_| ())
}

/// 原生目录选择器。**取消返回 `Ok(None)`**（不是错误 —— 前端不该弹"操作失败"）。
///
/// `window` 由 Tauri 注入（前端调用签名不变），用来让对话框模态于主窗口。
#[tauri::command]
pub fn pick_directory(window: tauri::Window, title: Option<String>) -> Result<Option<String>, String> {
    let title = title.unwrap_or_else(|| "选择目录".to_string());
    pick_directory_impl(&window, &title)
}

#[cfg(target_os = "windows")]
fn pick_directory_impl(window: &tauri::Window, title: &str) -> Result<Option<String>, String> {
    // IFileOpenDialog（资源管理器同款）：有地址栏、能粘贴路径、能新建文件夹、窗口可调大小。
    // 旧实现走 powershell 的 FolderBrowserDialog —— 那是 .NET Framework 的 Win95 树控件。
    let owner = match window.hwnd() {
        Ok(hwnd) => hwnd.0 as *mut std::ffi::c_void,
        Err(_) => std::ptr::null_mut(),
    };
    crate::folder_picker::pick(title, owner)
}

#[cfg(target_os = "macos")]
fn pick_directory_impl(_window: &tauri::Window, title: &str) -> Result<Option<String>, String> {
    let script = format!("POSIX path of (choose folder with prompt \"{}\")", title.replace('"', "\\\""));
    let output = Command::new("osascript")
        .args(["-e", &script])
        .output()
        .map_err(|err| format!("无法启动系统目录选择器：{err}"))?;
    if !output.status.success() {
        return Ok(None); // 用户取消
    }
    let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(if text.is_empty() { None } else { Some(text) })
}

#[cfg(all(unix, not(target_os = "macos")))]
fn pick_directory_impl(_window: &tauri::Window, title: &str) -> Result<Option<String>, String> {
    // zenity（GNOME 系）→ kdialog（KDE 系）依次尝试；都没有就明确报错，别假装取消
    if let Ok(output) = Command::new("zenity")
        .args(["--file-selection", "--directory", &format!("--title={title}")])
        .output()
    {
        if output.status.success() {
            let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
            return Ok(if text.is_empty() { None } else { Some(text) });
        }
        return Ok(None);
    }
    if let Ok(output) = Command::new("kdialog")
        .args(["--getexistingdirectory", ".", "--title", title])
        .output()
    {
        if output.status.success() {
            let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
            return Ok(if text.is_empty() { None } else { Some(text) });
        }
        return Ok(None);
    }
    Err("未找到 zenity 或 kdialog，请安装其一（Debian/Ubuntu: sudo apt install zenity）。".into())
}
