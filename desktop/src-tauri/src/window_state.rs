//! 记住窗口大小 / 位置 / 最大化状态。
//!
//! 不引 `tauri-plugin-window-state`：这里只需要一个 5 字段的 JSON，
//! 而且**必须**和本壳已有的「配置落在 `<appConfigDir>`」约定一致（排障时一个目录能找齐）。
//!
//! ## 两个刻意的取舍
//!
//! 1. **最大化时不覆盖窗口化尺寸**：最大化时 `outer_size()` 返回的是铺满屏幕的尺寸，
//!    照抄会让用户下次把窗口还原时得到一个屏幕那么大的「窗口」。
//! 2. **恢复前做可见性校验**：拔掉副屏后旧坐标可能落在屏幕外，窗口会在看不见的地方打开
//!    （表现像"启动失败"）。校验不过就不设位置，交给系统的 `center: true`。

use serde::{Deserialize, Serialize};
use std::path::Path;
use tauri::{AppHandle, Manager, Window};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct WindowState {
    pub width: u32,
    pub height: u32,
    pub x: i32,
    pub y: i32,
    pub maximized: bool,
}

impl Default for WindowState {
    fn default() -> Self {
        // 与 tauri.conf.json 的首窗尺寸一致：读不到状态时恢复成"出厂设置"
        Self {
            width: 1440,
            height: 900,
            x: 0,
            y: 0,
            maximized: false,
        }
    }
}

fn state_path(config_dir: &Path) -> std::path::PathBuf {
    config_dir.join("window.json")
}

fn load(config_dir: &Path) -> Option<WindowState> {
    let raw = std::fs::read_to_string(state_path(config_dir)).ok()?;
    serde_json::from_str(&raw).ok()
}

/// 启动时调用：把上次的尺寸/位置/最大化状态套回主窗口。
///
/// 调用方应在窗口 `visible: false` 的情况下调用，之后再 `show()`，避免"先闪一下默认尺寸"。
pub fn restore(app: &AppHandle) {
    use tauri::{PhysicalPosition, PhysicalSize};

    let Some(window) = app.get_webview_window("main") else {
        return;
    };
    let Some(state) = load(&crate::config::config_dir(app)) else {
        return;
    };

    // 尺寸下限兜一层：配置文件被手改成 10x10 时窗口会小到点不着
    let width = state.width.clamp(800, 10_000);
    let height = state.height.clamp(600, 10_000);
    if state.width > 0 && state.height > 0 {
        // 必须用 Physical：存的是物理像素，用 Logical 会在 125%/150% 缩放下被放大一遍
        let _ = window.set_size(PhysicalSize::new(width, height));
    }

    if state.x != 0 || state.y != 0 {
        if position_visible(&window, state.x, state.y) {
            let _ = window.set_position(PhysicalPosition::new(state.x, state.y));
        }
    }

    if state.maximized {
        let _ = window.maximize();
    }
}

/// 目标左上角是否落在一块真实屏幕内（留 320×80 的可抓取区域）。
fn position_visible(window: &tauri::WebviewWindow, x: i32, y: i32) -> bool {
    let Ok(monitors) = window.available_monitors() else {
        return false;
    };
    monitors.iter().any(|monitor| {
        let origin = monitor.position();
        let size = monitor.size();
        let (mx, my) = (origin.x, origin.y);
        let (mw, mh) = (size.width as i32, size.height as i32);
        let hit_x = x.max(mx) < (mx + mw).min(x + 320);
        let hit_y = y.max(my) < (my + mh).min(y + 80);
        hit_x && hit_y
    })
}

/// 关窗时调用：窗口还在，此时读到的才是用户真实使用的尺寸。
pub fn save(window: &Window) {
    let app = window.app_handle();
    let config_dir = crate::config::config_dir(app);
    let _ = std::fs::create_dir_all(&config_dir);

    let maximized = window.is_maximized().unwrap_or(false);
    let previous = load(&config_dir).unwrap_or_default();

    // 最大化时只更新标志位，尺寸/位置沿用上一次"窗口化"时的值
    let next = if maximized {
        WindowState {
            maximized: true,
            ..previous
        }
    } else {
        // 用 inner_size 而不是 outer_size：`set_size` 设的就是客户区，
        // 用外部尺寸存、客户区尺寸还原，每关一次窗口就会缩掉一圈边框。
        let Ok(size) = window.inner_size() else {
            return;
        };
        let position = window.outer_position().ok();
        WindowState {
            width: size.width,
            height: size.height,
            x: position.map(|p| p.x).unwrap_or(previous.x),
            y: position.map(|p| p.y).unwrap_or(previous.y),
            maximized: false,
        }
    };

    if let Ok(text) = serde_json::to_string_pretty(&next) {
        let _ = std::fs::write(state_path(&config_dir), text);
    }
}
