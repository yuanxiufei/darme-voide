//! 短剧工坊桌面壳（Tauri 2）。
//!
//! 职责边界**只有四件事**：开窗口、加载打包好的前端资源、拉起本地 Python 服务、退出时清理。
//! 业务逻辑一律留在前端与后端 —— 壳越薄，安装包越小、跨平台差异越少。

mod backend;
mod child_job;
mod commands;
mod config;
mod dialog;
#[cfg(target_os = "windows")]
mod folder_picker;
mod single_instance;
mod window_state;

use backend::AppState;
use tauri::Manager;

pub fn run() {
    // 单实例判定**必须**在 Builder 之前：第二个实例连窗口都不该创建，
    // 否则用户会看到窗口闪一下再关掉，还以为程序崩了。
    if !single_instance::is_primary() {
        single_instance::notify_primary();
        return;
    }
    tauri::Builder::default()
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            commands::app_bootstrap,
            commands::backend_status,
            commands::backend_start,
            commands::backend_stop,
            commands::open_path,
            commands::pick_directory,
        ])
        .setup(|app| {
            single_instance::watch_activate(app.handle().clone());

            let config_dir = config::config_dir(app.handle());
            let log_dir = config::log_dir(app.handle());
            let _ = std::fs::create_dir_all(&config_dir);
            let _ = std::fs::create_dir_all(&log_dir);

            let cfg = config::DesktopConfig::load(&config_dir);
            let state = app.state::<AppState>();
            *state.config.lock().unwrap() = cfg;
            *state.log_dir.lock().unwrap() = log_dir;
            *state.config_dir.lock().unwrap() = config_dir;

            // 配置里 `visible: false`：先按上次的尺寸/位置摆好，再显示，避免"先闪一下默认尺寸"
            window_state::restore(app.handle());
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.show();
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                // 先把尺寸落盘：这一刻窗口还活着，读到的是用户真实使用的尺寸
                window_state::save(window);
                // 后端在跑（可能正在出图）就先问一句；外部复用的实例不拦
                if backend::busy(window.app_handle()) && !dialog::confirm_quit() {
                    api.prevent_close();
                }
            }
        })
        .build(tauri::generate_context!())
        .expect("Tauri 初始化失败")
        .run(|app, event| {
            // 退出时必杀子进程：否则 uvicorn 会变孤儿，下次启动占着端口
            if let tauri::RunEvent::Exit = event {
                let state = app.state::<AppState>();
                backend::kill_child(&state);
            }
        });
}
