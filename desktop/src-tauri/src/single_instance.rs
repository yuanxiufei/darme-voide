//! 单实例：第二次双击**不再**起第二个后端。
//!
//! ## 为什么必须做
//!
//! 两个实例会各自 `pick_port()`：第二个顺延到 5791 后**同样**拉起一个 uvicorn，
//! 两个进程写同一个 `drama.db`。用户看到的是两份界面各刷各的，数据在背后互相踩。
//! 更糟的是关掉其中一个时，另一个的 `kill_child` 只管自己那个子进程，孤儿 uvicorn 会一直占着端口。
//!
//! ## 为什么不用 `tauri-plugin-single-instance`
//!
//! 体积是硬指标（见 `Cargo.toml` 抬头）。这个能力用十几个 Win32 调用就够：
//!
//! - **判重**：命名 Mutex `CreateMutexW` —— 名字已存在时返回句柄 + `ERROR_ALREADY_EXISTS`
//! - **通知**：命名 Event `CreateEventW` —— 第二个实例 `SetEvent`，第一个实例的等待线程收到后聚焦主窗口
//!
//! 两个对象都用 `Local\` 前缀：只在当前登录会话内生效，多用户/远程桌面下互不干扰。

/// 判定当前进程是否为主实例；主实例返回 `true`，否则 `false`（调用方应立即退出）。
pub fn is_primary() -> bool {
    imp::is_primary()
}

/// 第二个实例通知主实例「把窗口抬到前面」。非主实例退出前调用。
pub fn notify_primary() {
    imp::notify_primary()
}

/// 主实例在 `setup` 里调用一次：起等待线程，收到通知就聚焦主窗口。
pub fn watch_activate(app: tauri::AppHandle) {
    imp::watch_activate(app)
}

#[cfg(windows)]
mod imp {
    use std::ffi::c_void;

    type Handle = *mut c_void;

    const ERROR_ALREADY_EXISTS: u32 = 183;
    const INFINITE: u32 = 0xFFFF_FFFF;
    const WAIT_OBJECT_0: u32 = 0;

    #[link(name = "kernel32")]
    extern "system" {
        fn CreateMutexW(attrs: *const c_void, owner: i32, name: *const u16) -> Handle;
        fn CreateEventW(attrs: *const c_void, manual: i32, initial: i32, name: *const u16) -> Handle;
        fn SetEvent(handle: Handle) -> i32;
        fn WaitForSingleObject(handle: Handle, millis: u32) -> u32;
        fn GetLastError() -> u32;
    }

    const MUTEX_NAME: &str = "Local\\com.darme.voide.desktop.single";
    const EVENT_NAME: &str = "Local\\com.darme.voide.desktop.activate";

    fn wide(text: &str) -> Vec<u16> {
        text.encode_utf16().chain(std::iter::once(0)).collect()
    }

    pub fn is_primary() -> bool {
        let name = wide(MUTEX_NAME);
        // SAFETY: 只传一个合法的宽字符串指针；句柄由系统持有，进程退出自动释放。
        let handle = unsafe { CreateMutexW(std::ptr::null(), 0, name.as_ptr()) };
        // ⚠️ GetLastError 必须紧跟这一次调用读取：中间插任何 Win32 调用都会被覆盖。
        let already = unsafe { GetLastError() } == ERROR_ALREADY_EXISTS;
        if handle.is_null() {
            // 建不出互斥体（极端受限环境）时别拦用户：宁可多开，也不要打不开
            return true;
        }
        !already
    }

    pub fn notify_primary() {
        let name = wide(EVENT_NAME);
        // SAFETY: 同名对象已存在时 CreateEventW 返回已有句柄的副本；这里只用来 SetEvent。
        let handle = unsafe { CreateEventW(std::ptr::null(), 1, 0, name.as_ptr()) };
        if handle.is_null() {
            return;
        }
        unsafe { SetEvent(handle) };
    }

    pub fn watch_activate(app: tauri::AppHandle) {
        let name = wide(EVENT_NAME);
        // SAFETY: manual-reset 事件，初始未触发；句柄由等待线程长期持有。
        let handle = unsafe { CreateEventW(std::ptr::null(), 1, 0, name.as_ptr()) };
        if handle.is_null() {
            return;
        }
        // 裸指针不是 Send，转成地址再在线程内还原
        let addr = handle as usize;
        std::thread::spawn(move || {
            let handle = addr as Handle;
            loop {
                let waited = unsafe { WaitForSingleObject(handle, INFINITE) };
                if waited != WAIT_OBJECT_0 {
                    // 句柄出问题时不要忙等，退一步重试
                    std::thread::sleep(std::time::Duration::from_millis(500));
                    continue;
                }
                focus_main(&app);
            }
        });
    }

    fn focus_main(app: &tauri::AppHandle) {
        use tauri::Manager;
        if let Some(window) = app.get_webview_window("main") {
            // 顺序有讲究：先取消最小化再显示，否则任务栏图标会留在"最小化"状态
            let _ = window.unminimize();
            let _ = window.show();
            let _ = window.set_focus();
        }
    }
}

#[cfg(not(windows))]
mod imp {
    use std::io::Write;

    fn lock_path() -> std::path::PathBuf {
        std::env::temp_dir().join("com.darme.voide.desktop.lock")
    }

    /// 非 Windows 退化方案：`create_new` 即独占。
    ///
    /// 崩溃会留下陈旧锁文件，所以先读里面的 PID 判活：Linux 看 `/proc/<pid>` 是否还在，
    /// 不在就删掉重试一次；其他 unix（macOS 尚未真机实测）不猜，直接视为已有实例。
    pub fn is_primary() -> bool {
        let path = lock_path();
        match std::fs::OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(mut file) => {
                let _ = writeln!(file, "{}", std::process::id());
                true
            }
            Err(_) => {
                let stale = std::fs::read_to_string(&path)
                    .ok()
                    .and_then(|t| t.trim().parse::<u32>().ok())
                    .map(|pid| !std::path::Path::new(&format!("/proc/{pid}")).exists())
                    .unwrap_or(false);
                if stale && std::fs::remove_file(&path).is_ok() {
                    return std::fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(&path)
                        .map(|mut file| {
                            let _ = writeln!(file, "{}", std::process::id());
                        })
                        .is_ok();
                }
                false
            }
        }
    }

    /// 拿不到其他进程的窗口句柄，这里只保证"不开第二个"。
    pub fn notify_primary() {}

    pub fn watch_activate(_app: tauri::AppHandle) {}
}
