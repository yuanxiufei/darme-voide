//! 原生对话框。
//!
//! 为什么不用 `tauri-plugin-dialog`：本壳只需要「一个是/否确认框」和「一个选目录」，
//! 两者都能用系统 API 直接调，插件会连带一套 Rust 依赖进安装包（见 `Cargo.toml` 抬头）。
//!
//! 目前提供：
//! - [`confirm_quit`] 关窗前确认（后端在跑时避免中途杀掉任务）

/// 关窗前确认。返回 `true` 表示允许关闭。
///
/// 默认按钮刻意放在「否」（`MB_DEFBUTTON2`）：用户连按回车不该把正在出图的后端关掉。
pub fn confirm_quit() -> bool {
    imp::confirm_quit()
}

#[cfg(windows)]
mod imp {
    use std::ffi::c_void;

    type Handle = *mut c_void;

    const MB_YESNO: u32 = 0x0000_0004;
    const MB_ICONWARNING: u32 = 0x0000_0030;
    /// 把默认焦点放在第二个按钮（「否」）
    const MB_DEFBUTTON2: u32 = 0x0000_0100;
    const IDYES: i32 = 6;

    #[link(name = "user32")]
    extern "system" {
        fn MessageBoxW(hwnd: Handle, text: *const u16, caption: *const u16, utype: u32) -> i32;
    }

    fn wide(text: &str) -> Vec<u16> {
        text.encode_utf16().chain(std::iter::once(0)).collect()
    }

    pub fn confirm_quit() -> bool {
        let text = wide("本地服务正在运行，退出会中断正在进行的任务。\n\n确定退出吗？");
        let caption = wide("短剧工坊");
        // SAFETY: 两个指针都指向本函数内仍存活的以 NUL 结尾的宽字符串。
        let choice = unsafe {
            MessageBoxW(
                std::ptr::null_mut(),
                text.as_ptr(),
                caption.as_ptr(),
                MB_YESNO | MB_ICONWARNING | MB_DEFBUTTON2,
            )
        };
        choice == IDYES
    }
}

#[cfg(not(windows))]
mod imp {
    // macOS / Linux 尚未真机实测：这里不拦用户，直接放行
    pub fn confirm_quit() -> bool {
        true
    }
}
