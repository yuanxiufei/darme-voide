//! Windows 现代目录选择器（`IFileOpenDialog` + `FOS_PICKFOLDERS`）。
//!
//! ## 为什么换掉 `System.Windows.Forms.FolderBrowserDialog`
//!
//! 那是 Win95 时代的树控件：没有地址栏、不能粘贴路径、不能拖放、窗口不可调大小。
//! .NET Framework 下的 `FolderBrowserDialog` 尤其老（.NET Core 3.0+ 才改走本文件这套 COM），
//! 而 `powershell.exe`（5.1）加载的正是 Framework 版 —— 用户看到的就是那个古董框。
//!
//! ## 为什么不引 `rfd`
//!
//! 只用一个对话框，却要多带一整套 Rust 依赖进安装包。这里直接调 COM：
//! `CoCreateInstance(CLSID_FileOpenDialog)` → `SetOptions(FOS_PICKFOLDERS)` → `Show` → `GetResult`。
//!
//! ## COM 的两个坑（都在下面代码里处理了）
//!
//! 1. **vtable 索引必须精确**，而且**一律 0 基**（`vtable[0]` = `QueryInterface`）。
//!    推导法是「前置基接口的方法总数 + 本接口内的 0 基序号」：
//!    `IUnknown` 3 个（0~2）⇒ `IModalWindow::Show` = 3 ⇒ `IFileDialog` 从 4 起排
//!    （`SetFileTypes` 4、`SetFileTypeIndex` 5、`GetFileTypeIndex` 6、`Advise` 7、
//!    `Unadvise` 8、**`SetOptions` 9**、`GetOptions` 10、…、**`SetTitle` 17**、…、
//!    **`GetResult` 20**）。错一位不是"报错"，是把别的方法当这个调：
//!    曾经把 `SetOptions` 写成 10（＝`GetOptions`），标志位被当成出参指针写进 `0x60` ⇒ 立刻 AV。
//! 2. **取消不是错误**：用户点「取消」时 `Show` 返回 `HRESULT_FROM_WIN32(ERROR_CANCELLED)`，
//!    必须翻成 `Ok(None)`，否则前端会弹一个"操作失败"。

use std::ffi::c_void;
use std::ptr;

type HRESULT = i32;

const S_OK: HRESULT = 0;
const CLSCTX_INPROC_SERVER: u32 = 1;
const COINIT_APARTMENTTHREADED: u32 = 2;
/// 线程已是 MTA 时 `CoInitializeEx` 的返回值；COM 仍可继续用，不算失败
const RPC_E_CHANGED_MODE: HRESULT = -2_147_417_850;

const FOS_PICKFOLDERS: u32 = 0x0000_0020;
const FOS_FORCEFILESYSTEM: u32 = 0x0000_0040;
const SIGDN_FILESYSPATH: u32 = 0x8005_8000;

#[repr(C)]
struct Guid {
    d1: u32,
    d2: u16,
    d3: u16,
    d4: [u8; 8],
}

/// `{DC1C5A9C-E88A-4dde-A5A1-60F82A20AEF7}`
const CLSID_FILE_OPEN_DIALOG: Guid = Guid {
    d1: 0xDC1C_5A9C,
    d2: 0xE88A,
    d3: 0x4dde,
    d4: [0xA5, 0xA1, 0x60, 0xF8, 0x2A, 0x20, 0xAE, 0xF7],
};

/// `{d57c7288-d4ad-4768-be02-9d969532d960}`
const IID_IFILE_OPEN_DIALOG: Guid = Guid {
    d1: 0xd57c_7288,
    d2: 0xd4ad,
    d3: 0x4768,
    d4: [0xbe, 0x02, 0x9d, 0x96, 0x95, 0x32, 0xd9, 0x60],
};

// IFileOpenDialog vtable 槽位（0 基；推导见模块注释）
const V_RELEASE: usize = 2;
const V_SHOW: usize = 3;
const V_SET_OPTIONS: usize = 9;
const V_SET_TITLE: usize = 17;
const V_GET_RESULT: usize = 20;
// IShellItem 的 GetDisplayName 在第 5 槽（IUnknown 3 + BindToHandler + GetParent）
const ISHELLITEM_GET_DISPLAY_NAME: usize = 5;

/// 编译期把上面这张表钉死：槽位写错不是"报错"，是把别的方法当这个调（表现为 AV 或
/// 标志位被当成指针写）。这里用编译期断言把**推导结果**与常量对齐，改错一个数就直接编不过。
const _: () = {
    // IUnknown(3) + IModalWindow(1)
    assert!(V_SHOW == 3);
    assert!(V_RELEASE == 2);
    // IFileDialog：Show 之后第 5 个（SetFileTypes/SetFileTypeIndex/GetFileTypeIndex/Advise/Unadvise）
    assert!(V_SET_OPTIONS == 3 + 1 + 5);
    // SetOptions 之后第 7 个（GetOptions/SetDefaultFolder/SetFolder/GetFolder/
    // GetCurrentSelection/SetFileName/GetFileName）
    assert!(V_SET_TITLE == V_SET_OPTIONS + 8);
    // SetTitle 之后第 2 个（SetOkButtonLabel/SetFileNameLabel）
    assert!(V_GET_RESULT == V_SET_TITLE + 3);
};

type ReleaseFn = unsafe extern "system" fn(*mut c_void) -> u32;
type ShowFn = unsafe extern "system" fn(*mut c_void, *mut c_void) -> HRESULT;
type SetOptionsFn = unsafe extern "system" fn(*mut c_void, u32) -> HRESULT;
type SetTitleFn = unsafe extern "system" fn(*mut c_void, *const u16) -> HRESULT;
type GetResultFn = unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> HRESULT;
type GetDisplayNameFn = unsafe extern "system" fn(*mut c_void, u32, *mut *mut u16) -> HRESULT;

#[link(name = "ole32")]
extern "system" {
    fn CoInitializeEx(reserved: *mut c_void, coinit: u32) -> HRESULT;
    fn CoUninitialize();
    fn CoCreateInstance(
        rclsid: *const Guid,
        outer: *mut c_void,
        clsctx: u32,
        riid: *const Guid,
        ppv: *mut *mut c_void,
    ) -> HRESULT;
    fn CoTaskMemFree(pv: *mut c_void);
}

/// 取 COM 对象第 `index` 个 vtable 槽位并转成函数指针。
///
/// # Safety
/// `object` 必须是有效的 COM 接口指针，且 `index` 处的签名必须与 `T` 一致。
unsafe fn slot<T: Copy>(object: *mut c_void, index: usize) -> T {
    let vtbl = *(object as *const *const c_void);
    let entry = (vtbl as *const *const c_void).add(index);
    std::mem::transmute_copy(&*entry)
}

/// 弹出目录选择框。返回 `Ok(None)` 表示用户取消。
///
/// `owner` 传主窗口句柄可让对话框模态于它；传空指针则独立显示。
pub fn pick(title: &str, owner: *mut c_void) -> Result<Option<String>, String> {
    // SAFETY: 全程只操作本函数创建/校验过的 COM 指针，vtable 索引见模块注释。
    unsafe {
        let init = CoInitializeEx(ptr::null_mut(), COINIT_APARTMENTTHREADED);
        if init < 0 && init != RPC_E_CHANGED_MODE {
            return Err(format!("COM 初始化失败（0x{:08X}）", init as u32));
        }
        // 只有本次真正初始化了才配对 CoUninitialize，否则会把调用方的 COM 状态抽走
        let owns_com = init == S_OK;

        let outcome = (|| -> Result<Option<String>, String> {
            let mut dialog: *mut c_void = ptr::null_mut();
            let hr = CoCreateInstance(
                &CLSID_FILE_OPEN_DIALOG,
                ptr::null_mut(),
                CLSCTX_INPROC_SERVER,
                &IID_IFILE_OPEN_DIALOG,
                &mut dialog,
            );
            if hr < 0 || dialog.is_null() {
                return Err(format!("无法创建目录选择对话框（0x{:08X}）", hr as u32));
            }

            let set_options: SetOptionsFn = slot(dialog, V_SET_OPTIONS);
            set_options(dialog, FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM);

            let title_wide: Vec<u16> = title.encode_utf16().chain(std::iter::once(0)).collect();
            let set_title: SetTitleFn = slot(dialog, V_SET_TITLE);
            set_title(dialog, title_wide.as_ptr());

            let show: ShowFn = slot(dialog, V_SHOW);
            let shown = show(dialog, owner);
            if shown != S_OK {
                let release: ReleaseFn = slot(dialog, V_RELEASE);
                release(dialog);
                return Ok(None); // 取消
            }

            let mut item: *mut c_void = ptr::null_mut();
            let get_result: GetResultFn = slot(dialog, V_GET_RESULT);
            let mut picked = None;
            if get_result(dialog, &mut item) >= 0 && !item.is_null() {
                let mut buffer: *mut u16 = ptr::null_mut();
                let get_name: GetDisplayNameFn = slot(item, ISHELLITEM_GET_DISPLAY_NAME);
                if get_name(item, SIGDN_FILESYSPATH, &mut buffer) >= 0 && !buffer.is_null() {
                    let len = (0..).take_while(|&i| *buffer.add(i) != 0).count();
                    let text = std::slice::from_raw_parts(buffer, len);
                    picked = Some(String::from_utf16_lossy(text));
                    CoTaskMemFree(buffer as *mut c_void);
                }
                let release_item: ReleaseFn = slot(item, V_RELEASE);
                release_item(item);
            }

            let release: ReleaseFn = slot(dialog, V_RELEASE);
            release(dialog);
            Ok(picked)
        })();

        if owns_com {
            CoUninitialize();
        }
        outcome
    }
}
