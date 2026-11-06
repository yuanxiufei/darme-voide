//! 后端子进程的**所有权**：内核级兜底 + 落盘凭据。
//!
//! ## 为什么 `kill_child` 不够
//!
//! `RunEvent::Exit` 只在**正常退出路径**上跑。用户从任务管理器结束进程、`taskkill /F`、
//! 或壳自己 panic 时它不执行，uvicorn 就变成孤儿：下次启动 `reuse_external` 会把它当
//! "外部后端"复用 ⇒ 界面能用，但 `owned=false` ⇒ 关窗不再确认、退出也不再杀它，
//! 端口被永久占住。这是**测试里真踩到过**的形状（两次强杀留下两个孤儿）。
//!
//! ## 两层
//!
//! 1. **Job Object（Windows，真兜底）**：`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` 的语义是
//!    "最后一个句柄关闭时杀掉 job 内全部进程"。壳无论怎么死，内核都会关掉它的句柄，
//!    于是整棵子进程树被回收 —— 不需要壳"有机会执行代码"。
//!    句柄故意**泄漏到进程结束**（`static JOB` 持有），这不是疏忽：一旦 Drop 掉，
//!    job 就会在壳还活着的时候把后端杀掉。
//! 2. **`backend.pid` 凭据（跨平台，兜旧版本/兜 Job 没生效）**：落 `{pid, 创建时间, 端口,
//!    后端目录, 解释器}`。下次启动先认领：确认**还是那个进程**（Windows 比创建时间，
//!    Unix 比 `/proc/<pid>/cmdline`）才杀 —— 只比 pid 会在重启后误杀别人的同名进程。
//!
//! 认不出来就什么都不做（宁可留个孤儿，也不能杀错用户自己的服务）。

use std::path::{Path, PathBuf};

/// 凭据文件名（落在 `<appConfigDir>` 下，与 `desktop.json` / `window.json` 同层）
const RECORD_NAME: &str = "backend.pid";

/// 凭据是**跨版本契约**：`serde(default)` 让"新版本写的凭据被老版本读"与
/// "老版本写的凭据被新版本读"都不至于解析失败。缺字段解析失败 ⇒ 凭据被当废纸删掉
/// ⇒ 那次升级后的残留后端永远清不掉（一次性，但用户侧表现为"端口莫名被占"）。
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase", default)]
struct Record {
    pid: u32,
    /// 进程创建时间。Windows 是 FILETIME（100ns 精度）；`0` = 拿不到 ⇒ 不认领（宁可不动）
    created: u64,
    port: u16,
    backend_dir: String,
    python: String,
}

/// 认领回来的残留后端
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reclaimed {
    pub pid: u32,
    pub port: u16,
}

/// `reclaim` 的三种结局。**不能只是 `Option`**：`Clean`（没有残留）与
/// `Unverified`（有凭据但本平台认不了）在处理上完全相反 —— 后者既不能杀，
/// 更不能删凭据（那是唯一的证据），还得让用户看见。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reclaim {
    /// 确认是自家残留 ⇒ 已杀掉
    Killed(Reclaimed),
    /// 没有残留可清（无凭据 / 身份不符已作废）
    Clean,
    /// 有凭据，但本平台**没有判定身份的手段**（macOS 无 procfs）⇒ 一个字节都没动
    Unverified(String),
}

/// 身份判定三态。为什么必须三态：把 `Unknown` 当 `Mismatch` 处理，就是
/// "认不出来 ⇒ 把证据删掉"，孤儿再也清不掉；当 `Match` 处理则是随机杀人。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Identity {
    /// 确认就是凭据里那个进程
    Match,
    /// 确认不是（进程已退出 / pid 被复用 / 凭据残缺）
    Mismatch,
    /// 本平台没有可用的判定手段 ⇒ 什么都不许动
    ///
    /// Windows 侧永不构造（`GetProcessTimes` 总能给出确定答案），但类型里必须留着：
    /// 全仓只有一份 `reclaim`，它要同时服务有 procfs 与没有 procfs 的平台。
    #[cfg_attr(windows, allow(dead_code))]
    Unknown,
}

pub fn record_path(config_dir: &Path) -> PathBuf {
    config_dir.join(RECORD_NAME)
}

/// 把刚拉起的子进程纳入监管，并把凭据落盘。
///
/// 返回 `Some(警告)` 表示"少了某层保护"，调用方只记日志、**不中断启动**：
/// 极端受限环境（父进程已在不可 breakaway 的 job 里）下 assign 会失败，
/// 但后端本身是好的，为此拦住用户得不偿失。
pub fn adopt(
    config_dir: &Path,
    pid: u32,
    port: u16,
    backend_dir: &Path,
    python: &Path,
) -> Option<String> {
    let mut warn = None;
    if let Err(err) = imp::supervise(pid) {
        warn = Some(format!("内核级兜底未启用（{err}），仅靠退出时 kill + 下次启动认领"));
    }
    let rec = Record {
        pid,
        created: imp::created_at(pid).unwrap_or(0),
        port,
        backend_dir: backend_dir.to_string_lossy().into_owned(),
        python: python.to_string_lossy().into_owned(),
    };
    let write = serde_json::to_string_pretty(&rec)
        .map_err(|e| e.to_string())
        .and_then(|text| {
            std::fs::create_dir_all(config_dir).map_err(|e| e.to_string())?;
            std::fs::write(record_path(config_dir), text).map_err(|e| e.to_string())
        });
    if let Err(err) = write {
        warn.get_or_insert(format!("凭据写入失败（{err}），崩溃后将无法认领残留"));
    }
    warn
}

/// 交还所有权（子进程已经被杀掉之后调用）。
pub fn release(config_dir: &Path) {
    let _ = std::fs::remove_file(record_path(config_dir));
}

/// 认领**上次残留的自家后端**：确认身份 → 杀掉 → 删凭据。
///
/// 身份确认是硬条件。三种结局的区别见 [`Reclaim`]：只有"确认不是 / 凭据已坏"
/// 才删凭据；"认不出来"要原样留着。
pub fn reclaim(config_dir: &Path) -> Reclaim {
    let Ok(raw) = std::fs::read_to_string(record_path(config_dir)) else {
        return Reclaim::Clean;
    };
    // 别的工具（PowerShell `Set-Content -Encoding UTF8`、记事本）写出来会带 BOM；
    // 我们自己写的不带，但凭据被别人动过就整份认不出来 ⇒ 顺手剥掉
    let text = raw.trim_start_matches('\u{feff}');
    let Ok(rec) = serde_json::from_str::<Record>(text) else {
        // 凭据自身坏了：删掉，避免每次启动都白读一遍
        release(config_dir);
        return Reclaim::Clean;
    };
    if rec.pid == std::process::id() {
        return Reclaim::Clean;
    }
    match imp::identify(&rec) {
        Identity::Match => {
            imp::terminate(rec.pid);
            release(config_dir);
            Reclaim::Killed(Reclaimed {
                pid: rec.pid,
                port: rec.port,
            })
        }
        Identity::Mismatch => {
            // 不是那个进程了（PID 已复用 / 早就退了）：凭据作废，别留着误导
            release(config_dir);
            Reclaim::Clean
        }
        // 认不出来 ⇒ **不杀、不删**。凭据留着是下次仍能提示的唯一依据，
        // 端口真被占时用户还能照 pid 手动处理。
        Identity::Unknown => Reclaim::Unverified(format!(
            "上次可能有残留服务（记录 pid {}、端口 {}），但本平台无法确认它是否还在，未作处理；\
            若端口被占请手动结束该进程。",
            rec.pid, rec.port
        )),
    }
}

/// 在 `prefer..prefer+20` 里等端口真正空出来（被杀的进程还要一小会儿才松开监听）。
pub fn wait_port_free(port: u16, timeout: std::time::Duration) -> bool {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        if std::net::TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return true;
        }
        if std::time::Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_millis(120));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ds-child-job-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// 凭据是**跨版本契约**：字段改名会让老凭据读不出来（表现为"残留永远清不掉"），
    /// 所以字段名钉死成 camelCase 的字面量。
    #[test]
    fn record_fields_are_a_stable_contract() {
        let rec = Record {
            pid: 4242,
            created: 133_000_000_000_000_000,
            port: 5790,
            backend_dir: "D:/x/backend-py".into(),
            python: "D:/x/python.exe".into(),
        };
        let text = serde_json::to_string(&rec).unwrap();
        for key in ["\"pid\"", "\"created\"", "\"port\"", "\"backendDir\"", "\"python\""] {
            assert!(text.contains(key), "缺字段 {key}：{text}");
        }
        let back: Record = serde_json::from_str(&text).unwrap();
        assert_eq!(back.pid, rec.pid);
        assert_eq!(back.created, rec.created);
        assert_eq!(back.port, rec.port);
    }

    /// **跨版本兼容**：新版本加字段后，老版本写的凭据（缺字段）仍要能读出来，
    /// 且缺的那几格退化成"不认领"而不是"整份作废"。
    #[test]
    fn record_tolerates_missing_and_extra_fields() {
        // 只有 pid —— 模拟将来/过去某个版本的凭据形状
        let old: Record = serde_json::from_str(r#"{"pid": 4242}"#).expect("缺字段不该解析失败");
        assert_eq!(old.pid, 4242);
        assert_eq!(old.created, 0, "缺 created ⇒ 0 ⇒ 认领时按 Mismatch 作废");

        // 多出未知字段（未来版本写的）也不许炸
        let newer: Record =
            serde_json::from_str(r#"{"pid": 1, "futureField": {"a": [1,2]}}"#).expect("未知字段要忽略");
        assert_eq!(newer.pid, 1);
    }

    /// 没有凭据 ⇒ 什么都不做（不能因为"没记录"就随便清理端口）
    #[test]
    fn reclaim_without_record_is_a_noop() {
        let dir = tmp("norec");
        assert_eq!(reclaim(&dir), Reclaim::Clean);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 凭据坏了 ⇒ 不认领、且把坏凭据删掉（否则每次启动都白读一遍）
    #[test]
    fn reclaim_drops_corrupt_record_without_touching_anything() {
        let dir = tmp("corrupt");
        std::fs::write(record_path(&dir), "{ not json").unwrap();
        assert_eq!(reclaim(&dir), Reclaim::Clean);
        assert!(!record_path(&dir).exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `created == 0`（凭据残缺/被手改）⇒ 不认领、凭据作废，但**绝不误杀**
    #[cfg(windows)]
    #[test]
    fn reclaim_never_guesses_without_creation_time() {
        let mut bait = spawn_bait();
        let pid = bait.id();
        let dir = tmp("nocreated");
        let rec = Record {
            pid,
            created: 0,
            port: 5790,
            backend_dir: String::new(),
            python: String::new(),
        };
        std::fs::write(record_path(&dir), serde_json::to_string(&rec).unwrap()).unwrap();
        assert_eq!(reclaim(&dir), Reclaim::Clean);
        assert!(!record_path(&dir).exists(), "作废凭据要清掉");
        assert!(bait.try_wait().unwrap().is_none(), "没有创建时间却把进程杀了");
        let _ = bait.kill();
        let _ = bait.wait();
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 端口释放等待：自己占着的端口必须判"没空出来"
    #[test]
    fn wait_port_free_reports_busy_port() {
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        assert!(!wait_port_free(port, std::time::Duration::from_millis(300)));
        drop(listener);
        assert!(wait_port_free(port, std::time::Duration::from_millis(300)));
    }

    /// 造一个长命子进程当"残留后端"（不依赖 python/uvicorn：手机制不是测业务）
    #[cfg(windows)]
    fn spawn_bait() -> std::process::Child {
        std::process::Command::new("cmd")
            .args(["/c", "ping -n 120 127.0.0.1 >nul"])
            .stdout(std::process::Stdio::null())
            .spawn()
            .expect("起诱饵进程")
    }

    /// **正向**：创建时间对得上 ⇒ 认领并真的杀掉
    #[cfg(windows)]
    #[test]
    fn reclaim_kills_recorded_process_when_identity_matches() {
        let mut bait = spawn_bait();
        let pid = bait.id();
        let created = imp::created_at(pid).expect("取不到诱饵创建时间");
        let dir = tmp("kill");
        let rec = Record {
            pid,
            created,
            port: 5999,
            backend_dir: String::new(),
            python: String::new(),
        };
        std::fs::write(record_path(&dir), serde_json::to_string(&rec).unwrap()).unwrap();

        match reclaim(&dir) {
            Reclaim::Killed(got) => {
                assert_eq!(got.pid, pid);
                assert_eq!(got.port, 5999);
            }
            other => panic!("创建时间对得上就应当认领，实际 {other:?}"),
        }
        assert!(!record_path(&dir).exists(), "认领后凭据必须删掉");
        let _ = bait.wait(); // 已被 TerminateProcess，会立即返回
        std::fs::remove_dir_all(&dir).ok();
    }

    /// **反向**：创建时间差 1（等价于 PID 被复用）⇒ 一个字节都不许动
    #[cfg(windows)]
    #[test]
    fn reclaim_leaves_foreign_process_alone() {
        let mut bait = spawn_bait();
        let pid = bait.id();
        let created = imp::created_at(pid).expect("取不到诱饵创建时间") + 1;
        let dir = tmp("spare");
        let rec = Record {
            pid,
            created,
            port: 5999,
            backend_dir: String::new(),
            python: String::new(),
        };
        std::fs::write(record_path(&dir), serde_json::to_string(&rec).unwrap()).unwrap();

        assert_eq!(reclaim(&dir), Reclaim::Clean, "身份不符不许认领");
        assert!(!record_path(&dir).exists(), "作废凭据要清掉，别每次启动都白读");

        let alive = bait.try_wait().unwrap().is_none();
        assert!(alive, "身份不符却把进程杀了");
        let _ = bait.kill();
        let _ = bait.wait();
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 三态映射：本平台有这个能力，就**不许**用 `Unknown` 敷衍（那会让凭据永不作废）。
    #[cfg(windows)]
    #[test]
    fn identity_is_three_state_on_windows() {
        let mut bait = spawn_bait();
        let pid = bait.id();
        let created = imp::created_at(pid).expect("取不到诱饵创建时间");
        let good = Record {
            pid,
            created,
            port: 5790,
            backend_dir: String::new(),
            python: String::new(),
        };
        assert_eq!(imp::identify(&good), Identity::Match);

        let stale = Record {
            created: created + 1,
            ..good.clone()
        };
        assert_eq!(imp::identify(&stale), Identity::Mismatch, "创建时间不符");

        let no_stamp = Record {
            created: 0,
            ..good
        };
        assert_eq!(imp::identify(&no_stamp), Identity::Mismatch, "凭据缺时间戳");

        let _ = bait.kill();
        let _ = bait.wait();
    }

}

#[cfg(windows)]
mod imp {
    use super::{Identity, Record};
    use std::ffi::c_void;
    use std::sync::Mutex;

    type Handle = *mut c_void;

    const JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: u32 = 0x0000_2000;
    /// `JobObjectExtendedLimitInformation`
    const JOB_INFO_CLASS_EXTENDED_LIMIT: u32 = 9;
    const PROCESS_TERMINATE: u32 = 0x0001;
    const PROCESS_SET_QUOTA: u32 = 0x0100;
    const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;
    const SYNCHRONIZE: u32 = 0x0010_0000;

    /// 故意泄漏的 job 句柄（`0` = 未建）。
    ///
    /// 语义就是"壳活着 job 就活着"：进程一死内核关句柄 ⇒ `KILL_ON_JOB_CLOSE` 触发。
    static JOB: Mutex<usize> = Mutex::new(0);

    #[repr(C)]
    struct BasicLimit {
        per_process_user_time_limit: i64,
        per_job_user_time_limit: i64,
        limit_flags: u32,
        minimum_working_set_size: usize,
        maximum_working_set_size: usize,
        active_process_limit: u32,
        affinity: usize,
        priority_class: u32,
        scheduling_class: u32,
    }

    #[repr(C)]
    struct IoCounters {
        read_operation_count: u64,
        write_operation_count: u64,
        other_operation_count: u64,
        read_transfer_count: u64,
        write_transfer_count: u64,
        other_transfer_count: u64,
    }

    #[repr(C)]
    struct ExtendedLimit {
        basic: BasicLimit,
        io: IoCounters,
        process_memory_limit: usize,
        job_memory_limit: usize,
        peak_process_memory_used: usize,
        peak_job_memory_used: usize,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn CreateJobObjectW(attrs: *const c_void, name: *const u16) -> Handle;
        fn SetInformationJobObject(job: Handle, class: u32, info: *const c_void, len: u32) -> i32;
        fn AssignProcessToJobObject(job: Handle, process: Handle) -> i32;
        fn OpenProcess(access: u32, inherit: i32, pid: u32) -> Handle;
        fn CloseHandle(handle: Handle) -> i32;
        fn TerminateProcess(process: Handle, code: u32) -> i32;
        fn WaitForSingleObject(handle: Handle, millis: u32) -> u32;
        fn GetProcessTimes(
            process: Handle,
            creation: *mut u64,
            exit: *mut u64,
            kernel: *mut u64,
            user: *mut u64,
        ) -> i32;
        fn GetLastError() -> u32;
    }

    /// 建（或复用）job 并把子进程塞进去。
    pub fn supervise(pid: u32) -> Result<(), String> {
        let job = {
            let mut slot = JOB.lock().map_err(|_| "job 互斥体中毒".to_string())?;
            if *slot == 0 {
                // SAFETY: 无属性、匿名 job；句柄存进 static，不关闭（见模块注释）。
                let created = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
                if created.is_null() {
                    return Err(format!("CreateJobObjectW 失败（错误码 {}）", unsafe {
                        GetLastError()
                    }));
                }
                let info = ExtendedLimit {
                    basic: BasicLimit {
                        per_process_user_time_limit: 0,
                        per_job_user_time_limit: 0,
                        limit_flags: JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
                        minimum_working_set_size: 0,
                        maximum_working_set_size: 0,
                        active_process_limit: 0,
                        affinity: 0,
                        priority_class: 0,
                        scheduling_class: 0,
                    },
                    io: IoCounters {
                        read_operation_count: 0,
                        write_operation_count: 0,
                        other_operation_count: 0,
                        read_transfer_count: 0,
                        write_transfer_count: 0,
                        other_transfer_count: 0,
                    },
                    process_memory_limit: 0,
                    job_memory_limit: 0,
                    peak_process_memory_used: 0,
                    peak_job_memory_used: 0,
                };
                // SAFETY: 结构与长度都是本平台的 `JOBOBJECT_EXTENDED_LIMIT_INFORMATION`。
                let ok = unsafe {
                    SetInformationJobObject(
                        created,
                        JOB_INFO_CLASS_EXTENDED_LIMIT,
                        &info as *const ExtendedLimit as *const c_void,
                        std::mem::size_of::<ExtendedLimit>() as u32,
                    )
                };
                if ok == 0 {
                    return Err(format!("SetInformationJobObject 失败（错误码 {}）", unsafe {
                        GetLastError()
                    }));
                }
                *slot = created as usize;
                created
            } else {
                *slot as Handle
            }
        };

        // SAFETY: pid 是我们刚 spawn 的子进程；句柄用完立刻关。
        unsafe {
            let process = OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, 0, pid);
            if process.is_null() {
                return Err(format!("OpenProcess 失败（错误码 {}）", GetLastError()));
            }
            let ok = AssignProcessToJobObject(job, process);
            let code = GetLastError();
            CloseHandle(process);
            if ok == 0 {
                return Err(format!("AssignProcessToJobObject 失败（错误码 {code}）"));
            }
        }
        Ok(())
    }

    /// 进程创建时间（FILETIME）。拿不到就 `None` —— 调用方据此放弃认领。
    pub fn created_at(pid: u32) -> Option<u64> {
        // SAFETY: 只读查询；缓冲区是本栈上的四个 u64。
        unsafe {
            let process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
            if process.is_null() {
                return None;
            }
            let (mut creation, mut exit, mut kernel, mut user) = (0u64, 0u64, 0u64, 0u64);
            let ok = GetProcessTimes(process, &mut creation, &mut exit, &mut kernel, &mut user);
            CloseHandle(process);
            if ok == 0 || creation == 0 {
                None
            } else {
                Some(creation)
            }
        }
    }

    /// 凭据里记的还是不是同一个进程：创建时间逐位相等才算。
    ///
    /// 只比 pid 会在机器重启后误杀"恰好同号"的无关进程；`created == 0`（当时没取到）
    /// 按 `Mismatch` 处理 —— 本平台必然取得（自己 spawn 的进程），取不到只可能是
    /// 旧版本写的或被人手改的凭据，留着只会让每次启动都白读一遍。
    pub fn identify(rec: &Record) -> Identity {
        if rec.created == 0 {
            return Identity::Mismatch;
        }
        // 进程不存在（或没权限查）⇒ `created_at` 返回 None ⇒ 这是**能确认的**不符，
        // 不是"不知道"：`Mismatch` 才会去删凭据。
        match created_at(rec.pid) {
            Some(t) if t == rec.created => Identity::Match,
            _ => Identity::Mismatch,
        }
    }

    pub fn terminate(pid: u32) {
        // SAFETY: 上面已用创建时间确认过身份；拿到句柄就 SIGKILL 等价物 + 等它真退出。
        unsafe {
            let process = OpenProcess(PROCESS_TERMINATE | SYNCHRONIZE, 0, pid);
            if process.is_null() {
                return;
            }
            TerminateProcess(process, 1);
            WaitForSingleObject(process, 3_000);
            CloseHandle(process);
        }
    }
}

#[cfg(not(windows))]
mod imp {
    use super::{Identity, Record};
    use std::path::Path;
    use std::process::Command;

    /// 非 Windows 没有 Job Object。诚实地报"没启用"，**不许返回 Ok 谎报成功**：
    /// 假绿灯比没有保护更糟 —— 上层会把警告吞掉，用户以为有内核级兜底。
    ///
    /// 等价物是有的（Linux `prctl(PR_SET_PDEATHSIG)` / cgroup v2、macOS kqueue 监听
    /// 父进程），但都要 libc 或额外依赖，且本仓此平台尚无真机验证运行 ——
    /// 与其写一套没跑过的"看起来对"，不如让这层如实降级到凭据认领。
    pub fn supervise(_pid: u32) -> Result<(), String> {
        Err("非 Windows 平台未实现内核级子进程回收".to_string())
    }

    pub fn created_at(_pid: u32) -> Option<u64> {
        None
    }

    /// 认领判据：`/proc/<pid>/cmdline` 里同时有 `app.main:app` 与 `--port <端口>`。
    ///
    /// 比 pid 单值强得多：重启后 pid 复用也不会命中，且不会误伤用户自己在终端起的服务
    /// （那种服务端口不同，或不是本壳的子进程）。
    ///
    /// 没有 procfs（macOS/BSD）⇒ `Unknown`：那种平台**认不出来**，不等于"不是它"。
    /// 把这两件事混为一谈会在 macOS 上每次都把凭据删掉，孤儿从此再也清不掉。
    pub fn identify(rec: &Record) -> Identity {
        // `/proc/self/cmdline` 是"本机有没有 procfs"的判据：Linux 必有，macOS 必无
        if !Path::new("/proc/self/cmdline").exists() {
            return Identity::Unknown;
        }
        let Ok(raw) = std::fs::read(format!("/proc/{}/cmdline", rec.pid)) else {
            // procfs 在，但读不到这个 pid ⇒ 进程确实不在了
            return Identity::Mismatch;
        };
        let cmd = String::from_utf8_lossy(&raw).replace('\0', " ");
        if cmd.contains("app.main:app") && cmd.contains(&format!("--port {}", rec.port)) {
            Identity::Match
        } else {
            Identity::Mismatch
        }
    }

    pub fn terminate(pid: u32) {
        // 没有 libc 依赖，用系统 `kill`：它本来就带在 PATH 里。
        // 先收拾直接子进程：uvicorn 起 worker 时，只杀父会把 worker 留成孤儿。
        let _ = Command::new("pkill").arg("-9").arg("-P").arg(pid.to_string()).status();
        let _ = Command::new("kill").arg("-9").arg(pid.to_string()).status();
    }
}
