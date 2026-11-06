// release 构建不弹控制台窗口（Windows）
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    darme_voide_desktop_lib::run()
}
