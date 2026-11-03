//! A non-proprietary stand-in for a generated title library.
//!
//! It implements the Android bootstrap contract (`recomp_android_main`) and
//! logs the environment the bootstrap prepared. `scripts/build-apk.sh
//! --stub-game` packages it as `lib/arm64-v8a/librecomp_game.so` so the
//! dlopen entry path can be exercised before the ARM64 code generator can
//! produce a real title. It never contains game code or assets.

#![cfg(target_os = "android")]

use std::ffi::{c_char, c_int, CString};

#[link(name = "log")]
extern "C" {
    fn __android_log_print(
        priority: c_int,
        tag: *const c_char,
        format: *const c_char,
        ...
    ) -> c_int;
}

fn log(message: &str) {
    if let (Ok(tag), Ok(message)) = (CString::new("SwitchRecompStub"), CString::new(message)) {
        // SAFETY: both strings are NUL-terminated and "%s" cannot interpret
        // the message.
        unsafe {
            __android_log_print(4, tag.as_ptr(), b"%s\0".as_ptr().cast(), message.as_ptr());
        }
    }
}

/// Contract symbol the bootstrap resolves with dlsym.
#[no_mangle]
pub extern "C" fn recomp_android_main(argc: c_int, _argv: *mut *mut c_char) -> c_int {
    log(&format!("stub game entry called with argc={argc}"));
    for key in [
        "RECOMP_ASSETS",
        "RECOMP_SAVE_DIR",
        "RECOMP_CACHE_DIR",
        "RECOMP_ROMFS",
    ] {
        log(&format!(
            "{key}={}",
            std::env::var(key).unwrap_or_else(|_| "<unset>".to_string())
        ));
    }
    // Stand in for the first frame/loop setup without touching a GPU.
    std::thread::sleep(std::time::Duration::from_millis(500));
    log("stub game entry returning 0");
    0
}

/// Optional contract version. The bootstrap does not require it yet; bump it
/// only when the entry ABI changes.
#[no_mangle]
pub extern "C" fn recomp_android_abi() -> u32 {
    1
}
