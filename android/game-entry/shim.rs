// Append to the generated project's src/lib.rs after renaming `fn main` to
// `fn recomp_main`. Requires `recomp_abi` in scope.
//
// The generated project may set panic=abort in its profile; in that case the
// catch_unwind is compiled out and a Rust panic terminates the process, which
// is visible in logcat as an abort.

#[no_mangle]
pub extern "C" fn recomp_android_main(
    _argc: core::ffi::c_int,
    _argv: *mut *mut core::ffi::c_char,
) -> core::ffi::c_int {
    let result = std::panic::catch_unwind(recomp_main);
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(fault)) => {
            eprintln!("[recomp-android] guest fault: {fault}");
            1
        }
        Err(_) => {
            eprintln!("[recomp-android] panic in generated entry");
            2
        }
    }
}
