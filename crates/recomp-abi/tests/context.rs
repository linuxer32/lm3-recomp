//! The guest-context ABI generated code uses when the pipeline's
//! `guest_context` flag is on. The functions below are the shape the emitter
//! produces for a small caller (register loads, spills, a direct call, the
//! X30 return check, flag and vector hand-off), run against the same storage
//! the runtime uses.

use recomp_abi::{guest_context, GuestContext, GuestResult};
use std::sync::Mutex;

/// The runtime normally defines these; tests of the callback path supply
/// them so `recomp_abi::call` can fall back to `dispatch` without the
/// runtime crate.
#[no_mangle]
extern "Rust" fn __recomp_rt_dispatch(address: u64) -> GuestResult {
    DISPATCHED.with(|dispatched| dispatched.set(Some(address)));
    Ok(())
}

#[no_mangle]
extern "Rust" fn __recomp_rt_count_calls(_count: u32) {}

thread_local! {
    static DISPATCHED: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

/// `recomp_abi::allow_direct_calls` is process-wide, so call-path tests do
/// not run at the same time.
static CALL_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn the_context_is_the_threads_register_file() {
    let context = guest_context();
    // SAFETY: the pointer is this thread's context and lives for the thread.
    unsafe {
        (*context).x[0] = 7;
        (*context).v[0] = 0x0011_2233_4455_6677_8899_aabb_ccdd_eeff;
        (*context).nzcv = 0b1010;
    }
    assert_eq!(recomp_abi::cpu_get(0), 7);
    assert_eq!(recomp_abi::cpu_regs()[0], 7);
    assert_eq!(
        recomp_abi::cpu_vregs()[0],
        0x0011_2233_4455_6677_8899_aabb_ccdd_eeff
    );
    assert_eq!(recomp_abi::cpu_nzcv(), 0b1010);

    recomp_abi::cpu_set(1, 9);
    recomp_abi::cpu_vstore(&[(2, 3)]);
    recomp_abi::cpu_set_nzcv(0b0101);
    // SAFETY: as above.
    unsafe {
        assert_eq!((*context).x[1], 9);
        assert_eq!((*context).v[2], 3);
        assert_eq!((*context).nzcv, 0b0101);
    }
}

/// The callee shape: load the registers it uses, compute, store them back.
/// It leaves X30 alone, as `ret` does.
unsafe fn callee(ctx: *mut GuestContext) -> GuestResult {
    let mut x0 = (*ctx).x[0];
    let x1 = (*ctx).x[1];
    x0 = x0.wrapping_add(x1);
    (*ctx).x[0] = x0;
    // cmp x0, x1 would leave Z clear; stand in for a flag-setting callee.
    (*ctx).nzcv = 0b0100;
    (*ctx).v[0] = 0x4000_0000_0000_0000;
    Ok(())
}

/// A callee that returns through a different X30 (rtld's `mov x30, x0; ret`,
/// longjmp), which must unwind the caller too.
unsafe fn callee_changes_x30(ctx: *mut GuestContext) -> GuestResult {
    (*ctx).x[30] = 0x9999;
    Ok(())
}

/// The caller shape the emitter produces around one direct call.
unsafe fn caller(
    ctx: *mut GuestContext,
    callee: unsafe fn(*mut GuestContext) -> GuestResult,
) -> GuestResult {
    let mut x0 = (*ctx).x[0];
    let mut x1 = (*ctx).x[1];
    let mut x30 = (*ctx).x[30];
    let mut v0 = (*ctx).v[0];
    let nzcv = (*ctx).nzcv;
    let mut flag_z = nzcv & 4 != 0;

    // Spill the live set before the call.
    (*ctx).x[0] = x0;
    (*ctx).x[1] = x1;
    (*ctx).x[30] = x30;
    (*ctx).v[0] = v0;
    (*ctx).nzcv = (flag_z as u8) << 2;
    recomp_abi::call(0x2000, || callee(ctx))?;

    // Reload it and take the callee's flags.
    x0 = (*ctx).x[0];
    x1 = (*ctx).x[1];
    x30 = (*ctx).x[30];
    v0 = (*ctx).v[0];
    let nzcv = (*ctx).nzcv;
    flag_z = nzcv & 4 != 0;

    // A callee that returned elsewhere unwinds this frame too.
    if x30 as u64 != 0x1004 {
        (*ctx).x[0] = x0;
        (*ctx).x[1] = x1;
        (*ctx).x[30] = x30;
        (*ctx).v[0] = v0;
        return Ok(());
    }

    // Record what survived: the sum and the callee's flags (its vector is
    // asserted separately).
    (*ctx).x[2] = x0.wrapping_add(i64::from(flag_z));
    (*ctx).x[0] = x0;
    (*ctx).x[1] = x1;
    (*ctx).x[30] = x30;
    (*ctx).v[0] = v0;
    Ok(())
}

#[test]
fn the_generated_call_shape_preserves_registers_flags_and_vectors() {
    let _guard = CALL_LOCK.lock().expect("call lock");
    recomp_abi::allow_direct_calls(true);

    let context = guest_context();
    // SAFETY: this thread's context.
    unsafe {
        (*context).x = [0; 32];
        (*context).x[0] = 5;
        (*context).x[1] = 7;
        (*context).x[30] = 0x1004;
        (*context).v[0] = 0;
        (*context).nzcv = 0;
        caller(context, callee).expect("call");
        // x0 = 5 + 7 = 12 (X30 matched and its flags arrived with Z set).
        assert_eq!((*context).x[0], 12);
        assert_eq!((*context).x[2], 12 + 1);
        assert_eq!((*context).v[0], 0x4000_0000_0000_0000);
    }
}

#[test]
fn a_callee_that_changes_x30_unwinds_the_caller() {
    let _guard = CALL_LOCK.lock().expect("call lock");
    recomp_abi::allow_direct_calls(true);

    let context = guest_context();
    // SAFETY: this thread's context.
    unsafe {
        (*context).x = [0; 32];
        (*context).x[0] = 5;
        (*context).x[1] = 7;
        (*context).x[2] = 0x55;
        (*context).x[30] = 0x1004;
        caller(context, callee_changes_x30).expect("call");
        // The caller returned before recording its result.
        assert_eq!((*context).x[2], 0x55);
        assert_eq!((*context).x[30], 0x9999);
    }
}

#[test]
fn calls_fall_back_to_dispatch_when_direct_calls_are_off() {
    let _guard = CALL_LOCK.lock().expect("call lock");
    recomp_abi::allow_direct_calls(false);
    DISPATCHED.with(|dispatched| dispatched.set(None));
    recomp_abi::call(0x1234, || panic!("the direct path must not run")).expect("dispatch result");
    assert_eq!(
        DISPATCHED.with(std::cell::Cell::get),
        Some(0x1234),
        "the runtime dispatch path was not used"
    );
}
