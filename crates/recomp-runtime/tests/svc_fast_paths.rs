//! The direct SVC paths added ahead of the generic dispatcher. They must
//! return exactly what `thread_svc`/`KernelSvcDispatcher` returned, while
//! SVCs that can block or need the dispatcher keep their old path.

use std::sync::Once;
use std::time::{Duration, Instant};

use recomp_runtime::{
    init_kernel_services, svc_dispatch, svc_dispatch_id, KernelSvcDispatcher, RuntimeError,
};
use recomp_services::kernel::{RESULT_INVALID_HANDLE, RESULT_INVALID_STATE, RESULT_TIMED_OUT};

const HEAP: u64 = 0x1_0000_0000;

fn init() {
    // One process-wide dispatcher; every test in this binary uses the full
    // allow-list. `svc_fast_paths_disallowed` covers the filtered list.
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        init_kernel_services(KernelSvcDispatcher::all_allowed(), HEAP, 0x1_0000);
    });
}

#[test]
fn sleep_thread_returns_zero_and_honours_zero_negative_and_positive() {
    init();
    let start = Instant::now();
    let yielded = svc_dispatch_id(0x0B, &[0]).expect("sleep 0");
    assert_eq!((yielded.x0, yielded.x1, yielded.x2), (0, None, None));
    assert!(
        start.elapsed() < Duration::from_secs(1),
        "yield did not sleep"
    );

    let start = Instant::now();
    let negative = svc_dispatch_id(0x0B, &[-5]).expect("sleep negative");
    assert_eq!((negative.x0, negative.x1, negative.x2), (0, None, None));
    assert!(
        start.elapsed() < Duration::from_secs(1),
        "negative sleep did not yield"
    );

    let start = Instant::now();
    let slept = svc_dispatch_id(0x0B, &[5_000_000]).expect("sleep 5ms");
    assert_eq!((slept.x0, slept.x1, slept.x2), (0, None, None));
    assert!(
        start.elapsed() >= Duration::from_millis(4),
        "5ms sleep returned in {:?}",
        start.elapsed()
    );
}

#[test]
fn get_system_tick_returns_the_counter_in_x0_only() {
    init();
    let first = svc_dispatch_id(0x1E, &[]).expect("tick");
    let second = svc_dispatch_id(0x1E, &[]).expect("tick");
    assert!(first.x0 > 0, "{first:?}");
    assert!(second.x0 >= first.x0, "tick went backwards");
    assert_eq!((first.x1, first.x2), (None, None));
    assert_eq!((second.x1, second.x2), (None, None));

    // The named dispatcher takes the same direct path and agrees.
    let named = svc_dispatch("svcGetSystemTick", &[]).expect("named tick");
    assert!(named.x0 >= second.x0);
    assert_eq!((named.x1, named.x2), (None, None));
}

#[test]
fn event_signal_clear_and_reset_keep_their_results() {
    init();
    let created = svc_dispatch_id(0x45, &[]).expect("create event");
    assert_eq!(created.x0, 0);
    let write = created.x1.expect("writable handle") as u32;
    let read = created.x2.expect("readable handle") as u32;

    assert_eq!(
        svc_dispatch_id(0x11, &[i64::from(write)])
            .expect("signal")
            .x0,
        0
    );
    // ResetSignal clears a signalled event, then reports InvalidState.
    assert_eq!(
        svc_dispatch_id(0x17, &[i64::from(read)])
            .expect("reset signalled")
            .x0,
        0
    );
    assert_eq!(
        svc_dispatch_id(0x17, &[i64::from(read)])
            .expect("reset cleared")
            .x0,
        RESULT_INVALID_STATE as i64
    );
    assert_eq!(
        svc_dispatch_id(0x12, &[i64::from(read)]).expect("clear").x0,
        0
    );
    // Only the writable side signals.
    assert_eq!(
        svc_dispatch_id(0x11, &[i64::from(read)])
            .expect("signal read side")
            .x0,
        RESULT_INVALID_HANDLE as i64
    );
}

#[test]
fn thread_priority_and_core_mask_paths_match_the_dispatcher() {
    init();
    assert_eq!(
        svc_dispatch_id(0x0D, &[0x7777, 60])
            .expect("set priority")
            .x0,
        0
    );
    let priority = svc_dispatch_id(0x0C, &[0, 0x7777]).expect("get priority");
    assert_eq!((priority.x0, priority.x1), (0, Some(60)));

    let mask = svc_dispatch_id(0x0E, &[0x7777, 0, 0]).expect("get core mask");
    assert_eq!((mask.x0, mask.x1, mask.x2), (0, Some(0), Some(0xF)));
    assert_eq!(
        svc_dispatch_id(0x0F, &[0x7777, 0, 0])
            .expect("set core mask")
            .x0,
        0
    );
    assert_eq!(svc_dispatch_id(0x10, &[]).expect("processor number").x0, 0);

    let process = svc_dispatch_id(0x24, &[]).expect("process id");
    assert_eq!(
        (process.x0, process.x1, process.x2),
        (0, Some(0x1_234_567_890), None)
    );

    let created =
        svc_dispatch_id(0x08, &[0, 0x2000, 0, 0x7FFF_0000, 44, 0]).expect("create thread");
    let handle = created.x1.expect("thread handle");
    let id = svc_dispatch_id(0x25, &[0, handle]).expect("thread id");
    assert_eq!(id.x0, 0);
    assert!(id.x1.is_some_and(|id| id > 0), "{id:?}");
}

#[test]
fn light_synchronisation_signals_keep_the_thread_svc_results() {
    init();
    // No waiters and no mapped memory: these calls must still succeed and
    // return 0 rather than fault, exactly as `thread_svc` did.
    assert_eq!(
        svc_dispatch_id(0x1B, &[0x1234])
            .expect("arbitrate unlock")
            .x0,
        0
    );
    assert_eq!(
        svc_dispatch_id(0x1D, &[0x1234, 1])
            .expect("signal process-wide key")
            .x0,
        0
    );
    assert_eq!(
        svc_dispatch_id(0x35, &[0x1234, 0, 0, 1])
            .expect("signal to address")
            .x0,
        0
    );
}

#[test]
fn close_handle_closes_kernel_objects_and_keeps_the_generic_fallback() {
    init();
    let created = svc_dispatch_id(0x45, &[]).expect("create event");
    let write = created.x1.expect("writable handle");
    assert_eq!(svc_dispatch_id(0x16, &[write]).expect("close").x0, 0);
    assert_eq!(
        svc_dispatch_id(0x11, &[write]).expect("signal closed").x0,
        RESULT_INVALID_HANDLE as i64
    );
    // A handle the kernel does not own falls through to the dispatcher,
    // which succeeds vacuously without an IPC hub.
    assert_eq!(svc_dispatch_id(0x16, &[write]).expect("close again").x0, 0);
}

#[test]
fn blocking_wait_synchronization_still_uses_the_generic_path() {
    init();
    // No handles and a zero timeout: the wait poll returns TimedOut without
    // blocking or touching guest memory.
    let waited = svc_dispatch_id(0x18, &[0, 0, 0, 0]).expect("wait");
    assert_eq!((waited.x0, waited.x1), (RESULT_TIMED_OUT as i64, None));
}

#[test]
fn unknown_ids_still_get_the_diagnostic_error() {
    init();
    let error = svc_dispatch_id(0x1234, &[]).expect_err("unknown id");
    assert!(
        matches!(
            error,
            RuntimeError::StubbedSyscall { ref name } if name == "svc 0x1234"
        ),
        "{error}"
    );
}
