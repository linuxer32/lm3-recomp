//! A filtered allow-list must keep the old outcomes: ids the dispatcher
//! serves need `allows` before a direct path, while threads/events are
//! handled before the generic allow-list check (as `thread_svc` was).

use std::sync::Once;

use recomp_runtime::{init_kernel_services, svc_dispatch_id, KernelSvcDispatcher, RuntimeError};

const HEAP: u64 = 0x1_0000_0000;

fn init() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let mut allowed = KernelSvcDispatcher::all_allowed();
        for id in [0x0E, 0x10, 0x1E] {
            allowed.retain(|allowed| *allowed != id);
        }
        init_kernel_services(allowed, HEAP, 0x1_0000);
    });
}

#[test]
fn disallowed_dispatcher_ids_fall_back_to_the_normal_error() {
    init();
    for (id, name) in [
        (0x1E, "svcGetSystemTick"),
        (0x10, "svcGetCurrentProcessorNumber"),
    ] {
        let expected = format!("{name} (0x{id:02x})");
        let error = svc_dispatch_id(id, &[]).expect_err("disallowed id");
        assert!(
            matches!(
                error,
                RuntimeError::UnimplementedSvc { ref name } if name == &expected
            ),
            "{error}"
        );
    }
    let error = svc_dispatch_id(0x0E, &[0x7777, 0, 0]).expect_err("disallowed core mask");
    assert!(
        matches!(
            error,
            RuntimeError::UnimplementedSvc { ref name }
                if name == "svcGetThreadCoreMask (0x0e)"
        ),
        "{error}"
    );
}

#[test]
fn thread_svcs_keep_the_thread_svc_precedence_over_the_allow_list() {
    init();
    // svcSetThreadPriority/GetThreadPriority are served by `thread_svc`
    // before the dispatcher's allow-list check, so they still work.
    assert_eq!(
        svc_dispatch_id(0x0D, &[0x8888, 59])
            .expect("set priority")
            .x0,
        0
    );
    assert_eq!(
        svc_dispatch_id(0x0C, &[0, 0x8888])
            .expect("get priority")
            .x1,
        Some(59)
    );
    // svcSleepThread likewise.
    assert_eq!(svc_dispatch_id(0x0B, &[0]).expect("sleep").x0, 0);
}
