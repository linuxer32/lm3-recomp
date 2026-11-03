//! The numeric SVC entry point generated code uses. It must behave exactly
//! like the string dispatcher for known ids, and keep the diagnostic failure
//! for ids and names outside the kernel table.

use recomp_runtime::{
    init_kernel_services, svc_dispatch, svc_dispatch_id, KernelSvcDispatcher, RuntimeError,
};

const HEAP: u64 = 0x1000_0000;

#[test]
fn numeric_and_named_dispatch_round_trip() {
    init_kernel_services(KernelSvcDispatcher::all_allowed(), HEAP, 0x10000);

    let by_id = svc_dispatch_id(0x1E, &[]).expect("numeric svcGetSystemTick");
    let by_name = svc_dispatch("svcGetSystemTick", &[]).expect("named svcGetSystemTick");
    assert!(by_id.x0 > 0, "system tick in x0");
    assert!(by_name.x0 >= by_id.x0, "tick advances between calls");
    assert_eq!(by_id.x1, None);
    assert_eq!(by_name.x1, None);
}

#[test]
fn unknown_id_is_a_stubbed_syscall() {
    let error = svc_dispatch_id(0x1234, &[]).expect_err("unknown id");
    assert!(
        matches!(
            error,
            RuntimeError::StubbedSyscall { ref name } if name == "svc 0x1234"
        ),
        "{error}"
    );
}

#[test]
fn unknown_name_keeps_its_diagnostic() {
    let error = svc_dispatch("svcNotAThing", &[]).expect_err("unknown name");
    assert!(
        matches!(
            error,
            RuntimeError::StubbedSyscall { ref name } if name == "svcNotAThing"
        ),
        "{error}"
    );
}
