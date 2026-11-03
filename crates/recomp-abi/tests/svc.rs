//! `recomp_abi::svc_dispatch_id` is the generated code's SVC entry point: it
//! must forward the numeric id and arguments to the runtime without building
//! a name.

use recomp_abi::{svc_dispatch_id, Fault, SvcOutcome};
use std::cell::Cell;

thread_local! {
    static CAPTURED: Cell<Option<(u64, Vec<i64>)>> = const { Cell::new(None) };
}

/// Stands in for the runtime symbol the generated project links against.
#[no_mangle]
extern "Rust" fn __recomp_rt_svc_id(id: u64, args: &[i64]) -> Result<SvcOutcome, Fault> {
    CAPTURED.with(|captured| captured.set(Some((id, args.to_vec()))));
    Ok(SvcOutcome {
        x0: 7,
        x1: Some(8),
        x2: None,
    })
}

#[test]
fn numeric_dispatch_forwards_the_id_and_arguments() {
    let outcome = svc_dispatch_id(0x21, &[11, 22]).expect("dispatch");
    assert_eq!((outcome.x0, outcome.x1, outcome.x2), (7, Some(8), None));
    let (id, args) = CAPTURED
        .with(|captured| captured.take())
        .expect("runtime called");
    assert_eq!(id, 0x21);
    assert_eq!(args, vec![11, 22]);
}
