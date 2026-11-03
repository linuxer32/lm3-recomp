use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

pub mod account;
pub mod am;
pub mod assets;
pub mod audio;
pub mod clock;
mod file_at;
pub mod fs;
pub mod gpu;
pub mod hid;
pub mod hipc;
pub mod host1x;
pub mod ipc;
pub mod kernel;
pub mod network;
pub mod nvdrv;
pub mod online;
pub mod opus;
pub mod romfs;
pub mod server;
pub mod services;
pub mod system;
pub mod vi;
pub mod vp9;
mod vp9_tables;

pub use ipc::{DirectoryEntry, IpcHub, MemoryAccess, IPC_RESULT_BAD_MESSAGE, IPC_RESULT_OK};

#[derive(Debug, Clone)]
pub struct ServiceCall {
    pub client: String,
    pub service: String,
    pub args: Vec<i64>,
}

#[derive(Debug, thiserror::Error)]
pub enum ServiceError {
    #[error("service not found: {0}")]
    NotFound(String),
    #[error("access denied: {0}")]
    AccessDenied(String),
    #[error("stubbed service: {0}")]
    Stubbed(String),
}

pub type ServiceResult<T> = Result<T, ServiceError>;

type ServiceHandler = Box<dyn Fn(&ServiceCall) -> ServiceResult<()> + Send + Sync>;

#[derive(Default)]
pub struct ServiceRegistry {
    handlers: BTreeMap<String, ServiceHandler>,
}

impl ServiceRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<F>(&mut self, name: &str, handler: F)
    where
        F: Fn(&ServiceCall) -> ServiceResult<()> + Send + Sync + 'static,
    {
        self.handlers.insert(name.to_string(), Box::new(handler));
    }

    pub fn call(&self, call: &ServiceCall) -> ServiceResult<()> {
        let handler = self
            .handlers
            .get(&call.service)
            .ok_or_else(|| ServiceError::NotFound(call.service.clone()))?;
        handler(call)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StubBehavior {
    Log,
    Noop,
    Panic,
}

pub fn stub_handler(
    behavior: StubBehavior,
) -> impl Fn(&ServiceCall) -> ServiceResult<()> + Send + Sync {
    move |call: &ServiceCall| match behavior {
        StubBehavior::Log => {
            println!(
                "[recomp-services] stub {service} args={:?}",
                call.args,
                service = call.service
            );
            Ok(())
        }
        StubBehavior::Noop => Ok(()),
        StubBehavior::Panic => Err(ServiceError::Stubbed(call.service.clone())),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceStubSpec {
    pub name: String,
    pub behavior: StubBehavior,
}

impl ServiceStubSpec {
    pub fn new(name: impl Into<String>, behavior: StubBehavior) -> Self {
        Self {
            name: name.into(),
            behavior,
        }
    }
}

pub fn register_stubbed_services(registry: &mut ServiceRegistry, stubs: &[ServiceStubSpec]) {
    for stub in stubs {
        registry.register(&stub.name, stub_handler(stub.behavior));
    }
}

#[derive(Debug, Default)]
pub struct ServiceAccessControl {
    allowed: BTreeSet<String>,
}

impl ServiceAccessControl {
    pub fn from_allowed(names: impl IntoIterator<Item = String>) -> Self {
        Self {
            allowed: names.into_iter().collect(),
        }
    }

    pub fn check(&self, call: &ServiceCall) -> ServiceResult<()> {
        if self.allowed.is_empty() {
            return Ok(());
        }
        if self.allowed.contains(&call.service) {
            Ok(())
        } else {
            Err(ServiceError::AccessDenied(call.service.clone()))
        }
    }
}

/// Log every SVC the dispatcher handles and every IPC request
/// (`RECOMP_TRACE_IPC=1`, or `RECOMP_TRACE_IPC_AFTER=<seconds>` to start
/// that long after the first check). Off by default: titles make millions
/// of calls (LM3 polls hid and audio in tight loops), and with guest
/// threads in parallel the lines also serialize them on stdout.
pub fn trace_ipc() -> bool {
    static TRACE: std::sync::OnceLock<Option<std::time::Instant>> = std::sync::OnceLock::new();
    let start = TRACE.get_or_init(|| {
        if std::env::var_os("RECOMP_TRACE_IPC").is_some() {
            return Some(std::time::Instant::now());
        }
        let seconds: u64 = std::env::var("RECOMP_TRACE_IPC_AFTER").ok()?.parse().ok()?;
        Some(std::time::Instant::now() + std::time::Duration::from_secs(seconds))
    });
    start.is_some_and(|start| std::time::Instant::now() >= start)
}

#[derive(Debug, Default)]
pub struct ServiceLogger {
    counter: AtomicU64,
}

impl ServiceLogger {
    pub fn next_id(&self) -> u64 {
        self.counter.fetch_add(1, Ordering::SeqCst)
    }

    pub fn log_call(&self, call: &ServiceCall) {
        let id = self.next_id();
        if !trace_ipc() {
            return;
        }
        println!(
            "[recomp-services] #{id} client={} service={} args={:?}",
            call.client, call.service, call.args
        );
    }
}

pub struct ServiceDispatcher {
    registry: ServiceRegistry,
    access: ServiceAccessControl,
    logger: ServiceLogger,
}

impl ServiceDispatcher {
    pub fn new(
        registry: ServiceRegistry,
        access: ServiceAccessControl,
        logger: ServiceLogger,
    ) -> Self {
        Self {
            registry,
            access,
            logger,
        }
    }

    pub fn dispatch(&self, call: &ServiceCall) -> ServiceResult<()> {
        self.access.check(call)?;
        self.logger.log_call(call);
        self.registry.call(call)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_calls_handler() {
        let mut registry = ServiceRegistry::new();
        registry.register("svc_test", |_| Ok(()));

        let call = ServiceCall {
            client: "demo".to_string(),
            service: "svc_test".to_string(),
            args: vec![1, 2],
        };

        assert!(registry.call(&call).is_ok());
    }

    #[test]
    fn access_control_denies_unknown() {
        let access = ServiceAccessControl::from_allowed(vec!["svc_ok".to_string()]);
        let call = ServiceCall {
            client: "demo".to_string(),
            service: "svc_bad".to_string(),
            args: vec![],
        };

        let err = access.check(&call).unwrap_err();
        assert!(matches!(err, ServiceError::AccessDenied(_)));
    }

    #[test]
    fn stub_panic_returns_error() {
        let handler = stub_handler(StubBehavior::Panic);
        let call = ServiceCall {
            client: "demo".to_string(),
            service: "svc_stub".to_string(),
            args: vec![],
        };
        let err = handler(&call).unwrap_err();
        assert!(matches!(err, ServiceError::Stubbed(_)));
    }

    #[test]
    fn dispatcher_checks_access_and_logs() {
        let mut registry = ServiceRegistry::new();
        registry.register("svc_ok", |_| Ok(()));
        let access = ServiceAccessControl::from_allowed(vec!["svc_ok".to_string()]);
        let dispatcher = ServiceDispatcher::new(registry, access, ServiceLogger::default());
        let call = ServiceCall {
            client: "demo".to_string(),
            service: "svc_ok".to_string(),
            args: vec![42],
        };

        assert!(dispatcher.dispatch(&call).is_ok());
    }

    #[test]
    fn register_stubbed_services_installs_handlers() {
        let mut registry = ServiceRegistry::new();
        register_stubbed_services(
            &mut registry,
            &[ServiceStubSpec::new("svc_stub", StubBehavior::Noop)],
        );
        let call = ServiceCall {
            client: "demo".to_string(),
            service: "svc_stub".to_string(),
            args: vec![],
        };
        assert!(registry.call(&call).is_ok());
    }
}

/// Kernel (supervisor) SVC identifiers, using the NPDM allow-list numbering.
pub const KERNEL_SVC_TABLE: &[(u64, &str)] = &[
    (0x01, "svcSetHeapSize"),
    (0x02, "svcSetMemoryPermission"),
    (0x03, "svcSetMemoryAttribute"),
    (0x04, "svcMapMemory"),
    (0x05, "svcUnmapMemory"),
    (0x06, "svcQueryMemory"),
    (0x07, "svcExitProcess"),
    (0x08, "svcCreateThread"),
    (0x09, "svcStartThread"),
    (0x0A, "svcExitThread"),
    (0x0B, "svcSleepThread"),
    (0x0C, "svcGetThreadPriority"),
    (0x0D, "svcSetThreadPriority"),
    (0x0E, "svcGetThreadCoreMask"),
    (0x0F, "svcSetThreadCoreMask"),
    (0x10, "svcGetCurrentProcessorNumber"),
    (0x11, "svcSignalEvent"),
    (0x12, "svcClearEvent"),
    (0x13, "svcMapSharedMemory"),
    (0x14, "svcUnmapSharedMemory"),
    (0x15, "svcCreateTransferMemory"),
    (0x16, "svcCloseHandle"),
    (0x17, "svcResetSignal"),
    (0x18, "svcWaitSynchronization"),
    (0x19, "svcCancelSynchronization"),
    (0x1A, "svcArbitrateLock"),
    (0x1B, "svcArbitrateUnlock"),
    (0x1C, "svcWaitProcessWideKeyAtomic"),
    (0x1D, "svcSignalProcessWideKey"),
    (0x1E, "svcGetSystemTick"),
    (0x1F, "svcConnectToNamedPort"),
    (0x20, "svcSendSyncRequestLight"),
    (0x21, "svcSendSyncRequest"),
    (0x22, "svcSendSyncRequestWithUserBuffer"),
    (0x23, "svcSendAsyncRequestWithUserBuffer"),
    (0x24, "svcGetProcessId"),
    (0x25, "svcGetThreadId"),
    (0x26, "svcBreak"),
    (0x27, "svcOutputDebugString"),
    (0x28, "svcReturnFromException"),
    (0x29, "svcGetInfo"),
    (0x2A, "svcFlushEntireDataCache"),
    (0x2B, "svcFlushDataCache"),
    (0x2C, "svcMapPhysicalMemory"),
    (0x2D, "svcUnmapPhysicalMemory"),
    (0x2E, "svcGetDebugFutureThreadInfo"),
    (0x2F, "svcGetLastThreadInfo"),
    (0x30, "svcGetResourceLimitLimitValue"),
    (0x31, "svcGetResourceLimitCurrentValue"),
    (0x32, "svcSetThreadActivity"),
    (0x33, "svcGetThreadContext3"),
    (0x34, "svcWaitForAddress"),
    (0x35, "svcSignalToAddress"),
    (0x36, "svcSynchronizePreemptionState"),
    (0x37, "svcGetResourceLimitPeakValue"),
    (0x39, "svcCreateIoPool"),
    (0x3A, "svcCreateIoRegion"),
    (0x3C, "svcKernelDebug"),
    (0x3D, "svcChangeKernelTraceState"),
    (0x40, "svcCreateSession"),
    (0x41, "svcAcceptSession"),
    (0x42, "svcReplyAndReceiveLight"),
    (0x43, "svcReplyAndReceive"),
    (0x44, "svcReplyAndReceiveWithUserBuffer"),
    (0x45, "svcCreateEvent"),
    (0x46, "svcMapIoRegion"),
    (0x47, "svcUnmapIoRegion"),
    (0x48, "svcMapPhysicalMemoryUnsafe"),
    (0x49, "svcUnmapPhysicalMemoryUnsafe"),
    (0x4A, "svcSetUnsafeLimit"),
    (0x4B, "svcCreateCodeMemory"),
    (0x4C, "svcControlCodeMemory"),
    (0x4D, "svcSleepSystem"),
    (0x4E, "svcReadWriteRegister"),
    (0x4F, "svcSetProcessActivity"),
    (0x50, "svcCreateSharedMemory"),
    (0x51, "svcMapTransferMemory"),
    (0x52, "svcUnmapTransferMemory"),
    (0x53, "svcCreateInterruptEvent"),
    (0x54, "svcQueryPhysicalAddress"),
    (0x55, "svcQueryMemoryMapping"),
    (0x56, "svcCreateDeviceAddressSpace"),
    (0x57, "svcAttachDeviceAddressSpace"),
    (0x58, "svcDetachDeviceAddressSpace"),
    (0x59, "svcMapDeviceAddressSpaceByForce"),
    (0x5A, "svcMapDeviceAddressSpaceAligned"),
    (0x5B, "svcMapDeviceAddressSpace"),
    (0x5C, "svcUnmapDeviceAddressSpace"),
    (0x5D, "svcInvalidateProcessDataCache"),
    (0x5E, "svcStoreProcessDataCache"),
    (0x5F, "svcFlushProcessDataCache"),
    (0x60, "svcDebugActiveProcess"),
    (0x61, "svcBreakDebugProcess"),
    (0x62, "svcTerminateDebugProcess"),
    (0x63, "svcGetDebugEvent"),
    (0x64, "svcContinueDebugEvent"),
    (0x65, "svcGetProcessList"),
    (0x66, "svcGetThreadList"),
    (0x67, "svcGetDebugThreadContext"),
    (0x68, "svcSetDebugThreadContext"),
    (0x69, "svcQueryDebugProcessMemory"),
    (0x6A, "svcReadDebugProcessMemory"),
    (0x6B, "svcWriteDebugProcessMemory"),
    (0x6C, "svcSetHardwareBreakPoint"),
    (0x6D, "svcGetDebugThreadParam"),
    (0x6F, "svcGetSystemInfo"),
    (0x70, "svcCreatePort"),
    (0x71, "svcManageNamedPort"),
    (0x72, "svcConnectToPort"),
    (0x73, "svcSetProcessMemoryPermission"),
    (0x74, "svcMapProcessMemory"),
    (0x75, "svcUnmapProcessMemory"),
    (0x76, "svcQueryProcessMemory"),
    (0x77, "svcMapProcessCodeMemory"),
    (0x78, "svcUnmapProcessCodeMemory"),
    (0x79, "svcCreateProcess"),
    (0x7A, "svcStartProcess"),
    (0x7B, "svcTerminateProcess"),
    (0x7C, "svcGetProcessInfo"),
    (0x7D, "svcCreateResourceLimit"),
    (0x7E, "svcSetResourceLimitLimitValue"),
    (0x7F, "svcCallSecureMonitor"),
];

pub const KERNEL_SVC_NOT_ALLOWED: u64 = 0x340;
pub const KERNEL_SVC_UNSUPPORTED: u64 = 0x1;
/// Memory available to an application (Horizon's 3285 MiB application pool).
pub const APPLICATION_POOL_SIZE: u64 = 0xCD50_0000;
/// Memory in use before the heap: code, stacks, kernel objects.
const BASE_USED_MEMORY: u64 = 0x0400_0000;
/// svc::ResultInvalidSize (module 1, description 101).
pub const KERNEL_RESULT_INVALID_SIZE: u64 = 0xCA01;
/// svc::ResultInvalidPointer (module 1, description 106).
pub const KERNEL_RESULT_INVALID_POINTER: u64 = 0xD401;

pub fn kernel_svc_name(id: u64) -> Option<&'static str> {
    // Only tracing and cold diagnostics need the reverse lookup, but build
    // the map once from the same table rather than scanning per call.
    static BY_ID: OnceLock<HashMap<u64, &'static str>> = OnceLock::new();
    BY_ID
        .get_or_init(|| {
            KERNEL_SVC_TABLE
                .iter()
                .map(|(id, name)| (*id, *name))
                .collect()
        })
        .get(&id)
        .copied()
}

/// Which general-purpose registers an SVC writes in addition to W0, derived
/// from the kernel ABI and the SDK's wrapper convention. Used by codegen to
/// know which guest registers become live after the call.
pub fn kernel_svc_writes(id: u64) -> (bool, bool) {
    match id {
        // QueryMemory -> PageInfo in X1; GetInfo -> value in X1.
        0x06 | 0x29 => (true, false),
        // GetThreadCoreMask(handle, out_priority, out_core_mask):
        // priority in W1, core mask in X2.
        0x0E => (true, true),
        // CreateSession, CreateEvent and CreatePort return two handles.
        0x40 | 0x45 | 0x70 => (true, true),
        // All other SVCs with documented payloads return it in X1 only
        // (WaitSynchronization: the signaled handle index).
        0x01 | 0x08 | 0x0C | 0x15 | 0x18 | 0x1F | 0x24 | 0x25 | 0x41 | 0x50 | 0x53 | 0x72 => {
            (true, false)
        }
        _ => (false, false),
    }
}

pub fn kernel_svc_id(name: &str) -> Option<u64> {
    // Generated SVC calls bypass this lookup (the pipeline resolves the id
    // at recompilation time); string dispatch, thread yields and diagnostics
    // still use it, so build the map once instead of scanning the table.
    static BY_NAME: OnceLock<HashMap<&'static str, u64>> = OnceLock::new();
    BY_NAME
        .get_or_init(|| {
            KERNEL_SVC_TABLE
                .iter()
                .map(|(id, name)| (*name, *id))
                .collect()
        })
        .get(name)
        .copied()
}

/// Whether `id` names a kernel SVC in `KERNEL_SVC_TABLE`, without the map
/// probe `kernel_svc_name` makes. Generated code resolves names to ids at
/// recompilation time, so the numeric SVC path only needs this check and a
/// single small-table load.
#[inline]
pub fn kernel_svc_known(id: u64) -> bool {
    id < KERNEL_SVC_KNOWN.len() as u64 && KERNEL_SVC_KNOWN[id as usize]
}

/// The `KERNEL_SVC_TABLE` ids as a lookup table. Every id is below 0x80.
const KERNEL_SVC_KNOWN: [bool; 128] = {
    let mut known = [false; 128];
    let mut index = 0;
    while index < KERNEL_SVC_TABLE.len() {
        let id = KERNEL_SVC_TABLE[index].0 as usize;
        if id < known.len() {
            known[id] = true;
        }
        index += 1;
    }
    known
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelSvcResult {
    /// Syscall result code (W0 on return).
    pub code: u64,
    pub x0: u64,
    /// Some(value) when the kernel writes X1 for this SVC.
    pub x1: Option<u64>,
    /// Some(value) when the kernel writes X2 for this SVC.
    pub x2: Option<u64>,
    /// Set for svcExitProcess / svcReturnFromException style exits.
    pub terminates: bool,
}

impl KernelSvcResult {
    pub fn ok() -> Self {
        Self {
            code: 0,
            x0: 0,
            x1: None,
            x2: None,
            terminates: false,
        }
    }

    pub fn ok_with(x0: u64) -> Self {
        Self {
            code: 0,
            x0,
            x1: None,
            x2: None,
            terminates: false,
        }
    }

    /// Kernel convention for SVCs that report status in W0 and payloads in
    /// X1/X2 (handles, addresses, counts).
    pub fn ok_out(x1: u64, x2: u64) -> Self {
        Self {
            code: 0,
            x0: 0,
            x1: Some(x1),
            x2: Some(x2),
            terminates: false,
        }
    }

    /// SVC writing only X1 (svcQueryMemory PageInfo, svcGetInfo value).
    pub fn ok_out1(x1: u64) -> Self {
        Self {
            code: 0,
            x0: 0,
            x1: Some(x1),
            x2: None,
            terminates: false,
        }
    }

    pub fn err(code: u64) -> Self {
        Self {
            code,
            x0: 0,
            x1: None,
            x2: None,
            terminates: false,
        }
    }
}

#[derive(Debug)]
pub struct KernelSvcState {
    heap_base: Option<u64>,
    heap_size: u64,
    next_handle: u64,
    tick: u64,
    thread_id: u64,
    /// Recent PageInfo values; the kernel never returns the same one twice.
    page_info: u32,
}

impl Default for KernelSvcState {
    fn default() -> Self {
        Self {
            heap_base: None,
            heap_size: 0,
            next_handle: 1,
            tick: 0,
            thread_id: 1,
            page_info: 0,
        }
    }
}

/// The stack region a 39-bit process gets by default: 2 GiB at 128 GiB.
pub const DEFAULT_STACK_REGION: (u64, u64) = (0x20_0000_0000, 0x8000_0000);

#[derive(Debug)]
pub struct KernelSvcDispatcher {
    allowed: Vec<u64>,
    state: std::sync::Mutex<KernelSvcState>,
    heap_base_address: u64,
    max_heap_size: u64,
    /// StackRegionAddress and Size (GetInfo 14 and 15): where the SDK
    /// aliases thread stacks (svcMapMemory).
    stack_region: (u64, u64),
    logger: ServiceLogger,
    ipc: std::sync::Mutex<Option<crate::ipc::IpcHub>>,
    memory: std::sync::Mutex<Option<std::sync::Arc<dyn crate::ipc::MemoryAccess>>>,
}

impl KernelSvcDispatcher {
    pub fn new(allowed: Vec<u64>, heap_base_address: u64, max_heap_size: u64) -> Self {
        Self {
            allowed,
            state: std::sync::Mutex::new(KernelSvcState::default()),
            heap_base_address,
            max_heap_size,
            stack_region: DEFAULT_STACK_REGION,
            logger: ServiceLogger {
                counter: std::sync::atomic::AtomicU64::new(0),
            },
            ipc: std::sync::Mutex::new(None),
            memory: std::sync::Mutex::new(None),
        }
    }

    /// Report the stack region at `base` (`size` bytes) instead of
    /// [`DEFAULT_STACK_REGION`]: a host that keeps only part of the guest
    /// address space directly accessible places it there.
    pub fn with_stack_region(mut self, base: u64, size: u64) -> Self {
        self.stack_region = (base, size);
        self
    }

    /// Attach the IPC/fs: hub so svcConnectToNamedPort/svcSendSyncRequest*
    /// dispatch against emulated services instead of succeeding vacuously.
    pub fn attach_ipc(&self, hub: crate::ipc::IpcHub) {
        *self
            .ipc
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(hub);
    }

    fn ipc_locked(&self) -> std::sync::MutexGuard<'_, Option<crate::ipc::IpcHub>> {
        self.ipc
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Attach guest-memory access used by memory-reporting SVCs
    /// (svcQueryMemory, svcGetInfo).
    pub fn attach_memory(&self, memory: std::sync::Arc<dyn crate::ipc::MemoryAccess>) {
        *self
            .memory
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(memory);
    }

    fn memory_access(&self) -> Option<std::sync::Arc<dyn crate::ipc::MemoryAccess>> {
        self.memory
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Allow-list covering every kernel SVC in the table (dev-mode default).
    pub fn all_allowed() -> Vec<u64> {
        KERNEL_SVC_TABLE.iter().map(|(id, _)| *id).collect()
    }

    /// Whether `id` is on the allow-list. The hot numeric path uses this to
    /// take a direct handler without entering `dispatch`; a disallowed id
    /// falls back to `dispatch` for its normal error.
    #[inline]
    pub fn allows(&self, id: u64) -> bool {
        self.allowed.contains(&id)
    }

    /// The IPC SVCs, routed to the fs:/HIPC hub when one is attached. They
    /// run without the dispatcher's locks: a service may block (vi's
    /// DequeueBuffer waits for a vsync), and other threads' SVCs must go on
    /// meanwhile, or the guest deadlocks.
    fn dispatch_ipc(&self, id: u64, args: &[u64]) -> Option<KernelSvcResult> {
        if !matches!(id, 0x16 | 0x1F | 0x21 | 0x22 | 0x23) {
            return None;
        }
        let hub = self.ipc_locked().clone()?;
        let arg = |index: usize| args.get(index).copied().unwrap_or(0);
        Some(match id {
            0x1F => {
                let (code, handle) = hub.connect_to_named_port(arg(1));
                if code != IPC_RESULT_OK {
                    return Some(KernelSvcResult::err(code));
                }
                KernelSvcResult::ok_out(handle, 0)
            }
            0x16 => KernelSvcResult::ok_with(hub.close_handle(arg(0))),
            _ => {
                // SendSyncRequest(handle); SendSyncRequestWithUserBuffer
                // and its async form (buffer, size, handle).
                let (handle, buffer) = if id == 0x21 {
                    (arg(0), arg(1))
                } else {
                    (arg(2), arg(0))
                };
                let code = hub.send_sync_request(handle, buffer);
                if code != IPC_RESULT_OK {
                    return Some(KernelSvcResult::err(code));
                }
                KernelSvcResult::ok()
            }
        })
    }

    pub fn dispatch(&self, id: u64, args: &[u64]) -> KernelSvcResult {
        // The name is only needed for tracing. Resolving it unconditionally
        // put a map probe on every SVC, including the hot numeric path.
        if !self.allowed.contains(&id) {
            if trace_ipc() {
                self.logger.log_call(&ServiceCall {
                    client: "kernel".to_string(),
                    service: kernel_svc_name(id).unwrap_or("svc_unknown").to_string(),
                    args: args.iter().map(|value| *value as i64).collect(),
                });
            }
            return KernelSvcResult::err(KERNEL_SVC_NOT_ALLOWED);
        }
        if trace_ipc() {
            if let Some(name) = kernel_svc_name(id) {
                self.logger.log_call(&ServiceCall {
                    client: "kernel".to_string(),
                    service: name.to_string(),
                    args: args.iter().map(|value| *value as i64).collect(),
                });
            }
        }

        if let Some(result) = self.dispatch_ipc(id, args) {
            return result;
        }
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match id {
            0x01 => {
                // Kernel ABI: size in X1, heap address returned in X1.
                let requested = args.get(1).copied().unwrap_or(0);
                let size = requested.next_multiple_of(0x1000_u64);
                if size == 0 || size > self.max_heap_size {
                    return KernelSvcResult::err(KERNEL_RESULT_INVALID_SIZE);
                }
                state.heap_base = Some(self.heap_base_address);
                state.heap_size = size;
                KernelSvcResult::ok_out(self.heap_base_address, 0)
            }
            0x07 => KernelSvcResult {
                code: args.first().copied().unwrap_or(0),
                x0: 0,
                x1: None,
                x2: None,
                terminates: true,
            },
            0x28 => KernelSvcResult {
                code: 0,
                x0: 0,
                x1: None,
                x2: None,
                terminates: true,
            },
            0x06 => {
                // QueryMemory(out_memory_info=X0, out_page_info=X1, address=X2).
                let out_info = args.first().copied().unwrap_or(0);
                let address = args.get(2).copied().unwrap_or(0);
                let Some(block) = self
                    .memory_access()
                    .and_then(|memory| memory.query_block(address).map(|block| (memory, block)))
                else {
                    return KernelSvcResult::err(KERNEL_SVC_UNSUPPORTED);
                };
                let (memory, block) = block;
                // MemoryInfo: addr, size, state, attribute, permission,
                // ipc_refcount, device_refcount, padding.
                let mut info = [0_u8; 0x28];
                info[0..8].copy_from_slice(&block.base.to_le_bytes());
                info[8..16].copy_from_slice(&block.size.to_le_bytes());
                info[16..20].copy_from_slice(&block.state.to_le_bytes());
                info[24..28].copy_from_slice(&block.perm.to_le_bytes());
                if !memory.write_bytes(out_info, &info) {
                    return KernelSvcResult::err(KERNEL_RESULT_INVALID_POINTER);
                }
                // PageInfo bit layout is opaque to user code; unique per call.
                state.page_info = state.page_info.wrapping_add(1);
                KernelSvcResult::ok_out1((state.page_info & 0xFFFF) as u64)
            }
            0x29 => {
                // GetInfo(out=X0, info_type=X1, handle=X2, subtype=X3).
                let out = args.first().copied().unwrap_or(0);
                let info_type = args.get(1).copied().unwrap_or(0);
                let subtype = args.get(3).copied().unwrap_or(0);
                let value: u64 = match info_type {
                    0 => 0xF,
                    1 => 0x3F,
                    // AliasRegionAddress / Size: 64 GiB above the heap.
                    2 => 0x10_0000_0000,
                    3 => 0x10_0000_0000,
                    4 => state.heap_base.unwrap_or(self.heap_base_address),
                    5 => {
                        if state.heap_size == 0 {
                            self.max_heap_size
                        } else {
                            state.heap_size
                        }
                    }
                    // TotalMemorySize / UsedMemorySize: an application's
                    // pool (titles check how much of it is free at boot).
                    6 | 21 => APPLICATION_POOL_SIZE,
                    7 | 22 => BASE_USED_MEMORY + state.heap_size,
                    8 => 0,
                    9 => 0,
                    10 => {
                        state.tick = state.tick.wrapping_add(1000);
                        state.tick
                    }
                    11 => [
                        0x1234_5678_9ABC_DEF0_u64,
                        0x0F1E_2D3C_4B5A_6978,
                        0xA5A5_5A5A_C3C3_3C3C,
                        0xDEAD_BEEF_CAFE_F00D,
                    ][subtype as usize % 4],
                    // AslrRegionAddress / Size: where modules are loaded.
                    12 => 0x0800_0000,
                    13 => 0x7F_F800_0000,
                    // StackRegionAddress / Size: where thread stacks are
                    // aliased (svcMapMemory).
                    14 => self.stack_region.0,
                    15 => self.stack_region.1,
                    16 | 17 => 0,
                    18 => 0x0100_0000_0000_0000,
                    _ => 0,
                };
                if let Some(memory) = self.memory_access() {
                    if out != 0 {
                        let _ = memory.store_u64(out, value);
                    }
                }
                KernelSvcResult::ok_out1(value)
            }
            0x0B => {
                let ns = args.first().copied().unwrap_or(0);
                if ns > 0 && ns < 1_000_000_000 {
                    std::thread::sleep(std::time::Duration::from_nanos(ns));
                }
                KernelSvcResult::ok()
            }
            0x10 => KernelSvcResult::ok_with(0),
            0x1E => KernelSvcResult::ok_with(clock::ticks()),
            0x24 => KernelSvcResult::ok_out(0x1_234_567_890, 0),
            0x25 => {
                state.thread_id = state.thread_id.wrapping_add(1);
                KernelSvcResult::ok_out(state.thread_id, 0)
            }
            0x08 => {
                // CreateThread: outputs X1 = thread handle.
                state.next_handle = state.next_handle.wrapping_add(1);
                KernelSvcResult::ok_out(state.next_handle - 1, 0)
            }
            0x15 => {
                state.next_handle = state.next_handle.wrapping_add(1);
                KernelSvcResult::ok_out(state.next_handle - 1, 0)
            }
            0x0C => KernelSvcResult::ok_out(0x20, 0),
            0x0E => KernelSvcResult::ok_out(0, 0xF),
            // IPC SVCs without a hub attached (with one, `dispatch_ipc`
            // handles them): succeed vacuously so startup paths can still
            // progress.
            0x1F => {
                state.next_handle = state.next_handle.wrapping_add(1);
                KernelSvcResult::ok_with(state.next_handle - 1)
            }
            0x21..=0x23 => KernelSvcResult::ok_with(0),
            0x16 => KernelSvcResult::ok(),
            0x18 => KernelSvcResult::ok_out(0, 0),
            0x40 | 0x70 => {
                state.next_handle = state.next_handle.wrapping_add(2);
                KernelSvcResult::ok_out(state.next_handle - 2, state.next_handle - 1)
            }
            0x45 => {
                state.next_handle = state.next_handle.wrapping_add(2);
                KernelSvcResult::ok_out(state.next_handle - 2, state.next_handle - 1)
            }
            0x41 | 0x50 | 0x53 | 0x72 => {
                state.next_handle = state.next_handle.wrapping_add(1);
                KernelSvcResult::ok_out(state.next_handle - 1, 0)
            }
            // Structural handlers: succeed so startup paths can progress while
            // real service emulation (fs:, am:, gl:) is still being built.
            0x02 | 0x03 | 0x04 | 0x05 | 0x09 | 0x0A | 0x0D | 0x0F | 0x11 | 0x12 | 0x14 | 0x17
            | 0x19 | 0x1A | 0x1B | 0x1C | 0x1D | 0x26 | 0x27 | 0x2A | 0x2B | 0x2C | 0x2D | 0x2F
            | 0x30 | 0x31 | 0x32 | 0x33 | 0x34 | 0x35 | 0x36 | 0x37 | 0x4D | 0x4F | 0x51 | 0x52
            | 0x54 | 0x56 | 0x57 | 0x58 | 0x59 | 0x5A | 0x5B | 0x5C | 0x5D | 0x5E | 0x5F | 0x71
            | 0x73 | 0x74 | 0x75 | 0x76 | 0x77 | 0x78 | 0x79 | 0x7A | 0x7B | 0x7C | 0x7D | 0x7E => {
                KernelSvcResult::ok()
            }
            // SendSyncRequestLight (0x20) mirrors svcSendSyncRequest semantics.
            0x20 => KernelSvcResult::ok_with(0),
            _ => KernelSvcResult::err(KERNEL_SVC_UNSUPPORTED),
        }
    }

    pub fn heap_info(&self) -> (Option<u64>, u64) {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        (state.heap_base, state.heap_size)
    }
}

#[cfg(test)]
mod kernel_svc_tests {
    use super::*;

    #[test]
    fn set_heap_size_returns_base() {
        let dispatcher =
            KernelSvcDispatcher::new(KernelSvcDispatcher::all_allowed(), 0x20000000, 0x400000);
        let result = dispatcher.dispatch(0x01, &[0, 0x1000]);
        assert_eq!(result.code, 0);
        assert_eq!(result.x1, Some(0x20000000));
        assert_eq!(dispatcher.heap_info(), (Some(0x20000000), 0x1000));
    }

    #[test]
    fn disallowed_svc_is_rejected() {
        let dispatcher = KernelSvcDispatcher::new(vec![0x01], 0x20000000, 0x400000);
        assert_eq!(dispatcher.dispatch(0x21, &[0]).code, KERNEL_SVC_NOT_ALLOWED);
    }

    #[test]
    fn exit_process_terminates_with_code() {
        let dispatcher =
            KernelSvcDispatcher::new(KernelSvcDispatcher::all_allowed(), 0x20000000, 0x400000);
        let result = dispatcher.dispatch(0x07, &[3]);
        assert!(result.terminates);
        assert_eq!(result.code, 3);
    }

    #[derive(Debug, Default)]
    struct TestMemory {
        bytes: std::sync::Mutex<Vec<u8>>,
    }

    impl TestMemory {
        fn new(size: usize) -> Self {
            Self {
                bytes: std::sync::Mutex::new(vec![0; size]),
            }
        }
    }

    impl crate::ipc::MemoryAccess for TestMemory {
        fn load_u8(&self, addr: u64) -> Option<u8> {
            self.bytes.lock().unwrap().get(addr as usize).copied()
        }
        fn load_u32(&self, addr: u64) -> Option<u32> {
            let bytes = self.bytes.lock().unwrap();
            let slice = bytes.get(addr as usize..addr as usize + 4)?;
            Some(u32::from_le_bytes(slice.try_into().ok()?))
        }
        fn load_u64(&self, addr: u64) -> Option<u64> {
            let bytes = self.bytes.lock().unwrap();
            let slice = bytes.get(addr as usize..addr as usize + 8)?;
            Some(u64::from_le_bytes(slice.try_into().ok()?))
        }
        fn store_u32(&self, addr: u64, value: u32) -> bool {
            let mut bytes = self.bytes.lock().unwrap();
            let Some(slice) = bytes.get_mut(addr as usize..addr as usize + 4) else {
                return false;
            };
            slice.copy_from_slice(&value.to_le_bytes());
            true
        }
        fn store_u64(&self, addr: u64, value: u64) -> bool {
            let mut bytes = self.bytes.lock().unwrap();
            let Some(slice) = bytes.get_mut(addr as usize..addr as usize + 8) else {
                return false;
            };
            slice.copy_from_slice(&value.to_le_bytes());
            true
        }
        fn read_bytes(&self, addr: u64, len: usize) -> Option<Vec<u8>> {
            self.bytes
                .lock()
                .unwrap()
                .get(addr as usize..addr as usize + len)
                .map(|slice| slice.to_vec())
        }
        fn write_bytes(&self, addr: u64, bytes: &[u8]) -> bool {
            let mut memory = self.bytes.lock().unwrap();
            let Some(slice) = memory.get_mut(addr as usize..addr as usize + bytes.len()) else {
                return false;
            };
            slice.copy_from_slice(bytes);
            true
        }
        fn query_block(&self, _addr: u64) -> Option<crate::ipc::MemoryBlock> {
            Some(crate::ipc::MemoryBlock {
                base: 0,
                size: 0x1000,
                state: crate::ipc::memory_state::NORMAL,
                perm: 3,
            })
        }
    }

    #[test]
    fn query_memory_reports_region_and_page_info() {
        let dispatcher =
            KernelSvcDispatcher::new(KernelSvcDispatcher::all_allowed(), 0x20000000, 0x400000);
        let memory = std::sync::Arc::new(TestMemory::new(0x1000));
        dispatcher.attach_memory(memory.clone());
        let result = dispatcher.dispatch(0x06, &[0x200, 0x300, 0x100]);
        assert_eq!(result.code, 0);
        assert!(result.x1.is_some());
        assert_eq!(memory.load_u64(0x200), Some(0));
        assert_eq!(memory.load_u64(0x208), Some(0x1000));
        assert_eq!(
            memory.load_u32(0x210),
            Some(crate::ipc::memory_state::NORMAL)
        );
        assert_eq!(memory.load_u32(0x218), Some(3));
    }

    #[test]
    fn get_info_reports_core_mask_and_program_id() {
        let dispatcher =
            KernelSvcDispatcher::new(KernelSvcDispatcher::all_allowed(), 0x20000000, 0x400000);
        let memory = std::sync::Arc::new(TestMemory::new(0x100));
        dispatcher.attach_memory(memory.clone());
        let result = dispatcher.dispatch(0x29, &[0x10, 0, 0, 0]);
        assert_eq!(result.x1, Some(0xF));
        assert_eq!(memory.load_u64(0x10), Some(0xF));

        let result = dispatcher.dispatch(0x29, &[0x20, 18, 0, 0]);
        assert_eq!(result.x1, Some(0x0100_0000_0000_0000));
        assert_eq!(memory.load_u64(0x20), Some(0x0100_0000_0000_0000));
    }

    #[test]
    fn every_kernel_svc_name_maps_back_to_its_id() {
        for (id, name) in KERNEL_SVC_TABLE {
            assert_eq!(kernel_svc_id(name), Some(*id), "{name}");
            assert_eq!(kernel_svc_name(*id), Some(*name), "{name}");
            assert!(kernel_svc_known(*id), "{name}");
        }
        assert_eq!(kernel_svc_id("svcNotAThing"), None);
        assert_eq!(kernel_svc_name(0xFFFF), None);
    }

    #[test]
    fn kernel_svc_known_rejects_gaps_and_out_of_range_ids() {
        // 0x38 and 0x3B are holes in the NPDM numbering; every real id is
        // below 0x80.
        assert!(!kernel_svc_known(0x38));
        assert!(!kernel_svc_known(0x3B));
        assert!(!kernel_svc_known(0));
        assert!(!kernel_svc_known(0x80));
        assert!(!kernel_svc_known(u64::MAX));
        assert!(!kernel_svc_known(0x1_0000_0001));
        assert!(kernel_svc_known(0x01));
        assert!(kernel_svc_known(0x7F));
    }
}
