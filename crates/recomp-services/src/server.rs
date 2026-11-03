//! A CMIF server for the guest: sessions, domains, and the service objects
//! behind them.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::hipc::{kind, Buffer, CmifRequest, Message, Reply};
use crate::ipc::MemoryAccess;

/// sf::cmif::ResultUnknownCommandId (module 10, description 221).
pub const RESULT_UNKNOWN_COMMAND: u32 = 0x1BA0A;
/// sm::ResultNotRegistered (module 21, description 7).
pub const RESULT_NOT_REGISTERED: u32 = 0xE15;
/// Kernel result for a message the server cannot parse.
pub const RESULT_BAD_MESSAGE: u64 = 0xE001_C001;
/// Pointer-buffer size every session reports (QueryPointerBufferSize).
pub const POINTER_BUFFER_SIZE: u16 = 0x8000;
/// Process id the kernel attaches to messages that send a PID.
pub const PROCESS_ID: u64 = 0x51;

/// What services may need from the outside world.
#[derive(Debug, Clone, Default)]
pub struct ServerContext {
    /// Host path of the title's RomFS image.
    pub romfs_image: Option<PathBuf>,
    /// Host directory that holds save data (one subdirectory per save).
    pub save_root: Option<PathBuf>,
}

/// An object served over CMIF: a service, or an interface one returns.
pub trait Service: Send {
    fn name(&self) -> String;
    /// Handle one command. `Err` carries the failure Result.
    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32>;
}

pub type ObjectRef = Arc<Mutex<Box<dyn Service>>>;

fn object(service: Box<dyn Service>) -> ObjectRef {
    Arc::new(Mutex::new(service))
}

/// One command in flight: argument access and the reply being built.
pub struct Call<'a> {
    pub mem: &'a dyn MemoryAccess,
    pub message: &'a Message,
    pub request: &'a CmifRequest,
    pub context: &'a ServerContext,
    data: Vec<u8>,
    copy_handles: Vec<u32>,
    /// Objects returned as CMIF out-interfaces.
    out_objects: Vec<Box<dyn Service>>,
    /// Services returned as new sessions (move handles) regardless of domain.
    out_sessions: Vec<Box<dyn Service>>,
    statics: Vec<Buffer>,
}

impl<'a> Call<'a> {
    pub fn command(&self) -> u32 {
        self.request.command
    }

    /// Read argument bytes at `offset`; missing bytes read as zero.
    pub fn arg_bytes(&self, offset: u64, len: usize) -> Vec<u8> {
        if offset + len as u64 > self.request.args_len {
            let mut bytes = vec![0u8; len];
            let available = self.request.args_len.saturating_sub(offset) as usize;
            if let Some(read) = self
                .mem
                .read_bytes(self.request.args + offset, available.min(len))
            {
                bytes[..read.len()].copy_from_slice(&read);
            }
            return bytes;
        }
        self.mem
            .read_bytes(self.request.args + offset, len)
            .unwrap_or_else(|| vec![0; len])
    }

    pub fn arg_u32(&self, offset: u64) -> u32 {
        u32::from_le_bytes(self.arg_bytes(offset, 4).try_into().unwrap_or([0; 4]))
    }

    pub fn arg_u64(&self, offset: u64) -> u64 {
        u64::from_le_bytes(self.arg_bytes(offset, 8).try_into().unwrap_or([0; 8]))
    }

    pub fn push_u8(&mut self, value: u8) {
        self.data.push(value);
    }

    pub fn push_u32(&mut self, value: u32) {
        self.align(4);
        self.data.extend_from_slice(&value.to_le_bytes());
    }

    pub fn push_u64(&mut self, value: u64) {
        self.align(8);
        self.data.extend_from_slice(&value.to_le_bytes());
    }

    pub fn push_bytes(&mut self, bytes: &[u8]) {
        self.data.extend_from_slice(bytes);
    }

    fn align(&mut self, to: usize) {
        while !self.data.len().is_multiple_of(to) {
            self.data.push(0);
        }
    }

    pub fn push_copy_handle(&mut self, handle: u32) {
        self.copy_handles.push(handle);
    }

    /// Return an interface object (a new session, or a domain object).
    pub fn push_object(&mut self, service: Box<dyn Service>) {
        self.out_objects.push(service);
    }

    /// Return a service as a new session handle (sm GetServiceHandle).
    pub fn push_session(&mut self, service: Box<dyn Service>) {
        self.out_sessions.push(service);
    }

    /// An input buffer by its position among A descriptors, falling back to
    /// the X static with that index (CMIF "auto select" buffers).
    pub fn in_buffer(&self, index: usize) -> Vec<u8> {
        let send = self
            .message
            .sends
            .get(index)
            .filter(|buffer| buffer.size > 0);
        let static_ = self
            .message
            .statics
            .iter()
            .find(|buffer| buffer.index_or_mode as usize == index && buffer.size > 0);
        send.or(static_)
            .and_then(|buffer| {
                self.mem
                    .read_bytes(buffer.address, buffer.size.min(1 << 28) as usize)
            })
            .unwrap_or_default()
    }

    /// An output buffer by its position among B descriptors, falling back to
    /// the C receive-static at that position.
    pub fn out_buffer(&self, index: usize) -> Option<Buffer> {
        self.message
            .receives
            .get(index)
            .filter(|buffer| buffer.size > 0)
            .or_else(|| {
                self.message
                    .recv_statics
                    .get(index)
                    .filter(|buffer| buffer.size > 0)
            })
            .copied()
    }

    /// Write `bytes` to the output buffer at `index` (truncated to fit);
    /// returns the bytes written.
    pub fn write_out(&mut self, index: usize, bytes: &[u8]) -> usize {
        let Some(buffer) = self.out_buffer(index) else {
            return 0;
        };
        let len = bytes.len().min(buffer.size as usize);
        self.mem.write_bytes(buffer.address, &bytes[..len]);
        len
    }
}

/// The objects of one domain. Sessions cloned from a domain session join
/// the same domain.
struct Domain {
    objects: BTreeMap<u32, ObjectRef>,
    next_id: u32,
}

type DomainRef = Arc<Mutex<Domain>>;

fn lock_domain(domain: &DomainRef) -> std::sync::MutexGuard<'_, Domain> {
    domain
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

enum SessionTarget {
    Object(ObjectRef),
    Domain(DomainRef),
}

/// The guest's view of all server sessions.
///
/// The session registry is locked only for the short map operations, never
/// while a service call runs: a service that blocks (nvdrv waiting on a GPU
/// syncpoint) must not stop other threads from talking to other services, as
/// on hardware where every IPC request is handled concurrently.
pub struct Server {
    registry: Mutex<Registry>,
    pub context: ServerContext,
}

#[derive(Default)]
struct Registry {
    sessions: BTreeMap<u32, SessionTarget>,
    next_handle: u32,
}

impl Server {
    /// Session handles start well above the kernel object handles so the two
    /// namespaces never collide.
    pub fn new(context: ServerContext) -> Self {
        Server {
            registry: Mutex::new(Registry {
                sessions: BTreeMap::new(),
                next_handle: 0x0001_0000,
            }),
            context,
        }
    }

    fn registry(&self) -> std::sync::MutexGuard<'_, Registry> {
        self.registry
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// A new session for `service`; returns the handle the guest receives.
    pub fn open(&self, service: Box<dyn Service>) -> u32 {
        self.open_ref(object(service))
    }

    fn open_ref(&self, object: ObjectRef) -> u32 {
        let mut registry = self.registry();
        let handle = registry.next_handle;
        registry.next_handle += 1;
        registry
            .sessions
            .insert(handle, SessionTarget::Object(object));
        handle
    }

    pub fn owns(&self, handle: u64) -> bool {
        u32::try_from(handle).is_ok_and(|handle| self.registry().sessions.contains_key(&handle))
    }

    pub fn close(&self, handle: u64) -> bool {
        u32::try_from(handle).is_ok_and(|handle| self.registry().sessions.remove(&handle).is_some())
    }

    /// svcSendSyncRequest on one of our sessions: returns the kernel result.
    pub fn dispatch(&self, mem: &dyn MemoryAccess, handle: u64, base: u64) -> u64 {
        let Ok(handle) = u32::try_from(handle) else {
            return RESULT_BAD_MESSAGE;
        };
        let Some(message) = Message::parse(mem, base) else {
            return RESULT_BAD_MESSAGE;
        };
        match message.kind {
            kind::CLOSE => {
                self.registry().sessions.remove(&handle);
                0
            }
            kind::CONTROL | kind::CONTROL_WITH_CONTEXT => self.control(mem, handle, base, &message),
            kind::REQUEST | kind::REQUEST_WITH_CONTEXT => self.request(mem, handle, base, &message),
            other => {
                println!("[recomp-ipc] unsupported message type {other} on handle {handle:#x}");
                RESULT_BAD_MESSAGE
            }
        }
    }

    fn control(&self, mem: &dyn MemoryAccess, handle: u32, base: u64, message: &Message) -> u64 {
        let Some(request) = CmifRequest::parse(mem, message, false) else {
            return RESULT_BAD_MESSAGE;
        };
        let mut reply = Reply::default();
        {
            let mut registry = self.registry();
            let Some(target) = registry.sessions.get_mut(&handle) else {
                return RESULT_BAD_MESSAGE;
            };
            match request.command {
                // ConvertCurrentObjectToDomain -> object id of the session object.
                0 => {
                    if let SessionTarget::Object(object) = target {
                        let mut objects = BTreeMap::new();
                        objects.insert(1, object.clone());
                        *target = SessionTarget::Domain(Arc::new(Mutex::new(Domain {
                            objects,
                            next_id: 2,
                        })));
                    }
                    reply.data = 1u32.to_le_bytes().to_vec();
                }
                // CopyFromCurrentDomain(object id) -> session for that object.
                1 => {
                    let id = mem.load_u32(request.args).unwrap_or(0);
                    let found = match target {
                        SessionTarget::Domain(domain) => {
                            lock_domain(domain).objects.get(&id).cloned()
                        }
                        SessionTarget::Object(_) => None,
                    };
                    match found {
                        Some(object) => {
                            let cloned = registry.next_handle;
                            registry.next_handle += 1;
                            registry
                                .sessions
                                .insert(cloned, SessionTarget::Object(object));
                            reply.move_handles.push(cloned);
                        }
                        None => reply.result = RESULT_UNKNOWN_COMMAND,
                    }
                }
                // CloneCurrentObject / CloneCurrentObjectEx -> another session to
                // the same object, or into the same domain.
                2 | 4 => {
                    let clone = match target {
                        SessionTarget::Object(object) => SessionTarget::Object(object.clone()),
                        SessionTarget::Domain(domain) => SessionTarget::Domain(domain.clone()),
                    };
                    let cloned = registry.next_handle;
                    registry.next_handle += 1;
                    registry.sessions.insert(cloned, clone);
                    reply.move_handles.push(cloned);
                }
                // QueryPointerBufferSize -> u16.
                3 => reply.data = u32::from(POINTER_BUFFER_SIZE).to_le_bytes().to_vec(),
                other => {
                    println!("[recomp-ipc] UNIMPLEMENTED control command {other}");
                    reply.result = RESULT_UNKNOWN_COMMAND;
                }
            }
        }
        if crate::hipc::write_reply(mem, base, &reply, false) {
            0
        } else {
            RESULT_BAD_MESSAGE
        }
    }

    fn request(&self, mem: &dyn MemoryAccess, handle: u32, base: u64, message: &Message) -> u64 {
        let is_domain = matches!(
            self.registry().sessions.get(&handle),
            Some(SessionTarget::Domain { .. })
        );
        let Some(request) = CmifRequest::parse(mem, message, is_domain) else {
            let words: Vec<String> = (0..24)
                .map(|index| format!("{:08x}", mem.load_u32(base + index * 4).unwrap_or(0)))
                .collect();
            println!(
                "[recomp-ipc] malformed CMIF request on handle {handle:#x} (domain {is_domain}): {}",
                words.join(" ")
            );
            return RESULT_BAD_MESSAGE;
        };
        // The session is picked under the registry lock and then dropped: the
        // service call below holds only the object's own lock, so it can block
        // without stopping other sessions.
        let target = {
            let registry = self.registry();
            match registry.sessions.get(&handle) {
                Some(SessionTarget::Object(object)) => object.clone(),
                Some(SessionTarget::Domain(domain)) => {
                    let header = request.domain.unwrap_or_default();
                    let mut domain = lock_domain(domain);
                    if header.command == 2 {
                        domain.objects.remove(&header.object_id);
                        drop(domain);
                        drop(registry);
                        let ok = crate::hipc::write_reply(mem, base, &Reply::default(), true);
                        return if ok { 0 } else { RESULT_BAD_MESSAGE };
                    }
                    match domain.objects.get(&header.object_id) {
                        Some(object) => object.clone(),
                        None => {
                            println!(
                                "[recomp-ipc] domain object {} not found on handle {handle:#x}",
                                header.object_id
                            );
                            return RESULT_BAD_MESSAGE;
                        }
                    }
                }
                None => return RESULT_BAD_MESSAGE,
            }
        };

        let mut call = Call {
            mem,
            message,
            request: &request,
            context: &self.context,
            data: Vec::new(),
            copy_handles: Vec::new(),
            out_objects: Vec::new(),
            out_sessions: Vec::new(),
            statics: Vec::new(),
        };
        let (name, result) = {
            // A service may block inside a call (vi's DequeueBuffer waits for
            // a vsync); another thread calling it then waits with the
            // scheduler released, or the two would deadlock on the run lock.
            let mut service = match target.try_lock() {
                Ok(service) => service,
                Err(std::sync::TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
                Err(std::sync::TryLockError::WouldBlock) => {
                    crate::kernel::kernel().blocking(|| {
                        target
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                    })
                }
            };
            let result = service.call(&mut call);
            (service.name(), result)
        };
        let Call {
            data,
            copy_handles,
            out_objects,
            out_sessions,
            statics,
            ..
        } = call;
        match result {
            Ok(()) => {
                if crate::trace_ipc() {
                    println!("[recomp-ipc] {name} #{}", request.command);
                }
            }
            Err(code) => println!(
                "[recomp-ipc] {name} #{} -> {}",
                request.command,
                describe(code)
            ),
        }

        let mut reply = Reply {
            result: result.err().unwrap_or(0),
            data,
            copy_handles,
            statics,
            ..Reply::default()
        };
        if reply.result == 0 {
            for session in out_sessions {
                reply.move_handles.push(self.open(session));
            }
            let in_domain = {
                let registry = self.registry();
                matches!(
                    registry.sessions.get(&handle),
                    Some(SessionTarget::Domain { .. })
                )
            };
            if in_domain {
                let registry = self.registry();
                if let Some(SessionTarget::Domain(domain)) = registry.sessions.get(&handle) {
                    let mut domain = lock_domain(domain);
                    for out in out_objects {
                        let id = domain.next_id;
                        domain.next_id += 1;
                        domain.objects.insert(id, object(out));
                        reply.out_objects.push(id);
                    }
                }
            } else {
                for out in out_objects {
                    reply.move_handles.push(self.open(out));
                }
            }
        }
        if crate::hipc::write_reply(mem, base, &reply, is_domain) {
            0
        } else {
            RESULT_BAD_MESSAGE
        }
    }
}

/// A Result as `0xVALUE (2MMM-DDDD)`.
pub fn describe(code: u32) -> String {
    format!(
        "{code:#x} ({:04}-{:04})",
        (code & 0x1FF) + 2000,
        (code >> 9) & 0x1FFF
    )
}
