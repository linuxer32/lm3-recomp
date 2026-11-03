use recomp_services::hipc::{kind, Message, CMIF_OUT_MAGIC};
use recomp_services::server::{Call, Server, ServerContext, Service, RESULT_NOT_REGISTERED};
use recomp_services::MemoryAccess;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::sync::{Arc, Condvar};
use std::time::{Duration, Instant};

const BASE: u64 = 0x1000;

#[derive(Debug)]
struct Memory(Mutex<Vec<u8>>);

impl Memory {
    fn new() -> Self {
        Memory(Mutex::new(vec![0; 0x4000]))
    }
}

impl MemoryAccess for Memory {
    fn load_u8(&self, addr: u64) -> Option<u8> {
        self.0.lock().unwrap().get(addr as usize).copied()
    }
    fn load_u32(&self, addr: u64) -> Option<u32> {
        let bytes = self.read_bytes(addr, 4)?;
        Some(u32::from_le_bytes(bytes.try_into().ok()?))
    }
    fn load_u64(&self, addr: u64) -> Option<u64> {
        let bytes = self.read_bytes(addr, 8)?;
        Some(u64::from_le_bytes(bytes.try_into().ok()?))
    }
    fn store_u32(&self, addr: u64, value: u32) -> bool {
        self.write_bytes(addr, &value.to_le_bytes())
    }
    fn store_u64(&self, addr: u64, value: u64) -> bool {
        self.write_bytes(addr, &value.to_le_bytes())
    }
    fn read_bytes(&self, addr: u64, len: usize) -> Option<Vec<u8>> {
        let memory = self.0.lock().unwrap();
        memory
            .get(addr as usize..addr as usize + len)
            .map(<[u8]>::to_vec)
    }
    fn write_bytes(&self, addr: u64, bytes: &[u8]) -> bool {
        let mut memory = self.0.lock().unwrap();
        match memory.get_mut(addr as usize..addr as usize + bytes.len()) {
            Some(slice) => {
                slice.copy_from_slice(bytes);
                true
            }
            None => false,
        }
    }
}

/// Build a CMIF request the way libnx and the SDK lay it out.
fn request(
    mem: &Memory,
    kind: u16,
    command: u32,
    args: &[u8],
    send_pid: bool,
    domain_object: Option<u32>,
) {
    let mut words = Vec::new();
    let mut header1 = 0u32;
    let mut pos = BASE + 8;
    if send_pid {
        header1 |= 1 << 31;
        words.push((pos, 1u32));
        pos += 4 + 8;
    }
    let raw = pos;
    let payload = raw.next_multiple_of(16);
    let mut body = Vec::new();
    if let Some(object) = domain_object {
        let data_size = 16 + args.len() as u32;
        body.extend_from_slice(&(1 | (data_size << 16)).to_le_bytes());
        body.extend_from_slice(&object.to_le_bytes());
        body.extend_from_slice(&[0; 8]);
    }
    body.extend_from_slice(b"SFCI");
    let version = u32::from(kind == 6 || kind == 7);
    body.extend_from_slice(&version.to_le_bytes());
    body.extend_from_slice(&command.to_le_bytes());
    body.extend_from_slice(&0u32.to_le_bytes());
    body.extend_from_slice(args);
    header1 |= ((payload - raw) as u32 + body.len() as u32).div_ceil(4) + 4;
    mem.write_bytes(BASE, &[0; 0x100]);
    mem.store_u32(BASE, u32::from(kind));
    mem.store_u32(BASE + 4, header1);
    for (addr, value) in words {
        mem.store_u32(addr, value);
    }
    mem.write_bytes(payload, &body);
}

struct ParsedReply {
    result: u32,
    data: Vec<u8>,
    move_handles: Vec<u32>,
}

fn reply(mem: &Memory, domain: bool) -> ParsedReply {
    let message = Message::parse(mem, BASE).expect("reply parses");
    let mut header = message.payload();
    if domain {
        header += 16;
    }
    assert_eq!(mem.load_u32(header), Some(CMIF_OUT_MAGIC));
    let end = message.raw + message.raw_len;
    ParsedReply {
        result: mem.load_u32(header + 8).unwrap(),
        data: mem
            .read_bytes(header + 16, (end - header - 16) as usize)
            .unwrap(),
        move_handles: message.move_handles,
    }
}

fn name_arg(name: &str) -> [u8; 8] {
    let mut bytes = [0u8; 8];
    bytes[..name.len()].copy_from_slice(name.as_bytes());
    bytes
}

#[test]
fn request_with_context_and_pid_parses() {
    let mem = Memory::new();
    request(
        &mem,
        kind::REQUEST_WITH_CONTEXT,
        1,
        &0x1234u64.to_le_bytes(),
        true,
        None,
    );
    let message = Message::parse(&mem, BASE).expect("parse");
    assert_eq!(message.kind, kind::REQUEST_WITH_CONTEXT);
    assert!(message.send_pid);
    let request = recomp_services::hipc::CmifRequest::parse(&mem, &message, false).expect("cmif");
    assert_eq!(request.command, 1);
    assert_eq!(mem.load_u64(request.args), Some(0x1234));
}

#[test]
fn sm_hands_out_lm_and_lm_opens_a_logger() {
    let mem = Memory::new();
    let server = Server::new(ServerContext::default());
    let sm = u64::from(server.open(Box::new(recomp_services::services::ServiceManager)));

    // RegisterClient(pid), as nn::sm::Initialize sends it.
    request(&mem, kind::REQUEST_WITH_CONTEXT, 0, &[0; 8], true, None);
    assert_eq!(server.dispatch(&mem, sm, BASE), 0);
    assert_eq!(reply(&mem, false).result, 0);

    // GetServiceHandle("lm") -> a move handle to a new session.
    request(
        &mem,
        kind::REQUEST_WITH_CONTEXT,
        1,
        &name_arg("lm"),
        false,
        None,
    );
    assert_eq!(server.dispatch(&mem, sm, BASE), 0);
    let got = reply(&mem, false);
    assert_eq!(got.result, 0);
    assert_eq!(got.move_handles.len(), 1);
    let lm = u64::from(got.move_handles[0]);
    assert!(server.owns(lm));

    // ILogService::OpenLogger -> ILogger session.
    request(&mem, kind::REQUEST, 0, &[0; 8], true, None);
    assert_eq!(server.dispatch(&mem, lm, BASE), 0);
    let logger = reply(&mem, false);
    assert_eq!(logger.result, 0);
    assert_eq!(logger.move_handles.len(), 1);

    // An unknown service is reported as not registered.
    request(&mem, kind::REQUEST, 1, &name_arg("nope"), false, None);
    assert_eq!(server.dispatch(&mem, sm, BASE), 0);
    assert_eq!(reply(&mem, false).result, RESULT_NOT_REGISTERED);
}

#[test]
fn domains_return_object_ids() {
    let mem = Memory::new();
    let server = Server::new(ServerContext::default());
    let sm = u64::from(server.open(Box::new(recomp_services::services::ServiceManager)));
    request(&mem, kind::REQUEST, 1, &name_arg("lm"), false, None);
    server.dispatch(&mem, sm, BASE);
    let lm = u64::from(reply(&mem, false).move_handles[0]);

    // QueryPointerBufferSize, then ConvertCurrentObjectToDomain -> id 1.
    request(&mem, kind::CONTROL, 3, &[], false, None);
    assert_eq!(server.dispatch(&mem, lm, BASE), 0);
    assert_eq!(&reply(&mem, false).data[..2], &0x8000u16.to_le_bytes());
    request(&mem, kind::CONTROL, 0, &[], false, None);
    assert_eq!(server.dispatch(&mem, lm, BASE), 0);
    assert_eq!(&reply(&mem, false).data[..4], &1u32.to_le_bytes());

    // OpenLogger on domain object 1 returns a new object id, not a handle.
    request(&mem, kind::REQUEST, 0, &[0; 8], false, Some(1));
    assert_eq!(server.dispatch(&mem, lm, BASE), 0);
    let message = Message::parse(&mem, BASE).unwrap();
    assert!(message.move_handles.is_empty());
    let header = message.payload();
    assert_eq!(mem.load_u32(header), Some(1), "one out object");
    assert_eq!(mem.load_u32(header + 16), Some(CMIF_OUT_MAGIC));
    assert_eq!(mem.load_u32(header + 16 + 8), Some(0));
    let ids = header + 32;
    assert_eq!(mem.load_u32(ids), Some(2), "object id 2");
}

/// A CMIF request (no PID, no domain) with X (`statics`), A (`sends`) and B
/// (`receives`) buffers as `(address, size)`.
fn buffered_request(
    mem: &Memory,
    command: u32,
    args: &[u8],
    statics: &[(u64, u64)],
    sends: &[(u64, u64)],
    receives: &[(u64, u64)],
) {
    mem.write_bytes(BASE, &[0; 0x100]);
    let header0 = u32::from(kind::REQUEST)
        | ((statics.len() as u32) << 16)
        | ((sends.len() as u32) << 20)
        | ((receives.len() as u32) << 24);
    let mut pos = BASE + 8;
    for (index, (address, size)) in statics.iter().enumerate() {
        mem.store_u32(pos, index as u32 | ((*size as u32) << 16));
        mem.store_u32(pos + 4, *address as u32);
        pos += 8;
    }
    for (address, size) in sends.iter().chain(receives) {
        mem.store_u32(pos, *size as u32);
        mem.store_u32(pos + 4, *address as u32);
        mem.store_u32(pos + 8, 0);
        pos += 12;
    }
    let payload = pos.next_multiple_of(16);
    let mut body = Vec::new();
    body.extend_from_slice(b"SFCI");
    body.extend_from_slice(&[0; 4]);
    body.extend_from_slice(&command.to_le_bytes());
    body.extend_from_slice(&[0; 4]);
    body.extend_from_slice(args);
    let data_words = ((payload - pos) as u32 + body.len() as u32).div_ceil(4);
    mem.store_u32(BASE, header0);
    mem.store_u32(BASE + 4, data_words);
    mem.write_bytes(payload, &body);
}

fn args(values: &[(usize, u64)]) -> Vec<u8> {
    let mut bytes = vec![0u8; 0x40];
    for (offset, value) in values {
        bytes[*offset..*offset + 8].copy_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Write a NUL-terminated guest path at `address`; returns its X buffer.
fn path(mem: &Memory, address: u64, text: &str) -> (u64, u64) {
    let mut bytes = text.as_bytes().to_vec();
    bytes.push(0);
    mem.write_bytes(address, &bytes);
    (address, 0x301)
}

#[test]
fn fsp_srv_serves_romfs_and_host_save_data() {
    let dir = std::env::temp_dir().join(format!("recomp-fsp-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let romfs: Vec<u8> = (0..1024u32).map(|i| i as u8).collect();
    std::fs::write(dir.join("romfs.bin"), &romfs).expect("romfs");
    let mem = Memory::new();
    let server = Server::new(ServerContext {
        romfs_image: Some(dir.join("romfs.bin")),
        save_root: Some(dir.join("save")),
    });
    let sm = u64::from(server.open(Box::new(recomp_services::services::ServiceManager)));
    request(&mem, kind::REQUEST, 1, &name_arg("fsp-srv"), false, None);
    assert_eq!(server.dispatch(&mem, sm, BASE), 0);
    let fsp = u64::from(reply(&mem, false).move_handles[0]);
    let call = |handle: u64,
                command,
                args: &[u8],
                x: &[(u64, u64)],
                a: &[(u64, u64)],
                b: &[(u64, u64)]| {
        buffered_request(&mem, command, args, x, a, b);
        assert_eq!(server.dispatch(&mem, handle, BASE), 0);
        reply(&mem, false)
    };

    // RomFS: OpenDataStorageByCurrentProcess, GetSize, Read.
    let storage = u64::from(call(fsp, 200, &[], &[], &[], &[]).move_handles[0]);
    assert_eq!(
        call(storage, 4, &[], &[], &[], &[]).data[..8],
        1024u64.to_le_bytes()
    );
    assert_eq!(
        call(
            storage,
            0,
            &args(&[(0, 0x10), (8, 8)]),
            &[],
            &[],
            &[(0x2000, 8)]
        )
        .result,
        0
    );
    assert_eq!(mem.read_bytes(0x2000, 8).unwrap(), romfs[0x10..0x18]);

    // Save data: OpenSaveDataFileSystem(account save), then a file round trip.
    let save = call(fsp, 51, &args(&[(0, 1), (0x28, 1)]), &[], &[], &[]);
    assert_eq!(save.result, 0);
    let save = u64::from(save.move_handles[0]);
    let data_dir = path(&mem, 0x3000, "/data");
    assert_eq!(call(save, 2, &[], &[data_dir], &[], &[]).result, 0);
    let file_path = path(&mem, 0x3000, "/data/save.bin");
    assert_eq!(
        call(save, 0, &args(&[(0, 0)]), &[file_path], &[], &[]).result,
        0
    );
    let file = call(save, 8, &args(&[(0, 3)]), &[file_path], &[], &[]);
    let file = u64::from(file.move_handles[0]);
    mem.write_bytes(0x3400, b"hello");
    assert_eq!(
        call(file, 1, &args(&[(8, 0), (16, 5)]), &[], &[(0x3400, 5)], &[]).result,
        0
    );
    assert_eq!(
        call(file, 4, &[], &[], &[], &[]).data[..8],
        5u64.to_le_bytes()
    );
    let read = call(file, 0, &args(&[(8, 1), (16, 4)]), &[], &[], &[(0x3500, 4)]);
    assert_eq!(read.data[..8], 4u64.to_le_bytes());
    assert_eq!(mem.read_bytes(0x3500, 4).unwrap(), b"ello");
    assert_eq!(
        std::fs::read(
            dir.join("save/1-00000000000000000000000000000000-0000000000000000-0/data/save.bin")
        )
        .unwrap(),
        b"hello"
    );

    // OpenDirectory lists the file with its type and size.
    let data_dir = path(&mem, 0x3000, "/data");
    let listing = call(save, 9, &args(&[(0, 3)]), &[data_dir], &[], &[]);
    let listing = u64::from(listing.move_handles[0]);
    let entries = call(listing, 0, &[], &[], &[], &[(0x3600, 0x620)]);
    assert_eq!(entries.data[..8], 1u64.to_le_bytes());
    assert_eq!(&mem.read_bytes(0x3600, 9).unwrap(), b"save.bin\0");
    assert_eq!(mem.load_u8(0x3600 + 0x304), Some(1));
    assert_eq!(mem.load_u64(0x3600 + 0x308), Some(5));

    // Missing paths and escapes report PathNotFound.
    let missing = path(&mem, 0x3000, "/missing");
    assert_eq!(
        call(save, 7, &[], &[missing], &[], &[]).result,
        recomp_services::fs::RESULT_PATH_NOT_FOUND
    );
    let escape = path(&mem, 0x3000, "/../escape");
    assert_eq!(
        call(save, 2, &[], &[escape], &[], &[]).result,
        recomp_services::fs::RESULT_PATH_NOT_FOUND
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// Open `name` through sm: and return its session handle.
fn open_service(server: &Server, mem: &Memory, name: &str) -> u64 {
    let sm = u64::from(server.open(Box::new(recomp_services::services::ServiceManager)));
    request(mem, kind::REQUEST, 1, &name_arg(name), false, None);
    assert_eq!(server.dispatch(mem, sm, BASE), 0);
    u64::from(reply(mem, false).move_handles[0])
}

#[test]
fn system_services_answer_like_a_single_user_console() {
    let mem = Memory::new();
    let server = Server::new(ServerContext::default());
    let call = |server: &Server, handle: u64, command, args: &[u8], b: &[(u64, u64)]| {
        buffered_request(&mem, command, args, &[], &[], b);
        assert_eq!(server.dispatch(&mem, handle, BASE), 0);
        reply(&mem, false)
    };

    // time: calendar conversions in UTC and the steady clock's source id.
    let time = open_service(&server, &mem, "time:u");
    let zone = u64::from(call(&server, time, 3, &[], &[]).move_handles[0]);
    let cal = call(&server, zone, 101, &1_700_000_000u64.to_le_bytes(), &[]).data;
    assert_eq!(
        &cal[..8],
        &[0xE7, 0x07, 11, 14, 22, 13, 20, 0],
        "2023-11-14 22:13:20"
    );
    assert_eq!(&cal[8..16], &[2, 0, 0, 0, 61, 1, 0, 0], "Tuesday, day 317");
    let leap = [0xD0, 0x07, 2, 29, 12, 0, 0, 0];
    call(&server, zone, 202, &leap, &[(0x2000, 8)]);
    assert_eq!(mem.load_u64(0x2000), Some(951_825_600));
    let steady = u64::from(call(&server, time, 2, &[], &[]).move_handles[0]);
    assert_eq!(
        &call(&server, steady, 0, &[], &[]).data[8..24],
        b"recomp-steadyclk"
    );

    // set: American English and 15 language codes.
    let set = open_service(&server, &mem, "set");
    assert_eq!(&call(&server, set, 0, &[], &[]).data[..8], b"en-US\0\0\0");
    assert_eq!(
        call(&server, set, 1, &[], &[(0x2100, 0x100)]).data[..4],
        15u32.to_le_bytes()
    );
    assert_eq!(mem.read_bytes(0x2108, 5).unwrap(), b"en-US");

    // acc:u0: one user, who exists; the rest of the list is invalid ids.
    let acc = open_service(&server, &mem, "acc:u0");
    mem.write_bytes(0x2200, &[0xAA; 0x80]);
    call(&server, acc, 2, &[], &[(0x2200, 0x80)]);
    let user = mem.read_bytes(0x2200, 16).unwrap();
    assert_eq!(user, recomp_services::account::user_id_bytes());
    assert_eq!(mem.read_bytes(0x2210, 0x70).unwrap(), vec![0; 0x70]);
    assert_eq!(call(&server, acc, 1, &user, &[]).data[0], 1);

    // appletOE: the preselected user is passed as a launch parameter.
    let applet = open_service(&server, &mem, "appletOE");
    let proxy = u64::from(call(&server, applet, 0, &[0; 8], &[]).move_handles[0]);
    let functions = u64::from(call(&server, proxy, 20, &[], &[]).move_handles[0]);
    let storage = call(&server, functions, 1, &2u32.to_le_bytes(), &[]);
    let storage = u64::from(storage.move_handles[0]);
    let accessor = u64::from(call(&server, storage, 0, &[], &[]).move_handles[0]);
    call(
        &server,
        accessor,
        11,
        &0u64.to_le_bytes(),
        &[(0x2300, 0x88)],
    );
    assert_eq!(mem.load_u32(0x2300), Some(0xC794_97CA));
    assert_eq!(mem.read_bytes(0x2308, 16).unwrap(), user);

    // The controller support applet completes as soon as it starts, with
    // one player on the handheld controller.
    let creator = u64::from(call(&server, proxy, 11, &[], &[]).move_handles[0]);
    let mut args = 0x0Cu32.to_le_bytes().to_vec();
    args.extend_from_slice(&0u32.to_le_bytes());
    let applet = u64::from(call(&server, creator, 0, &args, &[]).move_handles[0]);
    call(&server, applet, 10, &[], &[]);
    assert_eq!(call(&server, applet, 1, &[], &[]).data[0], 1);
    let output = u64::from(call(&server, applet, 101, &[], &[]).move_handles[0]);
    let accessor = u64::from(call(&server, output, 0, &[], &[]).move_handles[0]);
    call(&server, accessor, 11, &0u64.to_le_bytes(), &[(0x2400, 0xC)]);
    assert_eq!(mem.read_bytes(0x2400, 1).unwrap(), [1]);
    assert_eq!(mem.load_u32(0x2404), Some(0x20));
    assert_eq!(mem.load_u32(0x2408), Some(0));

    // The user picker returns success and the one user.
    let mut args = 0x10u32.to_le_bytes().to_vec();
    args.extend_from_slice(&0u32.to_le_bytes());
    let applet = u64::from(call(&server, creator, 0, &args, &[]).move_handles[0]);
    call(&server, applet, 10, &[], &[]);
    let output = u64::from(call(&server, applet, 101, &[], &[]).move_handles[0]);
    let accessor = u64::from(call(&server, output, 0, &[], &[]).move_handles[0]);
    call(
        &server,
        accessor,
        11,
        &0u64.to_le_bytes(),
        &[(0x2500, 0x18)],
    );
    assert_eq!(mem.load_u64(0x2500), Some(0));
    assert_eq!(mem.read_bytes(0x2508, 16).unwrap(), user);
}

#[test]
fn hid_shared_memory_shows_connected_controllers() {
    let mem = Memory::new();
    let server = Server::new(ServerContext::default());
    let hid = open_service(&server, &mem, "hid");
    buffered_request(&mem, 0, &[0; 8], &[], &[], &[]);
    assert_eq!(server.dispatch(&mem, hid, BASE), 0);
    let resource = u64::from(reply(&mem, false).move_handles[0]);
    buffered_request(&mem, 0, &[], &[], &[], &[]);
    assert_eq!(server.dispatch(&mem, resource, BASE), 0);
    let message = Message::parse(&mem, BASE).unwrap();
    let handle = message.copy_handles[0];
    let shared = recomp_services::kernel::kernel()
        .map_shared_memory(handle, 0x10_0000)
        .expect("shared memory handle");
    assert_eq!(shared.len(), 0x40000);
    let word = |at: usize| u32::from_le_bytes(shared[at..at + 4].try_into().unwrap());
    // No1: full key style; its full key LIFO's newest entry is connected.
    assert_eq!(word(0x9A00), 1);
    assert_eq!(word(0x9A00 + 0x28 + 0x20 + 0x28), 0b11);
    // Handheld (entry 8): handheld style, state in the second LIFO.
    let handheld = 0x9A00 + 8 * 0x5000;
    assert_eq!(word(handheld), 2);
    assert_eq!(word(handheld + 0x28 + 0x350 + 0x20 + 0x28), 0b11);
}

#[test]
fn clones_of_a_domain_session_share_its_objects() {
    let mem = Memory::new();
    let server = Server::new(ServerContext::default());
    let lm = open_service(&server, &mem, "lm");
    // ConvertCurrentObjectToDomain, then CloneCurrentObjectEx.
    request(&mem, kind::CONTROL, 0, &[], false, None);
    assert_eq!(server.dispatch(&mem, lm, BASE), 0);
    request(&mem, kind::CONTROL, 4, &[0; 4], false, None);
    assert_eq!(server.dispatch(&mem, lm, BASE), 0);
    let clone = u64::from(reply(&mem, false).move_handles[0]);
    // The clone addresses object 1 with a domain header, as the SDK's
    // session pools do, and new objects continue the shared id sequence.
    for expected_id in [2, 3] {
        request(&mem, kind::REQUEST_WITH_CONTEXT, 0, &[0; 8], true, Some(1));
        assert_eq!(server.dispatch(&mem, clone, BASE), 0);
        let message = Message::parse(&mem, BASE).unwrap();
        let ids = message.payload() + 32;
        assert_eq!(mem.load_u32(ids), Some(expected_id));
    }
}

#[test]
fn nvdrv_tracks_memory_handles_and_gpu_mappings() {
    let mem = Memory::new();
    let server = Server::new(ServerContext::default());
    let nvdrv = open_service(&server, &mem, "nvdrv");
    let call = |command, args: &[u8], a: &[(u64, u64)], b: &[(u64, u64)]| {
        buffered_request(&mem, command, args, &[], a, b);
        assert_eq!(server.dispatch(&mem, nvdrv, BASE), 0);
        reply(&mem, false).data
    };
    let open = |path: &str| {
        mem.write_bytes(0x2000, format!("{path}\0").as_bytes());
        let data = call(0, &[], &[(0x2000, path.len() as u64 + 1)], &[]);
        assert_eq!(&data[4..8], &[0; 4], "{path} opens");
        u32::from_le_bytes(data[..4].try_into().unwrap())
    };
    let nvmap = open("/dev/nvmap");
    let address_space = open("/dev/nvhost-as-gpu");
    let ctrl_gpu = open("/dev/nvhost-ctrl-gpu");
    let ctrl = open("/dev/nvhost-ctrl");
    let ioctl = |fd: u32, cmd: u32, input: &[u8]| {
        mem.write_bytes(0x2100, input);
        let mut args = fd.to_le_bytes().to_vec();
        args.extend_from_slice(&cmd.to_le_bytes());
        let size = u64::from((cmd >> 16) & 0x3FFF);
        let result = call(1, &args, &[(0x2100, size)], &[(0x2400, size)]);
        assert_eq!(&result[..4], &[0; 4], "ioctl {cmd:#x}");
        mem.read_bytes(0x2400, size as usize).unwrap()
    };
    // nvmap Create(0x1000) -> handle; Alloc at CPU address 0x5000.
    let created = ioctl(nvmap, 0xC008_0101, &0x1000u32.to_le_bytes());
    let handle = u32::from_le_bytes(created[4..8].try_into().unwrap());
    let mut alloc = vec![0u8; 0x20];
    alloc[..4].copy_from_slice(&handle.to_le_bytes());
    alloc[24..32].copy_from_slice(&0x5000u64.to_le_bytes());
    ioctl(nvmap, 0xC020_0104, &alloc);
    // as-gpu MapBufferEx(handle) -> a GPU address.
    let mut map = vec![0u8; 0x28];
    map[8..12].copy_from_slice(&handle.to_le_bytes());
    let mapped = ioctl(address_space, 0xC028_4106, &map);
    let gpu_va = u64::from_le_bytes(mapped[32..40].try_into().unwrap());
    assert!(gpu_va >= 0x4_0000_0000);
    // ctrl-gpu GetCharacteristics: a GM20B with the Maxwell 3D class.
    let characteristics = ioctl(ctrl_gpu, 0xC0B0_4705, &[0; 0xB0]);
    let word = |at: usize| u32::from_le_bytes(characteristics[at..at + 4].try_into().unwrap());
    assert_eq!((word(0x10), word(0x10 + 0x5C)), (0x120, 0xB197));
    // ctrl EventWait with a zero timeout polls: nothing has been submitted on
    // this syncpoint, so the value is 0 rather than the threshold asked for.
    let mut wait = vec![0u8; 0x14];
    wait[..4].copy_from_slice(&0x10u32.to_le_bytes());
    wait[4..8].copy_from_slice(&1u32.to_le_bytes());
    let polled = ioctl(ctrl, 0xC014_001D, &wait);
    assert_eq!(u32::from_le_bytes(polled[12..16].try_into().unwrap()), 0);
}

#[test]
fn vi_opens_a_layer_and_ticks_vsync() {
    let mem = Memory::new();
    let server = Server::new(ServerContext::default());
    let vi = open_service(&server, &mem, "vi:m");
    let call = |handle: u64, command, args: &[u8], a: &[(u64, u64)], b: &[(u64, u64)]| {
        buffered_request(&mem, command, args, &[], a, b);
        assert_eq!(server.dispatch(&mem, handle, BASE), 0);
        let message = Message::parse(&mem, BASE).unwrap();
        (reply(&mem, false), message.copy_handles)
    };
    let display = u64::from(call(vi, 2, &[0; 8], &[], &[]).0.move_handles[0]);
    let relay = u64::from(call(display, 100, &[], &[], &[]).0.move_handles[0]);
    // OpenLayer: a native window parcel naming binder 1 and "dispdrv".
    let (opened, _) = call(display, 2020, &[0; 0x50], &[], &[(0x2000, 0x100)]);
    assert_eq!(opened.data[..8], 0x3Cu64.to_le_bytes());
    assert_eq!(mem.load_u32(0x2008), Some(4), "one binder object");
    assert_eq!(mem.load_u32(0x2000 + 0x38), Some(0), "at data offset 0");
    assert_eq!(mem.load_u32(0x2000 + 0x10 + 8), Some(1));
    assert_eq!(
        &mem.read_bytes(0x2000 + 0x10 + 0x18, 7).unwrap(),
        b"dispdrv"
    );
    // Connect over the binder: QueueBufferOutput says 1280x720.
    let mut request = vec![0u8; 0x10];
    request[4..8].copy_from_slice(&0x10u32.to_le_bytes());
    mem.write_bytes(0x2100, &request);
    let mut args = 1u32.to_le_bytes().to_vec();
    args.extend_from_slice(&10u32.to_le_bytes());
    call(relay, 0, &args, &[(0x2100, 0x10)], &[(0x2200, 0x100)]);
    assert_eq!(mem.load_u32(0x2200 + 0x10), Some(1280));
    assert_eq!(mem.load_u32(0x2200 + 0x14), Some(720));
    // GetBufferHistory for no entries: an OK status, a zero count, and the
    // transaction's OK status.
    args[4..8].copy_from_slice(&17u32.to_le_bytes());
    mem.write_bytes(0x2200, &[0xAA; 0x20]);
    call(relay, 0, &args, &[(0x2100, 0x10)], &[(0x2200, 0x100)]);
    assert_eq!(mem.load_u32(0x2200), Some(12), "three words of data");
    assert_eq!(mem.load_u64(0x2200 + 0x10), Some(0));
    assert_eq!(mem.load_u32(0x2200 + 0x18), Some(0));
    // The vsync event fires within a frame or two.
    let (_, handles) = call(display, 5202, &1u64.to_le_bytes(), &[], &[]);
    let kernel = recomp_services::kernel::kernel();
    assert_eq!(
        kernel.wait_synchronization(&[handles[0]], 100_000_000, 0),
        Ok(0)
    );
}

#[test]
fn audio_renderer_answers_updates_and_ticks() {
    let mem = Memory::new();
    let server = Server::new(ServerContext::default());
    let audren = open_service(&server, &mem, "audren:u");
    let mut parameters = vec![0u8; 0x38];
    for (at, value) in [
        (0x00, 48_000u32),
        (0x04, 240),
        (0x08, 24),
        (0x10, 64),
        (0x14, 2),
        (0x18, 4),
        (0x30, u32::from_le_bytes(*b"REV8")),
    ] {
        parameters[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }
    let call = |handle: u64, command, args: &[u8], b: &[(u64, u64)]| {
        buffered_request(&mem, command, args, &[], &[], b);
        assert_eq!(server.dispatch(&mem, handle, BASE), 0);
        let message = Message::parse(&mem, BASE).unwrap();
        (reply(&mem, false), message.copy_handles)
    };
    let size = u64::from_le_bytes(
        call(audren, 1, &parameters, &[]).0.data[..8]
            .try_into()
            .unwrap(),
    );
    assert!(size >= 0x10_0000 && size % 0x1000 == 0);
    let renderer = u64::from(call(audren, 0, &parameters, &[]).0.move_handles[0]);
    call(renderer, 4, &[], &[(0x2000, 0x2000), (0x3000, 0x100)]);
    let pools = (4 + 64 * 4) * 0x10;
    assert_eq!(mem.load_u32(0x2000), Some(u32::from_le_bytes(*b"REV8")));
    assert_eq!(mem.load_u32(0x2008), Some(pools), "memory pool statuses");
    assert_eq!(mem.load_u32(0x200C), Some(64 * 0x10), "voice statuses");
    assert_eq!(mem.load_u32(0x201C), Some(2 * 0x20), "sink statuses");
    let total = 0x40 + 0xB0 + pools + 0x400 + 0x40 + 0x40 + 0x10 + 0x10;
    assert_eq!(mem.load_u32(0x203C), Some(total));
    // The renderer's system event ticks every few milliseconds.
    let (_, handles) = call(renderer, 7, &[], &[]);
    let kernel = recomp_services::kernel::kernel();
    assert_eq!(
        kernel.wait_synchronization(&[handles[0]], 50_000_000, 0),
        Ok(0)
    );
}

#[test]
fn opus_decoders_measure_packets() {
    let mem = Memory::new();
    let server = Server::new(ServerContext::default());
    let hwopus = open_service(&server, &mem, "hwopus");
    let call = |handle: u64, command, args: &[u8], a: &[(u64, u64)], b: &[(u64, u64)]| {
        buffered_request(&mem, command, args, &[], a, b);
        assert_eq!(server.dispatch(&mem, handle, BASE), 0);
        reply(&mem, false)
    };
    let parameters = args(&[(0, 48_000 | (2 << 32))]);
    let size = call(hwopus, 1, &parameters, &[], &[]).data;
    assert!(u32::from_le_bytes(size[..4].try_into().unwrap()) >= 0x1000);
    let decoder = u64::from(call(hwopus, 0, &parameters, &[], &[]).move_handles[0]);

    // A 20 ms CELT packet (TOC 0xF8) behind its size/final-range header.
    mem.write_bytes(0x2000, &[0, 0, 0, 3, 0, 0, 0, 0, 0xF8, 0xFF, 0xFE]);
    mem.write_bytes(0x2100, &[0xAA; 0x1000]);
    let reply = call(decoder, 4, &[], &[(0x2000, 11)], &[(0x2100, 0x1000)]);
    assert_eq!(reply.result, 0);
    assert_eq!(
        &reply.data[..8],
        &[11, 0, 0, 0, 0xC0, 3, 0, 0],
        "11 bytes, 960 samples"
    );
    assert_eq!(reply.data.len(), 16, "decode time follows");
    assert_eq!(mem.read_bytes(0x2100, 960 * 4), Some(vec![0; 960 * 4]));
    assert_eq!(mem.load_u8(0x2100 + 960 * 4), Some(0xAA));

    // Code 3 with three frames; a payload past the input fails.
    mem.write_bytes(0x2000, &[0, 0, 0, 2, 0, 0, 0, 0, 0xFB, 0x03]);
    let reply = call(decoder, 0, &[], &[(0x2000, 10)], &[(0x2100, 0x1000)]);
    assert_eq!(&reply.data[..8], &[10, 0, 0, 0, 0x40, 0xB, 0, 0]);
    mem.write_bytes(0x2000, &[0, 0, 0, 9]);
    assert_ne!(
        call(decoder, 0, &[], &[(0x2000, 10)], &[(0x2100, 0x1000)]).result,
        0
    );
}

/// A session that blocks until another session lets it go, the way nvdrv
/// blocks waiting for a GPU syncpoint.
struct Gate {
    open: Arc<(Mutex<bool>, Condvar)>,
    entered: Arc<AtomicBool>,
}

impl Service for Gate {
    fn name(&self) -> String {
        "gate".to_string()
    }

    fn call(&mut self, _call: &mut Call<'_>) -> Result<(), u32> {
        self.entered.store(true, Ordering::Relaxed);
        let mut open = self.open.0.lock().unwrap();
        while !*open {
            open = self.open.1.wait(open).unwrap();
        }
        Ok(())
    }
}

/// The session that lets the gate through.
struct Opener {
    open: Arc<(Mutex<bool>, Condvar)>,
}

impl Service for Opener {
    fn name(&self) -> String {
        "opener".to_string()
    }

    fn call(&mut self, _call: &mut Call<'_>) -> Result<(), u32> {
        *self.open.0.lock().unwrap() = true;
        self.open.1.notify_all();
        Ok(())
    }
}

/// A blocking service call must not stop other sessions from being served: on
/// hardware every IPC request is handled at the same time, so a thread waiting
/// on a GPU syncpoint leaves the others free to map buffers and submit.
#[test]
fn a_blocked_service_call_does_not_stop_other_sessions() {
    let mem = Memory::new();
    let server = Server::new(ServerContext::default());
    let open = Arc::new((Mutex::new(false), Condvar::new()));
    let entered = Arc::new(AtomicBool::new(false));
    let gate = u64::from(server.open(Box::new(Gate {
        open: Arc::clone(&open),
        entered: Arc::clone(&entered),
    })));
    let opener = u64::from(server.open(Box::new(Opener {
        open: Arc::clone(&open),
    })));

    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        request(&mem, kind::REQUEST, 0, &[0; 8], false, None);
        scope.spawn(|| {
            tx.send(server.dispatch(&mem, gate, BASE)).expect("result");
        });

        let start = Instant::now();
        while !entered.load(Ordering::Relaxed) && start.elapsed() < Duration::from_secs(2) {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(
            entered.load(Ordering::Relaxed),
            "the call reached the service"
        );

        request(&mem, kind::REQUEST, 0, &[0; 8], false, None);
        scope.spawn(|| {
            server.dispatch(&mem, opener, BASE);
        });

        match rx.recv_timeout(Duration::from_secs(3)) {
            Ok(result) => assert_eq!(result, 0, "the blocked call finishes on time"),
            Err(_) => {
                // Another session was never served: let the gate through so
                // the scope finishes instead of hanging.
                *open.0.lock().unwrap() = true;
                open.1.notify_all();
                panic!("a blocked service call stopped other sessions");
            }
        }
    });
}
