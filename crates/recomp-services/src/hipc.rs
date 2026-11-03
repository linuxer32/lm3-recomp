//! Horizon IPC message framing (HIPC) and the CMIF command layer on top.
//!
//! A message lives in the calling thread's TLS buffer:
//!
//! ```text
//! u32 header0: type:16 | num_x:4 | num_a:4 | num_b:4 | num_w:4
//! u32 header1: data_words:10 | recv_static_mode:4 | pad:6 | recv_list_offset:11 | has_special:1
//! [special header: send_pid:1 | num_copy:4 | num_move:4] [u64 pid] [copy handles] [move handles]
//! X descriptors (8 bytes), A, B, W descriptors (12 bytes each)
//! raw data (data_words words; the CMIF payload starts 16-byte aligned)
//! C descriptors (receive list, 8 bytes each)
//! ```
//!
//! CMIF requests carry `SFCI`, version, command id and token ahead of their
//! arguments; replies carry `SFCO`, version, result and token. Sessions
//! converted to domains prefix both with a 16-byte domain header.

use crate::ipc::MemoryAccess;

pub const CMIF_IN_MAGIC: u32 = 0x4943_4653; // "SFCI"
pub const CMIF_OUT_MAGIC: u32 = 0x4F43_4653; // "SFCO"

/// HIPC/CMIF message types.
pub mod kind {
    pub const CLOSE: u16 = 2;
    pub const REQUEST: u16 = 4;
    pub const CONTROL: u16 = 5;
    pub const REQUEST_WITH_CONTEXT: u16 = 6;
    pub const CONTROL_WITH_CONTEXT: u16 = 7;
}

/// A guest buffer named by a descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Buffer {
    pub address: u64,
    pub size: u64,
    /// X descriptors: the static index. A/B/W: the transfer mode.
    pub index_or_mode: u32,
}

/// A parsed HIPC request.
#[derive(Debug, Clone, Default)]
pub struct Message {
    pub kind: u16,
    pub send_pid: bool,
    pub copy_handles: Vec<u32>,
    pub move_handles: Vec<u32>,
    /// X (send static) descriptors.
    pub statics: Vec<Buffer>,
    /// A (send) descriptors.
    pub sends: Vec<Buffer>,
    /// B (receive) descriptors.
    pub receives: Vec<Buffer>,
    /// W (exchange) descriptors.
    pub exchanges: Vec<Buffer>,
    /// C (receive static) descriptors.
    pub recv_statics: Vec<Buffer>,
    /// Guest address and length of the raw data words.
    pub raw: u64,
    pub raw_len: u64,
}

fn read_u32(mem: &dyn MemoryAccess, addr: u64) -> Option<u32> {
    mem.load_u32(addr)
}

impl Message {
    pub fn parse(mem: &dyn MemoryAccess, base: u64) -> Option<Self> {
        let header0 = read_u32(mem, base)?;
        let header1 = read_u32(mem, base + 4)?;
        let mut message = Message {
            kind: (header0 & 0xFFFF) as u16,
            ..Message::default()
        };
        let num_x = (header0 >> 16) & 0xF;
        let num_a = (header0 >> 20) & 0xF;
        let num_b = (header0 >> 24) & 0xF;
        let num_w = (header0 >> 28) & 0xF;
        let data_words = header1 & 0x3FF;
        let recv_static_mode = (header1 >> 10) & 0xF;
        let recv_list_offset = (header1 >> 20) & 0x7FF;

        let mut pos = base + 8;
        if header1 >> 31 != 0 {
            let special = read_u32(mem, pos)?;
            pos += 4;
            message.send_pid = special & 1 != 0;
            if message.send_pid {
                pos += 8;
            }
            for _ in 0..(special >> 1) & 0xF {
                message.copy_handles.push(read_u32(mem, pos)?);
                pos += 4;
            }
            for _ in 0..(special >> 5) & 0xF {
                message.move_handles.push(read_u32(mem, pos)?);
                pos += 4;
            }
        }
        for _ in 0..num_x {
            let word = read_u32(mem, pos)?;
            let low = u64::from(read_u32(mem, pos + 4)?);
            message.statics.push(Buffer {
                address: low
                    | (u64::from((word >> 12) & 0xF) << 32)
                    | (u64::from((word >> 6) & 0x3F) << 36),
                size: u64::from(word >> 16),
                index_or_mode: word & 0x3F,
            });
            pos += 8;
        }
        for (count, list) in [
            (num_a, &mut message.sends),
            (num_b, &mut message.receives),
            (num_w, &mut message.exchanges),
        ] {
            for _ in 0..count {
                let size_low = u64::from(read_u32(mem, pos)?);
                let address_low = u64::from(read_u32(mem, pos + 4)?);
                let word = read_u32(mem, pos + 8)?;
                list.push(Buffer {
                    address: address_low
                        | (u64::from((word >> 28) & 0xF) << 32)
                        | (u64::from((word >> 2) & 0x3F_FFFF) << 36),
                    size: size_low | (u64::from((word >> 24) & 0xF) << 32),
                    index_or_mode: word & 0x3,
                });
                pos += 12;
            }
        }
        message.raw = pos;
        message.raw_len = u64::from(data_words) * 4;

        if recv_static_mode >= 2 {
            let count = if recv_static_mode == 2 {
                1
            } else {
                recv_static_mode - 2
            };
            let mut entry = if recv_list_offset != 0 {
                base + u64::from(recv_list_offset) * 4
            } else {
                message.raw + message.raw_len
            };
            for _ in 0..count {
                let low = u64::from(read_u32(mem, entry)?);
                let word = read_u32(mem, entry + 4)?;
                message.recv_statics.push(Buffer {
                    address: low | (u64::from(word & 0xFFFF) << 32),
                    size: u64::from(word >> 16),
                    index_or_mode: 0,
                });
                entry += 8;
            }
        }
        Some(message)
    }

    /// Where the CMIF payload starts: the raw data aligned to 16 bytes.
    pub fn payload(&self) -> u64 {
        self.raw.next_multiple_of(16)
    }
}

/// A domain request header (sessions converted to domains).
#[derive(Debug, Clone, Copy, Default)]
pub struct DomainHeader {
    /// 1 = send message, 2 = close object.
    pub command: u8,
    pub object_id: u32,
    /// Bytes of CMIF header plus arguments.
    pub data_size: u16,
    pub num_in_objects: u8,
}

/// A parsed CMIF request.
#[derive(Debug, Clone, Default)]
pub struct CmifRequest {
    pub command: u32,
    pub token: u32,
    pub domain: Option<DomainHeader>,
    /// Object ids passed into a domain request.
    pub in_objects: Vec<u32>,
    /// Guest address and length of the command arguments.
    pub args: u64,
    pub args_len: u64,
}

impl CmifRequest {
    pub fn parse(mem: &dyn MemoryAccess, message: &Message, is_domain: bool) -> Option<Self> {
        let mut header = message.payload();
        let end = message.raw + message.raw_len;
        let mut request = CmifRequest::default();
        if is_domain {
            let word = read_u32(mem, header)?;
            let domain = DomainHeader {
                command: (word & 0xFF) as u8,
                num_in_objects: ((word >> 8) & 0xFF) as u8,
                data_size: (word >> 16) as u16,
                object_id: read_u32(mem, header + 4)?,
            };
            header += 16;
            let objects = header + u64::from(domain.data_size);
            for index in 0..u64::from(domain.num_in_objects) {
                request.in_objects.push(read_u32(mem, objects + index * 4)?);
            }
            request.domain = Some(domain);
            if domain.command == 2 {
                return Some(request);
            }
        }
        if read_u32(mem, header)? != CMIF_IN_MAGIC {
            return None;
        }
        request.command = read_u32(mem, header + 8)?;
        request.token = read_u32(mem, header + 12)?;
        request.args = header + 16;
        request.args_len = end.saturating_sub(request.args);
        Some(request)
    }
}

/// Everything a reply carries besides the CMIF result.
#[derive(Debug, Clone, Default)]
pub struct Reply {
    pub result: u32,
    pub data: Vec<u8>,
    pub copy_handles: Vec<u32>,
    pub move_handles: Vec<u32>,
    /// For domain sessions: ids of objects returned by this call.
    pub out_objects: Vec<u32>,
    /// Static descriptors the reply carries (server-written C buffers).
    pub statics: Vec<Buffer>,
}

impl Reply {
    pub fn error(result: u32) -> Self {
        Reply {
            result,
            ..Reply::default()
        }
    }
}

/// Write a CMIF reply into the message buffer at `base`. `domain` adds the
/// domain output header and the returned object ids.
pub fn write_reply(mem: &dyn MemoryAccess, base: u64, reply: &Reply, domain: bool) -> bool {
    let special = !reply.copy_handles.is_empty() || !reply.move_handles.is_empty();
    let header0 = (reply.statics.len() as u32 & 0xF) << 16;
    let mut pos = base + 8;
    if special {
        pos += 4 + 4 * (reply.copy_handles.len() + reply.move_handles.len()) as u64;
    }
    pos += 8 * reply.statics.len() as u64;
    let raw = pos;
    let payload = raw.next_multiple_of(16);

    let mut body = Vec::new();
    if domain {
        body.extend_from_slice(&(reply.out_objects.len() as u32).to_le_bytes());
        body.extend_from_slice(&[0u8; 12]);
    }
    body.extend_from_slice(&CMIF_OUT_MAGIC.to_le_bytes());
    body.extend_from_slice(&0u32.to_le_bytes());
    body.extend_from_slice(&reply.result.to_le_bytes());
    body.extend_from_slice(&0u32.to_le_bytes());
    body.extend_from_slice(&reply.data);
    while !body.len().is_multiple_of(4) {
        body.push(0);
    }
    if domain {
        for object in &reply.out_objects {
            body.extend_from_slice(&object.to_le_bytes());
        }
    }
    // Clients locate the payload by aligning the raw data start, so the word
    // count covers the alignment padding as well.
    let data_words = ((payload - raw) + body.len() as u64).div_ceil(4) as u32;
    let header1 = data_words | if special { 1 << 31 } else { 0 };

    let mut ok = mem.store_u32(base, header0) && mem.store_u32(base + 4, header1);
    let mut cursor = base + 8;
    if special {
        let special_header = ((reply.copy_handles.len() as u32 & 0xF) << 1)
            | ((reply.move_handles.len() as u32 & 0xF) << 5);
        ok &= mem.store_u32(cursor, special_header);
        cursor += 4;
        for handle in reply.copy_handles.iter().chain(&reply.move_handles) {
            ok &= mem.store_u32(cursor, *handle);
            cursor += 4;
        }
    }
    for buffer in &reply.statics {
        let word = (buffer.index_or_mode & 0x3F)
            | (((buffer.address >> 36) as u32 & 0x3F) << 6)
            | (((buffer.address >> 32) as u32 & 0xF) << 12)
            | ((buffer.size as u32 & 0xFFFF) << 16);
        ok &= mem.store_u32(cursor, word) && mem.store_u32(cursor + 4, buffer.address as u32);
        cursor += 8;
    }
    if payload > raw {
        ok &= mem.write_bytes(raw, &vec![0u8; (payload - raw) as usize]);
    }
    ok && mem.write_bytes(payload, &body)
}
