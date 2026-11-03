//! `vi:u` / `vi:s` / `vi:m`: one 1280x720 display at 60 Hz with layers
//! whose buffer queues (Android IGraphicBufferProducer over
//! IHOSBinderDriver) accept frames.
//!
//! The queue paces the title as the console's compositor does: a queued
//! frame goes on screen at a vsync once the frame before it has been shown
//! for its swap interval, and the buffer it replaces is released then.
//! DequeueBuffer blocks until a buffer is free. Without that a title renders
//! as fast as it can and its render thread starves the others (LM3's loader
//! barely ran while its loading screen spun at 70+ frames per second).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use crate::server::{Call, Service};
use crate::services::unimplemented;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const DISPLAY_ID: u64 = 1;
/// Binder id of the one layer's buffer queue.
const BINDER_ID: u32 = 1;
const LAYER_ID: u64 = 1;
/// Frames presented so far.
static FRAMES: AtomicU64 = AtomicU64::new(0);

/// Frames the display has shown (for frame rate counters).
pub fn frames_shown() -> u64 {
    FRAMES.load(Ordering::Relaxed)
}

/// The frame rate the player picked: 0 for the title's own (its swap
/// interval decides), or 60, 120, 180 or 240 to show a new frame every
/// vsync, at that refresh rate (`RECOMP_FPS`, a script's
/// `setting("fps", 60)`, or the menu). A title that times its frames itself
/// needs its own patch as well (LM3's frame budget constant, which
/// `game.rhai` sets in `on_frame_rate`).
pub fn frame_rate() -> u32 {
    match FRAME_RATE.load(Ordering::Acquire) {
        u32::MAX => {
            let fps = std::env::var("RECOMP_FPS")
                .ok()
                .and_then(|value| value.parse().ok())
                .map_or(0, |fps: u32| if fps > 30 { fps } else { 0 });
            FRAME_RATE.store(fps, Ordering::Release);
            fps
        }
        fps => fps,
    }
}

pub fn set_frame_rate(fps: u32) {
    FRAME_RATE.store(if fps > 30 { fps } else { 0 }, Ordering::Release);
}

static FRAME_RATE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(u32::MAX);

/// The display's vsync period at the picked frame rate: 60 Hz for the
/// title's own mode, otherwise 120, 180 or 240 Hz (any rate past 60 runs at
/// that period). A title's own mode stays at 60 Hz: its swap interval of 2
/// shows each frame for two vsyncs, 30 fps.
fn vsync_period() -> Duration {
    vsync_period_for(frame_rate())
}

/// The period for `fps` on the 60 Hz display: `fps` vsyncs' worth of time.
/// Rates at or below 60 (including the title's own 0) stay at 60 Hz.
fn vsync_period_for(fps: u32) -> Duration {
    (VSYNC * 60 / fps.max(60)).max(Duration::from_micros(1))
}

/// The interval a queued frame is shown for: one vsync at every picked rate
/// (60 and up), otherwise the title's own interval (2 for its 30 fps mode).
fn swap_interval_for(fps: u32, title_interval: u32) -> u32 {
    if fps > 30 {
        1
    } else {
        title_interval
    }
}

/// The crop rectangle of the last queued frame (right and bottom; the left
/// and top edges are 0 for titles that render into the corner), packed
/// `right << 32 | bottom`; 0 before any.
static DISPLAY_CROP: AtomicU64 = AtomicU64::new(0);

/// The part of a display buffer the last queued frame shows: (width,
/// height) from its crop rectangle, if it had one.
pub fn display_crop() -> Option<(u32, u32)> {
    let crop = DISPLAY_CROP.load(Ordering::Relaxed);
    let (right, bottom) = ((crop >> 32) as u32, crop as u32);
    (right > 0 && bottom > 0).then_some((right, bottom))
}

/// GPU addresses of the buffers the title has queued for display (each is
/// reported once).
static DISPLAY_BUFFERS: std::sync::Mutex<Vec<u64>> = std::sync::Mutex::new(Vec::new());

/// The memory of a flattened GraphicBuffer: the nvmap id at 0x2C (in the
/// NvGraphicBuffer after the 0x28-byte header) and the first surface's
/// offset at 0x78 (surfaces start at 0x5C; offset is their field 0x1C).
fn graphic_buffer_memory(buffer: &[u8]) -> Option<(u32, u64)> {
    let word = |at: usize| {
        buffer
            .get(at..at + 4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
    };
    Some((word(0x2C)?, u64::from(word(0x78)?)))
}
/// The GPU addresses a queued slot's buffer is mapped at, resolved once per
/// buffer (the lookup takes the GPU mappings, which the GPU worker holds
/// while it decodes) and remembered as display buffers.
fn display_buffer_addresses(buffer: &[u8]) -> Vec<u64> {
    type Resolved = Vec<((u32, u64), Vec<u64>)>;
    static RESOLVED: std::sync::Mutex<Resolved> = std::sync::Mutex::new(Vec::new());
    let Some(memory) = graphic_buffer_memory(buffer) else {
        return Vec::new();
    };
    let mut resolved = RESOLVED
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some((_, addresses)) = resolved.iter().find(|(known, _)| *known == memory) {
        return addresses.clone();
    }
    let (handle, offset) = memory;
    let Some(cpu) = crate::nvdrv::nvmap_address(handle) else {
        return Vec::new();
    };
    let addresses = crate::gpu::gpu_addresses_of(cpu + offset);
    if addresses.is_empty() {
        return addresses;
    }
    resolved.push((memory, addresses.clone()));
    let mut known = DISPLAY_BUFFERS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for address in &addresses {
        if !known.contains(address) {
            println!("[recomp-vi] display buffer at GPU address {address:#x}");
            known.push(*address);
        }
    }
    addresses
}

/// The display's refresh period.
const VSYNC: Duration = Duration::from_micros(16_667);

static FRAME_SHOWN: OnceLock<fn(u64)> = OnceLock::new();

/// Call `callback` with the frame count each time a frame goes on screen.
/// It runs on the vsync thread, so it must return quickly (the runtime's
/// script support only wakes its own thread).
pub fn on_frame_shown(callback: fn(u64)) {
    let _ = FRAME_SHOWN.set(callback);
}

/// The display's vsync event, signalled at the picked refresh rate by the
/// compositor thread, which also shows queued frames. Started on first use.
fn vsync_event() -> u32 {
    static EVENT: OnceLock<u32> = OnceLock::new();
    *EVENT.get_or_init(|| {
        let (write, read) = crate::kernel::kernel().create_event();
        let _ = std::thread::Builder::new()
            .name("recomp-vsync".to_string())
            .spawn(move || {
                let mut next = Instant::now();
                loop {
                    next += vsync_period();
                    let now = Instant::now();
                    if next > now {
                        std::thread::sleep(next - now);
                    } else {
                        // Fell behind (the host was busy): skip the missed
                        // vsyncs rather than run them back to back.
                        next = now;
                    }
                    if queue().vsync() {
                        let frame = FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
                        if frame == 1 || frame.is_power_of_two() {
                            println!("[recomp-vi] frame {frame} shown");
                        }
                        if let Some(callback) = FRAME_SHOWN.get() {
                            callback(frame);
                        }
                    }
                    let kernel = crate::kernel::kernel();
                    kernel.signal_event(write);
                    // A buffer may have been released: wake DequeueBuffer.
                    kernel.notify_waiters();
                }
            });
        read
    })
}

/// Signalled when the buffer queue can hand out a buffer; always.
fn buffer_event() -> u32 {
    static EVENT: OnceLock<u32> = OnceLock::new();
    *EVENT.get_or_init(|| {
        let (write, read) = crate::kernel::kernel().create_event();
        crate::kernel::kernel().signal_event(write);
        read
    })
}

/// A serialized binder parcel: 16-byte header (data size and offset,
/// object list size and offset), the data, then the object list (offsets
/// of flattened binder objects within the data).
fn parcel_with_objects(data: &[u8], objects: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(0x10 + data.len() + 4 * objects.len());
    bytes.extend_from_slice(&(data.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&0x10u32.to_le_bytes());
    bytes.extend_from_slice(&((4 * objects.len()) as u32).to_le_bytes());
    bytes.extend_from_slice(&((0x10 + data.len()) as u32).to_le_bytes());
    bytes.extend_from_slice(data);
    for object in objects {
        bytes.extend_from_slice(&object.to_le_bytes());
    }
    bytes
}

fn parcel(data: &[u8]) -> Vec<u8> {
    parcel_with_objects(data, &[])
}

/// The native window an OpenLayer returns: a flattened binder object
/// (type 2, process 1, the binder id, the "dispdrv" service name), listed
/// as the parcel's one object.
fn native_window() -> Vec<u8> {
    let mut window = vec![0u8; 0x28];
    window[0..4].copy_from_slice(&2u32.to_le_bytes());
    window[4..8].copy_from_slice(&1u32.to_le_bytes());
    window[8..12].copy_from_slice(&BINDER_ID.to_le_bytes());
    window[0x18..0x20].copy_from_slice(b"dispdrv\0");
    parcel_with_objects(&window, &[0])
}

/// `vi:u`, `vi:s`, `vi:m` (IApplicationRootService and friends)
pub struct ViRootService;

impl Service for ViRootService {
    fn name(&self) -> String {
        "vi".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // GetDisplayService (vi:u 0, vi:s 1, vi:m 2),
            // GetDisplayServiceWithProxyNameExchange
            0..=3 => {
                call.push_object(Box::new(DisplayService));
                Ok(())
            }
            _ => unimplemented(call, "vi"),
        }
    }
}

/// IApplicationDisplayService
struct DisplayService;

impl Service for DisplayService {
    fn name(&self) -> String {
        "vi:IApplicationDisplayService".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // GetRelayService, GetIndirectDisplayTransactionService
            100 | 103 => call.push_object(Box::new(BinderDriver)),
            // GetSystemDisplayService, GetManagerDisplayService
            101 | 102 => call.push_object(Box::new(DisplayControl)),
            // ListDisplays -> (infos, count)
            1000 => {
                let mut info = vec![0u8; 0x60];
                info[..7].copy_from_slice(b"Default");
                info[0x40] = 1;
                info[0x48..0x50].copy_from_slice(&1u64.to_le_bytes());
                info[0x50..0x58].copy_from_slice(&u64::from(WIDTH).to_le_bytes());
                info[0x58..0x60].copy_from_slice(&u64::from(HEIGHT).to_le_bytes());
                call.write_out(0, &info);
                call.push_u64(1);
            }
            // OpenDisplay(name), OpenDefaultDisplay -> display id
            1010 | 1011 => call.push_u64(DISPLAY_ID),
            // CloseDisplay, SetDisplayEnabled, CloseLayer, DestroyStrayLayer,
            // SetLayerScalingMode
            1020 | 1101 | 2021 | 2031 | 2101 => {}
            // GetDisplayResolution -> (width, height)
            1102 => {
                call.push_u64(u64::from(WIDTH));
                call.push_u64(u64::from(HEIGHT));
            }
            // OpenLayer(name, layer, aruid) -> (native window, size)
            2020 => {
                let written = call.write_out(0, &native_window());
                call.push_u64(written as u64);
            }
            // CreateStrayLayer(flags, display) -> (layer, size, native window)
            2030 => {
                let written = call.write_out(0, &native_window());
                call.push_u64(LAYER_ID);
                call.push_u64(written as u64);
            }
            // ConvertScalingMode(mode) -> mode
            2102 => call.push_u64(u64::from(call.arg_u32(0))),
            // GetIndirectLayerImageRequiredMemoryInfo(width, height)
            // -> (size, alignment)
            2460 => {
                let (width, height) = (call.arg_u64(0), call.arg_u64(8));
                call.push_u64((width * height * 4).next_multiple_of(0x20000));
                call.push_u64(0x1000);
            }
            // GetDisplayVsyncEvent(display), GetDisplayVsyncEventForDebug
            5202 | 5203 => call.push_copy_handle(vsync_event()),
            _ => return unimplemented(call, "vi:IApplicationDisplayService"),
        }
        Ok(())
    }
}

/// ISystemDisplayService / IManagerDisplayService: layer placement and
/// display modes.
struct DisplayControl;

impl Service for DisplayControl {
    fn name(&self) -> String {
        "vi:ISystemDisplayService".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // Layer position, size, z order and visibility; layer stacks;
            // managed layer teardown.
            1200 | 2201 | 2203 | 2205 | 2207 | 2209 | 2011 | 2402 | 6000..=6010 => {}
            // GetZOrderCountMin / Max
            1201 => call.push_u64(0),
            1202 => call.push_u64(255),
            // CreateManagedLayer -> layer id
            2010 => call.push_u64(LAYER_ID),
            // GetDisplayMode -> (width, height, refresh rate, unknown)
            3200 => {
                call.push_u32(WIDTH);
                call.push_u32(HEIGHT);
                call.push_u32(60f32.to_bits());
                call.push_u32(0);
            }
            _ => return unimplemented(call, "vi:ISystemDisplayService"),
        }
        Ok(())
    }
}

/// IGraphicBufferProducer transaction codes.
mod transaction {
    pub const REQUEST_BUFFER: u32 = 1;
    pub const SET_BUFFER_COUNT: u32 = 2;
    pub const DEQUEUE_BUFFER: u32 = 3;
    pub const DETACH_BUFFER: u32 = 4;
    pub const QUEUE_BUFFER: u32 = 7;
    pub const CANCEL_BUFFER: u32 = 8;
    pub const QUERY: u32 = 9;
    pub const CONNECT: u32 = 10;
    pub const DISCONNECT: u32 = 11;
    pub const SET_PREALLOCATED_BUFFER: u32 = 14;
    pub const GET_BUFFER_HISTORY: u32 = 17;
}

/// IHOSBinderDriver: the layer's buffer queue.
struct BinderDriver;

/// The buffer queue of the one layer, shared by every binder session.
fn queue() -> std::sync::MutexGuard<'static, BufferQueue> {
    static QUEUE: std::sync::Mutex<BufferQueue> = std::sync::Mutex::new(BufferQueue::new());
    QUEUE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

const SLOTS: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotState {
    Free,
    /// Handed to the title by DequeueBuffer.
    Dequeued,
    /// Queued, waiting for a vsync.
    Queued,
    /// On screen.
    Shown,
}

struct BufferQueue {
    /// Each slot's flattened GraphicBuffer, as SetPreallocatedBuffer gave it.
    slots: [Option<Vec<u8>>; SLOTS],
    states: [SlotState; SLOTS],
    /// Queued frames in order: slot and swap interval (vsyncs on screen).
    queued: VecDeque<(usize, u32)>,
    /// Vsyncs the frame on screen has been shown for, and how many it stays.
    shown_for: u32,
    shown_interval: u32,
    /// Frames queued so far, and each slot's latest frame number.
    frames: i32,
    slot_frames: [i32; SLOTS],
    /// The last frames queued, for GetBufferHistory: a ring, `history_at`
    /// the newest.
    history: [HistoryEntry; HISTORY],
    history_at: usize,
}

/// Frames GetBufferHistory remembers (as Nintendo's buffer queue does).
const HISTORY: usize = 8;

/// One queued frame, as GetBufferHistory reports it: frame number, when it
/// was queued and shown (nanoseconds), and its state (0 free, 1 dequeued,
/// 2 queued, 3 acquired by the compositor).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HistoryEntry {
    frame: i32,
    queued_at: i64,
    shown_at: i64,
    state: u32,
}

impl HistoryEntry {
    const EMPTY: Self = Self {
        frame: 0,
        queued_at: 0,
        shown_at: 0,
        state: 0,
    };

    /// The packed 28-byte layout: 64-bit frame number, the two times,
    /// state (Ryujinx's `BufferInfo`, size 0x1C).
    fn bytes(&self) -> [u8; 28] {
        let mut out = [0u8; 28];
        out[0..8].copy_from_slice(&i64::from(self.frame).to_le_bytes());
        out[8..16].copy_from_slice(&self.queued_at.to_le_bytes());
        out[16..24].copy_from_slice(&self.shown_at.to_le_bytes());
        out[24..28].copy_from_slice(&self.state.to_le_bytes());
        out
    }
}

/// Nanoseconds on the compositor's clock: the system counter's, which the
/// SDK converts these times to (`nn::vi::ListFrameHistory`) and compares
/// with `GetSystemTick`.
fn now_ns() -> i64 {
    crate::clock::now_ns() as i64
}

impl BufferQueue {
    const fn new() -> Self {
        Self {
            slots: [const { None }; SLOTS],
            states: [SlotState::Free; SLOTS],
            queued: VecDeque::new(),
            shown_for: 0,
            shown_interval: 1,
            frames: 0,
            slot_frames: [0; SLOTS],
            history: [HistoryEntry::EMPTY; HISTORY],
            history_at: 0,
        }
    }

    /// The newest `count` history entries, newest first.
    fn history(&self, count: usize) -> Vec<HistoryEntry> {
        (0..count.min(HISTORY))
            .map(|age| self.history[(self.history_at + HISTORY - age) % HISTORY])
            .collect()
    }

    /// A free slot that has a buffer, now dequeued; `None` while every buffer
    /// is in use. Before any buffer is set, slot 0.
    fn dequeue(&mut self) -> Option<usize> {
        if self.slots.iter().all(Option::is_none) {
            return Some(0);
        }
        let slot = (0..SLOTS)
            .find(|slot| self.slots[*slot].is_some() && self.states[*slot] == SlotState::Free)?;
        self.states[slot] = SlotState::Dequeued;
        Some(slot)
    }

    fn queue(&mut self, slot: usize, swap_interval: u32) {
        if slot < SLOTS {
            self.states[slot] = SlotState::Queued;
            self.queued.push_back((slot, swap_interval.max(1)));
            self.frames += 1;
            self.slot_frames[slot] = self.frames;
            self.history_at = (self.history_at + 1) % HISTORY;
            self.history[self.history_at] = HistoryEntry {
                frame: self.frames,
                queued_at: now_ns(),
                shown_at: 0,
                state: 2,
            };
        }
    }

    fn cancel(&mut self, slot: usize) {
        if self.states.get(slot) == Some(&SlotState::Dequeued) {
            self.states[slot] = SlotState::Free;
        }
    }

    /// Forget every frame: buffers are all free again (Disconnect).
    fn reset(&mut self) {
        self.states = [SlotState::Free; SLOTS];
        self.queued.clear();
        self.shown_for = 0;
        self.shown_interval = 1;
    }

    /// One vsync: show the next queued frame once the frame on screen has
    /// had its interval, releasing that frame's buffer. True when a frame
    /// was shown.
    fn vsync(&mut self) -> bool {
        self.shown_for = self.shown_for.saturating_add(1);
        if self.shown_for < self.shown_interval {
            return false;
        }
        let Some((slot, interval)) = self.queued.pop_front() else {
            return false;
        };
        for state in &mut self.states {
            if *state == SlotState::Shown {
                *state = SlotState::Free;
            }
        }
        self.states[slot] = SlotState::Shown;
        self.shown_for = 0;
        self.shown_interval = interval;
        let frame = self.slot_frames[slot];
        if let Some(entry) = self.history.iter_mut().find(|entry| entry.frame == frame) {
            entry.state = 3;
            entry.shown_at = now_ns();
        }
        true
    }
}

/// QueueBufferOutput: width, height, transform hint, pending buffers, and
/// the 64-bit number of the frame last queued. The client keeps that number
/// (`Surface::getQueuedFrameNumber`) and LM3's title screen waits until
/// GetBufferHistory reports that frame on screen; without it the number read
/// past the reply and never matched.
fn queue_output() -> Vec<u8> {
    let (pending, frame) = {
        let queue = queue();
        (queue.queued.len() as u32, queue.frames)
    };
    let mut out = Vec::new();
    for value in [WIDTH, HEIGHT, 0, pending] {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out.extend_from_slice(&i64::from(frame).to_le_bytes());
    out
}

impl BinderDriver {
    /// Answer one IGraphicBufferProducer transaction; returns the reply
    /// data (the status word last).
    fn transact(code: u32, input: &[u8]) -> Vec<u8> {
        // The request: parcel header, then the interface token (strict
        // mode policy, UTF-16 length, characters), then arguments.
        let data_offset = input
            .get(4..8)
            .map_or(0x10, |b| u32::from_le_bytes(b.try_into().unwrap()) as usize);
        let data = input.get(data_offset..).unwrap_or(&[]);
        let token_chars = data
            .get(4..8)
            .map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()) as usize);
        let args_at = 8 + ((token_chars + 1) * 2).next_multiple_of(4);
        let arg = |index: usize| {
            data.get(args_at + index * 4..args_at + index * 4 + 4)
                .map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()))
        };
        let mut reply = Vec::new();
        let push = |reply: &mut Vec<u8>, value: u32| reply.extend_from_slice(&value.to_le_bytes());
        if crate::trace_ipc() {
            println!(
                "[recomp-ipc] vi: transaction {code} ({:#x}, {:#x})",
                arg(0),
                arg(1)
            );
        }
        match code {
            transaction::CONNECT => reply.extend_from_slice(&queue_output()),
            transaction::QUERY => {
                let value = match arg(0) {
                    0 | 6 => WIDTH,
                    1 | 7 => HEIGHT,
                    // Format: RGBA_8888.
                    2 => 1,
                    // Minimum undequeued buffers.
                    3 => 1,
                    _ => 0,
                };
                push(&mut reply, value);
            }
            // SetPreallocatedBuffer(slot, flattened GraphicBuffer: valid,
            // size, bytes)
            transaction::SET_PREALLOCATED_BUFFER => {
                let slot = arg(0) as usize;
                let size = u64::from(arg(2)) | (u64::from(arg(3)) << 32);
                let start = args_at + 16;
                let buffer = (arg(1) != 0)
                    .then(|| data.get(start..start + size as usize).map(<[u8]>::to_vec))
                    .flatten();
                if let Some(entry) = queue().slots.get_mut(slot) {
                    *entry = buffer;
                }
            }
            transaction::DEQUEUE_BUFFER => {
                // Wait for a vsync to release a buffer, with the scheduler
                // released (the compositor thread wakes the waiters).
                vsync_event();
                let slot = std::cell::Cell::new(None);
                crate::kernel::kernel().wait_for(-1, || {
                    slot.set(queue().dequeue());
                    slot.get().is_some()
                });
                let slot = slot.get().unwrap_or(0);
                // Slot, a fence follows: flattened MultiFence with none.
                push(&mut reply, slot as u32);
                push(&mut reply, 1);
                push(&mut reply, 0x24);
                push(&mut reply, 0);
                reply.extend_from_slice(&[0; 0x24]);
            }
            // QueueBuffer(slot, flattened QueueBufferInput: size, fd count,
            // timestamp (2 words), auto timestamp, crop (4), scaling mode,
            // transform, sticky transform, async, swap interval, fence)
            transaction::QUEUE_BUFFER => {
                vsync_event();
                static FIRST: std::sync::Once = std::sync::Once::new();
                FIRST.call_once(|| {
                    println!(
                        "[recomp-vi] first QueueBuffer: slot {}, input size {:#x}, crop ({}, {})-({}, {}), swap interval {}",
                        arg(0),
                        arg(1),
                        arg(6),
                        arg(7),
                        arg(8),
                        arg(9),
                        arg(14)
                    );
                });
                let crop = (u64::from(arg(8)) << 32) | u64::from(arg(9));
                if DISPLAY_CROP.swap(crop, Ordering::Relaxed) != crop
                    && FRAMES.load(Ordering::Relaxed) > 0
                {
                    println!("[recomp-vi] frames are now {}x{}", arg(8), arg(9));
                }
                // A picked frame rate shows every frame for one vsync.
                let interval = swap_interval_for(frame_rate(), arg(14));
                let buffer = {
                    let mut queue = queue();
                    queue.queue(arg(0) as usize, interval);
                    queue.slots.get(arg(0) as usize).cloned().flatten()
                };
                // The frame's fence (MultiFence: count, then up to four
                // syncpoint/value pairs; id -1 is no fence): the host shows
                // the buffer once the GPU work reaches it.
                let fences: Vec<(u32, u32)> = (0..arg(15).min(4) as usize)
                    .map(|index| (arg(16 + 2 * index), arg(17 + 2 * index)))
                    .filter(|(syncpoint, _)| *syncpoint != u32::MAX)
                    .collect();
                if let Some(buffer) = buffer {
                    let addresses = display_buffer_addresses(&buffer);
                    if !addresses.is_empty() {
                        crate::gpu::present(addresses, display_crop(), fences);
                    }
                }
                reply.extend_from_slice(&queue_output());
            }
            // RequestBuffer(slot) -> flattened GraphicBuffer (valid, size,
            // bytes) of that slot.
            transaction::REQUEST_BUFFER => {
                let slot = arg(0) as usize;
                match queue().slots.get(slot).cloned().flatten() {
                    Some(buffer) => {
                        push(&mut reply, 1);
                        reply.extend_from_slice(&(buffer.len() as u64).to_le_bytes());
                        reply.extend_from_slice(&buffer);
                    }
                    None => push(&mut reply, 0),
                }
            }
            // GetBufferHistory(count) -> entries: none are kept (the
            // reply is a zero count and the status either way round).
            // GetBufferHistory(count): status, how many entries, then the
            // entries, newest first (as Ryujinx answers it). The title
            // screen paces its frames by this: answering "no history" left
            // LM3 waiting for its last frame to reach the screen, forever.
            transaction::GET_BUFFER_HISTORY => {
                let count = (arg(0) as i32).clamp(0, HISTORY as i32) as usize;
                let entries = queue().history(count);
                if crate::trace_ipc() {
                    let summary: Vec<String> = entries
                        .iter()
                        .map(|entry| format!("{}:{}", entry.frame, entry.state))
                        .collect();
                    println!("[recomp-ipc] vi: history {}", summary.join(" "));
                }
                push(&mut reply, 0);
                push(&mut reply, entries.len() as u32);
                for entry in &entries {
                    reply.extend_from_slice(&entry.bytes());
                }
            }
            transaction::CANCEL_BUFFER => queue().cancel(arg(0) as usize),
            transaction::DISCONNECT => queue().reset(),
            transaction::SET_BUFFER_COUNT | transaction::DETACH_BUFFER => {}
            other => println!("[recomp-ipc] UNIMPLEMENTED binder transaction {other}"),
        }
        // Status: OK.
        push(&mut reply, 0);
        parcel(&reply)
    }
}

impl Service for BinderDriver {
    fn name(&self) -> String {
        "vi:IHOSBinderDriver".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // TransactParcel(binder, code, flags, in parcel, out parcel),
            // TransactParcelAuto
            0 | 3 => {
                let code = call.arg_u32(4);
                let input = call.in_buffer(0);
                let reply = Self::transact(code, &input);
                if crate::trace_ipc() {
                    let capacity = call.out_buffer(0).map_or(0, |buffer| buffer.size);
                    println!(
                        "[recomp-vi] binder transaction {code}: {} bytes in, {} bytes out (room for {capacity})",
                        input.len(),
                        reply.len()
                    );
                }
                call.write_out(0, &reply);
            }
            // AdjustRefcount(binder, add, type)
            1 => {}
            // GetNativeHandle(binder, type) -> event
            2 => call.push_copy_handle(buffer_event()),
            _ => return unimplemented(call, "vi:IHOSBinderDriver"),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_buffers(count: usize) -> BufferQueue {
        let mut queue = BufferQueue::new();
        for slot in 0..count {
            queue.slots[slot] = Some(vec![0; 4]);
        }
        queue
    }

    #[test]
    fn buffer_history_reports_queued_then_shown_frames() {
        let mut queue = BufferQueue::new();
        queue.slots[0] = Some(vec![0]);
        queue.slots[1] = Some(vec![0]);
        assert_eq!(queue.history(1)[0], HistoryEntry::EMPTY);
        let slot = queue.dequeue().expect("free slot");
        queue.queue(slot, 1);
        let newest = queue.history(1)[0];
        assert_eq!((newest.frame, newest.state), (1, 2));
        assert!(queue.vsync(), "shown at the next vsync");
        let shown = queue.history(2);
        assert_eq!((shown[0].frame, shown[0].state), (1, 3));
        assert!(shown[0].shown_at >= shown[0].queued_at);
        assert_eq!(shown[1], HistoryEntry::EMPTY);
    }

    #[test]
    fn dequeue_waits_for_a_vsync_to_release_a_buffer() {
        let mut queue = with_buffers(3);
        let first = queue.dequeue().expect("a free buffer");
        queue.queue(first, 1);
        assert!(queue.vsync(), "the first frame is shown at the next vsync");
        let second = queue.dequeue().expect("a free buffer");
        let third = queue.dequeue().expect("a free buffer");
        assert_eq!(queue.dequeue(), None, "all three are in use");
        queue.queue(second, 1);
        assert!(queue.vsync());
        // Showing the second frame released the first's buffer.
        assert_eq!(queue.dequeue(), Some(first));
        queue.cancel(third);
        assert_eq!(queue.dequeue(), Some(third));
    }

    #[test]
    fn frames_stay_on_screen_for_their_swap_interval() {
        let mut queue = with_buffers(3);
        for _ in 0..2 {
            let slot = queue.dequeue().expect("a free buffer");
            queue.queue(slot, 2);
        }
        assert!(queue.vsync());
        assert!(!queue.vsync(), "a swap interval of 2 holds the frame");
        assert!(queue.vsync());
        assert!(!queue.vsync());
        assert!(!queue.vsync(), "nothing left to show");
    }

    #[test]
    fn graphic_buffers_name_their_nvmap_memory() {
        // LM3's first swapchain buffer (1920x1080, nvmap id 0xc, offset 0),
        // with a non-zero offset patched in.
        let mut words = [0u32; 0x40];
        words[..12].copy_from_slice(&[
            0x4742_4652,
            0x780,
            0x438,
            0x780,
            1,
            0xb00,
            0x2a,
            0,
            0,
            0x51,
            0xffff_ffff,
            0xc,
        ]);
        words[29] = 0xc;
        words[30] = 0x2_0000;
        words[32] = 4;
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        assert_eq!(graphic_buffer_memory(&bytes), Some((0xc, 0x2_0000)));
        assert_eq!(graphic_buffer_memory(&bytes[..0x40]), None);
    }

    #[test]
    fn dequeue_hands_out_slot_zero_before_buffers_are_set() {
        let mut queue = BufferQueue::new();
        assert_eq!(queue.dequeue(), Some(0));
        assert_eq!(queue.dequeue(), Some(0));
    }

    #[test]
    fn vsync_period_maps_picked_rates_to_refresh_periods() {
        // The title's own mode (0) is 60 Hz; its swap interval of 2 then
        // shows a frame every other vsync.
        assert_eq!(vsync_period_for(0), VSYNC);
        assert_eq!(vsync_period_for(30), VSYNC);
        assert_eq!(vsync_period_for(60), VSYNC);
        assert_eq!(vsync_period_for(120), VSYNC / 2);
        assert_eq!(vsync_period_for(180), VSYNC / 3);
        assert_eq!(vsync_period_for(240), VSYNC / 4);
        // Any other picked rate paces at its own period too.
        assert_eq!(vsync_period_for(90), VSYNC * 2 / 3);
        assert!(vsync_period_for(u32::MAX) > Duration::ZERO);
    }

    #[test]
    fn picked_frame_rates_show_each_frame_for_one_vsync() {
        // The title's own interval is kept (LM3 queues at 2 for 30 fps).
        assert_eq!(swap_interval_for(0, 2), 2);
        assert_eq!(swap_interval_for(60, 2), 1);
        assert_eq!(swap_interval_for(120, 2), 1);
        assert_eq!(swap_interval_for(180, 2), 1);
        assert_eq!(swap_interval_for(240, 2), 1);
    }
}
