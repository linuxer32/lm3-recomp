//! `hid`: controllers. Player 1 is a Pro Controller and the console is in
//! handheld mode with both Joy-Con attached; both show the host controller
//! (`set_controller`, fed by the window backend), sampled every 5 ms.
//!
//! The SDK reads controller state from shared memory (0x40000 bytes). Its
//! Npad section at 0x9A00 has one 0x5000-byte entry per controller (No1-No8,
//! then Handheld, then Other): style set at 0x0, assignment mode at 0x4,
//! colors, then one LIFO per style. A LIFO is a 0x20-byte header
//! (timestamp, capacity 17, tail index, count) and 17 entries of a
//! sampling number and a 0x28-byte state.

use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use crate::ipc::MemoryAccess;
use crate::server::{Call, Service};
use crate::services::unimplemented;

const SHARED_MEMORY_SIZE: usize = 0x40000;
const NPAD_SECTION: usize = 0x9A00;
const NPAD_ENTRY: usize = 0x5000;
const LIFO_SIZE: usize = 0x350;
/// Offset of the first (full key) state LIFO in an Npad entry; the others
/// follow: handheld, dual, left, right.
const FIRST_LIFO: usize = 0x28;
const DEVICE_TYPE: usize = 0x4188;

/// NpadStyleTag bits.
const STYLE_FULL_KEY: u32 = 1 << 0;
const STYLE_HANDHELD: u32 = 1 << 1;
/// NpadAttribute: connected and wired.
const CONNECTED_WIRED: u32 = 0b11;
/// Assignment mode: dual.
const ASSIGNMENT_DUAL: u32 = 0;

/// NpadButton bits.
pub mod button {
    pub const A: u64 = 1 << 0;
    pub const B: u64 = 1 << 1;
    pub const X: u64 = 1 << 2;
    pub const Y: u64 = 1 << 3;
    pub const STICK_L: u64 = 1 << 4;
    pub const STICK_R: u64 = 1 << 5;
    pub const L: u64 = 1 << 6;
    pub const R: u64 = 1 << 7;
    pub const ZL: u64 = 1 << 8;
    pub const ZR: u64 = 1 << 9;
    pub const PLUS: u64 = 1 << 10;
    pub const MINUS: u64 = 1 << 11;
    pub const LEFT: u64 = 1 << 12;
    pub const UP: u64 = 1 << 13;
    pub const RIGHT: u64 = 1 << 14;
    pub const DOWN: u64 = 1 << 15;
}

/// The host controller as the guest sees it: NpadButton bits and stick
/// positions (-0x7FFF..=0x7FFF, y up).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ControllerState {
    pub buttons: u64,
    pub left: (i32, i32),
    pub right: (i32, i32),
}

impl ControllerState {
    /// The buttons with the stick-direction bits (left stick from bit 16,
    /// right from bit 20: left, up, right, down) for sticks past half way.
    fn buttons_with_sticks(&self) -> u64 {
        const HALF: i32 = 0x4000;
        let directions = |(x, y): (i32, i32)| {
            u64::from(x < -HALF)
                | (u64::from(y > HALF) << 1)
                | (u64::from(x > HALF) << 2)
                | (u64::from(y < -HALF) << 3)
        };
        self.buttons | (directions(self.left) << 16) | (directions(self.right) << 20)
    }
}

static CONTROLLER: Mutex<ControllerState> = Mutex::new(ControllerState {
    buttons: 0,
    left: (0, 0),
    right: (0, 0),
});

/// Show `state` to the guest from the next sample on.
pub fn set_controller(state: ControllerState) {
    *CONTROLLER
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = state;
}

/// One scripted input: buttons, or a stick pushed all the way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scripted {
    Buttons(u64),
    Stick(i32, i32),
    RightStick(i32, i32),
}

/// `RECOMP_HID_PRESS=<input>@<seconds>[+<held seconds>],...`: also hold
/// each input that many seconds after start, for 150 ms or the time given
/// (a scripted controller for testing). Inputs: a, b, x, y, l, r, zl, zr,
/// plus, minus, up, down, left, right, the left stick (stick_left,
/// stick_up, stick_right, stick_down), which walks a character, and the
/// right stick (rstick_left, ...), which turns the camera.
fn scripted_presses() -> &'static [(Scripted, Duration, Duration)] {
    static PRESSES: OnceLock<Vec<(Scripted, Duration, Duration)>> = OnceLock::new();
    PRESSES.get_or_init(|| parse_presses(&std::env::var("RECOMP_HID_PRESS").unwrap_or_default()))
}

fn parse_presses(text: &str) -> Vec<(Scripted, Duration, Duration)> {
    let names = [
        ("a", Scripted::Buttons(button::A)),
        ("b", Scripted::Buttons(button::B)),
        ("x", Scripted::Buttons(button::X)),
        ("y", Scripted::Buttons(button::Y)),
        ("l", Scripted::Buttons(button::L)),
        ("r", Scripted::Buttons(button::R)),
        ("zl", Scripted::Buttons(button::ZL)),
        ("zr", Scripted::Buttons(button::ZR)),
        ("plus", Scripted::Buttons(button::PLUS)),
        ("minus", Scripted::Buttons(button::MINUS)),
        ("up", Scripted::Buttons(button::UP)),
        ("down", Scripted::Buttons(button::DOWN)),
        ("left", Scripted::Buttons(button::LEFT)),
        ("right", Scripted::Buttons(button::RIGHT)),
        ("stick_left", Scripted::Stick(-1, 0)),
        ("stick_up", Scripted::Stick(0, 1)),
        ("stick_right", Scripted::Stick(1, 0)),
        ("stick_down", Scripted::Stick(0, -1)),
        ("rstick_left", Scripted::RightStick(-1, 0)),
        ("rstick_up", Scripted::RightStick(0, 1)),
        ("rstick_right", Scripted::RightStick(1, 0)),
        ("rstick_down", Scripted::RightStick(0, -1)),
    ];
    let seconds = |text: &str| Some(Duration::from_secs_f64(text.parse().ok()?));
    text.split(',')
        .filter_map(|press| {
            let (name, when) = press.trim().split_once('@')?;
            let input = names.iter().find(|(known, _)| *known == name)?.1;
            let (start, hold) = match when.split_once('+') {
                Some((start, hold)) => (seconds(start)?, seconds(hold)?),
                None => (seconds(when)?, Duration::from_millis(150)),
            };
            Some((input, start, hold))
        })
        .collect()
}

/// Add the scripted inputs held at `now` (time since the clock started).
fn apply_scripted(state: &mut ControllerState, now: Duration) {
    for &(input, at, hold) in scripted_presses() {
        if now >= at && now < at + hold {
            match input {
                Scripted::Buttons(bits) => state.buttons |= bits,
                Scripted::Stick(x, y) => state.left = (x * 0x7FFF, y * 0x7FFF),
                Scripted::RightStick(x, y) => state.right = (x * 0x7FFF, y * 0x7FFF),
            }
        }
    }
}

/// How often controllers are sampled into shared memory.
const SAMPLE_PERIOD: Duration = Duration::from_millis(5);

/// Start sampling the host controller into the shared memory's Npad LIFOs
/// (player 1's full key state and the handheld state). The runtime calls
/// this with the process-wide memory view before the guest runs.
pub fn attach_memory(memory: Arc<dyn MemoryAccess>) {
    static STARTED: OnceLock<()> = OnceLock::new();
    STARTED.get_or_init(|| {
        let _ = std::thread::Builder::new()
            .name("recomp-hid".to_string())
            .spawn(move || {
                let handle = shared_memory();
                let mut lifos = [Lifo::new(0, 0), Lifo::new(0x20, 1)];
                loop {
                    std::thread::sleep(SAMPLE_PERIOD);
                    let mut state = *CONTROLLER
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    let now = Duration::from_nanos(crate::clock::now_ns());
                    apply_scripted(&mut state, now);
                    for lifo in &mut lifos {
                        lifo.push(&*memory, handle, &state);
                    }
                }
            });
    });
}

/// One Npad state LIFO the sampler appends to.
struct Lifo {
    /// Offset of its header in shared memory.
    header: usize,
    /// The newest entry's index, how many entries are valid, and the next
    /// sampling number.
    tail: u64,
    count: u64,
    sampling: u64,
}

impl Lifo {
    /// The LIFO `lifo` (index after the full key one) of controller `id`, as
    /// `write_npad` left it: one entry, sampling number 1.
    fn new(id: u32, lifo: usize) -> Self {
        Lifo {
            header: NPAD_SECTION + npad_index(id) * NPAD_ENTRY + FIRST_LIFO + lifo * LIFO_SIZE,
            tail: 0,
            count: 1,
            sampling: 2,
        }
    }

    /// Write the next entry, then publish it in the header (the SDK reads
    /// the tail index, then the entry, checking its two sampling numbers).
    fn push(&mut self, memory: &dyn MemoryAccess, handle: u32, state: &ControllerState) {
        let tail = (self.tail + 1) % 17;
        let mut entry = [0u8; 0x30];
        entry[0..8].copy_from_slice(&self.sampling.to_le_bytes());
        entry[8..16].copy_from_slice(&self.sampling.to_le_bytes());
        entry[16..24].copy_from_slice(&state.buttons_with_sticks().to_le_bytes());
        for (at, value) in [
            (24, state.left.0),
            (28, state.left.1),
            (32, state.right.0),
            (36, state.right.1),
        ] {
            entry[at..at + 4].copy_from_slice(&value.to_le_bytes());
        }
        entry[40..44].copy_from_slice(&CONNECTED_WIRED.to_le_bytes());
        let kernel = crate::kernel::kernel();
        let at = self.header + 0x20 + tail as usize * 0x30;
        kernel.write_shared_memory(memory, handle, at, &entry);
        self.count = (self.count + 1).min(17);
        let mut indices = [0u8; 16];
        indices[0..8].copy_from_slice(&tail.to_le_bytes());
        indices[8..16].copy_from_slice(&self.count.to_le_bytes());
        kernel.write_shared_memory(memory, handle, self.header + 16, &indices);
        self.tail = tail;
        self.sampling += 1;
    }
}

/// Npad id to shared-memory entry: No1-No8 are 0-7, Handheld (0x20) is 8,
/// Other (0x10) is 9.
fn npad_index(id: u32) -> usize {
    match id {
        0x20 => 8,
        0x10 => 9,
        id => id as usize & 7,
    }
}

/// One controller entry: `style`, the LIFO `lifo` (index after the full
/// key one) holding a single connected state, and the device type bits.
fn write_npad(memory: &mut [u8], id: u32, style: u32, lifo: usize, device_type: u32) {
    let entry = NPAD_SECTION + npad_index(id) * NPAD_ENTRY;
    let put = |memory: &mut [u8], at: usize, bytes: &[u8]| {
        memory[entry + at..entry + at + bytes.len()].copy_from_slice(bytes);
    };
    put(memory, 0x0, &style.to_le_bytes());
    put(memory, 0x4, &ASSIGNMENT_DUAL.to_le_bytes());
    // Color attribute "Ok" (0) with a dark body and light buttons.
    put(memory, 0xC, &0xFF32_3232u32.to_le_bytes());
    put(memory, 0x10, &0xFFE6_E6E6u32.to_le_bytes());
    let header = FIRST_LIFO + lifo * LIFO_SIZE;
    // Header: timestamp 1, capacity 17, tail 0, one entry.
    put(memory, header, &1u64.to_le_bytes());
    put(memory, header + 8, &17u64.to_le_bytes());
    put(memory, header + 24, &1u64.to_le_bytes());
    // Entry 0: sampling number 1 (twice: storage and state), no buttons,
    // centred sticks, connected.
    let state = header + 0x20;
    put(memory, state, &1u64.to_le_bytes());
    put(memory, state + 8, &1u64.to_le_bytes());
    put(memory, state + 8 + 0x20, &CONNECTED_WIRED.to_le_bytes());
    put(memory, DEVICE_TYPE, &device_type.to_le_bytes());
}

fn shared_memory() -> u32 {
    static HANDLE: OnceLock<u32> = OnceLock::new();
    *HANDLE.get_or_init(|| {
        let mut memory = vec![0u8; SHARED_MEMORY_SIZE];
        // Player 1: Pro Controller (device type FullKey).
        write_npad(&mut memory, 0, STYLE_FULL_KEY, 0, 1 << 0);
        // Handheld: both Joy-Con attached (HandheldLeft | HandheldRight).
        write_npad(&mut memory, 0x20, STYLE_HANDHELD, 1, (1 << 2) | (1 << 3));
        crate::kernel::kernel().create_shared_memory(memory)
    })
}

/// `hid` (IHidServer)
pub struct HidService {
    supported_styles: u32,
    hold_type: u64,
    handheld_activation: u64,
}

impl HidService {
    pub fn new() -> Self {
        HidService {
            supported_styles: STYLE_FULL_KEY | STYLE_HANDHELD,
            hold_type: 0,
            handheld_activation: 0,
        }
    }
}

impl Default for HidService {
    fn default() -> Self {
        Self::new()
    }
}

impl Service for HidService {
    fn name(&self) -> String {
        "hid".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // CreateAppletResource(aruid)
            0 => call.push_object(Box::new(AppletResource)),
            // Activate* (debug pad, touch, mouse, keyboard, gesture, Npad),
            // keyboard lock keys, six-axis sensor start/stop and fusion
            // settings, Npad assignment and capture-button settings,
            // vibration permission and sessions, console six-axis sensors
            1
            | 11
            | 21
            | 31
            | 32
            | 66
            | 67
            | 69
            | 70
            | 72
            | 79
            | 91
            | 103
            | 104
            | 107
            | 109
            | 122..=127
            | 130
            | 132..=136
            | 201
            | 204
            | 206
            | 207
            | 209
            | 210
            | 300..=310
            | 1000
            | 1002 => {}
            // IsSixAxisSensorFusionEnabled, IsSixAxisSensorAtRest
            68 | 82 => call.push_u8(u8::from(call.command() == 82)),
            // GetGyroscopeZeroDriftMode
            74 => call.push_u32(1),
            // SetSupportedNpadStyleSet(styles, aruid)
            100 => self.supported_styles = call.arg_u32(0),
            101 => call.push_u32(self.supported_styles),
            // SetSupportedNpadIdType(aruid, ids)
            102 => {}
            // AcquireNpadStyleSetUpdateEventHandle(aruid, id, unknown)
            106 => {
                let (write, read) = crate::kernel::kernel().create_event();
                crate::kernel::kernel().signal_event(write);
                call.push_copy_handle(read);
            }
            // GetPlayerLedPattern(id)
            108 => call.push_u64(1 << call.arg_u32(0).min(3)),
            // SetNpadJoyHoldType(aruid, type), GetNpadJoyHoldType
            120 => self.hold_type = call.arg_u64(8),
            121 => call.push_u64(self.hold_type),
            // SetNpadHandheldActivationMode(aruid, mode),
            // GetNpadHandheldActivationMode
            128 => self.handheld_activation = call.arg_u64(8),
            129 => call.push_u64(self.handheld_activation),
            // IsUnintendedHomeButtonInputProtectionEnabled,
            // IsVibrationPermitted, IsVibrationDeviceMounted
            131 | 205 | 211 => call.push_u8(1),
            // GetVibrationDeviceInfo(handle): linear resonant actuator,
            // position by Npad side.
            200 => {
                call.push_u32(1);
                call.push_u32(0);
            }
            // GetActualVibrationValue: still.
            202 => call.push_bytes(&[0; 16]),
            // CreateActiveVibrationDeviceList
            203 => call.push_object(Box::new(VibrationDeviceList)),
            // GetActualVibrationGcErmCommand
            208 => call.push_u64(0),
            // GetNpadCommunicationMode
            1001 => call.push_u64(0),
            // IsFirmwareUpdateNeededForNotification
            1003 => call.push_u8(0),
            _ => return unimplemented(call, "hid"),
        }
        Ok(())
    }
}

/// IAppletResource
struct AppletResource;

impl Service for AppletResource {
    fn name(&self) -> String {
        "hid:IAppletResource".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // GetSharedMemoryHandle
            0 => {
                call.push_copy_handle(shared_memory());
                Ok(())
            }
            _ => unimplemented(call, "hid:IAppletResource"),
        }
    }
}

/// IActiveVibrationDeviceList
struct VibrationDeviceList;

impl Service for VibrationDeviceList {
    fn name(&self) -> String {
        "hid:IActiveVibrationDeviceList".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // ActivateVibrationDevice(handle)
            0 => Ok(()),
            _ => unimplemented(call, "hid:IActiveVibrationDeviceList"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scripted_presses_hold_buttons_and_the_stick() {
        let presses = parse_presses("a@20, stick_up@70+1.5,zr@3.5,bogus@1,b@x");
        assert_eq!(
            presses,
            [
                (
                    Scripted::Buttons(button::A),
                    Duration::from_secs(20),
                    Duration::from_millis(150)
                ),
                (
                    Scripted::Stick(0, 1),
                    Duration::from_secs(70),
                    Duration::from_millis(1500)
                ),
                (
                    Scripted::Buttons(button::ZR),
                    Duration::from_millis(3500),
                    Duration::from_millis(150)
                ),
            ]
        );
    }

    /// No guest mappings: shared memory writes land in its stored bytes.
    #[derive(Debug)]
    struct NoGuest;

    impl MemoryAccess for NoGuest {
        fn load_u8(&self, _addr: u64) -> Option<u8> {
            None
        }
        fn load_u32(&self, _addr: u64) -> Option<u32> {
            None
        }
        fn load_u64(&self, _addr: u64) -> Option<u64> {
            None
        }
        fn store_u32(&self, _addr: u64, _value: u32) -> bool {
            false
        }
        fn store_u64(&self, _addr: u64, _value: u64) -> bool {
            false
        }
        fn read_bytes(&self, _addr: u64, _len: usize) -> Option<Vec<u8>> {
            None
        }
        fn write_bytes(&self, _addr: u64, _bytes: &[u8]) -> bool {
            false
        }
    }

    fn u64_at(bytes: &[u8], at: usize) -> u64 {
        u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
    }

    #[test]
    fn samples_append_to_the_handheld_lifo() {
        let kernel = crate::kernel::kernel();
        let handle = kernel.create_shared_memory(vec![0; SHARED_MEMORY_SIZE]);
        let mut lifo = Lifo::new(0x20, 1);
        let state = ControllerState {
            buttons: button::A | button::PLUS,
            left: (-0x7FFF, 0),
            right: (0, 0x7FFF),
        };
        for _ in 0..20 {
            lifo.push(&NoGuest, handle, &state);
        }
        let memory = kernel.map_shared_memory(handle, 0).expect("shared memory");
        let header = NPAD_SECTION + 8 * NPAD_ENTRY + FIRST_LIFO + LIFO_SIZE;
        // 20 samples after the initial one: the tail wrapped to 20 % 17,
        // the LIFO is full.
        assert_eq!(u64_at(&memory, header + 16), 3);
        assert_eq!(u64_at(&memory, header + 24), 17);
        let entry = header + 0x20 + 3 * 0x30;
        assert_eq!(u64_at(&memory, entry), 21);
        assert_eq!(u64_at(&memory, entry + 8), 21);
        // A, Plus, left stick left (bit 16) and right stick up (bit 21).
        assert_eq!(
            u64_at(&memory, entry + 16),
            button::A | button::PLUS | (1 << 16) | (1 << 21)
        );
        assert_eq!(&memory[entry + 24..entry + 28], &(-0x7FFFi32).to_le_bytes());
        assert_eq!(
            &memory[entry + 40..entry + 44],
            &CONNECTED_WIRED.to_le_bytes()
        );
    }
}
