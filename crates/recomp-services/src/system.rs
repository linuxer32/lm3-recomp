//! System services: time, settings, power state and multimedia clocks.

use std::sync::OnceLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::server::{Call, Service};
use crate::services::{new_event, unimplemented};

/// Whether the console reports itself docked (`RECOMP_DOCKED=1`, or a
/// script's `setting("docked", 1)`): titles then render for a 1920x1080 TV
/// instead of the 1280x720 handheld screen.
pub fn docked() -> bool {
    match DOCKED.load(std::sync::atomic::Ordering::Acquire) {
        UNREAD => {
            let docked =
                std::env::var("RECOMP_DOCKED").is_ok_and(|value| value != "0" && !value.is_empty());
            DOCKED.store(u8::from(docked), std::sync::atomic::Ordering::Release);
            docked
        }
        docked => docked == 1,
    }
}

/// Dock or undock the console while the title runs: it is told as a
/// docked Switch tells it (operation and performance mode messages), asks
/// again and renders for the new screen.
pub fn set_docked(dock: bool) {
    if docked() != dock {
        DOCKED.store(u8::from(dock), std::sync::atomic::Ordering::Release);
        crate::am::post_message(crate::am::MESSAGE_OPERATION_MODE_CHANGED);
        crate::am::post_message(crate::am::MESSAGE_PERFORMANCE_MODE_CHANGED);
    }
}

/// `docked()`'s state: 0 or 1, or `UNREAD` before `RECOMP_DOCKED` is read.
const UNREAD: u8 = 2;
static DOCKED: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(UNREAD);

/// Source id shared by the steady clock and every system clock context,
/// which marks the contexts as valid for the steady clock.
const CLOCK_SOURCE: [u8; 16] = *b"recomp-steadyclk";

fn steady_seconds() -> i64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs() as i64
}

fn posix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs() as i64)
}

/// SteadyClockTimePoint: seconds and the clock source id (0x18 bytes).
fn steady_time_point() -> Vec<u8> {
    let mut bytes = steady_seconds().to_le_bytes().to_vec();
    bytes.extend_from_slice(&CLOCK_SOURCE);
    bytes
}

/// SystemClockContext: the offset from the steady clock to posix time and
/// the steady time point it was taken at (0x20 bytes).
fn system_clock_context() -> Vec<u8> {
    let mut bytes = (posix_now() - steady_seconds()).to_le_bytes().to_vec();
    bytes.extend_from_slice(&steady_time_point());
    bytes
}

/// The time service's shared memory, which the SDK reads lock-free: the
/// steady clock context at 0x00, the local and network system clock
/// contexts at 0x38 and 0x80, automatic correction at 0xC8. Each holds a
/// counter and two copies; the counter (0) selects the first.
fn time_shared_memory() -> u32 {
    static HANDLE: OnceLock<u32> = OnceLock::new();
    *HANDLE.get_or_init(|| {
        let mut bytes = vec![0u8; 0x1000];
        // Steady clock: internal offset 0 ns, so its time points count from
        // the guest's tick counter, which starts with the process.
        for copy in 0..2 {
            let at = 0x8 + copy * 0x18;
            bytes[at + 8..at + 24].copy_from_slice(&CLOCK_SOURCE);
        }
        let context = system_clock_context();
        for base in [0x38, 0x80] {
            for copy in 0..2 {
                let at = base + 0x8 + copy * 0x20;
                bytes[at..at + 0x20].copy_from_slice(&context);
            }
        }
        crate::kernel::kernel().create_shared_memory(bytes)
    })
}

/// `time:u`, `time:a`, `time:s` (IStaticService)
pub struct TimeService;

impl Service for TimeService {
    fn name(&self) -> String {
        "time".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // GetStandardUserSystemClock, GetStandardNetworkSystemClock,
            // GetStandardLocalSystemClock, GetEphemeralNetworkSystemClock
            0 | 1 | 4 | 5 => call.push_object(Box::new(SystemClock)),
            2 => call.push_object(Box::new(SteadyClock)),
            3 => call.push_object(Box::new(TimeZoneService)),
            // GetSharedMemoryNativeHandle
            20 => call.push_copy_handle(time_shared_memory()),
            // SetStandardSteadyClockInternalOffset,
            // SetStandardUserSystemClockAutomaticCorrectionEnabled
            50 | 101 => {}
            // IsStandardUserSystemClockAutomaticCorrectionEnabled,
            // IsStandardNetworkSystemClockAccuracySufficient
            100 | 200 => call.push_u8(0),
            // CalculateMonotonicSystemClockBaseTimePoint(context)
            300 => call.push_u64((posix_now() - steady_seconds()) as u64),
            _ => return unimplemented(call, "time"),
        }
        Ok(())
    }
}

/// ISystemClock
struct SystemClock;

impl Service for SystemClock {
    fn name(&self) -> String {
        "time:ISystemClock".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // GetCurrentTime
            0 => call.push_u64(posix_now() as u64),
            // SetCurrentTime, SetSystemClockContext
            1 | 3 => {}
            // GetSystemClockContext
            2 => call.push_bytes(&system_clock_context()),
            // GetOperationEventReadableHandle
            4 => call.push_copy_handle(new_event()),
            _ => return unimplemented(call, "time:ISystemClock"),
        }
        Ok(())
    }
}

/// ISteadyClock
struct SteadyClock;

impl Service for SteadyClock {
    fn name(&self) -> String {
        "time:ISteadyClock".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // GetCurrentTimePoint
            0 => call.push_bytes(&steady_time_point()),
            // GetTestOffset, GetInternalOffset
            2 | 200 => call.push_u64(0),
            // SetTestOffset, SetInternalOffset
            3 | 201 => {}
            // GetRtcValue
            100 => call.push_u64(posix_now() as u64),
            // IsRtcResetDetected
            101 => call.push_u8(0),
            // GetSetupResultValue
            102 => call.push_u32(0),
            _ => return unimplemented(call, "time:ISteadyClock"),
        }
        Ok(())
    }
}

/// Days since 1970-01-01 to (year, month, day) in the proleptic Gregorian
/// calendar.
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(month <= 2), month, day)
}

fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = year.div_euclid(400);
    let yoe = year - era * 400;
    let doy = (153 * (if month > 2 { month - 3 } else { month + 9 }) + 2) / 5 + day - 1;
    era * 146_097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719_468
}

/// CalendarTime (8 bytes) and CalendarAdditionalInfo (0x18 bytes) for a
/// posix time, in UTC.
fn calendar(posix: i64) -> Vec<u8> {
    let days = posix.div_euclid(86_400);
    let seconds = posix.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let mut bytes = (year as i16).to_le_bytes().to_vec();
    bytes.extend_from_slice(&[
        month as u8,
        day as u8,
        (seconds / 3600) as u8,
        (seconds / 60 % 60) as u8,
        (seconds % 60) as u8,
        0,
    ]);
    bytes.extend_from_slice(&((days + 4).rem_euclid(7) as u32).to_le_bytes());
    bytes.extend_from_slice(&((days - days_from_civil(year, 1, 1)) as u32).to_le_bytes());
    bytes.extend_from_slice(b"UTC\0\0\0\0\0");
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&0i32.to_le_bytes());
    bytes
}

/// ITimeZoneService: every location is UTC.
struct TimeZoneService;

impl Service for TimeZoneService {
    fn name(&self) -> String {
        "time:ITimeZoneService".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        let mut location = [0u8; 0x24];
        location[..3].copy_from_slice(b"UTC");
        match call.command() {
            // GetDeviceLocationName
            0 => call.push_bytes(&location),
            // SetDeviceLocationName
            1 => {}
            // GetTotalLocationNameCount
            2 => call.push_u32(1),
            // LoadLocationNameList -> (names, count)
            3 => {
                call.write_out(0, &location);
                call.push_u32(1);
            }
            // LoadTimeZoneRule: the rule is opaque to the SDK; ours is empty.
            4 => {
                let size = call.out_buffer(0).map_or(0, |buffer| buffer.size);
                call.write_out(0, &vec![0; size as usize]);
            }
            // GetTimeZoneRuleVersion
            5 => call.push_bytes(b"2018e\0\0\0\0\0\0\0\0\0\0\0"),
            // ToCalendarTime(posix, rule), ToCalendarTimeWithMyRule(posix)
            100 | 101 => {
                let posix = call.arg_u64(0) as i64;
                call.push_bytes(&calendar(posix));
            }
            // ToPosixTime(calendar, rule), ToPosixTimeWithMyRule(calendar)
            // -> (times, count)
            201 | 202 => {
                let raw = call.arg_bytes(0, 8);
                let year = i64::from(i16::from_le_bytes([raw[0], raw[1]]));
                let days = days_from_civil(year, i64::from(raw[2]), i64::from(raw[3]));
                let posix = days * 86_400
                    + i64::from(raw[4]) * 3600
                    + i64::from(raw[5]) * 60
                    + i64::from(raw[6]);
                call.write_out(0, &posix.to_le_bytes());
                call.push_u32(1);
            }
            _ => return unimplemented(call, "time:ITimeZoneService"),
        }
        Ok(())
    }
}

/// Language codes in `nn::settings::Language` order.
const LANGUAGES: [&[u8]; 18] = [
    b"ja", b"en-US", b"fr", b"de", b"it", b"es", b"zh-CN", b"ko", b"nl", b"pt", b"ru", b"zh-TW",
    b"en-GB", b"fr-CA", b"es-419", b"zh-Hans", b"zh-Hant", b"pt-BR",
];

fn language_code(language: &[u8]) -> u64 {
    let mut bytes = [0u8; 8];
    bytes[..language.len()].copy_from_slice(language);
    u64::from_le_bytes(bytes)
}

/// `set` (ISettingsServer): American English, region USA.
pub struct SettingsService;

impl Service for SettingsService {
    fn name(&self) -> String {
        "set".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // GetLanguageCode
            0 => call.push_u64(language_code(b"en-US")),
            // GetAvailableLanguageCodes (first 15), ...Codes2 -> (codes, count)
            1 | 5 => {
                let count = if call.command() == 1 {
                    15
                } else {
                    LANGUAGES.len()
                };
                let codes: Vec<u8> = LANGUAGES[..count]
                    .iter()
                    .flat_map(|language| language_code(language).to_le_bytes())
                    .collect();
                let written = call.write_out(0, &codes);
                call.push_u32((written / 8) as u32);
            }
            // MakeLanguageCode(language)
            2 => {
                let index = call.arg_u32(0) as usize;
                call.push_u64(LANGUAGES.get(index).map_or(0, |code| language_code(code)));
            }
            // GetAvailableLanguageCodeCount, ...Count2
            3 => call.push_u32(15),
            6 => call.push_u32(LANGUAGES.len() as u32),
            // GetRegionCode: 1 = USA.
            4 => call.push_u32(1),
            // GetQuestFlag
            8 => call.push_u8(0),
            _ => return unimplemented(call, "set"),
        }
        Ok(())
    }
}

/// `set:sys` (ISystemSettingsServer): firmware version and display
/// settings.
pub struct SystemSettingsService;

impl Service for SystemSettingsService {
    fn name(&self) -> String {
        "set:sys".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // GetFirmwareVersion, GetFirmwareVersion2: 16.0.3.
            3 | 4 => {
                let mut version = vec![0u8; 0x100];
                version[..3].copy_from_slice(&[16, 0, 3]);
                version[4..6].copy_from_slice(&[1, 0]);
                version[8..10].copy_from_slice(b"NX");
                version[0x68..0x6E].copy_from_slice(b"16.0.3");
                let title = b"NintendoSDK Firmware for NX 16.0.3-1.0";
                version[0x80..0x80 + title.len()].copy_from_slice(title);
                call.write_out(0, &version);
            }
            // GetColorSetId: basic white.
            23 => call.push_u32(0),
            _ => return unimplemented(call, "set:sys"),
        }
        Ok(())
    }
}

/// `psm` (IPsmServer): a full battery on a sufficient charger.
pub struct PowerStateService;

impl Service for PowerStateService {
    fn name(&self) -> String {
        "psm".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // GetBatteryChargePercentage
            0 => call.push_u32(100),
            // GetChargerType: EnoughPower.
            1 => call.push_u32(1),
            // Enable/DisableBatteryCharging, Acquire/ReleaseControllerPowerSupply,
            // charge emulation and fast charging toggles
            2 | 3 | 5 | 6 | 8..=11 => {}
            // IsBatteryChargingEnabled, IsEnoughPowerSupplied
            4 | 14 => call.push_u8(1),
            7 => call.push_object(Box::new(PowerStateSession)),
            // GetBatteryVoltageState: Good.
            12 => call.push_u32(3),
            // GetRawBatteryChargePercentage, GetBatteryAgePercentage
            13 | 15 => call.push_u64(100f64.to_bits()),
            _ => return unimplemented(call, "psm"),
        }
        Ok(())
    }
}

/// `psm` IPsmSession
struct PowerStateSession;

impl Service for PowerStateSession {
    fn name(&self) -> String {
        "psm:IPsmSession".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // BindStateChangeEvent
            0 => call.push_copy_handle(new_event()),
            // UnbindStateChangeEvent and the change-event toggles
            1..=4 => {}
            _ => return unimplemented(call, "psm:IPsmSession"),
        }
        Ok(())
    }
}

/// `mm:u` (IRequest): multimedia clock requests, all granted.
pub struct MultimediaService;

impl Service for MultimediaService {
    fn name(&self) -> String {
        "mm:u".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // InitializeOld, FinalizeOld, SetAndWaitOld, Finalize, SetAndWait
            0 | 1 | 2 | 5 | 6 => {}
            // GetOld, Get
            3 | 7 => call.push_u32(0),
            // Initialize -> request id
            4 => call.push_u32(1),
            _ => return unimplemented(call, "mm:u"),
        }
        Ok(())
    }
}
