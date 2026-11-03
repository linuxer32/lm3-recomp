//! Horizon services the guest reaches through `sm:`.

use crate::server::{Call, Service, RESULT_NOT_REGISTERED, RESULT_UNKNOWN_COMMAND};

/// A service by its `sm:` name, if it is implemented.
pub fn open(name: &str) -> Option<Box<dyn Service>> {
    match name {
        "lm" => Some(Box::new(LogService)),
        "apm" | "apm:p" => Some(Box::new(PerformanceManager)),
        "appletOE" => Some(Box::new(crate::am::ApplicationProxyService)),
        "aoc:u" => Some(Box::new(AddOnContentManager)),
        "fsp-srv" => Some(Box::new(crate::fs::FileSystemProxy)),
        "acc:u0" | "acc:u1" => Some(Box::new(crate::account::AccountService)),
        "time:u" | "time:a" | "time:s" => Some(Box::new(crate::system::TimeService)),
        "set" => Some(Box::new(crate::system::SettingsService)),
        "set:sys" => Some(Box::new(crate::system::SystemSettingsService)),
        "psm" => Some(Box::new(crate::system::PowerStateService)),
        "hid" => Some(Box::new(crate::hid::HidService::new())),
        "nifm:u" | "nifm:a" | "nifm:s" => Some(Box::new(crate::network::NetworkService)),
        "bsd:u" | "bsd:s" => Some(Box::new(crate::online::SocketService::new())),
        "sfdnsres" => Some(Box::new(crate::online::ResolverService)),
        "ssl" => Some(Box::new(crate::online::SslService)),
        "friend:u" | "friend:a" | "friend:m" | "friend:s" | "friend:v" => {
            Some(Box::new(crate::online::FriendServiceCreator))
        }
        "prepo:u" | "prepo:s" | "prepo:a" | "prepo:m" => {
            Some(Box::new(crate::online::PlayReportService))
        }
        "bcat:u" => Some(Box::new(crate::online::BcatServiceCreator)),
        "vi:u" | "vi:s" | "vi:m" => Some(Box::new(crate::vi::ViRootService)),
        "audren:u" => Some(Box::new(crate::audio::AudioRendererManager)),
        "audout:u" => Some(Box::new(crate::audio::AudioOutManager)),
        "hwopus" => Some(Box::new(crate::opus::HardwareOpusDecoderManager)),
        "nvdrv" | "nvdrv:a" | "nvdrv:s" | "nvdrv:t" => {
            Some(Box::new(crate::nvdrv::NvDrvService::new()))
        }
        "mm:u" => Some(Box::new(crate::system::MultimediaService)),
        "pctl" | "pctl:s" | "pctl:r" | "pctl:a" => Some(Box::new(ParentalControlFactory)),
        _ => None,
    }
}

/// The readable side of a new kernel event, for services that hand out
/// events nothing signals yet.
pub fn new_event() -> u32 {
    crate::kernel::kernel().create_event().1
}

/// The readable side of a new kernel event that is already signalled.
pub fn signalled_event() -> u32 {
    let (write, read) = crate::kernel::kernel().create_event();
    crate::kernel::kernel().signal_event(write);
    read
}

pub fn unimplemented(call: &Call<'_>, service: &str) -> Result<(), u32> {
    println!(
        "[recomp-ipc] UNIMPLEMENTED {service} command {}",
        call.command()
    );
    Err(RESULT_UNKNOWN_COMMAND)
}

/// `sm:` (IUserInterface): the service manager.
pub struct ServiceManager;

impl Service for ServiceManager {
    fn name(&self) -> String {
        "sm:".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // RegisterClient(pid), RegisterService, UnregisterService,
            // DetachClient: nothing to track for a single process.
            0 | 2 | 3 | 4 => Ok(()),
            // GetServiceHandle(name: u64, NUL padded) -> session handle.
            1 => {
                let raw = call.arg_bytes(0, 8);
                let name: String = raw
                    .iter()
                    .take_while(|byte| **byte != 0)
                    .map(|byte| char::from(*byte))
                    .collect();
                match open(&name) {
                    Some(service) => {
                        println!("[recomp-ipc] sm: GetServiceHandle(\"{name}\")");
                        call.push_session(service);
                        Ok(())
                    }
                    None => {
                        println!(
                            "[recomp-ipc] UNIMPLEMENTED service \"{name}\" (sm: GetServiceHandle)"
                        );
                        Err(RESULT_NOT_REGISTERED)
                    }
                }
            }
            _ => unimplemented(call, "sm:"),
        }
    }
}

/// `lm` (ILogService): the system log.
struct LogService;

impl Service for LogService {
    fn name(&self) -> String {
        "lm".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // OpenLogger(pid) -> ILogger
            0 => {
                call.push_object(Box::new(Logger));
                Ok(())
            }
            _ => unimplemented(call, "lm"),
        }
    }
}

/// `lm` ILogger: prints the text fields of each log packet.
struct Logger;

impl Service for Logger {
    fn name(&self) -> String {
        "lm:ILogger".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // Log(packet)
            0 => {
                let text = printable_runs(&call.in_buffer(0));
                if !text.is_empty() {
                    println!("[guest log] {text}");
                }
                Ok(())
            }
            // SetDestination(u32)
            1 => Ok(()),
            _ => unimplemented(call, "lm:ILogger"),
        }
    }
}

/// The readable text in a binary log packet: runs of at least 4 printable
/// characters, joined.
fn printable_runs(bytes: &[u8]) -> String {
    let mut runs = Vec::new();
    let mut current = String::new();
    for &byte in bytes {
        if byte == b'\n' || byte == b'\t' || (0x20..0x7F).contains(&byte) {
            current.push(char::from(byte));
        } else {
            if current.trim().len() >= 4 {
                runs.push(current.trim().to_string());
            }
            current.clear();
        }
    }
    if current.trim().len() >= 4 {
        runs.push(current.trim().to_string());
    }
    runs.join(" | ")
}

/// `apm` (IManager): performance modes.
struct PerformanceManager;

impl Service for PerformanceManager {
    fn name(&self) -> String {
        "apm".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // OpenSession -> ISession
            0 => call.push_object(Box::new(PerformanceSession)),
            // GetPerformanceMode: normal (handheld) or boost (docked).
            1 => call.push_u32(u32::from(crate::system::docked())),
            // IsCpuOverclockEnabled
            6 => call.push_u8(0),
            _ => return unimplemented(call, "apm"),
        }
        Ok(())
    }
}

/// `apm` ISession
struct PerformanceSession;

impl Service for PerformanceSession {
    fn name(&self) -> String {
        "apm:ISession".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // SetPerformanceConfiguration(mode, config), SetCpuOverclockEnabled
            0 | 2 => {}
            // GetPerformanceConfiguration(mode): the default handheld one.
            1 => call.push_u32(0x0002_0003),
            _ => return unimplemented(call, "apm:ISession"),
        }
        Ok(())
    }
}

/// `aoc:u` (IAddOnContentManager): no add-on content installed.
struct AddOnContentManager;

impl Service for AddOnContentManager {
    fn name(&self) -> String {
        "aoc:u".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // CountAddOnContent{,ByApplicationId}, ListAddOnContent{,...}
            0..=3 => call.push_u32(0),
            // GetAddOnContentBaseId{,ByApplicationId}: program id + 0x1000.
            4 | 5 => call.push_u64(0x0100_DCA0_064A_7000),
            // PrepareAddOnContent{,ByApplicationId}
            6 | 7 => {}
            // GetAddOnContentListChangedEvent{,WithProcessId}
            8 | 10 => call.push_copy_handle(new_event()),
            _ => return unimplemented(call, "aoc:u"),
        }
        Ok(())
    }
}

/// `pctl` (IParentalControlServiceFactory)
struct ParentalControlFactory;

impl Service for ParentalControlFactory {
    fn name(&self) -> String {
        "pctl".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // CreateService{,WithoutInitialize}(pid) -> IParentalControlService
            0 | 1 => {
                call.push_object(Box::new(ParentalControl));
                Ok(())
            }
            _ => unimplemented(call, "pctl"),
        }
    }
}

/// `pctl` IParentalControlService: no restrictions configured.
struct ParentalControl;

impl Service for ParentalControl {
    fn name(&self) -> String {
        "pctl:IParentalControlService".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // Initialize and the Check*/Confirm* permission queries: allowed.
            1 | 1001..=1017 | 1064 => {}
            // IsFreeCommunicationAvailable, IsRestrictionEnabled,
            // IsPairingActive, IsRestrictedByPlayTimer,
            // ConfirmStereoVisionRestrictionConfigurable,
            // GetStereoVisionRestriction
            1018 | 1031 | 1043 | 1047 | 1061 | 1062 => call.push_u8(0),
            // GetSafetyLevel
            1032 => call.push_u32(0),
            // IsStereoVisionPermitted
            1065 => call.push_u8(1),
            _ => return unimplemented(call, "pctl:IParentalControlService"),
        }
        Ok(())
    }
}
