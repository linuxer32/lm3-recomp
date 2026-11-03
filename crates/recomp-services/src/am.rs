//! `appletOE`: the applet manager as an application sees it.

use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};

use crate::server::{Call, Service};
use crate::services::{new_event, unimplemented};

/// am::ResultNoDataInChannel (module 128, description 2).
const RESULT_NO_DATA_IN_CHANNEL: u32 = 0x480;
/// am::ResultNoMessages (module 128, description 3).
const RESULT_NO_MESSAGES: u32 = 0x680;
/// LaunchParameterKind_PreselectedUser.
const LAUNCH_PRESELECTED_USER: u32 = 2;
/// Magic of the preselected-user launch parameter.
const PRESELECTED_USER_MAGIC: u32 = 0xC794_97CA;
/// AppletMessage_FocusStateChanged.
const MESSAGE_FOCUS_STATE_CHANGED: u32 = 15;
/// AppletMessage_OperationModeChanged and _PerformanceModeChanged: sent
/// when the console is docked or undocked.
pub const MESSAGE_OPERATION_MODE_CHANGED: u32 = 30;
pub const MESSAGE_PERFORMANCE_MODE_CHANGED: u32 = 31;

/// The application's applet messages: every ICommonStateGetter reads the
/// same queue.
struct Messages {
    queue: Mutex<VecDeque<u32>>,
    /// Signalled while messages are queued (writable, readable handles).
    event: (u32, u32),
}

fn messages() -> &'static Messages {
    static MESSAGES: OnceLock<Messages> = OnceLock::new();
    MESSAGES.get_or_init(|| {
        // An application starts in focus and is told so once.
        let event = crate::kernel::kernel().create_event();
        crate::kernel::kernel().signal_event(event.0);
        Messages {
            queue: Mutex::new(VecDeque::from([MESSAGE_FOCUS_STATE_CHANGED])),
            event,
        }
    })
}

/// Queue an applet message for the application (docking, for one).
pub fn post_message(message: u32) {
    let messages = messages();
    messages
        .queue
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .push_back(message);
    crate::kernel::kernel().signal_event(messages.event.0);
}

/// FocusState_InFocus.
const FOCUS_IN_FOCUS: u8 = 1;

/// IApplicationProxyService
pub struct ApplicationProxyService;

impl Service for ApplicationProxyService {
    fn name(&self) -> String {
        "appletOE".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // OpenApplicationProxy(pid, process handle) -> IApplicationProxy
            0 => {
                call.push_object(Box::new(ApplicationProxy));
                Ok(())
            }
            _ => unimplemented(call, "appletOE"),
        }
    }
}

struct ApplicationProxy;

impl Service for ApplicationProxy {
    fn name(&self) -> String {
        "appletOE:IApplicationProxy".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        let object: Box<dyn Service> = match call.command() {
            0 => Box::new(CommonStateGetter::new()),
            1 => Box::new(SelfController),
            2 => Box::new(WindowController),
            3 => Box::new(AudioController),
            4 => Box::new(DisplayController),
            11 => Box::new(LibraryAppletCreator),
            20 => Box::new(ApplicationFunctions),
            1000 => Box::new(DebugFunctions),
            _ => return unimplemented(call, "IApplicationProxy"),
        };
        call.push_object(object);
        Ok(())
    }
}

/// ICommonStateGetter: applet messages, focus and operation mode.
struct CommonStateGetter;

impl CommonStateGetter {
    fn new() -> Self {
        messages();
        CommonStateGetter
    }
}

impl Service for CommonStateGetter {
    fn name(&self) -> String {
        "appletOE:ICommonStateGetter".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // GetEventHandle
            0 => {
                call.push_copy_handle(messages().event.1);
            }
            // ReceiveMessage
            1 => {
                let messages = messages();
                let mut queue = messages
                    .queue
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner());
                let message = queue.pop_front();
                if queue.is_empty() {
                    crate::kernel::kernel().clear_event(messages.event.0);
                }
                call.push_u32(message.ok_or(RESULT_NO_MESSAGES)?);
            }
            // GetOperationMode: handheld (0) or docked (1).
            5 => call.push_u8(u8::from(crate::system::docked())),
            // GetPerformanceMode: normal (0) or boost, as docked (1).
            6 => call.push_u32(u32::from(crate::system::docked())),
            // GetBootMode: normal.
            8 => call.push_u8(0),
            // GetCurrentFocusState
            9 => call.push_u8(FOCUS_IN_FOCUS),
            // RequestToAcquireSleepLock, SetCpuBoostMode,
            // SetRequestExitToLibraryAppletAtExecuteNextProgramEnabled
            10 | 66 | 900 => {}
            // GetAcquiredSleepLockEvent, GetDefaultDisplayResolutionChangeEvent
            13 | 61 => call.push_copy_handle(new_event()),
            // IsVrModeEnabled
            50 => call.push_u8(0),
            // GetDefaultDisplayResolution: handheld 1280x720, docked
            // 1920x1080.
            60 => {
                let (width, height) = if crate::system::docked() {
                    (1920, 1080)
                } else {
                    (1280, 720)
                };
                call.push_u32(width);
                call.push_u32(height);
            }
            // GetSettingsPlatformRegion: global.
            300 => call.push_u8(1),
            _ => return unimplemented(call, "ICommonStateGetter"),
        }
        Ok(())
    }
}

/// ISelfController: notifications and display layer setup.
struct SelfController;

impl Service for SelfController {
    fn name(&self) -> String {
        "appletOE:ISelfController".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // Exit/Lock/Unlock, fatal sections, permission and notification
            // toggles, focus handling, display requests, idle and sleep
            // settings, media and album options.
            0..=4
            | 10..=20
            | 50
            | 51
            | 60
            | 61
            | 62
            | 64
            | 65
            | 68
            | 70
            | 72
            | 80
            | 100
            | 110
            | 120 => {}
            // GetLibraryAppletLaunchableEvent: library applets can always
            // be launched (the SDK waits for this before creating one; LM3's
            // title screen asks for the controller support applet and hid
            // its frames until it returned).
            9 => call.push_copy_handle(crate::services::signalled_event()),
            // GetAccumulatedSuspendedTickChangedEvent
            91 => call.push_copy_handle(new_event()),
            // CreateManagedDisplayLayer -> layer id
            40 => call.push_u64(1),
            // IsSystemBufferSharingEnabled
            41 => {}
            // CreateManagedDisplaySeparableLayer -> two layer ids
            44 => {
                call.push_u64(1);
                call.push_u64(2);
            }
            // GetIdleTimeDetectionExtension
            63 => call.push_u32(0),
            // GetCurrentIlluminance -> f32 lux
            66 => call.push_u32(500f32.to_bits()),
            // IsIlluminanceAvailable, IsAutoSleepDisabled
            67 | 69 => call.push_u8(0),
            // GetCurrentIlluminanceEx -> (overlimit, lux)
            71 => {
                call.push_u8(0);
                call.push_u32(500f32.to_bits());
            }
            // GetAccumulatedSuspendedTickValue
            90 => call.push_u64(0),
            _ => return unimplemented(call, "ISelfController"),
        }
        Ok(())
    }
}

/// IWindowController
struct WindowController;

impl Service for WindowController {
    fn name(&self) -> String {
        "appletOE:IWindowController".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // GetAppletResourceUserId: the process id serves as the ARUID.
            1 => call.push_u64(crate::server::PROCESS_ID),
            // GetAppletResourceUserIdOfCallerApplet
            2 => call.push_u64(0),
            // AcquireForegroundRights, ReleaseForegroundRights,
            // RejectToChangeIntoBackground, SetAppletWindowVisibility,
            // SetAppletGpuTimeSlice
            10 | 11 | 12 | 20 | 21 => {}
            _ => return unimplemented(call, "IWindowController"),
        }
        Ok(())
    }
}

/// IAudioController
struct AudioController;

impl Service for AudioController {
    fn name(&self) -> String {
        "appletOE:IAudioController".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // SetExpectedMasterVolume, ChangeMainAppletMasterVolume,
            // SetTransparentVolumeRate
            0 | 3 | 4 => {}
            // Get{Main,Library}AppletExpectedMasterVolume -> 1.0
            1 | 2 => call.push_u32(1f32.to_bits()),
            _ => return unimplemented(call, "IAudioController"),
        }
        Ok(())
    }
}

/// IDisplayController
struct DisplayController;

impl Service for DisplayController {
    fn name(&self) -> String {
        "appletOE:IDisplayController".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        unimplemented(call, "IDisplayController")
    }
}

/// ILibraryAppletCreator
struct LibraryAppletCreator;

impl Service for LibraryAppletCreator {
    fn name(&self) -> String {
        "appletOE:ILibraryAppletCreator".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // CreateLibraryApplet(applet id, mode) -> ILibraryAppletAccessor
            0 => {
                let (applet, mode) = (call.arg_u32(0), call.arg_u32(4));
                println!("[recomp-ipc] library applet {applet:#x} (mode {mode}) created");
                call.push_object(Box::new(LibraryAppletAccessor::new(applet, mode)));
                Ok(())
            }
            // CreateStorage(size) -> IStorage
            10 => {
                let size = call.arg_u64(0) as usize;
                call.push_object(Box::new(Storage(vec![0; size.min(1 << 24)])));
                Ok(())
            }
            _ => unimplemented(call, "ILibraryAppletCreator"),
        }
    }
}

/// Library applet ids (`AppletId`).
const APPLET_CONTROLLER: u32 = 0x0C;
const APPLET_PLAYER_SELECT: u32 = 0x10;

/// ILibraryAppletAccessor: a library applet the application launched. No
/// applet really runs: it completes as soon as it starts, with the output a
/// user who accepted its defaults would leave (for controller support, the
/// handheld controller, which hid reports connected; for the user picker,
/// the one user account).
struct LibraryAppletAccessor {
    applet: u32,
    mode: u32,
    started: bool,
    /// Signalled once the applet has completed.
    state_changed: (u32, u32),
    out_data: VecDeque<Vec<u8>>,
}

impl LibraryAppletAccessor {
    fn new(applet: u32, mode: u32) -> Self {
        Self {
            applet,
            mode,
            started: false,
            state_changed: crate::kernel::kernel().create_event(),
            out_data: VecDeque::new(),
        }
    }

    /// The applet's output, by applet.
    fn output(&self) -> Option<Vec<u8>> {
        match self.applet {
            // ControllerSupportResultInfo: player count, padding, the
            // selected npad id (Handheld), Result.
            APPLET_CONTROLLER => {
                let mut info = vec![0u8; 0xC];
                info[0] = 1;
                info[4..8].copy_from_slice(&0x20u32.to_le_bytes());
                Some(info)
            }
            // UiReturnArg: result (success), then the selected user.
            APPLET_PLAYER_SELECT => {
                let mut arg = vec![0u8; 0x18];
                arg[8..].copy_from_slice(&crate::account::user_id_bytes());
                Some(arg)
            }
            _ => None,
        }
    }
}

impl Service for LibraryAppletAccessor {
    fn name(&self) -> String {
        "appletOE:ILibraryAppletAccessor".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // GetAppletStateChangedEvent
            0 => call.push_copy_handle(self.state_changed.1),
            // IsCompleted
            1 => call.push_u8(u8::from(self.started)),
            // Start: the applet runs to completion at once.
            10 => {
                self.started = true;
                if let Some(output) = self.output() {
                    self.out_data.push_back(output);
                }
                crate::kernel::kernel().signal_event(self.state_changed.0);
                println!(
                    "[recomp-ipc] library applet {:#x} (mode {}) completed",
                    self.applet, self.mode
                );
            }
            // RequestExit, Terminate, GetResult (success),
            // SetOutOfFocusApplicationSuspendingEnabled,
            // PresetLibraryAppletGpuTimeSliceZero, PushInData (the input is
            // not needed), PushExtraStorage, PushInteractiveInData,
            // RequestForAppletToGetForeground
            20 | 25 | 30 | 50 | 60 | 100 | 102 | 103 | 150 => {}
            // PopOutData -> IStorage
            101 => match self.out_data.pop_front() {
                Some(bytes) => call.push_object(Box::new(Storage(bytes))),
                None => return Err(RESULT_NO_DATA_IN_CHANNEL),
            },
            // PopInteractiveOutData: nothing.
            104 => return Err(RESULT_NO_DATA_IN_CHANNEL),
            // GetPopOutDataEvent, GetPopInteractiveOutDataEvent
            105 | 106 => call.push_copy_handle(crate::services::signalled_event()),
            // NeedsToExitProcess
            110 => call.push_u8(0),
            // GetLibraryAppletInfo -> applet id, mode
            120 => {
                call.push_u32(self.applet);
                call.push_u32(self.mode);
            }
            // GetIndirectLayerConsumerHandle
            160 => call.push_u64(0),
            _ => return unimplemented(call, "ILibraryAppletAccessor"),
        }
        Ok(())
    }
}

/// IApplicationFunctions
struct ApplicationFunctions;

impl Service for ApplicationFunctions {
    fn name(&self) -> String {
        "appletOE:IApplicationFunctions".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // PopLaunchParameter(kind): the preselected user, as if the
            // system had picked the only account before launch.
            1 if call.arg_u32(0) == LAUNCH_PRESELECTED_USER => {
                let mut parameter = vec![0u8; 0x88];
                parameter[..4].copy_from_slice(&PRESELECTED_USER_MAGIC.to_le_bytes());
                parameter[4..8].copy_from_slice(&1u32.to_le_bytes());
                parameter[8..24].copy_from_slice(&crate::account::user_id_bytes());
                call.push_object(Box::new(Storage(parameter)));
            }
            1 => return Err(RESULT_NO_DATA_IN_CHANNEL),
            // EnsureSaveData(uid) -> required size (0: already present)
            20 => call.push_u64(0),
            // GetDesiredLanguage -> language code "en-US"
            21 => call.push_u64(u64::from_le_bytes(*b"en-US\0\0\0")),
            // SetTerminateResult(result): the SDK reports the Result it is
            // about to abort with.
            22 => println!(
                "[recomp-ipc] application terminate result {}",
                crate::server::describe(call.arg_u32(0))
            ),
            // Begin/EndBlockingHomeButton*,
            // SetMediaPlaybackStateForApplication, SetGamePlayRecordingState,
            // EnableApplicationCrashReport,
            // SetApplicationCopyrightVisibility,
            // EnableApplicationAllThreadDumpOnCrash, SetHdcpAuthenticationActivated
            30..=33 | 60 | 67 | 90 | 102 | 124 | 170 => {}
            // GetDisplayVersion -> 16-byte string
            23 => {
                let mut version = [0u8; 16];
                version[..5].copy_from_slice(b"5.0.0");
                call.push_bytes(&version);
            }
            // NotifyRunning -> bool
            40 => call.push_u8(0),
            // GetPseudoDeviceId -> u128
            50 => {
                call.push_u64(0x5245_434F_4D50_0001);
                call.push_u64(0);
            }
            // IsGamePlayRecordingSupported
            65 => call.push_u8(0),
            // InitializeGamePlayRecording,
            // InitializeApplicationCopyrightFrameBuffer,
            // SetApplicationCopyrightImage
            66 | 100 | 101 => {}
            // GetPreviousProgramIndex: none.
            123 => call.push_u32(u32::MAX),
            // Gpu error, friend invitation, notification and health-warning
            // events.
            130 | 140 | 150 | 160 => call.push_copy_handle(new_event()),
            // TryPopFromFriendInvitationStorageChannel,
            // TryPopFromNotificationStorageChannel: empty.
            141 | 151 => return Err(RESULT_NO_DATA_IN_CHANNEL),
            _ => return unimplemented(call, "IApplicationFunctions"),
        }
        Ok(())
    }
}

/// IDebugFunctions
struct DebugFunctions;

impl Service for DebugFunctions {
    fn name(&self) -> String {
        "appletOE:IDebugFunctions".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        unimplemented(call, "IDebugFunctions")
    }
}

/// IStorage: a byte buffer passed between applets.
pub struct Storage(pub Vec<u8>);

impl Service for Storage {
    fn name(&self) -> String {
        "am:IStorage".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // Open -> IStorageAccessor
            0 => {
                call.push_object(Box::new(StorageAccessor(self.0.clone())));
                Ok(())
            }
            _ => unimplemented(call, "am:IStorage"),
        }
    }
}

/// IStorageAccessor over a snapshot of the storage bytes.
struct StorageAccessor(Vec<u8>);

impl Service for StorageAccessor {
    fn name(&self) -> String {
        "am:IStorageAccessor".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // GetSize
            0 => call.push_u64(self.0.len() as u64),
            // Write(offset, buffer)
            10 => {
                let offset = call.arg_u64(0) as usize;
                let bytes = call.in_buffer(0);
                if offset + bytes.len() <= self.0.len() {
                    self.0[offset..offset + bytes.len()].copy_from_slice(&bytes);
                }
            }
            // Read(offset) -> buffer
            11 => {
                let offset = (call.arg_u64(0) as usize).min(self.0.len());
                let bytes = self.0[offset..].to_vec();
                call.write_out(0, &bytes);
            }
            _ => return unimplemented(call, "am:IStorageAccessor"),
        }
        Ok(())
    }
}
