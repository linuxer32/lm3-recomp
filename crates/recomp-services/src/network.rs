//! `nifm:u` / `nifm:a` / `nifm:s`: the network interface manager of a
//! console that is offline. Requests are processed and refused.

use crate::server::{Call, Service};
use crate::services::unimplemented;

/// nifm::ResultInternetConnectionNotAvailable (module 110, description
/// 3127), the error a title shows as "unable to connect".
const RESULT_NOT_AVAILABLE: u32 = (3127 << 9) | 110;
/// RequestState_Free: not submitted, or finished without a connection.
const REQUEST_FREE: u32 = 1;
const CLIENT_ID: u32 = 1;

/// IStaticService
pub struct NetworkService;

impl Service for NetworkService {
    fn name(&self) -> String {
        "nifm".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // CreateGeneralServiceOld, CreateGeneralService(pid)
            4 | 5 => {
                call.push_object(Box::new(GeneralService));
                Ok(())
            }
            _ => unimplemented(call, "nifm"),
        }
    }
}

/// IGeneralService
struct GeneralService;

impl Service for GeneralService {
    fn name(&self) -> String {
        "nifm:IGeneralService".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // GetClientId -> buffer
            1 => {
                call.write_out(0, &CLIENT_ID.to_le_bytes());
            }
            // CreateScanRequest, CreateRequest(preset)
            2 | 4 => call.push_object(Box::new(Request::new())),
            // GetCurrentNetworkProfile, GetCurrentIpAddress,
            // GetCurrentAccessPoint, GetCurrentIpConfigInfo,
            // GetInternetConnectionStatus: there is no connection.
            5 | 12 | 13 | 15 | 18 => return Err(RESULT_NOT_AVAILABLE),
            // Set*CommunicationEnabled, PutToSleep, WakeUp,
            // SetExclusiveClient, ConfirmSystemAvailability,
            // SetBackgroundRequestEnabled
            16 | 19 | 23 | 24 | 26 | 29 | 30 | 33 | 34 => {}
            // IsWirelessCommunicationEnabled, IsEthernetCommunicationEnabled
            17 | 20 => call.push_u8(1),
            // IsAnyInternetRequestAccepted, IsAnyForegroundRequestAccepted
            21 | 22 => call.push_u8(0),
            _ => return unimplemented(call, "nifm:IGeneralService"),
        }
        Ok(())
    }
}

/// IRequest: submitting finishes at once, without a connection.
struct Request {
    /// Signalled when the request state changes: (writable, readable).
    events: [(u32, u32); 2],
    submitted: bool,
}

impl Request {
    fn new() -> Self {
        let kernel = crate::kernel::kernel();
        Request {
            events: [kernel.create_event(), kernel.create_event()],
            submitted: false,
        }
    }
}

impl Service for Request {
    fn name(&self) -> String {
        "nifm:IRequest".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // GetRequestState
            0 => call.push_u32(REQUEST_FREE),
            // GetResult
            1 if self.submitted => return Err(RESULT_NOT_AVAILABLE),
            1 => {}
            // GetSystemEventReadableHandles
            2 => {
                call.push_copy_handle(self.events[0].1);
                call.push_copy_handle(self.events[1].1);
            }
            // Cancel
            3 => self.submitted = false,
            // Submit: done, and refused.
            4 => {
                self.submitted = true;
                for (write, _) in self.events {
                    crate::kernel::kernel().signal_event(write);
                }
            }
            // Requirement, priority, profile and option setters,
            // RegisterSocketDescriptor, UnregisterSocketDescriptor
            5 | 6 | 8..=18 | 23..=25 => {}
            // GetRevision
            20 => call.push_u32(0),
            // GetAppletInfo: no applet to show.
            21 => return Err(RESULT_NOT_AVAILABLE),
            _ => return unimplemented(call, "nifm:IRequest"),
        }
        Ok(())
    }
}
