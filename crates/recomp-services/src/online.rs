//! Online services for a console without a network connection: sockets
//! open but cannot reach anything, name lookups fail, the friend list is
//! empty, play reports are accepted and dropped, and BCAT has no content.

use crate::server::{Call, Service};
use crate::services::unimplemented;

/// FreeBSD errno values, as the Switch socket library uses them.
const ENETDOWN: u32 = 50;
const EAI_FAIL: u32 = 4;

/// `bsd:u` / `bsd:s` (ISocketClient)
pub struct SocketService {
    next_fd: u32,
}

impl SocketService {
    pub fn new() -> Self {
        // 0-2 are taken by stdio in the BSD library's view.
        SocketService { next_fd: 3 }
    }
}

impl Default for SocketService {
    fn default() -> Self {
        Self::new()
    }
}

impl SocketService {
    /// The (return value, errno) pair most BSD commands reply with.
    fn reply(call: &mut Call<'_>, value: i32, errno: u32) {
        call.push_u32(value as u32);
        call.push_u32(errno);
    }
}

impl Service for SocketService {
    fn name(&self) -> String {
        "bsd:u".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // RegisterClient(config, pid, transfer memory size, memory) -> 0
            0 => call.push_u32(0),
            // StartMonitoring(pid)
            1 => {}
            // Socket, SocketExempt, Open, DuplicateSocket, EventFd -> fd
            2 | 3 | 4 | 27 | 31 => {
                let fd = self.next_fd;
                self.next_fd += 1;
                Self::reply(call, fd as i32, 0);
            }
            // Fcntl, SetSockOpt, Shutdown, ShutdownAllSockets, Close: fine.
            20 | 21 | 22 | 23 | 26 => Self::reply(call, 0, 0),
            // Select, Poll: nothing is ever ready.
            5 | 6 => Self::reply(call, 0, 0),
            // Everything that would touch the network.
            7..=19 | 24 | 25 | 29 | 30 => Self::reply(call, -1, ENETDOWN),
            _ => return unimplemented(call, "bsd:u"),
        }
        Ok(())
    }
}

/// `sfdnsres` (IResolver): lookups fail.
pub struct ResolverService;

impl Service for ResolverService {
    fn name(&self) -> String {
        "sfdnsres".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // SetDnsAddressesPrivate, CancelRequest
            0 | 9 => {}
            // GetHostByName(WithOptions), GetHostByAddr(WithOptions):
            // (h_errno HOST_NOT_FOUND, errno, size)
            2 | 3 | 10 | 11 => {
                call.push_u32(1);
                call.push_u32(ENETDOWN);
                call.push_u32(0);
            }
            // GetAddrInfo(WithOptions): (EAI_FAIL, errno, size)
            6 | 12 => {
                call.push_u32(EAI_FAIL);
                call.push_u32(ENETDOWN);
                call.push_u32(0);
            }
            // RequestCancelHandle
            8 => call.push_u32(1),
            _ => return unimplemented(call, "sfdnsres"),
        }
        Ok(())
    }
}

/// `ssl` (ISslService)
pub struct SslService;

impl Service for SslService {
    fn name(&self) -> String {
        "ssl".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // CreateContext(version, pid)
            0 => call.push_object(Box::new(SslContext)),
            // SetInterfaceVersion
            5 => {}
            // GetContextCount
            1 => call.push_u32(0),
            _ => return unimplemented(call, "ssl"),
        }
        Ok(())
    }
}

/// ISslContext: certificates import; connections cannot connect.
struct SslContext;

impl Service for SslContext {
    fn name(&self) -> String {
        "ssl:ISslContext".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // SetOption, RemovePki, RemoveCrl, FlushSessionCache
            0 | 7 | 10 | 12 => {}
            // GetOption
            1 => call.push_u32(0),
            // ImportServerPki, ImportClientPki, RegisterInternalPki, ImportCrl
            // -> id
            4 | 5 | 8 | 11 => call.push_u64(1),
            _ => return unimplemented(call, "ssl:ISslContext"),
        }
        Ok(())
    }
}

/// `friend:u` / `friend:a` / ... (IServiceCreator)
pub struct FriendServiceCreator;

impl Service for FriendServiceCreator {
    fn name(&self) -> String {
        "friend".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // CreateFriendService
            0 => call.push_object(Box::new(FriendService)),
            // CreateNotificationService(uid)
            1 => call.push_object(Box::new(FriendNotifications)),
            // CreateDaemonSuspendSessionService
            2 => call.push_object(Box::new(FriendService)),
            _ => return unimplemented(call, "friend"),
        }
        Ok(())
    }
}

/// IFriendService: no friends, presence and play history accepted.
struct FriendService;

impl Service for FriendService {
    fn name(&self) -> String {
        "friend:IFriendService".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // GetCompletionEvent
            0 => call.push_copy_handle(crate::services::new_event()),
            // Cancel, UpdateFriendInfo, Declare{Open,Close}OnlinePlaySession,
            // UpdateUserPresence, AddPlayHistory, SyncFriendList,
            // RequestSyncFriendList
            1 | 10102 | 10600 | 10601 | 10610 | 10702 | 20103 | 20104 => {}
            // GetFriendListIds, GetFriendList, GetBlockedUserListIds,
            // GetProfileList, GetFriendCount, GetNewlyFriendCount -> 0
            10100 | 10101 | 10400 | 10500 | 20100 | 20101 => call.push_u32(0),
            // GetPlayHistoryRegistrationKey(...) -> key buffer
            10700 | 10701 => {
                call.write_out(0, &[0; 0x40]);
            }
            // GetReceivedFriendRequestCount -> (count, unread)
            20200 => {
                call.push_u32(0);
                call.push_u32(0);
            }
            _ => return unimplemented(call, "friend:IFriendService"),
        }
        Ok(())
    }
}

/// INotificationService: never any notifications.
struct FriendNotifications;

impl Service for FriendNotifications {
    fn name(&self) -> String {
        "friend:INotificationService".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // GetEvent
            0 => call.push_copy_handle(crate::services::new_event()),
            // Clear
            1 => {}
            // Pop: friend::ResultNotificationNotFound (module 121, 304).
            2 => return Err((304 << 9) | 121),
            _ => return unimplemented(call, "friend:INotificationService"),
        }
        Ok(())
    }
}

/// `prepo:u` / `prepo:s` / ... (IPrepoService): reports are accepted and
/// dropped.
pub struct PlayReportService;

impl Service for PlayReportService {
    fn name(&self) -> String {
        "prepo".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // SaveReport* (all versions, with and without user),
            // RequestImmediateTransmission, SaveSystemReport*
            10100..=10105 | 10200 | 20100 | 20101 => {}
            // GetTransmissionStatus
            10300 => call.push_u32(0),
            // GetSystemSessionId
            10400 => call.push_u64(1),
            _ => return unimplemented(call, "prepo"),
        }
        Ok(())
    }
}

/// `bcat:u` (IServiceCreator): the delivery cache is empty.
pub struct BcatServiceCreator;

impl Service for BcatServiceCreator {
    fn name(&self) -> String {
        "bcat:u".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // CreateBcatService(pid)
            0 => call.push_object(Box::new(BcatService)),
            // CreateDeliveryCacheStorageService(pid),
            // CreateDeliveryCacheStorageServiceWithApplicationId(id)
            1 | 2 => call.push_object(Box::new(DeliveryCacheStorage)),
            _ => return unimplemented(call, "bcat:u"),
        }
        Ok(())
    }
}

/// IBcatService
struct BcatService;

impl Service for BcatService {
    fn name(&self) -> String {
        "bcat:IBcatService".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        unimplemented(call, "bcat:IBcatService")
    }
}

/// IDeliveryCacheStorageService: no directories.
struct DeliveryCacheStorage;

impl Service for DeliveryCacheStorage {
    fn name(&self) -> String {
        "bcat:IDeliveryCacheStorageService".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // EnumerateDeliveryCacheDirectory -> count
            10 => {
                call.push_u32(0);
                Ok(())
            }
            _ => unimplemented(call, "bcat:IDeliveryCacheStorageService"),
        }
    }
}
