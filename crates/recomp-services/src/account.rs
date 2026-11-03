//! `acc:u0` / `acc:u1`: one local user, always selected.

use crate::server::{Call, Service};
use crate::services::unimplemented;

/// The local user's id (two little-endian u64 halves; zero is invalid).
pub const USER_ID: [u64; 2] = [0x0000_0000_0000_0001, 0x5245_434F_4D50_0001];
const NICKNAME: &[u8] = b"Player";

pub fn user_id_bytes() -> [u8; 16] {
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&USER_ID[0].to_le_bytes());
    bytes[8..].copy_from_slice(&USER_ID[1].to_le_bytes());
    bytes
}

fn is_user(call: &Call<'_>, offset: u64) -> bool {
    [call.arg_u64(offset), call.arg_u64(offset + 8)] == USER_ID
}

/// ProfileBase: user id, last edit time, nickname (0x38 bytes).
fn profile_base() -> Vec<u8> {
    let mut bytes = user_id_bytes().to_vec();
    bytes.extend_from_slice(&0u64.to_le_bytes());
    let mut nickname = [0u8; 0x20];
    nickname[..NICKNAME.len()].copy_from_slice(NICKNAME);
    bytes.extend_from_slice(&nickname);
    bytes
}

/// IAccountServiceForApplication
pub struct AccountService;

impl Service for AccountService {
    fn name(&self) -> String {
        "acc:u0".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // GetUserCount
            0 => call.push_u32(1),
            // GetUserExistence(uid)
            1 => call.push_u8(u8::from(is_user(call, 0))),
            // ListAllUsers, ListOpenUsers, ListOpenContextStoredUsers,
            // ListQualifiedUsers: the one user, then invalid (zero) ids to
            // the end of the array, as Horizon fills it. The rest of the
            // title's buffer is its stack; LM3 took a stale float there for
            // a second user and aborted ensuring its save data.
            2 | 3 | 60 | 131 | 141 => {
                let size = call.out_buffer(0).map_or(0, |buffer| buffer.size as usize);
                let mut users = vec![0u8; size.max(16)];
                users[..16].copy_from_slice(&user_id_bytes());
                call.write_out(0, &users);
            }
            // GetLastOpenedUser, TrySelectUserWithoutInteraction
            4 | 51 => call.push_bytes(&user_id_bytes()),
            // GetProfile(uid)
            5 => call.push_object(Box::new(Profile)),
            // IsUserRegistrationRequestPermitted, IsUserAccountSwitchLocked
            50 | 150 => call.push_u8(0),
            // InitializeApplicationInfo(V2), DebugActivateOpenContextRetention,
            // StoreSaveDataThumbnail, ClearSaveDataThumbnail
            99 | 100 | 110 | 111 | 140 => {}
            // GetBaasAccountManagerForApplication(uid)
            101 => call.push_object(Box::new(AccountManager)),
            _ => return unimplemented(call, "acc:u0"),
        }
        Ok(())
    }
}

/// IProfile
struct Profile;

impl Service for Profile {
    fn name(&self) -> String {
        "acc:IProfile".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // Get -> (base, user data)
            0 => {
                call.write_out(0, &[0; 0x80]);
                call.push_bytes(&profile_base());
            }
            // GetBase
            1 => call.push_bytes(&profile_base()),
            // GetImageSize, LoadImage: no image.
            10 | 11 => call.push_u32(0),
            _ => return unimplemented(call, "acc:IProfile"),
        }
        Ok(())
    }
}

/// IManagerForApplication: the network account is never available.
struct AccountManager;

impl Service for AccountManager {
    fn name(&self) -> String {
        "acc:IManagerForApplication".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // CheckAvailability, StoreOpenContext
            0 | 160 => {}
            // GetAccountId
            1 => call.push_u64(USER_ID[1]),
            // LoadIdTokenCache -> size
            3 => call.push_u32(0),
            // EnsureIdTokenCacheAsync -> IAsyncContext
            2 => call.push_object(Box::new(AsyncContext)),
            _ => return unimplemented(call, "acc:IManagerForApplication"),
        }
        Ok(())
    }
}

/// IAsyncContext for requests that finish at once.
struct AsyncContext;

impl Service for AsyncContext {
    fn name(&self) -> String {
        "acc:IAsyncContext".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // GetSystemEvent: already signalled.
            0 => {
                let (write, read) = crate::kernel::kernel().create_event();
                crate::kernel::kernel().signal_event(write);
                call.push_copy_handle(read);
            }
            // Cancel
            1 => {}
            // HasDone
            2 => call.push_u8(1),
            // GetResult
            3 => {}
            _ => return unimplemented(call, "acc:IAsyncContext"),
        }
        Ok(())
    }
}
