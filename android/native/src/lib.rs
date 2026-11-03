//! Android host bootstrap for a recompiled title.
//!
//! SDL3's Android activity loads `libmain.so` and resolves `SDL_main` in it
//! (`SDLActivity.getMainSharedObject`/`getMainFunction`). This library is that
//! `libmain.so`. It prepares the app-private asset and save directories, then
//! loads the generated title library and calls its exported
//! [`RECOMP_ANDROID_ENTRY`] symbol.
//!
//! The title library is kept separate so an APK can be built and launched
//! before the ARM64 code generator is ready, and so the generated project can
//! be rebuilt with the pipeline without repackaging the Java app.
//!
//! Contract with the generated library (also documented in
//! `android/README.md` and `docs/android-port.md`):
//!
//! * shared object name `librecomp_game.so` (or `RECOMP_GAME_LIB` path);
//! * exported `extern "C" fn recomp_android_main(argc: c_int, argv: *mut *mut c_char) -> c_int`;
//! * return 0 for a normal exit, nonzero for a fault.
//!
//! Before the call the bootstrap sets `RECOMP_ASSETS`, `RECOMP_SAVE_DIR`,
//! `RECOMP_CACHE_DIR`, and (when a romfs image is present) `RECOMP_ROMFS`.
//! Relative paths in the runtime should be rooted at `RECOMP_ASSETS`.

#![cfg(target_os = "android")]

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::fs;
use std::path::{Path, PathBuf};

const LOG_TAG: &str = "SwitchRecomp";
const GAME_LIB_NAME: &str = "librecomp_game.so";
const GAME_LIB_ENV: &str = "RECOMP_GAME_LIB";
const GAME_ENTRY: &str = "recomp_android_main";

const ANDROID_LOG_INFO: c_int = 4;
const ANDROID_LOG_WARN: c_int = 5;
const ANDROID_LOG_ERROR: c_int = 6;

/// RTLD_NOW / RTLD_GLOBAL from bionic's `<dlfcn.h>`.
const RTLD_NOW: c_int = 2;
const RTLD_GLOBAL: c_int = 0x00100;

type GameMain = unsafe extern "C" fn(c_int, *mut *mut c_char) -> c_int;

#[link(name = "log")]
extern "C" {
    fn __android_log_print(
        priority: c_int,
        tag: *const c_char,
        format: *const c_char,
        ...
    ) -> c_int;
}

#[link(name = "dl")]
extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *mut c_char;
}

fn log(priority: c_int, message: &str) {
    if let Ok(tag) = CString::new(LOG_TAG) {
        if let Ok(message) = CString::new(message) {
            // SAFETY: both strings are NUL-terminated for the duration of the
            // call, and the "%s" format cannot interpret the message.
            unsafe {
                __android_log_print(
                    priority,
                    tag.as_ptr(),
                    b"%s\0".as_ptr().cast(),
                    message.as_ptr(),
                );
            }
        }
    }
}

/// The entry point SDL3's Android bridge looks up in `libmain.so`.
///
/// `SDLActivity` calls `nativeRunMain("libmain.so", "SDL_main", argv)` and
/// SDL's `SDL_RunApp` invokes this on the Java-created main thread. Keep the
/// name and ABI exactly as SDL expects.
#[no_mangle]
pub extern "C" fn SDL_main(argc: c_int, argv: *mut *mut c_char) -> c_int {
    log(ANDROID_LOG_INFO, "bootstrap: SDL_main entered");
    // The release profile aborts on panic, so this is only for debug builds.
    match std::panic::catch_unwind(|| run(argc, argv)) {
        Ok(code) => code,
        Err(_) => {
            log(ANDROID_LOG_ERROR, "bootstrap: panic escaped the game entry");
            1
        }
    }
}

fn run(argc: c_int, argv: *mut *mut c_char) -> c_int {
    // The title renders at 16:9; force both landscape orientations before
    // SDL creates the window, otherwise SDLActivity's full-sensor default can
    // leave the app in portrait on phones with auto-rotate on.
    sdl3::hint::set("SDL_ORIENTATIONS", "LandscapeLeft LandscapeRight");

    // SDL_RunApp has already set up the Android JNI bridge. SDL_Init gives us
    // the storage paths and a defined library state before the guest starts.
    let sdl = match sdl3::init() {
        Ok(sdl) => sdl,
        Err(error) => {
            log(
                ANDROID_LOG_ERROR,
                &format!("bootstrap: SDL_Init failed: {error}"),
            );
            return 1;
        }
    };
    log(
        ANDROID_LOG_INFO,
        &format!("bootstrap: SDL {} initialized", sdl3::version::version()),
    );

    let paths = match prepare_paths() {
        Ok(paths) => paths,
        Err(error) => {
            log(
                ANDROID_LOG_ERROR,
                &format!("bootstrap: could not prepare app storage: {error}"),
            );
            return 1;
        }
    };
    log(
        ANDROID_LOG_INFO,
        &format!(
            "bootstrap: assets={} save={} cache={}",
            paths.assets.display(),
            paths.save.display(),
            paths.cache.display()
        ),
    );
    export_paths(&paths);

    // The APK carries no game data. Fail visibly instead of returning to the
    // launcher when the user has not copied the segments and RomFS yet.
    let segments = paths.assets.join("segments");
    let romfs_dir = paths.assets.join("romfs");
    let romfs_bin = paths.assets.join("romfs.bin");
    if !segments.is_dir() || !(romfs_dir.is_dir() || romfs_bin.is_file()) {
        show_message(
            &sdl,
            "SwitchRecomp",
            &format!(
                "Game data is missing.\n\nCopy the extracted title data here:\n{}\n\nRequired:\n  segments/\n  romfs/  (or romfs.bin)\n\nThen restart the app.",
                paths.assets.display()
            ),
        );
        return 1;
    }

    let library = match open_game_library() {
        Ok(handle) => handle,
        Err(error) => {
            log(ANDROID_LOG_ERROR, &error);
            // A visible window proves that the APK, SDL3 and the activity work
            // even when the generated game library has not been packaged yet.
            splash(&sdl, &error);
            return 1;
        }
    };

    let entry = match resolve_entry(library) {
        Ok(entry) => entry,
        Err(error) => {
            log(ANDROID_LOG_ERROR, &error);
            splash(&sdl, &error);
            return 1;
        }
    };

    log(
        ANDROID_LOG_INFO,
        &format!("bootstrap: calling {GAME_ENTRY} in {GAME_LIB_NAME}"),
    );
    // SAFETY: the symbol has the documented SDL_main-shaped ABI. The game
    // library stays loaded for the life of the process.
    let status = unsafe { entry(argc, argv) };
    log(
        ANDROID_LOG_INFO,
        &format!("bootstrap: {GAME_ENTRY} returned {status}"),
    );
    if status != 0 {
        show_message(
            &sdl,
            "SwitchRecomp - game stopped",
            &format!(
                "The game stopped with status {status}.\n\nCheck that segments/ and romfs/ are complete.\nFor logs on a computer:\n  adb logcat -s SwitchRecompGame"
            ),
        );
    }
    status
}

/// Show a dialog through SDL's Android message box. Falls back to the blank
/// splash window when the dialog cannot be created.
fn show_message(sdl: &sdl3::Sdl, title: &str, message: &str) {
    log(ANDROID_LOG_ERROR, message);
    if sdl3::messagebox::show_simple_message_box(
        sdl3::messagebox::MessageBoxFlag::ERROR,
        title,
        message,
        None::<&sdl3::video::Window>,
    )
    .is_err()
    {
        splash(sdl, message);
    }
}

struct AppPaths {
    assets: PathBuf,
    save: PathBuf,
    cache: PathBuf,
}

fn prepare_paths() -> Result<AppPaths, String> {
    let internal = android_path(sdl3::sys::system::SDL_GetAndroidInternalStoragePath)
        .ok_or_else(|| "SDL_GetAndroidInternalStoragePath returned NULL".to_string())?;
    let external = android_path(sdl3::sys::system::SDL_GetAndroidExternalStoragePath);
    let cache = android_path(sdl3::sys::system::SDL_GetAndroidCachePath)
        .unwrap_or_else(|| internal.join("cache"));

    // External app storage is easy to fill with adb and survives a rebuild of
    // the APK; fall back to internal storage when it is unavailable.
    let assets = external
        .as_ref()
        .map(|root| root.join("assets"))
        .filter(|path| fs::create_dir_all(path).is_ok())
        .unwrap_or_else(|| internal.join("assets"));
    let save = internal.join("save");

    for dir in [&assets, &save, &cache] {
        fs::create_dir_all(dir).map_err(|error| format!("create {}: {error}", dir.display()))?;
    }

    Ok(AppPaths {
        assets,
        save,
        cache,
    })
}

fn android_path(getter: unsafe extern "C" fn() -> *const c_char) -> Option<PathBuf> {
    // SAFETY: these SDL getters return a NUL-terminated string owned by SDL.
    let pointer = unsafe { getter() };
    if pointer.is_null() {
        return None;
    }
    // SAFETY: non-NULL return from SDL_GetAndroid*Path is a valid C string.
    let value = unsafe { CStr::from_ptr(pointer) }.to_string_lossy();
    if value.is_empty() {
        None
    } else {
        Some(PathBuf::from(value.as_ref()))
    }
}

fn export_paths(paths: &AppPaths) {
    set_env_default("RECOMP_ASSETS", &paths.assets);
    set_env_default("RECOMP_SAVE_DIR", &paths.save);
    set_env_default("RECOMP_CACHE_DIR", &paths.cache);
    set_env_default("RECOMP_VULKAN", "1");

    // The generated entry currently bakes in the host path it was emitted
    // with. Let a build override it with the on-device image.
    let romfs = paths.assets.join("romfs.bin");
    if romfs.is_file() {
        set_env_default("RECOMP_ROMFS", &romfs);
    }
}

fn set_env_default(key: &str, value: impl AsRef<std::ffi::OsStr>) {
    if std::env::var_os(key).is_none() {
        std::env::set_var(key, value);
    }
}

fn open_game_library() -> Result<*mut c_void, String> {
    let mut errors = Vec::new();
    if let Some(path) = std::env::var_os(GAME_LIB_ENV) {
        if !path.is_empty() {
            let c_path = CString::new(path.to_string_lossy().as_bytes())
                .map_err(|_| format!("{GAME_LIB_ENV} contains a NUL byte"))?;
            // SAFETY: c_path is NUL-terminated; flags are bionic RTLD values.
            let handle = unsafe { dlopen(c_path.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
            if !handle.is_null() {
                log(
                    ANDROID_LOG_INFO,
                    &format!("bootstrap: loaded {}", Path::new(&path).display()),
                );
                return Ok(handle);
            }
            errors.push(format!(
                "{GAME_LIB_ENV}={}: {}",
                Path::new(&path).display(),
                dlerror_string()
            ));
        }
    }

    // A bare soname is resolved by the Android linker against the app's
    // nativeLibraryDir, where the APK's lib/arm64-v8a entries are extracted.
    let name = CString::new(GAME_LIB_NAME).expect("static soname");
    // SAFETY: name is NUL-terminated; flags are bionic RTLD values.
    let handle = unsafe { dlopen(name.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
    if !handle.is_null() {
        log(
            ANDROID_LOG_INFO,
            &format!("bootstrap: loaded {GAME_LIB_NAME} from the native library directory"),
        );
        return Ok(handle);
    }
    errors.push(format!("{GAME_LIB_NAME}: {}", dlerror_string()));

    Err(format!(
        "bootstrap: game library not found; package lib/arm64-v8a/{GAME_LIB_NAME} or set {GAME_LIB_ENV} ({})",
        errors.join("; ")
    ))
}

fn resolve_entry(library: *mut c_void) -> Result<GameMain, String> {
    let symbol = CString::new(GAME_ENTRY).expect("static symbol");
    // SAFETY: library is a live handle from dlopen.
    let pointer = unsafe { dlsym(library, symbol.as_ptr()) };
    if pointer.is_null() {
        return Err(format!(
            "bootstrap: {GAME_LIB_NAME} does not export {GAME_ENTRY}: {}",
            dlerror_string()
        ));
    }
    // SAFETY: the generated library exports this function with the documented
    // SDL_main-compatible signature.
    Ok(unsafe { std::mem::transmute::<*mut c_void, GameMain>(pointer) })
}

fn dlerror_string() -> String {
    // SAFETY: dlerror returns a thread-local NUL-terminated string or NULL.
    let pointer = unsafe { dlerror() };
    if pointer.is_null() {
        "no dl error".to_string()
    } else {
        unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned()
    }
}

/// Show the activity is alive when the title library is missing or invalid.
/// The message itself goes to logcat; drawing text would require SDL_ttf.
fn splash(sdl: &sdl3::Sdl, message: &str) {
    log(ANDROID_LOG_WARN, message);
    let result = (|| -> Result<(), String> {
        let video = sdl.video().map_err(|error| error.to_string())?;
        let window = video
            .window("SwitchRecomp", 960, 540)
            .position_centered()
            .build()
            .map_err(|error| error.to_string())?;
        let mut canvas = window.into_canvas();
        canvas.set_draw_color(sdl3::pixels::Color::RGB(24, 32, 48));
        canvas.clear();
        canvas.present();
        let mut events = sdl.event_pump().map_err(|error| error.to_string())?;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while std::time::Instant::now() < deadline {
            for event in events.poll_iter() {
                if matches!(event, sdl3::event::Event::Quit { .. }) {
                    return Ok(());
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        Ok(())
    })();
    if let Err(error) = result {
        log(
            ANDROID_LOG_WARN,
            &format!("bootstrap: splash window unavailable: {error}"),
        );
    }
}
