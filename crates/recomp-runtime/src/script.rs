//! Rhai scripts change a recompiled game without a compile: settings,
//! memory patches, per-frame cheats and hooks on any guest function.
//!
//! `RECOMP_SCRIPT=<file.rhai>` loads a script at start (a title's
//! `[runtime] script` is the default, so the game gets its script however
//! it is started; an empty `RECOMP_SCRIPT` loads none). Saving the file
//! reloads it while the game runs (settings apply at start only). The
//! script's top level registers things; the functions it defines are called
//! later:
//!
//! ```text
//! setting("pad_layout", "positional");       // RECOMP_PAD_LAYOUT, at start
//! hook(0x8022b50, |cpu| { log(`fatal, called from ${cpu.lr.to_hex()}`); false });
//! hook_after(0x87ad1e0, |cpu| { store("jobs", load("jobs") + 1); });
//! fn on_start() { log("ready"); }              // at start and after a reload
//! fn on_frame(frame) { if frame % 300 == 0 { log(`frame ${frame}`); } }
//! fn on_frame_rate(fps) { log(`now ${fps} fps`); }  // at start and on a change
//! ```
//!
//! `on_start` runs before the game's own startup code: data it initializes
//! or relocates is not there yet, so patch data from `on_frame`.
//!
//! A `hook` that returns `true` skips the original function (set `x0` for
//! its result first); `hook_after` runs once the original returned. Hooks
//! run on the calling guest thread, `on_frame` on the script's own thread.
//!
//! Hooked functions go through a trampoline in the call table the runtime
//! publishes (`recomp_abi::publish_targets`), and nothing the generated code
//! links against changes. While a hook is set, generated code's direct calls
//! go through the table too (`recomp_abi::allow_direct_calls`): in LM3's
//! gameplay that costs a few frames a second.

use crate::GuestFn;
use recomp_abi::GuestResult;
use rhai::{CallFnOptions, Dynamic, Engine, EvalAltResult, FnPtr, Scope, AST, FLOAT, INT};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, RwLock};
use std::time::{Duration, SystemTime};

/// Distinct guest functions one session can hook.
const MAX_HOOKS: usize = 64;

struct Script {
    engine: Engine,
    ast: AST,
}

/// A hooked guest function. Its slot (the trampoline that runs it) stays
/// the same for the session, so a reload never has to move a function in
/// a table other threads are calling through.
#[derive(Clone)]
struct Hook {
    address: u64,
    original: Option<GuestFn>,
    before: Option<FnPtr>,
    after: Option<FnPtr>,
}

static SCRIPT: RwLock<Option<Arc<Script>>> = RwLock::new(None);
static HOOKS: RwLock<Vec<Hook>> = RwLock::new(Vec::new());
/// Hooks the script's top level registers while it runs.
static PENDING: Mutex<Vec<(u64, bool, FnPtr)>> = Mutex::new(Vec::new());
/// Settings may change only while the script loads at start.
static AT_START: AtomicBool = AtomicBool::new(false);
/// `store`/`load` values: they outlive a call and a reload.
static VALUES: Mutex<BTreeMap<String, Dynamic>> = Mutex::new(BTreeMap::new());
/// Hooks whose script failed, reported once each until the next reload.
static FAILED: Mutex<BTreeSet<u64>> = Mutex::new(BTreeSet::new());
static PATH: OnceLock<PathBuf> = OnceLock::new();
static MODIFIED: Mutex<Option<SystemTime>> = Mutex::new(None);
/// The program's functions, kept to rebuild the call table.
static FUNCTIONS: OnceLock<Vec<(u64, GuestFn)>> = OnceLock::new();
/// The latest frame shown, and the script thread's wake-up.
static FRAME: AtomicU64 = AtomicU64::new(0);
static FRAME_SIGNAL: (Mutex<()>, Condvar) = (Mutex::new(()), Condvar::new());
/// The frame rate `on_frame_rate` last heard of.
static FRAME_RATE: AtomicU32 = AtomicU32::new(0);

macro_rules! trampolines {
    ($($slot:literal)*) => {
        [$({
            fn trampoline() -> GuestResult {
                run_hook($slot)
            }
            trampoline as GuestFn
        }),*]
    };
}

static TRAMPOLINES: [GuestFn; MAX_HOOKS] = trampolines!(
    0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
    32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61
    62 63
);

/// Load `RECOMP_SCRIPT`, if set (an empty one is none), before anything
/// reads the settings it may change (no guest memory yet: `on_start` runs
/// later).
pub(crate) fn load_at_start() {
    let Some(path) = std::env::var_os("RECOMP_SCRIPT").filter(|path| !path.is_empty()) else {
        return;
    };
    let path = PATH.get_or_init(|| PathBuf::from(path));
    AT_START.store(true, Ordering::Release);
    load(path);
    AT_START.store(false, Ordering::Release);
}

/// Keep the program's functions for the call tables built from them.
pub(crate) fn set_functions(functions: &[(u64, GuestFn)]) {
    let _ = FUNCTIONS.set(functions.to_vec());
}

/// Whether a script hooks any function (see `install_targets`); a reload
/// keeps the slots of hooks it dropped, with nothing to run.
pub(crate) fn hooks_set() -> bool {
    HOOKS
        .read()
        .unwrap_or_else(|poison| poison.into_inner())
        .iter()
        .any(|hook| hook.before.is_some() || hook.after.is_some())
}

/// The call table: the program's functions, hooked ones through their
/// trampolines.
pub(crate) fn table() -> recomp_abi::Targets {
    let functions = FUNCTIONS.get().map(Vec::as_slice).unwrap_or_default();
    let mut hooks = HOOKS.write().unwrap_or_else(|poison| poison.into_inner());
    let mut listed = functions.to_vec();
    for (slot, hook) in hooks.iter_mut().enumerate() {
        match listed
            .iter_mut()
            .find(|(address, _)| *address == hook.address)
        {
            Some(entry) => {
                hook.original = Some(entry.1);
                entry.1 = TRAMPOLINES[slot];
            }
            None => eprintln!(
                "[script] {:#x} is not the start of a recompiled function; its hook never runs",
                hook.address
            ),
        }
    }
    recomp_abi::Targets::new(&listed)
}

/// Once the call table is in place: run `on_start` and start the thread
/// that runs `on_frame` and reloads the script when its file changes.
pub(crate) fn start() {
    let Some(path) = PATH.get() else {
        return;
    };
    call_function("on_start", ());
    let fps = recomp_services::vi::frame_rate();
    FRAME_RATE.store(fps, Ordering::Release);
    call_function("on_frame_rate", (frame_rate_value(fps),));
    recomp_services::vi::on_frame_shown(frame_shown);
    let _ = std::thread::Builder::new()
        .name("recomp-script".to_string())
        .spawn(move || script_thread(path));
}

fn frame_shown(frame: u64) {
    FRAME.store(frame, Ordering::Release);
    FRAME_SIGNAL.1.notify_one();
}

fn script_thread(path: &'static Path) {
    let mut last_frame = 0;
    let mut last_check = std::time::Instant::now();
    loop {
        {
            let guard = FRAME_SIGNAL
                .0
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            if FRAME.load(Ordering::Acquire) == last_frame {
                let _ = FRAME_SIGNAL
                    .1
                    .wait_timeout(guard, Duration::from_millis(500));
            }
        }
        if last_check.elapsed() >= Duration::from_millis(500) {
            last_check = std::time::Instant::now();
            if modified(path) != *MODIFIED.lock().unwrap_or_else(|poison| poison.into_inner())
                && load(path)
            {
                crate::install_targets(table());
                call_function("on_start", ());
            }
        }
        // The player picked another frame rate (the menu).
        let fps = recomp_services::vi::frame_rate();
        if FRAME_RATE.swap(fps, Ordering::AcqRel) != fps {
            call_function("on_frame_rate", (frame_rate_value(fps),));
        }
        let frame = FRAME.load(Ordering::Acquire);
        if frame != last_frame {
            last_frame = frame;
            call_function("on_frame", (frame as INT,));
        }
    }
}

/// The frame rate a script's `on_frame_rate` gets: 30 for the title's own,
/// otherwise the picked rate unchanged (60, 120, 180 or 240).
fn frame_rate_value(fps: u32) -> INT {
    INT::from(if fps == 0 { 30 } else { fps })
}

fn modified(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path)
        .and_then(|meta| meta.modified())
        .ok()
}

/// Compile and run the script's top level. On success its hooks replace
/// the previous ones; on an error the previous script stays.
fn load(path: &Path) -> bool {
    *MODIFIED.lock().unwrap_or_else(|poison| poison.into_inner()) = modified(path);
    PENDING
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .clear();
    let engine = engine();
    let ast = match engine.compile_file(path.to_path_buf()) {
        Ok(ast) => ast,
        Err(error) => {
            eprintln!("[script] {}: {error}", path.display());
            return false;
        }
    };
    if let Err(error) = engine.run_ast(&ast) {
        eprintln!("[script] {}: {error}", path.display());
        return false;
    }
    let pending = std::mem::take(&mut *PENDING.lock().unwrap_or_else(|poison| poison.into_inner()));
    let mut hooks = HOOKS.write().unwrap_or_else(|poison| poison.into_inner());
    for hook in hooks.iter_mut() {
        hook.before = None;
        hook.after = None;
    }
    for (address, after, function) in pending {
        let slot = match hooks.iter().position(|hook| hook.address == address) {
            Some(slot) => slot,
            None if hooks.len() < MAX_HOOKS => {
                hooks.push(Hook {
                    address,
                    original: None,
                    before: None,
                    after: None,
                });
                hooks.len() - 1
            }
            None => {
                eprintln!("[script] more than {MAX_HOOKS} hooked functions: {address:#x} skipped");
                continue;
            }
        };
        if after {
            hooks[slot].after = Some(function);
        } else {
            hooks[slot].before = Some(function);
        }
    }
    let count = hooks
        .iter()
        .filter(|hook| hook.before.is_some() || hook.after.is_some())
        .count();
    drop(hooks);
    FAILED
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .clear();
    *SCRIPT.write().unwrap_or_else(|poison| poison.into_inner()) =
        Some(Arc::new(Script { engine, ast }));
    eprintln!("[script] loaded {} ({count} hooks)", path.display());
    true
}

fn script() -> Option<Arc<Script>> {
    SCRIPT
        .read()
        .unwrap_or_else(|poison| poison.into_inner())
        .clone()
}

/// Call a function the script defines, if it does.
fn call_function(name: &str, args: impl rhai::FuncArgs) {
    let Some(script) = script() else {
        return;
    };
    if !script
        .ast
        .iter_functions()
        .any(|function| function.name == name)
    {
        return;
    }
    let options = CallFnOptions::new().eval_ast(false).rewind_scope(true);
    if let Err(error) = script.engine.call_fn_with_options::<Dynamic>(
        options,
        &mut Scope::new(),
        &script.ast,
        name,
        args,
    ) {
        eprintln!("[script] {name}: {error}");
    }
}

/// What a hooked function's trampoline runs.
fn run_hook(slot: usize) -> GuestResult {
    let Some(hook) = HOOKS
        .read()
        .unwrap_or_else(|poison| poison.into_inner())
        .get(slot)
        .cloned()
    else {
        return Ok(());
    };
    let script = script();
    if let (Some(before), Some(script)) = (&hook.before, &script) {
        if call_hook(script, before, hook.address) {
            return Ok(());
        }
    }
    if let Some(original) = hook.original {
        original()?;
    }
    if let (Some(after), Some(script)) = (&hook.after, &script) {
        call_hook(script, after, hook.address);
    }
    Ok(())
}

/// Whether the hook asked to skip the original function.
fn call_hook(script: &Script, function: &FnPtr, address: u64) -> bool {
    match function.call::<Dynamic>(&script.engine, &script.ast, (Cpu,)) {
        Ok(value) => value.as_bool().unwrap_or(false),
        Err(error) => {
            if FAILED
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .insert(address)
            {
                eprintln!("[script] hook {address:#x}: {error}");
            }
            false
        }
    }
}

/// The calling guest thread's registers, as a hook sees them.
#[derive(Clone, Copy)]
struct Cpu;

type ScriptResult<T> = Result<T, Box<EvalAltResult>>;

fn general(index: INT) -> ScriptResult<usize> {
    usize::try_from(index)
        .ok()
        .filter(|index| *index <= 30)
        .ok_or_else(|| {
            format!("x{index}: registers are x0-x30 (sp and lr are cpu.sp, cpu.lr)").into()
        })
}

fn vector(index: INT) -> ScriptResult<usize> {
    usize::try_from(index)
        .ok()
        .filter(|index| *index <= 31)
        .ok_or_else(|| format!("s{index}: registers are s0-s31").into())
}

fn engine() -> Engine {
    let mut engine = Engine::new();
    engine.on_print(|text| eprintln!("[script] {text}"));
    engine.on_debug(|text, _, position| eprintln!("[script] {position}: {text}"));
    engine.register_fn("log", |text: &str| eprintln!("[script] {text}"));
    engine.register_fn("log", |value: Dynamic| eprintln!("[script] {value}"));
    engine.register_fn("setting", |name: &str, value: Dynamic| {
        let variable = format!("RECOMP_{}", name.to_ascii_uppercase());
        if AT_START.load(Ordering::Acquire) {
            eprintln!("[script] {variable}={value}");
            // Only while loading at start, before any other thread runs.
            std::env::set_var(&variable, value.to_string());
        } else {
            eprintln!("[script] {variable}: settings apply at start; restart the game");
        }
    });
    engine.register_fn("hook", |address: INT, function: FnPtr| {
        pend(address, false, function)
    });
    engine.register_fn("hook_after", |address: INT, function: FnPtr| {
        pend(address, true, function)
    });
    engine.register_fn("store", |key: &str, value: Dynamic| {
        VALUES
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .insert(key.to_string(), value);
    });
    engine.register_fn("load", |key: &str| {
        VALUES
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .get(key)
            .cloned()
            .unwrap_or(Dynamic::from_int(0))
    });
    register_memory(&mut engine);
    engine
        .register_type_with_name::<Cpu>("Cpu")
        .register_fn("x", |_: &mut Cpu, index: INT| -> ScriptResult<INT> {
            Ok(recomp_abi::cpu_get(general(index)?))
        })
        .register_fn(
            "set_x",
            |_: &mut Cpu, index: INT, value: INT| -> ScriptResult<()> {
                recomp_abi::cpu_set(general(index)?, value);
                Ok(())
            },
        )
        .register_fn("s", |_: &mut Cpu, index: INT| -> ScriptResult<FLOAT> {
            let bits = recomp_abi::cpu_vregs()[vector(index)?] as u32;
            Ok(FLOAT::from(f32::from_bits(bits)))
        })
        .register_fn(
            "set_s",
            |_: &mut Cpu, index: INT, value: FLOAT| -> ScriptResult<()> {
                // A scalar write zeroes the rest of the vector register.
                let bits = (value as f32).to_bits();
                recomp_abi::cpu_vstore(&[(vector(index)?, u128::from(bits))]);
                Ok(())
            },
        )
        .register_get("lr", |_: &mut Cpu| recomp_abi::cpu_get(30))
        .register_get("sp", |_: &mut Cpu| recomp_abi::cpu_get(31));
    engine
}

fn pend(address: INT, after: bool, function: FnPtr) {
    PENDING
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .push((address as u64, after, function));
}

/// Guest memory through the runtime's checked path: a wrong address is a
/// script error, not a fault.
fn register_memory(engine: &mut Engine) {
    fn read(
        name: &'static str,
        load: fn(u64) -> crate::RuntimeResult<u64>,
    ) -> impl Fn(INT) -> ScriptResult<INT> {
        move |address| {
            load(address as u64)
                .map(|value| value as INT)
                .map_err(|error| format!("{name}({address:#x}): {error}").into())
        }
    }
    fn write(
        name: &'static str,
        store: fn(u64, u64) -> crate::RuntimeResult<()>,
    ) -> impl Fn(INT, INT) -> ScriptResult<()> {
        move |address, value| {
            store(address as u64, value as u64)
                .map_err(|error| format!("{name}({address:#x}): {error}").into())
        }
    }
    engine.register_fn("read_u8", read("read_u8", crate::mem_load_u8));
    engine.register_fn("read_u16", read("read_u16", crate::mem_load_u16));
    engine.register_fn("read_u32", read("read_u32", crate::mem_load_u32));
    engine.register_fn("read_u64", read("read_u64", crate::mem_load_u64));
    engine.register_fn("write_u8", write("write_u8", crate::mem_store_u8));
    engine.register_fn("write_u16", write("write_u16", crate::mem_store_u16));
    engine.register_fn("write_u32", write("write_u32", crate::mem_store_u32));
    engine.register_fn("write_u64", write("write_u64", crate::mem_store_u64));
    engine.register_fn("read_f32", |address: INT| -> ScriptResult<FLOAT> {
        let bits = read("read_f32", crate::mem_load_u32)(address)?;
        Ok(FLOAT::from(f32::from_bits(bits as u32)))
    });
    engine.register_fn(
        "write_f32",
        |address: INT, value: FLOAT| -> ScriptResult<()> {
            let bits = (value as f32).to_bits();
            write("write_f32", crate::mem_store_u32)(address, INT::from(bits))
        },
    );
    engine.register_fn("read_string", |address: INT| {
        crate::guest_c_string(address as u64)
    });
    // What the F1 menu changes, from a script.
    engine.register_fn("set_docked", recomp_services::system::set_docked);
    engine.register_fn("set_frame_rate", |fps: INT| {
        recomp_services::vi::set_frame_rate(u32::try_from(fps).unwrap_or(0));
    });
    // Mods patch read-only data too (a constant in .rodata).
    engine.register_fn(
        "patch_u32",
        |address: INT, value: INT| -> ScriptResult<()> {
            crate::memory::patch_block(address as u64, &(value as u32).to_le_bytes())
                .map_err(|status| format!("patch_u32({address:#x}): {status:?}").into())
        },
    );
    engine.register_fn(
        "patch_f32",
        |address: INT, value: FLOAT| -> ScriptResult<()> {
            crate::memory::patch_block(address as u64, &(value as f32).to_le_bytes())
                .map_err(|status| format!("patch_f32({address:#x}): {status:?}").into())
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn returns_one() -> GuestResult {
        recomp_abi::cpu_set(0, 1);
        Ok(())
    }

    fn sets_x1() -> GuestResult {
        recomp_abi::cpu_set(1, 10);
        Ok(())
    }

    #[test]
    fn frame_rate_values_pass_native_targets_through() {
        assert_eq!(frame_rate_value(0), 30);
        assert_eq!(frame_rate_value(60), 60);
        assert_eq!(frame_rate_value(120), 120);
        assert_eq!(frame_rate_value(180), 180);
        assert_eq!(frame_rate_value(240), 240);
    }

    // One test: the script state is global.
    #[test]
    fn scripts_hook_functions_change_settings_and_keep_values() {
        let dir = std::env::temp_dir().join(format!("recomp-script-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.rhai");
        std::fs::write(
            &path,
            r#"
            setting("script_test", 7);
            store("calls", 0);
            hook(0x1000, |cpu| { cpu.set_x(0, 42); true });
            hook_after(0x2000, |cpu| {
                cpu.set_x(1, cpu.x(1) + 1);
                store("calls", load("calls") + 1);
            });
            hook(0x3000, |cpu| { cpu.set_x(40, 0); false });
            "#,
        )
        .unwrap();
        AT_START.store(true, Ordering::Release);
        assert!(load(&path));
        AT_START.store(false, Ordering::Release);
        assert_eq!(std::env::var("RECOMP_SCRIPT_TEST").as_deref(), Ok("7"));

        set_functions(&[
            (0x1000, returns_one as GuestFn),
            (0x2000, sets_x1 as GuestFn),
            (0x3000, returns_one as GuestFn),
            (0x4000, sets_x1 as GuestFn),
        ]);
        let table = table();
        // A hook that returns true replaces the function.
        table.get(0x1000).unwrap()().unwrap();
        assert_eq!(recomp_abi::cpu_get(0), 42);
        // An after-hook sees what the function did.
        table.get(0x2000).unwrap()().unwrap();
        assert_eq!(recomp_abi::cpu_get(1), 11);
        // A failing hook leaves the function to run.
        table.get(0x3000).unwrap()().unwrap();
        assert_eq!(recomp_abi::cpu_get(0), 1);
        // Unhooked functions are listed as they are.
        assert_eq!(
            table.get(0x4000).unwrap() as usize,
            sets_x1 as GuestFn as usize
        );
        let calls = VALUES.lock().unwrap().get("calls").cloned().unwrap();
        assert_eq!(calls.as_int(), Ok(1));

        // A reload without the 0x1000 hook lets the function run again,
        // through the same table.
        std::fs::write(&path, "hook_after(0x2000, |cpu| { cpu.set_x(1, 0); });").unwrap();
        assert!(load(&path));
        table.get(0x1000).unwrap()().unwrap();
        assert_eq!(recomp_abi::cpu_get(0), 1);
        table.get(0x2000).unwrap()().unwrap();
        assert_eq!(recomp_abi::cpu_get(1), 0);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
