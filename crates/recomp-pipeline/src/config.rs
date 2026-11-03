use crate::memory::{MemoryLayoutDescriptor, MemoryPermissionsDescriptor, MemoryRegionDescriptor};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StubBehavior {
    Log,
    Noop,
    Panic,
}

impl FromStr for StubBehavior {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "log" => Ok(StubBehavior::Log),
            "noop" => Ok(StubBehavior::Noop),
            "panic" => Ok(StubBehavior::Panic),
            other => Err(format!("unknown stub behavior: {other}")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceMode {
    Handheld,
    Docked,
}

impl FromStr for PerformanceMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "handheld" => Ok(PerformanceMode::Handheld),
            "docked" => Ok(PerformanceMode::Docked),
            other => Err(format!("unknown performance mode: {other}")),
        }
    }
}

/// The host graphics backend a title is built with. `Recording` keeps
/// rendering off-screen (the command processor runs, nothing is shown);
/// `Vulkan` links `recomp-gfx-vulkan`, which opens a window when the title
/// runs with `RECOMP_VULKAN=1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GraphicsBackend {
    #[default]
    Recording,
    Vulkan,
}

impl FromStr for GraphicsBackend {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "recording" => Ok(GraphicsBackend::Recording),
            "vulkan" => Ok(GraphicsBackend::Vulkan),
            other => Err(format!("unknown graphics backend: {other}")),
        }
    }
}

#[derive(Debug, Deserialize, Default)]
struct RawRuntimeConfig {
    #[serde(default)]
    performance_mode: Option<String>,
    /// `graphics = "vulkan"`: build with the Vulkan window backend.
    #[serde(default)]
    graphics: Option<String>,
    /// `parallel = true`: guest threads run in parallel on host cores by
    /// default (`RECOMP_PARALLEL=0` still serializes them).
    #[serde(default)]
    parallel: Option<bool>,
    /// `block_tracking = false`: record the block address only at function
    /// entry, not at every block. Together with `parallel = true` this also
    /// drops loop backedges, which then only feed the hang diagnostic; such
    /// a build must not be run with `RECOMP_PARALLEL=0`, as loops that make
    /// no calls would lose their preemption point.
    #[serde(default)]
    block_tracking: Option<bool>,
    /// `target_cpu = "native"`: compile the program for this CPU (FMA and
    /// AVX2 make lifted floating-point code much faster; the binary then
    /// runs only on CPUs with the same features). For a cross-compiled title
    /// (`target` below), `native` resolves to a portable CPU for the target
    /// architecture (`generic` on aarch64, which still has Advanced SIMD).
    #[serde(default)]
    target_cpu: Option<String>,
    /// `target = "aarch64-linux-android"`: the Cargo target triple the
    /// generated program is built for. The pipeline writes it to the
    /// project's `.cargo/config.toml` (including target-specific
    /// `target_cpu` flags), so `cargo build` cross-compiles without
    /// `--target` on the command line.
    #[serde(default)]
    target: Option<String>,
    /// `guest_context = true`: generate functions that take a
    /// `*mut GuestContext` and read and write its fields directly instead
    /// of snapshotting the whole register file at every entry and call.
    /// Off by default: it is a prototype ABI, not yet the default.
    #[serde(default)]
    guest_context: Option<bool>,
    /// `direct_registers = true`: in guest-context mode, lower each guest
    /// register read and write to a context field (`(*ctx).x[N]`,
    /// `(*ctx).v[N]`, `(*ctx).nzcv`) instead of loading the whole function's
    /// register set into locals in the prologue. Requires
    /// `guest_context = true`. Off by default.
    #[serde(default)]
    direct_registers: Option<bool>,
    /// `script = "scripts/game.rhai"`: the script the program loads when
    /// `RECOMP_SCRIPT` is not set (relative to the config file).
    #[serde(default)]
    script: Option<String>,
    #[serde(default)]
    memory_layout: Option<RawMemoryLayoutConfig>,
    #[serde(default)]
    ipc: Option<RawIpcConfig>,
    /// Initial stack pointer value for the entry function (x31).
    #[serde(default)]
    initial_sp: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct RawIpcConfig {
    /// Guest address window holding a RomFS image (legacy in-memory mode).
    #[serde(default)]
    romfs_base: u64,
    #[serde(default)]
    romfs_size: u64,
    /// Guest address of the IPC buffer (pointed to by TPIDRRO_EL0).
    #[serde(default)]
    ipc_buffer: Option<u64>,
    /// Host path to a decompressed RomFS image, served through fsp-srv
    /// storages/file systems. Preferred for full titles: images reach
    /// gigabytes and cannot live in the guest address space.
    #[serde(default)]
    romfs_image: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawMemoryLayoutConfig {
    regions: Vec<RawMemoryRegionConfig>,
}

#[derive(Debug, Deserialize)]
struct RawMemoryRegionConfig {
    name: String,
    base: u64,
    size: u64,
    permissions: RawMemoryPermissionsConfig,
}

#[derive(Debug, Deserialize)]
struct RawMemoryPermissionsConfig {
    read: bool,
    write: bool,
    execute: bool,
}

/// A runtime file (a script or a RomFS image) that the launcher may supply
/// beside the executable instead of the path recorded when the project was
/// generated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeAsset {
    /// Path as written in the config, or just the file name of an absolute
    /// config path, tried under each runtime asset root.
    pub relative: Option<String>,
    /// Absolute path resolved against the config file. It is used when no
    /// packaged copy is found, so an unpacked build keeps working.
    pub fallback: String,
}

impl RuntimeAsset {
    /// The asset as configured, before its fallback is resolved.
    pub fn unresolved(configured: String) -> Self {
        Self {
            relative: Some(configured.clone()),
            fallback: configured,
        }
    }

    /// Resolve the configured path against the directory holding the title
    /// config, and derive the relative path used beside the executable.
    pub fn resolve(self, config_dir: &Path) -> Self {
        let configured = PathBuf::from(&self.fallback);
        let absolute = if configured.is_absolute() {
            configured.clone()
        } else {
            config_dir.join(&configured)
        };
        let relative = if configured.is_absolute() {
            configured
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
        } else {
            Some(self.fallback)
        };
        Self {
            relative,
            fallback: absolute.to_string_lossy().into_owned(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct IpcConfig {
    pub romfs_base: u64,
    pub romfs_size: u64,
    pub romfs_image: Option<RuntimeAsset>,
    pub ipc_buffer: Option<u64>,
}

#[derive(Debug)]
pub struct RuntimeConfig {
    pub performance_mode: PerformanceMode,
    pub graphics: GraphicsBackend,
    pub parallel: bool,
    /// Whether generated code records the guest block it is about to run
    /// (`recomp_abi::enter_block`). On by default: faults and
    /// `RECOMP_WATCH_STORE` report the block's address. Off saves a
    /// thread-local store per block and, with `parallel = true`, a
    /// thread-local increment per loop iteration; diagnostics then see
    /// only the function entry (or the block where a caller left off) and
    /// the 5-second progress monitor no longer hears from loops that make
    /// no calls. A build with both flags off keeps no loop preemption
    /// point, so it must run in parallel mode (`RECOMP_PARALLEL=0` would
    /// leave loops that make no calls unable to let other threads run).
    pub block_tracking: bool,
    pub target_cpu: Option<String>,
    /// The Cargo target triple the generated program is built for, if it is
    /// not the build host's (`[runtime] target`).
    pub target: Option<String>,
    /// Generate the guest-context ABI (`*mut GuestContext` parameters).
    pub guest_context: bool,
    /// Lower guest register accesses to context fields per instruction
    /// (`direct_registers`). Only meaningful with `guest_context`.
    pub direct_registers: bool,
    /// The default script, as written in the config.
    pub script: Option<RuntimeAsset>,
    pub ipc: Option<IpcConfig>,
    pub initial_sp: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct RawTitleConfig {
    title: String,
    entry: String,
    abi_version: String,
    #[serde(default)]
    stubs: BTreeMap<String, String>,
    #[serde(default)]
    host_functions: BTreeMap<String, String>,
    #[serde(default)]
    runtime: Option<RawRuntimeConfig>,
}

#[derive(Debug)]
pub struct TitleConfig {
    pub title: String,
    pub entry: String,
    pub abi_version: String,
    pub stubs: BTreeMap<String, StubBehavior>,
    /// Lifted functions with a host implementation (function name to the
    /// runtime's `__recomp_host_<name>`), tried before the lifted code.
    pub host_functions: BTreeMap<String, String>,
    pub runtime: RuntimeConfig,
    pub memory_layout: MemoryLayoutDescriptor,
}

impl TitleConfig {
    pub fn parse(toml_src: &str) -> Result<Self, String> {
        let raw: RawTitleConfig =
            toml::from_str(toml_src).map_err(|err| format!("invalid config: {err}"))?;
        let mut stubs = BTreeMap::new();
        for (name, behavior) in raw.stubs {
            let parsed = StubBehavior::from_str(&behavior)?;
            stubs.insert(name, parsed);
        }
        for (function, host) in &raw.host_functions {
            let identifier = !host.is_empty()
                && host
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');
            if !identifier {
                return Err(format!(
                    "host function for {function}: {host:?} is not a lowercase identifier"
                ));
            }
        }
        let runtime = raw.runtime.unwrap_or_default();
        if let Some(target) = runtime.target.as_deref() {
            if !valid_cargo_component(target) || !target.contains('-') {
                return Err(format!(
                    "runtime target {target:?} is not a Cargo target triple"
                ));
            }
        }
        if let Some(cpu) = runtime.target_cpu.as_deref() {
            if !valid_cargo_component(cpu) {
                return Err(format!("runtime target_cpu {cpu:?} is not a CPU name"));
            }
        }
        let guest_context = runtime.guest_context.unwrap_or(false);
        let direct_registers = runtime.direct_registers.unwrap_or(false);
        if direct_registers && !guest_context {
            return Err(
                "direct_registers = true requires guest_context = true: direct register \
                 access reads and writes the function's GuestContext"
                    .to_string(),
            );
        }
        let ipc = runtime.ipc.map(|ipc| IpcConfig {
            romfs_base: ipc.romfs_base,
            romfs_size: ipc.romfs_size,
            romfs_image: ipc.romfs_image.map(RuntimeAsset::unresolved),
            ipc_buffer: ipc.ipc_buffer,
        });
        let runtime_mode = runtime
            .performance_mode
            .unwrap_or_else(|| "handheld".to_string());
        let performance_mode = PerformanceMode::from_str(&runtime_mode)?;
        let graphics = runtime
            .graphics
            .as_deref()
            .map(GraphicsBackend::from_str)
            .transpose()?
            .unwrap_or_default();
        let memory_layout = match runtime.memory_layout {
            Some(layout) => parse_memory_layout(layout)?,
            None => {
                let layout = MemoryLayoutDescriptor::minimal_default();
                layout.validate()?;
                layout
            }
        };
        Ok(TitleConfig {
            title: raw.title,
            entry: raw.entry,
            abi_version: raw.abi_version,
            stubs,
            host_functions: raw.host_functions,
            runtime: RuntimeConfig {
                performance_mode,
                graphics,
                parallel: runtime.parallel.unwrap_or(false),
                block_tracking: runtime.block_tracking.unwrap_or(true),
                target_cpu: runtime.target_cpu,
                target: runtime.target,
                guest_context,
                direct_registers,
                script: runtime.script.map(RuntimeAsset::unresolved),
                ipc,
                initial_sp: runtime.initial_sp,
            },
            memory_layout,
        })
    }
}

/// Characters Cargo and rustc accept in a target triple or CPU name. The
/// value is written into `.cargo/config.toml`, so quotes and whitespace must
/// not get through.
fn valid_cargo_component(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '+'))
}

fn parse_memory_layout(layout: RawMemoryLayoutConfig) -> Result<MemoryLayoutDescriptor, String> {
    let regions = layout
        .regions
        .into_iter()
        .map(|region| {
            MemoryRegionDescriptor::new(
                region.name,
                region.base,
                region.size,
                MemoryPermissionsDescriptor::new(
                    region.permissions.read,
                    region.permissions.write,
                    region.permissions.execute,
                ),
            )
        })
        .collect();
    let descriptor = MemoryLayoutDescriptor { regions };
    descriptor.validate()?;
    Ok(descriptor)
}
