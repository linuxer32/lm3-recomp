//! `RECOMP_VULKAN_PROBE`: a log of chosen draws' inputs over a few frames,
//! started while the title runs, for glitches that change from frame to
//! frame (a material that flickers between two colours) and only show at a
//! place in the game that takes a while to reach.
//!
//! `RECOMP_VULKAN_PROBE=<fragment program>[:<count>],...` names the draws
//! by their fragment program's GPU address (as `RECOMP_VULKAN_SKIP_FS`
//! does), each optionally only with `count` vertices or indices
//! (`RECOMP_VULKAN_PROBE_COUNT=<n>` gives the count to programs without
//! one). Nothing is logged until the file `probe.on` appears in
//! `RECOMP_VULKAN_PROBE_DIR` (default `probe`, in the working directory).
//! The probe then removes it and, for `RECOMP_VULKAN_PROBE_FRAMES` frames
//! (default 40), appends each probed draw to `probe.txt` there (its state,
//! the constant buffers its programs read and the textures they sample,
//! with what each one sampled: a render target, a depth target or an image
//! prepared from guest memory), and saves each frame as `frame-<n>/`.
//! Creating the file again starts another round. The file can change the
//! round, with words separated by spaces: programs as above (replacing the
//! list), `frames=<n>`, `capture=<address>` to save the render target at
//! that address (as the log names it) instead of the frame, and
//! `target=<address>,...` to probe every draw into those color targets too,
//! `crop=<x>,<y>,<width>,<height>` with `snap=<address>,...` to copy that
//! rectangle of those targets (color or depth, as the log names them) when
//! a probed draw is about to run, and of the draw's own target after it,
//! into `frame-<n>/` as raw texels
//! (`d<draw>-<when>-<address>-<w>x<h>-<format>.bin`).

use std::io::Write;
use std::path::PathBuf;

use recomp_gfx::maxwell::render::{DrawCall, GpuMemory};

pub struct Probe {
    /// Fragment programs, each with the vertex or index count it is
    /// probed at (any without one).
    programs: Vec<(u64, Option<u32>)>,
    /// The render target saved each frame instead of the frame.
    capture: Option<u64>,
    /// The rectangle (guest pixels) and targets copied at probed draws.
    crop: Option<[u32; 4]>,
    /// Color targets every draw into which is probed.
    targets: Vec<u64>,
    snaps: Vec<u64>,
    dir: PathBuf,
    frames: u64,
    /// Frames still to log; zero while waiting for `probe.on`.
    left: u64,
    lines: Vec<String>,
    /// The lines of the draw being recorded, put after its description.
    draw_lines: Vec<String>,
    /// The draw being recorded is logged in detail.
    pub current: bool,
    /// ... because its fragment program is one of `programs` (or it draws
    /// into one of `targets`).
    primary: bool,
    /// ... because its fragment program is one of `programs`: only those
    /// are read again for later writes (every draw into a target would
    /// make that slow).
    by_program: bool,
    /// Render and depth targets probed draws sampled: draws into them are
    /// logged too (in detail, the first `PRODUCERS` a frame), and clears
    /// and copies into them, to show what the probed draws read.
    watched: std::collections::BTreeSet<u64>,
    /// Draws into watched targets logged in detail this frame.
    producers: u32,
    /// Constant buffers probed draws read, read again at every later draw
    /// for two frames: a word that changes was written after the draw
    /// read it (on hardware the draw may have run after the write).
    rereads: Vec<Reread>,
    /// Programs already saved to the directory.
    dumped: std::collections::BTreeSet<u64>,
}

struct Reread {
    label: String,
    address: u64,
    bytes: Vec<u8>,
    frames: u32,
}

/// Draws into watched targets logged in detail per frame; the others get
/// one line.
const PRODUCERS: u32 = 64;

fn address(text: &str) -> Option<u64> {
    let text = text.trim();
    let digits = text.strip_prefix("0x").unwrap_or(text);
    u64::from_str_radix(digits, 16).ok()
}

/// `<address>[:<count>]`.
fn program(text: &str, count: Option<u32>) -> Option<(u64, Option<u32>)> {
    match text.split_once(':') {
        Some((at, count)) => Some((address(at)?, Some(count.trim().parse().ok()?))),
        None => Some((address(text)?, count)),
    }
}

impl Probe {
    pub fn from_env() -> Option<Self> {
        let count = std::env::var("RECOMP_VULKAN_PROBE_COUNT")
            .ok()
            .and_then(|value| value.parse().ok());
        let programs: Vec<(u64, Option<u32>)> = std::env::var("RECOMP_VULKAN_PROBE")
            .ok()?
            .split(',')
            .filter_map(|text| program(text, count))
            .collect();
        if programs.is_empty() {
            return None;
        }
        let dir = crate::paths::resolve(
            std::env::var_os("RECOMP_VULKAN_PROBE_DIR").unwrap_or_else(|| "probe".into()),
        );
        let _ = std::fs::create_dir_all(&dir);
        println!(
            "[recomp-vulkan] probe ready: create {} to log fragment programs {programs:#x?}",
            dir.join("probe.on").display()
        );
        Some(Self {
            programs,
            capture: None,
            crop: None,
            snaps: Vec::new(),
            targets: Vec::new(),
            dir,
            frames: std::env::var("RECOMP_VULKAN_PROBE_FRAMES")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(40),
            left: 0,
            lines: Vec::new(),
            draw_lines: Vec::new(),
            current: false,
            primary: false,
            by_program: false,
            watched: std::collections::BTreeSet::new(),
            producers: 0,
            rereads: Vec::new(),
            dumped: std::collections::BTreeSet::new(),
        })
    }

    /// A probed draw sampled the target at `address`.
    pub fn watch(&mut self, address: u64) {
        if self.primary {
            self.watched.insert(address);
        }
    }

    /// Whether `draw` renders into a target a probed draw sampled (watched
    /// by canonical address, as the renderer keys targets).
    pub fn produces(&self, memory: &dyn GpuMemory, draw: &DrawCall) -> bool {
        let watched = |address: u64| {
            self.watched
                .contains(&memory.resolve(address).unwrap_or(address))
        };
        self.left > 0
            && !self.watched.is_empty()
            && ((0..draw.color_target_count.min(8)).any(|buffer| {
                let target = draw.color_targets[draw.color_target_index(buffer)];
                target.enabled() && watched(target.address)
            }) || (draw.depth_target_enable && watched(draw.depth_target.address)))
    }

    /// Whether a clear or copy into `address` is logged.
    pub fn watches(&self, address: u64) -> bool {
        self.left > 0 && self.watched.contains(&address)
    }

    /// One line, outside any probed draw (a draw, clear or copy into a
    /// watched target).
    pub fn line(&mut self, line: String) {
        if self.left > 0 {
            self.lines.push(line);
        }
    }

    /// A draw is about to be recorded: whether it is logged in detail (a
    /// probed program, or one of the first draws a frame into a watched
    /// target), and so `current` while it is recorded.
    pub fn begin_draw(&mut self, memory: &dyn GpuMemory, draw: &DrawCall) -> bool {
        self.by_program = self.wants(draw);
        self.primary = self.by_program
            || (self.left > 0
                && !self.targets.is_empty()
                && draw.first_color_target_address().is_some_and(|target| {
                    self.targets
                        .contains(&memory.resolve(target).unwrap_or(target))
                }));
        let producer = !self.primary && self.producers < PRODUCERS && self.produces(memory, draw);
        if producer {
            self.producers += 1;
        }
        self.current = self.primary || producer;
        self.current
    }

    fn wants(&self, draw: &DrawCall) -> bool {
        let program = draw.fragment_shader().unwrap_or(0);
        self.left > 0
            && self.programs.iter().any(|(address, count)| {
                *address == program && count.is_none_or(|count| count == draw.parameters.count)
            })
    }

    pub fn note(&mut self, line: String) {
        if self.current {
            self.draw_lines.push(line);
        }
    }

    /// The probed draw was recorded: its description, then its lines.
    pub fn finish_draw(&mut self, description: String) {
        self.current = false;
        self.lines.push(if self.primary {
            description
        } else {
            format!("into a watched target: {description}")
        });
        self.primary = false;
        self.lines.append(&mut self.draw_lines);
    }

    /// A frame is presented: starts the probe when `probe.on` has appeared,
    /// and writes what the frame's draws logged. The directory to save the
    /// frame in when its draws were logged, and the render target to save
    /// instead of the frame.
    pub fn frame(&mut self, presented: u64) -> Option<(PathBuf, Option<u64>)> {
        if self.left == 0 {
            let trigger = self.dir.join("probe.on");
            let Ok(round) = std::fs::read_to_string(&trigger) else {
                return None;
            };
            let _ = std::fs::remove_file(&trigger);
            let mut programs = Vec::new();
            self.capture = None;
            self.crop = None;
            self.snaps.clear();
            self.targets.clear();
            for word in round.split_whitespace() {
                if let Some(value) = word.strip_prefix("frames=") {
                    self.frames = value.parse().unwrap_or(self.frames);
                } else if let Some(value) = word.strip_prefix("capture=") {
                    self.capture = address(value);
                } else if let Some(value) = word.strip_prefix("crop=") {
                    let numbers: Vec<u32> =
                        value.split(',').filter_map(|n| n.parse().ok()).collect();
                    self.crop = numbers.try_into().ok();
                } else if let Some(value) = word.strip_prefix("snap=") {
                    self.snaps = value.split(',').filter_map(address).collect();
                } else if let Some(value) = word.strip_prefix("target=") {
                    self.targets = value.split(',').filter_map(address).collect();
                } else if let Some(program) = program(word, None) {
                    programs.push(program);
                }
            }
            if !programs.is_empty() || !self.targets.is_empty() {
                self.programs = programs;
                self.watched.clear();
            }
            self.left = self.frames;
            self.lines.push(format!(
                "== probe started after frame {presented}: programs {:x?}, capture {:x?}",
                self.programs, self.capture
            ));
            println!("[recomp-vulkan] probe started: {} frames", self.frames);
            self.flush();
            return None;
        }
        self.lines.push(format!("--- frame {presented} presented"));
        self.flush();
        self.producers = 0;
        for reread in &mut self.rereads {
            reread.frames -= 1;
        }
        self.rereads.retain(|reread| reread.frames > 0);
        self.left -= 1;
        if self.left == 0 {
            println!(
                "[recomp-vulkan] probe finished: {}",
                self.dir.join("probe.txt").display()
            );
        }
        Some((self.dir.join(format!("frame-{presented}")), self.capture))
    }

    /// Keep `bytes` (read at `address` for a probed draw) to compare with
    /// what is there at the draws after it.
    pub fn remember(&mut self, label: String, address: u64, bytes: Vec<u8>) {
        if self.by_program && !bytes.is_empty() {
            self.rereads.push(Reread {
                label,
                address,
                bytes,
                frames: 2,
            });
        }
    }

    /// Read the remembered constant buffers again (before draw `draw` of
    /// submission `submission`), and log the words that changed.
    pub fn reread(&mut self, memory: &dyn GpuMemory, draw: u64, submission: u64) {
        for reread in &mut self.rereads {
            let Some(now) = memory.read_gpu(reread.address, reread.bytes.len()) else {
                continue;
            };
            if now == reread.bytes {
                continue;
            }
            let changes: Vec<String> = now
                .as_chunks::<4>()
                .0
                .iter()
                .zip(reread.bytes.as_chunks::<4>().0)
                .enumerate()
                .filter(|(_, (new, old))| new != old)
                .map(|(index, (new, old))| {
                    format!(
                        "[{:#05x}] {:08x} -> {:08x}",
                        index * 4,
                        u32::from_le_bytes(*old),
                        u32::from_le_bytes(*new)
                    )
                })
                .collect();
            self.lines.push(format!(
                "  written after the draw read it: {} at {:#x}, seen before draw {draw} of submission {submission}: {}",
                reread.label,
                reread.address,
                changes.join(", ")
            ));
            reread.bytes = now;
        }
    }

    /// The rectangle and targets to copy at the probed draw being recorded.
    pub fn crops(&self) -> Option<([u32; 4], Vec<u64>)> {
        self.primary
            .then_some(self.crop)
            .flatten()
            .map(|crop| (crop, self.snaps.clone()))
    }

    pub fn rereading(&self) -> bool {
        !self.rereads.is_empty()
    }

    /// Save a probed draw's program (and its constant buffer 1) the first
    /// time, as `RECOMP_CAPTURE_SHADERS` does: `<address>-<stage>.bin`.
    pub fn dump_program(
        &mut self,
        memory: &dyn GpuMemory,
        address: u64,
        stage: &str,
        cb1: Option<(u64, u32)>,
    ) {
        if !self.current || !self.dumped.insert(address) {
            return;
        }
        if let Some(code) = memory.read_gpu(address, 0x10000) {
            let _ = std::fs::write(self.dir.join(format!("{address:x}-{stage}.bin")), code);
        }
        if let Some(bytes) =
            cb1.and_then(|(at, size)| memory.read_gpu(at, size.min(0x10000) as usize))
        {
            let _ = std::fs::write(self.dir.join(format!("{address:x}-{stage}.cb1")), bytes);
        }
    }

    fn flush(&mut self) {
        let path = self.dir.join("probe.txt");
        let written = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .and_then(|mut file| file.write_all((self.lines.join("\n") + "\n").as_bytes()));
        if let Err(error) = written {
            println!("[recomp-vulkan] probe: {}: {error}", path.display());
        }
        self.lines.clear();
    }
}

/// A constant buffer a probed program reads: where it is and its words
/// (as hex and as floats), up to what the program reads.
pub fn constant_buffer(
    memory: &dyn GpuMemory,
    stage: usize,
    slot: u32,
    buffer: Option<(u64, u32)>,
    extent: u32,
) -> (Vec<String>, Vec<u8>) {
    let Some((address, size)) = buffer else {
        return (
            vec![format!("  stage {stage} c{slot}: unbound")],
            Vec::new(),
        );
    };
    let len = size.min(extent).min(0x400) as usize & !3;
    let bytes = memory.read_gpu(address, len).unwrap_or_default();
    let mut lines = vec![format!(
        "  stage {stage} c{slot} at {address:#x}, {size} bytes bound, {extent} read"
    )];
    for (row, chunk) in bytes.chunks(32).enumerate() {
        let words: Vec<u32> = chunk
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        let hex: Vec<String> = words.iter().map(|word| format!("{word:08x}")).collect();
        let floats: Vec<String> = words
            .iter()
            .map(|word| format!("{:.6}", f32::from_bits(*word)))
            .collect();
        lines.push(format!(
            "    [{:#05x}] {} | {}",
            row * 32,
            hex.join(" "),
            floats.join(" ")
        ));
    }
    (lines, bytes)
}
