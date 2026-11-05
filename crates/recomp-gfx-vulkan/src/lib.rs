//! A host Vulkan renderer behind an SDL3 window.
//!
//! `recomp-gfx` and `recomp-services` stay display-free: this crate builds a
//! Vulkan device through `ash` and a window through `sdl3`, and registers
//! itself with `recomp_services::gpu::register_renderer` when the title asks
//! for it (`RECOMP_VULKAN=1`). The GPU worker thread drives the renderer, so
//! Vulkan objects are created once and then live on that thread.
//!
//! Draws are recorded into one command buffer per submission by
//! [`gpu::Gpu`] (translated shaders, render targets, textures). When the
//! submission finishes, the largest target it drew is blitted to the window
//! and presented, and the renderer waits for the GPU before the next one. The
//! console backend still sees every call, for its statistics and dumps.

mod bcn;
mod decode;
mod formats;
mod gpu;
mod input;
mod overlay;
mod paths;
mod probe;
mod shader_pack;
mod texture;

use std::ffi::CString;

use ash::khr::{surface, swapchain};
use ash::vk;
use ash::{Device, Entry, Instance};

use recomp_gfx::maxwell::compute::ComputeDispatch;
use recomp_gfx::maxwell::render::{
    ClearColorCall, ClearDepthStencilCall, DrawCall, GpuMemory, Renderer,
};
use recomp_gfx::maxwell::twod::Blit;

use gpu::Gpu;
use recomp_gfx::GraphicsError;
use recomp_services::gpu::{console_renderer, register_renderer};

/// How long the window may go without a new frame before `refresh` shows
/// the last one again.
const REFRESH_AFTER: std::time::Duration = std::time::Duration::from_millis(250);

/// How long (ns) a present waits for a swapchain image. An X11 window that
/// is minimized keeps its size but stops giving images back; waiting
/// forever there hung the renderer, and the restored window stayed empty.
const ACQUIRE_TIMEOUT: u64 = 50_000_000;

/// The window size the backend opens with: the Switch's handheld
/// resolution.
const WINDOW_WIDTH: u32 = 1280;
/// The window height the backend opens with.
const WINDOW_HEIGHT: u32 = 720;

/// The color shown before the guest draws anything.
const INITIAL_CLEAR: [f32; 4] = [0.05, 0.05, 0.08, 1.0];

/// `VK_IMAGE_LAYOUT_PRESENT_KHR`. ash 0.38 does not name the promoted
/// swapchain layout constant, so it is the raw enum value here.
pub(crate) const PRESENT_LAYOUT: vk::ImageLayout = vk::ImageLayout::from_raw(1_000_001_001);

/// The renderer waiting for a GPU channel. The slot hands the window to the
/// first channel only; further channels get the recording backend.
static RENDERER_SLOT: std::sync::Mutex<Option<VulkanRenderer>> = std::sync::Mutex::new(None);

fn to_error(error: impl std::fmt::Display) -> String {
    format!("{error}")
}

/// A failed Vulkan library load, with the Windows DLL rule spelled out:
/// `ash` and SDL both load `vulkan-1.dll` through the normal DLL search
/// path, whose first entry is the executable's directory.
fn vulkan_load_error(error: impl std::fmt::Display) -> String {
    #[cfg(windows)]
    {
        format!("{error} (is vulkan-1.dll next to the executable, or installed by the GPU driver?)")
    }
    #[cfg(not(windows))]
    {
        format!("{error}")
    }
}

/// What the window thread hands the renderer: the Vulkan instance and the
/// window's surface.
type WindowSurface = (Entry, Instance, vk::SurfaceKHR);

/// Opens the window and Vulkan device and registers the backend. Called from
/// the target's `main`, before the runtime starts any GPU channel.
pub fn install() -> Result<(), String> {
    // SDL would turn SIGINT and SIGTERM into quit events nothing reads,
    // leaving the title unkillable by `timeout` and Ctrl-C.
    sdl3::hint::set("SDL_NO_SIGNAL_HANDLERS", "1");
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("recomp-window".to_string())
        .spawn(move || window_thread(sender))
        .map_err(to_error)?;
    let (entry, instance, surface_khr) = receiver.recv().map_err(to_error)??;
    let renderer = VulkanRenderer::new(entry, instance, surface_khr)?;
    let mut slot = RENDERER_SLOT.lock().expect("renderer slot poisoned");
    *slot = Some(renderer);
    register_renderer(|| {
        let mut slot = RENDERER_SLOT.lock().expect("renderer slot poisoned");
        slot.take()
            .map(|renderer| Box::new(renderer) as Box<dyn Renderer + Send>)
    });
    Ok(())
}

/// The window's thread: SDL is initialized and the window and its surface
/// made here, and the window's events handled here for the whole run (SDL
/// wants one thread for all of it; the main thread runs the guest). Each
/// pass hands the guest the gamepads' and keyboard's state through `hid`.
/// Closing the window ends the title.
///
/// On Windows this is also SDL's "main thread": SDL records the thread that
/// initializes video as its video and event thread, and Win32 messages are
/// pumped on the thread that created the window. macOS is the platform that
/// insists on the process's first thread; the title's `main` would have to
/// move off it there, which this backend does not do.
fn window_thread(sender: std::sync::mpsc::Sender<Result<WindowSurface, String>>) {
    let setup = || -> Result<_, String> {
        let sdl = sdl3::init().map_err(to_error)?;
        let video = sdl.video().map_err(to_error)?;
        video
            .vulkan_load_library_default()
            .map_err(vulkan_load_error)?;
        let window =
            sdl3::video::WindowBuilder::new(&video, "SwitchRecomp", WINDOW_WIDTH, WINDOW_HEIGHT)
                .vulkan()
                .resizable()
                .build()
                .map_err(to_error)?;
        let instance_extensions = window.vulkan_instance_extensions().map_err(to_error)?;
        let entry = unsafe { Entry::load() }.map_err(vulkan_load_error)?;
        let instance = create_instance(&entry, &instance_extensions)?;
        let surface_khr =
            unsafe { window.vulkan_create_surface(instance.handle()) }.map_err(to_error)?;
        Ok((sdl, video, window, (entry, instance, surface_khr)))
    };
    // The window and SDL stay alive on this thread: the surface needs them.
    let (sdl, _video, mut window, surface) = match setup() {
        Ok(parts) => parts,
        Err(error) => {
            let _ = sender.send(Err(error));
            return;
        }
    };
    let _ = sender.send(Ok(surface));
    // Audio out plays here; the stream (and its device) lives as long as
    // this thread.
    let spec = sdl3::audio::AudioSpec {
        freq: Some(48_000),
        channels: Some(2),
        format: Some(sdl3::audio::AudioFormat::S16LE),
    };
    let audio_stream = sdl.audio().ok().and_then(|audio| {
        audio
            .default_playback_device()
            .open_device_stream(Some(&spec))
            .ok()
    });
    match &audio_stream {
        Some(stream) if stream.resume().is_ok() => {
            recomp_services::audio::register_audio_sink(Box::new(SdlAudio(stream.raw())));
        }
        _ => println!("[recomp-vulkan] no audio output"),
    }
    let gamepads = sdl.gamepad().ok();
    let Ok(mut events) = sdl.event_pump() else {
        return;
    };
    let mut pads: Vec<sdl3::gamepad::Gamepad> = Vec::new();
    let controls = overlay::controls();
    let mut fullscreen = false;
    'run: loop {
        for event in events.poll_iter() {
            match event {
                // Leave the loop instead of `process::exit`: the renderer's
                // destructor writes the final pipeline cache before the
                // process goes away.
                sdl3::event::Event::Quit { .. } => break 'run,
                // F1: the menu, F11: fullscreen, Escape closes the menu.
                sdl3::event::Event::KeyDown {
                    keycode: Some(key),
                    repeat: false,
                    ..
                } => match key {
                    sdl3::keyboard::Keycode::F1 => {
                        controls
                            .menu
                            .fetch_xor(true, std::sync::atomic::Ordering::AcqRel);
                    }
                    sdl3::keyboard::Keycode::F11 => {
                        controls
                            .fullscreen
                            .fetch_xor(true, std::sync::atomic::Ordering::AcqRel);
                    }
                    sdl3::keyboard::Keycode::Escape => {
                        controls
                            .menu
                            .store(false, std::sync::atomic::Ordering::Release);
                    }
                    _ => {}
                },
                sdl3::event::Event::MouseMotion { x, y, .. } => {
                    controls.push(egui::Event::PointerMoved(egui::pos2(x, y)));
                }
                sdl3::event::Event::MouseButtonDown {
                    mouse_btn, x, y, ..
                }
                | sdl3::event::Event::MouseButtonUp {
                    mouse_btn, x, y, ..
                } => {
                    let button = match mouse_btn {
                        sdl3::mouse::MouseButton::Right => egui::PointerButton::Secondary,
                        sdl3::mouse::MouseButton::Middle => egui::PointerButton::Middle,
                        _ => egui::PointerButton::Primary,
                    };
                    controls.push(egui::Event::PointerButton {
                        pos: egui::pos2(x, y),
                        button,
                        pressed: matches!(event, sdl3::event::Event::MouseButtonDown { .. }),
                        modifiers: egui::Modifiers::default(),
                    });
                }
                sdl3::event::Event::MouseWheel { x, y, .. } => {
                    controls.push(egui::Event::MouseWheel {
                        unit: egui::MouseWheelUnit::Line,
                        delta: egui::vec2(x, y),
                        phase: egui::TouchPhase::Move,
                        modifiers: egui::Modifiers::default(),
                    });
                }
                sdl3::event::Event::GamepadAdded { which, .. } => {
                    let known = pads.iter().any(|pad| pad.id().ok() == Some(which));
                    let opened = gamepads
                        .as_ref()
                        .filter(|_| !known)
                        .and_then(|subsystem| subsystem.open(which).ok());
                    if let Some(pad) = opened {
                        println!(
                            "[recomp-vulkan] controller connected: {}",
                            pad.name().unwrap_or_default()
                        );
                        pads.push(pad);
                    }
                }
                sdl3::event::Event::JoyDeviceAdded { which, .. } => {
                    // Joysticks SDL has no gamepad mapping for are not read.
                    if let Some(subsystem) = gamepads.as_ref().filter(|s| !s.is_gamepad(which)) {
                        println!(
                            "[recomp-vulkan] joystick {} has no gamepad mapping, not read",
                            subsystem.name_for_id(which).unwrap_or_default()
                        );
                    }
                }
                sdl3::event::Event::GamepadRemoved { which, .. } => {
                    pads.retain(|pad| pad.id().ok() != Some(which));
                }
                _ => {}
            }
        }
        // The menu asks for fullscreen (or F11 did).
        let wanted = controls
            .fullscreen
            .load(std::sync::atomic::Ordering::Acquire);
        if wanted != fullscreen {
            fullscreen = wanted;
            if let Err(error) = window.set_fullscreen(fullscreen) {
                println!("[recomp-vulkan] fullscreen: {error}");
            }
        }
        let mut state = input::keyboard_state(&events.keyboard_state());
        for pad in &pads {
            state = input::combine(state, input::gamepad_state(pad));
        }
        // While the menu is open the title gets no input.
        if controls.menu.load(std::sync::atomic::Ordering::Acquire) {
            state = recomp_services::hid::ControllerState::default();
        }
        recomp_services::hid::set_controller(state);
        std::thread::sleep(std::time::Duration::from_millis(4));
    }
}

/// An SDL audio stream audio out plays into. SDL's audio streams take data
/// from any thread; the window thread keeps the stream open.
struct SdlAudio(*mut sdl3::sys::audio::SDL_AudioStream);

// SAFETY: SDL audio streams lock internally; the stream outlives the run.
unsafe impl Send for SdlAudio {}
unsafe impl Sync for SdlAudio {}

impl recomp_services::audio::AudioSink for SdlAudio {
    fn play(&self, samples: &[i16]) {
        let bytes = std::mem::size_of_val(samples) as i32;
        // SAFETY: the stream is open; SDL copies the samples.
        unsafe {
            sdl3::sys::audio::SDL_PutAudioStreamData(self.0, samples.as_ptr().cast(), bytes);
        }
    }

    fn queued(&self) -> usize {
        // SAFETY: as for `play`.
        let bytes = unsafe { sdl3::sys::audio::SDL_GetAudioStreamQueued(self.0) };
        bytes.max(0) as usize / 2
    }

    fn clear(&self) {
        // SAFETY: as for `play`.
        unsafe {
            sdl3::sys::audio::SDL_ClearAudioStream(self.0);
        }
    }
}

/// The transform baked into the swapchain images. The guest's frame is
/// rendered upright in the window's current orientation, so the image is not
/// pre-rotated: `IDENTITY` has the presentation engine rotate it onto the
/// panel (a 90-degree phone's panel is portrait even when the app is
/// landscape). `currentTransform` here claims the image is already
/// pre-rotated, which shows the frame sideways — `RECOMP_PRE_TRANSFORM=current`
/// reproduces that for comparison.
fn image_transform(capabilities: &vk::SurfaceCapabilitiesKHR) -> vk::SurfaceTransformFlagsKHR {
    match std::env::var("RECOMP_PRE_TRANSFORM").as_deref() {
        Ok("current") => capabilities.current_transform,
        _ => vk::SurfaceTransformFlagsKHR::IDENTITY,
    }
}

fn create_instance(entry: &Entry, extensions: &[String]) -> Result<Instance, String> {
    let names: Vec<CString> = extensions
        .iter()
        .map(|name| CString::new(name.as_str()).expect("extension name has no interior null"))
        .collect();
    let pointers: Vec<*const std::ffi::c_char> = names.iter().map(|name| name.as_ptr()).collect();
    let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
    let info = vk::InstanceCreateInfo::default()
        .application_info(&app)
        .enabled_extension_names(&pointers);
    unsafe { entry.create_instance(&info, None) }.map_err(to_error)
}

/// The swapchain's images, refreshed whenever the swapchain is recreated.
struct Frame {
    images: Vec<vk::Image>,
    extent: vk::Extent2D,
    format: vk::Format,
}

/// The swapchain needs to be written by transfers: frames are blitted in.
const SWAPCHAIN_USAGE: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
    vk::ImageUsageFlags::COLOR_ATTACHMENT.as_raw()
        | vk::ImageUsageFlags::TRANSFER_DST.as_raw()
        | vk::ImageUsageFlags::TRANSFER_SRC.as_raw(),
);

/// The Vulkan renderer: one device, one swapchain, and the [`Gpu`] that
/// records the guest's draws.
pub struct VulkanRenderer {
    _entry: Entry,
    // The instance and queue family are kept alive with the device; the swapchain
    // extension object borrows nothing after construction.
    _instance: Instance,
    surface: surface::Instance,
    surface_khr: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    _queue_family: u32,
    device: Device,
    swapchain: swapchain::Device,
    queue: vk::Queue,
    swapchain_khr: vk::SwapchainKHR,
    present_mode: vk::PresentModeKHR,
    command_pool: vk::CommandPool,
    /// Per recording slot (`Gpu::slot`; `gpu::SLOTS` submissions may be in
    /// flight): signalled when an acquired image is ready to be written.
    acquired: [vk::Semaphore; gpu::SLOTS],
    /// Per recording slot: signalled when a frame is ready to present.
    semaphore: [vk::Semaphore; gpu::SLOTS],
    frame: Frame,
    gpu: Gpu,
    /// The F1 menu and frame rate counter (`overlay`).
    overlay: Option<overlay::Overlay>,
    inner: Box<dyn Renderer + Send>,
    last_clear: Option<[f32; 4]>,
    presented: u64,
    submissions: u64,
    /// Every queue submission, flushes included (flush mode's stamps).
    queue_submits: u64,
    /// The title queues frames to VI: the window shows those (`present`).
    queued_frames: bool,
    /// The last frame shown and when, for `refresh`.
    last_frame: Option<(gpu::TargetKey, Option<(u32, u32)>)>,
    last_shown: std::time::Instant,
    /// `RECOMP_SLOW_FRAMES=<ms>`: frames presented longer than this after
    /// the one before are logged (a hitch while saving, say, lines up with
    /// the log lines around it).
    slow_frame_ms: Option<f64>,
    last_presented: std::time::Instant,
    /// Draws recorded since the last `RECOMP_VULKAN_FLUSH_DRAWS` flush.
    pending_draws: u32,
    /// When an acquire last found no free image: frames are not shown for a
    /// while after, so a minimized window does not cost a wait per frame.
    acquire_failed: Option<std::time::Instant>,
    /// The Vulkan device was lost (Adreno hangs or resets): rendering is off
    /// from then on, but the guest keeps running (sound and input still
    /// work). Reported once, with the draws that came last.
    device_lost: bool,
    /// `RECOMP_VULKAN_DUMP=<dir>`: write a trace of the draws of submission
    /// `RECOMP_VULKAN_DUMP_AT` (default 300; the first after it with at
    /// least 100 draws) to the directory, then every
    /// render target, one per following submission.
    dump: Option<(std::path::PathBuf, u64)>,
    dump_queue: Vec<gpu::TargetKey>,
    /// `RECOMP_VULKAN_CAPTURE=<dir>`: save every `RECOMP_VULKAN_CAPTURE_EVERY`th
    /// (default 300) presented frame as `<dir>/frame-<n>/<target>.ppm`, as
    /// the renderer drew it (no desktop screenshot needed), from frame
    /// `RECOMP_VULKAN_CAPTURE_FROM` on (default 0).
    capture: Option<(std::path::PathBuf, u64)>,
    capture_from: u64,
}

impl VulkanRenderer {
    /// Send the commands recorded so far (showing nothing) when the
    /// submission's stream or descriptor pool is nearly used up.
    fn send_if_full(&mut self) -> Result<(), GraphicsError> {
        if self.gpu.nearly_full() {
            self.submit(None)?;
        }
        Ok(())
    }

    fn new(entry: Entry, instance: Instance, surface_khr: vk::SurfaceKHR) -> Result<Self, String> {
        let surface = surface::Instance::new(&entry, &instance);

        let (physical_device, queue_family) = {
            let mut devices = unsafe { instance.enumerate_physical_devices() }.map_err(to_error)?;
            // Prefer real GPUs to software rasterizers.
            devices.sort_by_key(|device| {
                match unsafe { instance.get_physical_device_properties(*device) }.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                    vk::PhysicalDeviceType::CPU => 3,
                    _ => 2,
                }
            });
            let found = devices.iter().copied().find_map(|device| {
                let families =
                    unsafe { instance.get_physical_device_queue_family_properties(device) };
                families.iter().enumerate().find_map(|(index, family)| {
                    if !family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                        return None;
                    }
                    unsafe {
                        surface
                            .get_physical_device_surface_support(device, index as u32, surface_khr)
                            .ok()
                    }
                    .filter(|supported| *supported)
                    .map(|_| (device, index as u32))
                })
            });
            found.ok_or("no Vulkan device supports the window surface")?
        };

        let properties = unsafe { instance.get_physical_device_properties(physical_device) };
        // Vulkan 1.3 has dynamic rendering; older Adreno drivers (LM3's
        // Snapdragon 870 phone reports 1.1) use cached render passes.
        let dynamic_rendering = properties.api_version >= vk::API_VERSION_1_3;
        let supported = unsafe { instance.get_physical_device_features(physical_device) };
        // Robust access makes constant buffer reads past the bound size
        // zero, as the guest's do; the rest are used when available.
        let features = vk::PhysicalDeviceFeatures::default()
            .robust_buffer_access(supported.robust_buffer_access == vk::TRUE)
            .independent_blend(supported.independent_blend == vk::TRUE)
            .texture_compression_bc(supported.texture_compression_bc == vk::TRUE)
            .shader_image_gather_extended(supported.shader_image_gather_extended == vk::TRUE)
            .shader_clip_distance(supported.shader_clip_distance == vk::TRUE)
            .depth_clamp(supported.depth_clamp == vk::TRUE)
            .depth_bias_clamp(supported.depth_bias_clamp == vk::TRUE)
            .sampler_anisotropy(supported.sampler_anisotropy == vk::TRUE);
        let mut features13 = vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true);
        // Depth clip control lets pipelines keep the guest's -1..1 depth.
        let extensions = unsafe { instance.enumerate_device_extension_properties(physical_device) }
            .map_err(to_error)?;
        let depth_clip_control = extensions.iter().any(|extension| {
            extension.extension_name_as_c_str() == Ok(ash::ext::depth_clip_control::NAME)
        });
        let mut depth_clip_features =
            vk::PhysicalDeviceDepthClipControlFeaturesEXT::default().depth_clip_control(true);
        let mut extension_names = vec![swapchain::NAME.as_ptr()];
        if depth_clip_control {
            extension_names.push(ash::ext::depth_clip_control::NAME.as_ptr());
        }
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&[1.0]);
        let mut device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&extension_names)
            .enabled_features(&features);
        if dynamic_rendering {
            device_info = device_info.push_next(&mut features13);
        }
        if depth_clip_control {
            device_info = device_info.push_next(&mut depth_clip_features);
        }
        let device = unsafe { instance.create_device(physical_device, &device_info, None) }
            .map_err(to_error)?;
        let swapchain = swapchain::Device::new(&instance, &device);
        let queue = unsafe { device.get_device_queue(queue_family, 0) };

        let formats =
            unsafe { surface.get_physical_device_surface_formats(physical_device, surface_khr) }
                .map_err(to_error)?;
        let capabilities = unsafe {
            surface.get_physical_device_surface_capabilities(physical_device, surface_khr)
        }
        .map_err(to_error)?;
        // A UNORM swapchain: the guest's frame is already sRGB encoded, and
        // the blit copies it unconverted (sRGB targets are stored as UNORM).
        let format = formats
            .iter()
            .find(|format| {
                format.format == vk::Format::B8G8R8A8_UNORM
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .or_else(|| formats.first())
            .ok_or("the surface offers no format")?;

        // Mailbox presents without tearing and without blocking the GPU
        // worker on vsync (a stalled worker crashes LM3; see `gpu.rs`).
        let modes = unsafe {
            surface.get_physical_device_surface_present_modes(physical_device, surface_khr)
        }
        .map_err(to_error)?;
        // `RECOMP_PRESENT_MODE=fifo|mailbox|immediate` asks for one mode
        // (when the surface has it), for comparing drivers.
        let requested = match std::env::var("RECOMP_PRESENT_MODE").ok().as_deref() {
            Some("fifo") => Some(vk::PresentModeKHR::FIFO),
            Some("mailbox") => Some(vk::PresentModeKHR::MAILBOX),
            Some("immediate") => Some(vk::PresentModeKHR::IMMEDIATE),
            _ => None,
        };
        let present_mode = requested
            .into_iter()
            .chain([vk::PresentModeKHR::MAILBOX, vk::PresentModeKHR::IMMEDIATE])
            .find(|mode| modes.contains(mode))
            .unwrap_or(vk::PresentModeKHR::FIFO);
        let min_images = capabilities.min_image_count.max(3);
        let image_count = if capabilities.max_image_count == 0 {
            min_images
        } else {
            min_images.min(capabilities.max_image_count)
        };
        let extent = if capabilities.current_extent.width == u32::MAX {
            vk::Extent2D {
                width: WINDOW_WIDTH,
                height: WINDOW_HEIGHT,
            }
        } else {
            capabilities.current_extent
        };
        // Android reports the surface in the display's natural orientation
        // and a transform the presentation engine applies; the log says
        // which, because a mismatch shows the frame sideways or stretched.
        println!(
            "[recomp-vulkan] surface {}x{} transform {:?} (image {:?}), image extent {}x{}..{}x{}, {} format(s)",
            extent.width,
            extent.height,
            capabilities.current_transform,
            image_transform(&capabilities),
            capabilities.min_image_extent.width,
            capabilities.min_image_extent.height,
            capabilities.max_image_extent.width,
            capabilities.max_image_extent.height,
            formats.len()
        );

        let command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .map_err(to_error)?;
        let new_semaphore = || {
            unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
                .map_err(to_error)
        };
        let mut semaphore = [vk::Semaphore::null(); gpu::SLOTS];
        let mut acquired = [vk::Semaphore::null(); gpu::SLOTS];
        for (semaphore, acquired) in semaphore.iter_mut().zip(&mut acquired) {
            *semaphore = new_semaphore()?;
            *acquired = new_semaphore()?;
        }

        let swapchain_khr = unsafe {
            swapchain.create_swapchain(
                &vk::SwapchainCreateInfoKHR::default()
                    .surface(surface_khr)
                    .min_image_count(image_count)
                    .image_format(format.format)
                    .image_color_space(format.color_space)
                    .image_extent(extent)
                    .image_array_layers(1)
                    .image_usage(SWAPCHAIN_USAGE)
                    .pre_transform(image_transform(&capabilities))
                    .present_mode(present_mode)
                    .clipped(true),
                None,
            )
        }
        .map_err(to_error)?;
        let images = unsafe { swapchain.get_swapchain_images(swapchain_khr) }.map_err(to_error)?;

        let frame = Frame {
            images,
            extent,
            format: format.format,
        };
        let gpu = Gpu::new(
            &instance,
            physical_device,
            &device,
            command_pool,
            depth_clip_control,
            supported.depth_clamp == vk::TRUE,
            supported.depth_bias_clamp == vk::TRUE,
            dynamic_rendering,
        )?;
        // The menu uses dynamic rendering; on Vulkan 1.1/1.2 it is skipped
        // (the title still runs, without the F1 overlay).
        let overlay = if dynamic_rendering {
            overlay::Overlay::new(
                &instance,
                physical_device,
                &device,
                queue,
                command_pool,
                format.format,
            )
            .map_err(|error| println!("[recomp-vulkan] no menu: {error}"))
            .ok()
        } else {
            None
        };
        println!(
            "[recomp-vulkan] rendering on {}, presenting with {present_mode:?}",
            properties
                .device_name_as_c_str()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default()
        );

        Ok(Self {
            _entry: entry,
            _instance: instance,
            surface,
            surface_khr,
            physical_device,
            _queue_family: queue_family,
            device,
            swapchain,
            queue,
            swapchain_khr,
            present_mode,
            command_pool,
            acquired,
            semaphore,
            frame,
            gpu,
            overlay,
            inner: console_renderer(),
            last_clear: None,
            presented: 0,
            submissions: 0,
            queue_submits: 0,
            queued_frames: false,
            last_frame: None,
            last_shown: std::time::Instant::now(),
            slow_frame_ms: std::env::var("RECOMP_SLOW_FRAMES")
                .ok()
                .and_then(|value| value.parse().ok()),
            last_presented: std::time::Instant::now(),
            pending_draws: 0,
            acquire_failed: None,
            device_lost: false,
            dump: std::env::var_os("RECOMP_VULKAN_DUMP").map(|dir| {
                let at = std::env::var("RECOMP_VULKAN_DUMP_AT")
                    .ok()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(300);
                (paths::resolve(dir), at)
            }),
            dump_queue: Vec::new(),
            capture: std::env::var_os("RECOMP_VULKAN_CAPTURE").map(|dir| {
                let every = std::env::var("RECOMP_VULKAN_CAPTURE_EVERY")
                    .ok()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(300u64)
                    .max(1);
                (paths::resolve(dir), every)
            }),
            capture_from: std::env::var("RECOMP_VULKAN_CAPTURE_FROM")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(0),
        })
    }

    /// The device is gone (Adreno hangs or resets on a bad draw): say so
    /// once, with the draws from just before, and stop submitting.
    fn note_device_lost(&mut self) {
        if self.device_lost {
            return;
        }
        self.device_lost = true;
        println!(
            "[recomp-vulkan] device lost (the GPU reset; rendering stops, the guest keeps running): {}\n the last draws were:{}",
            self.gpu.cache_sizes(),
            self.gpu.recent_draws()
        );
        // `RECOMP_VULKAN_LOST_LOG=<file>`: every draw and operation kept
        // (thousands in flush mode, with when each ran), for matching a
        // kernel GPU fault report to the operation that caused it.
        if let Some(path) = std::env::var_os("RECOMP_VULKAN_LOST_LOG") {
            match std::fs::write(&path, self.gpu.all_recent()) {
                Ok(()) => println!(
                    "[recomp-vulkan] recent operations written to {}",
                    std::path::Path::new(&path).display()
                ),
                Err(error) => {
                    println!("[recomp-vulkan] could not write the recent operations: {error}")
                }
            }
        }
    }

    /// Submit the commands a guest submission recorded, and present what it
    /// drew. Submissions that drew nothing are submitted without presenting
    /// (until the first frame, the window shows the last clear color). The
    /// renderer waits for the GPU here: the next submission reuses the
    /// stream buffer and descriptor pool.
    fn submit(
        &mut self,
        frame: Option<(gpu::TargetKey, Option<(u32, u32)>)>,
    ) -> Result<(), GraphicsError> {
        // Nothing can be submitted again after a device loss; the guest keeps
        // running rather than spinning on a dead device.
        if self.device_lost {
            return Ok(());
        }
        // Once the title queues frames to VI, only those are shown;
        // before that, the largest target a submission drew stands in.
        let frame = frame.or_else(|| {
            (!self.queued_frames)
                .then_some(self.gpu.presentable)
                .flatten()
                .map(|key| (key, None))
        });
        let show = frame.is_some() || (self.presented == 0 && !self.queued_frames);
        if !show && !self.gpu.recording {
            return Ok(());
        }
        // A minimized window has no area: submit the work, show nothing.
        let backing_off = self
            .acquire_failed
            .is_some_and(|failed| failed.elapsed() < REFRESH_AFTER);
        let show = show && !backing_off && self.window_visible();
        // In flush mode the frame's own submission holds only the present:
        // whatever was recorded since the last flush is run (and checked)
        // first.
        if self.gpu.recording {
            self.flush_operation(|| "end of frame".to_string())?;
        }
        self.gpu.begin().map_err(failed("begin"))?;
        let index = if show { self.acquire()? } else { None };
        if show {
            self.acquire_failed = index.is_none().then(std::time::Instant::now);
        }
        if index.is_some() {
            self.last_shown = std::time::Instant::now();
        }
        if let Some(index) = index {
            let image = self.frame.images[index as usize];
            self.gpu.show_in(
                image,
                self.frame.extent,
                self.last_clear.unwrap_or(INITIAL_CLEAR),
                frame,
            );
            if let Some(overlay) = self.overlay.as_mut() {
                if let Err(error) = overlay.draw(self.gpu.command, image, self.frame.extent) {
                    println!("[recomp-vulkan] {error}; the menu is off");
                    self.overlay = None;
                }
            }
        }
        // `RECOMP_VULKAN_PROBE` saves each frame whose draws it logged.
        let probe_round = frame.and_then(|_| {
            let presented = self.presented + 1;
            self.gpu.probe.as_mut()?.frame(presented)
        });
        let probed = probe_round
            .as_ref()
            .and_then(|(dir, capture)| match capture {
                Some(address) => Some((self.gpu.target_at_any(&[*address])?, dir.clone())),
                None => frame.map(|(key, _)| (key, dir.clone())),
            });
        // Crops wait for the frame they belong to (LM3 records a frame in
        // two submissions).
        let probe_crops = if probe_round.is_some() {
            self.gpu.take_probe_crops()
        } else {
            Vec::new()
        };
        // At most one readback per submission: a dumped target, or the
        // frame being presented.
        let wanted = match (self.dump_queue.pop(), &self.dump, &self.capture) {
            (Some(key), Some((dir, _)), _) => Some((key, dir.clone())),
            _ if probed.is_some() => probed,
            (_, _, Some((dir, every)))
                if index.is_some()
                    && self.presented + 1 >= self.capture_from
                    && (self.presented + 1).is_multiple_of(*every) =>
            {
                frame.map(|(key, _)| (key, dir.join(format!("frame-{}", self.presented + 1))))
            }
            _ => None,
        };
        // `RECOMP_VULKAN_CAPTURE_WINDOW`: captures show the window's image
        // (menu and black bars included) instead of the guest's frame.
        let window_capture = std::env::var_os("RECOMP_VULKAN_CAPTURE_WINDOW").is_some()
            && self.dump_queue.is_empty();
        let readback = wanted.and_then(|(key, dir)| {
            let recorded = match index {
                Some(index) if window_capture => self
                    .gpu
                    .record_window_readback(self.frame.images[index as usize], self.frame.extent),
                _ => self.gpu.record_readback(key),
            };
            recorded
                .map_err(failed("readback"))
                .ok()
                .map(|readback| (readback, dir))
        });
        self.gpu.end().map_err(failed("end"))?;

        let command = self.gpu.command;
        let slot = self.gpu.slot();
        let wait_stage = vk::PipelineStageFlags::ALL_COMMANDS;
        let mut submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command));
        if index.is_some() {
            submit = submit
                .wait_semaphores(std::slice::from_ref(&self.acquired[slot]))
                .wait_dst_stage_mask(std::slice::from_ref(&wait_stage))
                .signal_semaphores(std::slice::from_ref(&self.semaphore[slot]));
        }
        let submitted = unsafe {
            self.device
                .queue_submit(self.queue, std::slice::from_ref(&submit), self.gpu.fence)
        };
        if let Err(error) = submitted {
            println!("[recomp-vulkan] submit failed: {error}");
            if error == vk::Result::ERROR_DEVICE_LOST {
                self.note_device_lost();
            }
            return Err(GraphicsError::Unsupported);
        }
        // In flush mode the frame's own submission (the present blit, after
        // the window's image is acquired) is timed too.
        if flush_every_draws() != 0 && self.presented >= flush_from() {
            let started = std::time::Instant::now();
            let waited = unsafe {
                self.device
                    .wait_for_fences(&[self.gpu.fence], true, 5_000_000_000)
            };
            let took = started.elapsed();
            if took >= std::time::Duration::from_millis(50) || waited.is_err() {
                println!(
                    "[recomp-vulkan] frame submission (image {index:?}) took {} ms: {waited:?}",
                    took.as_millis()
                );
            }
            let finished = monotonic_seconds();
            self.gpu
                .note_recent(format!("frame submission (image {index:?})"));
            self.gpu
                .stamp_recent(self.queue_submits, finished - took.as_secs_f64(), finished);
        }
        // The worker goes on recording into the next slot while the GPU
        // runs this one.
        self.gpu.submitted();

        let mut result = Ok(());
        if let Some(index) = index {
            let present = vk::PresentInfoKHR::default()
                .wait_semaphores(std::slice::from_ref(&self.semaphore[slot]))
                .swapchains(std::slice::from_ref(&self.swapchain_khr))
                .image_indices(std::slice::from_ref(&index));
            result = match unsafe { self.swapchain.queue_present(self.queue, &present) } {
                Ok(suboptimal) => {
                    self.presented += 1;
                    let presented = self.presented;
                    if let Some(limit) = self.slow_frame_ms {
                        let now = std::time::Instant::now();
                        let took = (now - self.last_presented).as_secs_f64() * 1000.0;
                        if took > limit {
                            println!("[recomp-vulkan] frame {presented} took {took:.1} ms");
                        }
                        self.last_presented = now;
                    }
                    if presented == 1 || presented.is_multiple_of(128) {
                        println!(
                            "[recomp-vulkan] presented {presented} frames ({} draws, {} skipped)",
                            self.gpu.draws, self.gpu.skipped
                        );
                        if std::env::var_os("RECOMP_GPU_STATS").is_some() {
                            let [shaders, vertices, descriptors, commands] = self.gpu.take_phases();
                            println!(
                                "[recomp-vulkan] draw recording per frame: {:.1} ms shaders, {:.1} ms vertices, {:.1} ms descriptors and pipeline, {:.1} ms commands; {:.1} ms waiting for the GPU",
                                shaders / 128.0,
                                vertices / 128.0,
                                descriptors / 128.0,
                                commands / 128.0,
                                self.gpu.take_gpu_waited() / 128.0
                            );
                            let [instances, barriers, cleared_in, cleared_out] = self
                                .gpu
                                .take_pass_counts()
                                .map(|count| count as f64 / 128.0);
                            println!(
                                "[recomp-vulkan] per frame: {instances:.1} rendering instances, {barriers:.1} barriers, {cleared_in:.1} clears in instances and {cleared_out:.1} outside"
                            );
                            let [vertex, index, cached] = self.gpu.take_copied();
                            println!(
                                "[recomp-vulkan] copied per frame: {:.1} MB vertices ({:.1} MB more from the vertex cache), {:.1} MB indices",
                                vertex as f64 / 128.0 / 1e6,
                                cached as f64 / 128.0 / 1e6,
                                index as f64 / 128.0 / 1e6
                            );
                        }
                    }
                    if suboptimal && self.surface_resized() {
                        self.recreate_swapchain()
                    } else {
                        Ok(())
                    }
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.recreate_swapchain(),
                Err(error) => {
                    println!("[recomp-vulkan] present failed: {error}");
                    if error == vk::Result::ERROR_DEVICE_LOST {
                        self.note_device_lost();
                    }
                    Err(GraphicsError::Unsupported)
                }
            };
        }

        self.submissions += 1;
        self.queue_submits += 1;
        if let (false, Some((dir, _))) = (probe_crops.is_empty(), &probe_round) {
            self.gpu.wait_submitted().map_err(failed("fence wait"))?;
            self.gpu.write_probe_crops(probe_crops, dir);
        }
        if let Some((readback, dir)) = readback {
            // A readback is read once the submission that copied it has run.
            self.gpu.wait_submitted().map_err(failed("fence wait"))?;
            self.gpu.write_readback(readback, &dir);
            if self.dump.is_some() && self.dump_queue.is_empty() {
                println!(
                    "[recomp-vulkan] dumped the render targets to {}",
                    dir.display()
                );
            }
        }
        self.dump_if_due();
        result
    }

    /// Trace the submission before the dump, then write the dump.
    fn dump_if_due(&mut self) {
        let Some((dir, at)) = self.dump.clone() else {
            return;
        };
        if self.submissions + 1 == at {
            self.gpu.trace = Some(Vec::new());
        }
        if self.submissions != at {
            return;
        }
        let mut lines = self.gpu.trace.take().unwrap_or_default();
        // A submission with few draws (LM3 alternates them with ones that
        // only copy or compute) says little: trace the next one instead.
        if lines.len() < 100 {
            self.dump = Some((dir, at + 1));
            self.gpu.trace = Some(Vec::new());
            return;
        }
        lines.push(format!("presented target: {:x?}", self.gpu.presentable));
        self.dump_queue = self.gpu.target_keys();
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(dir.join("trace.txt"), lines.join("\n") + "\n");
        println!(
            "[recomp-vulkan] traced submission {at} to {}",
            dir.display()
        );
    }

    /// Whether the window has an area to present to (not minimized).
    fn window_visible(&self) -> bool {
        unsafe {
            self.surface
                .get_physical_device_surface_capabilities(self.physical_device, self.surface_khr)
        }
        .map_or(true, |capabilities| {
            capabilities.current_extent.width != 0 && capabilities.current_extent.height != 0
        })
    }

    /// The next swapchain image index, or `None` when no image is free
    /// within `ACQUIRE_TIMEOUT` (the frame is then not shown); recreates the
    /// swapchain once if the surface changed.
    fn acquire(&mut self) -> Result<Option<u32>, GraphicsError> {
        let first = unsafe {
            self.swapchain.acquire_next_image(
                self.swapchain_khr,
                ACQUIRE_TIMEOUT,
                self.acquired[self.gpu.slot()],
                vk::Fence::default(),
            )
        };
        match first {
            Ok((index, _suboptimal)) => Ok(Some(index)),
            Err(vk::Result::TIMEOUT | vk::Result::NOT_READY) => Ok(None),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                self.recreate_swapchain()?;
                let second = unsafe {
                    self.swapchain.acquire_next_image(
                        self.swapchain_khr,
                        ACQUIRE_TIMEOUT,
                        self.acquired[self.gpu.slot()],
                        vk::Fence::default(),
                    )
                };
                match second {
                    Ok((index, _suboptimal)) => Ok(Some(index)),
                    Err(vk::Result::TIMEOUT | vk::Result::NOT_READY) => Ok(None),
                    Err(error) => {
                        println!("[recomp-vulkan] acquire after recreate failed: {error}");
                        Err(GraphicsError::Unsupported)
                    }
                }
            }
            Err(error) => {
                println!("[recomp-vulkan] acquire failed: {error}");
                Err(GraphicsError::Unsupported)
            }
        }
    }

    /// Whether the surface's size differs from the swapchain's: the reason
    /// a suboptimal present calls for a new swapchain. Android also calls
    /// every present suboptimal when the swapchain's image transform is not
    /// the display's rotation, which this renderer leaves to the compositor
    /// (`image_transform`); recreating the swapchain for that rebuilt it,
    /// after waiting for the whole GPU, on every frame.
    fn surface_resized(&self) -> bool {
        let capabilities = unsafe {
            self.surface
                .get_physical_device_surface_capabilities(self.physical_device, self.surface_khr)
        };
        capabilities.is_ok_and(|capabilities| {
            let extent = capabilities.current_extent;
            extent.width != u32::MAX && extent != self.frame.extent
        })
    }

    /// Recreate the swapchain (resize, display mode change) and everything
    /// that references its images.
    fn recreate_swapchain(&mut self) -> Result<(), GraphicsError> {
        unsafe { self.device.device_wait_idle() }.ok();
        let old = self.swapchain_khr;
        let format = self.frame.format;
        let capabilities = unsafe {
            self.surface
                .get_physical_device_surface_capabilities(self.physical_device, self.surface_khr)
        }
        .map_err(|error| {
            println!("[recomp-vulkan] capabilities failed: {error}");
            GraphicsError::Unsupported
        })?;
        let extent = if capabilities.current_extent.width == u32::MAX {
            self.frame.extent
        } else {
            capabilities.current_extent
        };
        let mut min_images = capabilities.min_image_count.max(3);
        if capabilities.max_image_count != 0 {
            min_images = min_images.min(capabilities.max_image_count);
        }
        let swapchain_khr = unsafe {
            self.swapchain.create_swapchain(
                &vk::SwapchainCreateInfoKHR::default()
                    .surface(self.surface_khr)
                    .min_image_count(min_images)
                    .image_format(format)
                    .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
                    .image_extent(extent)
                    .image_array_layers(1)
                    .image_usage(SWAPCHAIN_USAGE)
                    .pre_transform(image_transform(&capabilities))
                    .present_mode(self.present_mode)
                    .clipped(true)
                    .old_swapchain(old),
                None,
            )
        }
        .map_err(|error| {
            println!("[recomp-vulkan] swapchain recreate failed: {error}");
            GraphicsError::Unsupported
        })?;
        unsafe { self.swapchain.destroy_swapchain(old, None) };
        self.swapchain_khr = swapchain_khr;
        if let Some(overlay) = self.overlay.as_mut() {
            overlay.forget_views();
        }
        let images =
            unsafe { self.swapchain.get_swapchain_images(swapchain_khr) }.map_err(|error| {
                println!("[recomp-vulkan] image list failed: {error}");
                GraphicsError::Unsupported
            })?;
        self.frame = Frame {
            images,
            extent,
            format,
        };
        Ok(())
    }
}

impl VulkanRenderer {
    /// End and submit what has been recorded so far, and wait for the GPU
    /// (`RECOMP_VULKAN_FLUSH_DRAWS`).
    /// In `RECOMP_VULKAN_FLUSH_DRAWS` mode, name a non-draw operation in the
    /// recent draws and submit it alone, like each draw: a hang in a clear
    /// or copy is then reported there too.
    fn flush_operation(&mut self, operation: impl FnOnce() -> String) -> Result<(), GraphicsError> {
        if flush_every_draws() == 0 || self.presented < flush_from() || !self.gpu.recording {
            return Ok(());
        }
        self.gpu.note_recent(operation());
        self.flush_recorded()
    }

    fn flush_recorded(&mut self) -> Result<(), GraphicsError> {
        self.gpu.end().map_err(failed("end"))?;
        let submitted_at = monotonic_seconds();
        self.queue_submits += 1;
        let command = self.gpu.command;
        let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command));
        if let Err(error) = unsafe {
            self.device
                .queue_submit(self.queue, std::slice::from_ref(&submit), self.gpu.fence)
        } {
            println!("[recomp-vulkan] flushed submit failed: {error}");
            if error == vk::Result::ERROR_DEVICE_LOST {
                self.note_device_lost();
            }
            return Err(GraphicsError::Unsupported);
        }
        // Five seconds, not forever: a hung GPU must come back as a timeout
        // so the draws that were in flight are reported.
        let timeout = if flush_every_draws() != 0 {
            5_000_000_000
        } else {
            u64::MAX
        };
        let started = std::time::Instant::now();
        let waited = unsafe {
            self.device
                .wait_for_fences(&[self.gpu.fence], true, timeout)
        };
        // When it ran, on the clock the kernel's log uses (CLOCK_MONOTONIC):
        // a GPU fault the kernel reports can be matched to the operation.
        if flush_every_draws() != 0 {
            self.gpu
                .stamp_recent(self.queue_submits, submitted_at, monotonic_seconds());
            // The driver may skip a faulting command and go on: keep the
            // history on disk as well (`<file>.<submission>`, every 10000).
            if self.queue_submits.is_multiple_of(10_000) {
                if let Some(path) = std::env::var_os("RECOMP_VULKAN_LOST_LOG") {
                    let mut path = path;
                    path.push(format!(".{}", self.queue_submits));
                    let _ = std::fs::write(&path, self.gpu.all_recent());
                }
            }
        }
        // A submission the GPU spends long on (Adreno kills one at about
        // two seconds and still signals its fence) is named here.
        let took = started.elapsed();
        if flush_every_draws() != 0 && took >= std::time::Duration::from_millis(50) {
            println!(
                "[recomp-vulkan] flushed submission took {} ms: {}",
                took.as_millis(),
                self.gpu.last_recent()
            );
            // A draw that long: what it sampled too (the draw trace).
            if took >= std::time::Duration::from_millis(500) {
                if let Some(line) = self.gpu.trace.as_ref().and_then(|trace| trace.last()) {
                    println!("[recomp-vulkan]   {line}");
                }
            }
        }
        // Flush mode traces each draw (with its textures) for the line above.
        if flush_every_draws() != 0 && self.dump.is_none() {
            self.gpu.trace = Some(Vec::new());
        }
        if let Err(error) = waited {
            // The GPU is gone (a hang the driver reset): report it here, at
            // the draws that were in flight, not at the next present.
            println!("[recomp-vulkan] fence wait failed: {error}");
            if error == vk::Result::TIMEOUT || error == vk::Result::ERROR_DEVICE_LOST {
                self.note_device_lost();
            }
            return Err(GraphicsError::Unsupported);
        }
        // Adreno signals the fence of a submission that faulted and reports
        // the loss only at the next submit, after more work was recorded:
        // ask now, so the last operation named is the one that faulted.
        if flush_every_draws() != 0 {
            if let Err(error) = unsafe { self.device.device_wait_idle() } {
                println!("[recomp-vulkan] device check after a flush failed: {error}");
                if error == vk::Result::ERROR_DEVICE_LOST {
                    self.note_device_lost();
                }
                return Err(GraphicsError::Unsupported);
            }
        }
        // The fence stays signaled: `submitted` marks this slot in flight,
        // and its next `wait` returns at once and resets it. Resetting it
        // here left that wait on a fence nothing would signal again.
        self.gpu.submitted();
        Ok(())
    }
}

impl Renderer for VulkanRenderer {
    fn draw(&mut self, memory: &dyn GpuMemory, draw: &DrawCall) -> Result<(), GraphicsError> {
        if self.device_lost {
            return Ok(());
        }
        self.inner.draw(memory, draw)?;
        self.send_if_full()?;
        self.gpu.draw(memory, draw).map_err(failed("draw"))?;
        // `RECOMP_VULKAN_FLUSH_DRAWS=<n>` submits every n draws and waits:
        // a GPU hang is then reported at the draws that caused it instead of
        // at the end of a whole frame. Slow, for diagnosis only.
        let every = flush_every_draws();
        if every != 0 && self.presented >= flush_from() {
            self.pending_draws += 1;
            if self.pending_draws >= every {
                self.pending_draws = 0;
                self.flush_recorded()?;
            }
        }
        Ok(())
    }

    fn clear_color(
        &mut self,
        memory: &dyn GpuMemory,
        clear: &ClearColorCall,
    ) -> Result<(), GraphicsError> {
        if clear.index == 0 {
            self.last_clear = Some(clear.color);
        }
        self.inner.clear_color(memory, clear)?;
        self.gpu.clear(memory, clear).map_err(failed("clear"))?;
        self.flush_operation(|| {
            let target = &clear.target;
            format!(
                "clear color {} [{:#x}/{:#x}/{}/{}] layers {}+{}",
                clear.index,
                target.address,
                target.format,
                target.width_or_stride,
                target.height,
                clear.layer,
                clear.layer_count
            )
        })
    }

    fn clear_depth_stencil(
        &mut self,
        memory: &dyn GpuMemory,
        clear: &ClearDepthStencilCall,
    ) -> Result<(), GraphicsError> {
        self.inner.clear_depth_stencil(memory, clear)?;
        self.gpu
            .clear_depth(memory, clear)
            .map_err(failed("depth clear"))?;
        self.flush_operation(|| "clear depth/stencil".to_string())
    }

    fn dispatch(
        &mut self,
        memory: &dyn GpuMemory,
        dispatch: &ComputeDispatch,
    ) -> Result<(), GraphicsError> {
        self.inner.dispatch(memory, dispatch)?;
        self.send_if_full()?;
        self.gpu
            .dispatch(memory, dispatch)
            .map_err(failed("dispatch"))?;
        self.flush_operation(|| format!("dispatch {:#x}", dispatch.program_address))
    }

    fn blit(&mut self, memory: &dyn GpuMemory, blit: &Blit) -> Result<bool, GraphicsError> {
        self.inner.blit(memory, blit)?;
        self.send_if_full()?;
        let done = self.gpu.blit(memory, blit).map_err(failed("2D copy"))?;
        self.flush_operation(|| {
            format!(
                "2D copy {:#x}/{:#x}/{}x{} -> {:#x}/{:#x}/{}x{} rect {:?}",
                blit.src.address,
                blit.src.format,
                blit.src.width,
                blit.src.height,
                blit.dst.address,
                blit.dst.format,
                blit.dst.width,
                blit.dst.height,
                blit.dst_rect
            )
        })?;
        Ok(done)
    }

    fn finish(&mut self) -> Result<(), GraphicsError> {
        self.inner.finish()?;
        self.submit(None)
    }

    fn present(
        &mut self,
        addresses: &[u64],
        crop: Option<(u32, u32)>,
    ) -> Result<(), GraphicsError> {
        self.queued_frames = true;
        match self.gpu.target_at_any(addresses) {
            Some(key) => {
                self.last_frame = Some((key, crop));
                self.submit(Some((key, crop)))
            }
            // Not drawn by the host (yet): nothing to show.
            None => Ok(()),
        }
    }

    fn refresh(&mut self) -> Result<(), GraphicsError> {
        // While the title is busy and queues no frames, keep the window
        // showing its last one (a minimized and restored window would
        // otherwise stay empty).
        match self.last_frame {
            Some(frame) if self.last_shown.elapsed() >= REFRESH_AFTER => self.submit(Some(frame)),
            _ => Ok(()),
        }
    }
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            self.overlay = None;
            if std::env::var_os("RECOMP_GPU_STATS").is_some() {
                println!("[recomp-vulkan] cache at exit: {}", self.gpu.cache_sizes());
            }
            self.gpu.destroy();
            self.swapchain.destroy_swapchain(self.swapchain_khr, None);
            for semaphore in self.semaphore.iter().chain(&self.acquired) {
                self.device.destroy_semaphore(*semaphore, None);
            }
            self.device.destroy_command_pool(self.command_pool, None);
            self.surface.destroy_surface(self.surface_khr, None);
        }
    }
}

/// CLOCK_MONOTONIC in seconds, the clock of the kernel log's timestamps.
#[cfg(unix)]
fn monotonic_seconds() -> f64 {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `now` is a valid timespec to write.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut now) };
    now.tv_sec as f64 + now.tv_nsec as f64 / 1e9
}

/// Windows has no `clock_gettime`: use a process-local monotonic clock. The
/// values are only used for the renderer's submission/hitch log and its
/// differences, so the epoch does not matter.
#[cfg(windows)]
fn monotonic_seconds() -> f64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    START
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_secs_f64()
}

/// Report a failed Vulkan step and turn it into the renderer's error.
/// `RECOMP_VULKAN_FLUSH_FROM=<frame>`: only start flushing after that many
/// presented frames, so a run is fast until just before the frame that hangs.
fn flush_from() -> u64 {
    static FROM: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *FROM.get_or_init(|| {
        std::env::var("RECOMP_VULKAN_FLUSH_FROM")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0)
    })
}

/// `RECOMP_VULKAN_FLUSH_DRAWS=n`: submit every n draws (0: never).
fn flush_every_draws() -> u32 {
    static EVERY: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *EVERY.get_or_init(|| {
        std::env::var("RECOMP_VULKAN_FLUSH_DRAWS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0)
    })
}

fn failed<E: std::fmt::Display>(step: &'static str) -> impl Fn(E) -> GraphicsError {
    move |error| {
        println!("[recomp-vulkan] {step} failed: {error}");
        GraphicsError::Unsupported
    }
}
