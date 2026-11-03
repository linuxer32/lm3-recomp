//! The in-game menu and frame rate counter: egui drawn over each frame the
//! window shows. F1 opens and closes the menu (Escape closes it), F11
//! toggles fullscreen. While the menu is open the title gets no input.
//!
//! The window thread turns SDL events into egui events (`Controls::push`)
//! and applies what the menu changes on the window (fullscreen); the GPU
//! worker runs the menu and draws it into the swapchain image after the
//! frame's blit. Settings of the console go to the services directly: docked
//! mode (`system::set_docked`, which tells the title as docking does) and the
//! frame rate (`vi::set_frame_rate`, which scripts hear of in
//! `on_frame_rate`). The resolution scale goes to the renderer
//! (`Controls::resolution`).

use ash::vk;
use egui_ash_renderer::allocator::DefaultAllocator;
use egui_ash_renderer::{DynamicRendering, Options, RenderMode, Renderer};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// What the window thread, the menu and the renderer share.
pub struct Controls {
    pub menu: AtomicBool,
    pub show_fps: AtomicBool,
    pub fullscreen: AtomicBool,
    /// The resolution scale, in halves (2: the game's own resolution, 4:
    /// twice as wide and tall); the renderer makes its large render targets
    /// at it.
    pub resolution: AtomicU32,
    events: Mutex<Vec<egui::Event>>,
}

impl Controls {
    /// An input event for the menu (dropped while it is closed).
    pub fn push(&self, event: egui::Event) {
        if self.menu.load(Ordering::Acquire) {
            self.events
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .push(event);
        }
    }

    fn take(&self) -> Vec<egui::Event> {
        std::mem::take(
            &mut *self
                .events
                .lock()
                .unwrap_or_else(|poison| poison.into_inner()),
        )
    }
}

/// The shared controls. `RECOMP_SHOW_FPS=1` starts with the counter shown,
/// `RECOMP_FULLSCREEN=1` in fullscreen, `RECOMP_MENU=1` with the menu open
/// and `RECOMP_RESOLUTION=2` at twice the game's resolution (1, 1.5, 2, 3
/// or 4); a script's `setting(...)` sets any of them.
pub fn controls() -> &'static Controls {
    static CONTROLS: OnceLock<Controls> = OnceLock::new();
    CONTROLS.get_or_init(|| {
        let on =
            |name: &str| std::env::var(name).is_ok_and(|value| value != "0" && !value.is_empty());
        Controls {
            menu: AtomicBool::new(on("RECOMP_MENU")),
            show_fps: AtomicBool::new(on("RECOMP_SHOW_FPS")),
            fullscreen: AtomicBool::new(on("RECOMP_FULLSCREEN")),
            resolution: AtomicU32::new(
                std::env::var("RECOMP_RESOLUTION")
                    .ok()
                    .and_then(|value| value.trim().trim_end_matches('x').parse::<f32>().ok())
                    .map_or(2, |scale| ((scale * 2.0).round() as u32).clamp(2, 8)),
            ),
            events: Mutex::new(Vec::new()),
        }
    })
}

/// The resolution scales the menu offers, in halves.
const RESOLUTIONS: [(u32, &str); 5] = [
    (2, "1x (the game's own)"),
    (3, "1.5x"),
    (4, "2x"),
    (6, "3x"),
    (8, "4x"),
];

/// The frame rates the menu offers: 0 is the title's own.
const FRAME_RATES: [(u32, &str); 5] = [
    (0, "30 (the game's own)"),
    (60, "60 (FPS mod)"),
    (120, "120 (FPS mod)"),
    (180, "180 (FPS mod)"),
    (240, "240 (FPS mod)"),
];

pub struct Overlay {
    context: egui::Context,
    renderer: Renderer<DefaultAllocator>,
    device: ash::Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    /// Views of the swapchain images drawn into.
    views: HashMap<vk::Image, vk::ImageView>,
    format: vk::Format,
    /// Textures egui let go of, freed once the frames that drew with them
    /// have run.
    frees: VecDeque<(u64, Vec<egui::TextureId>)>,
    frame: u64,
    started: Instant,
    /// Frames shown per second: since when, the count then, the rate.
    fps: (Instant, u64, f32),
}

impl Overlay {
    pub fn new(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        format: vk::Format,
    ) -> Result<Self, String> {
        let renderer = Renderer::with_default_allocator(
            instance,
            physical_device,
            device.clone(),
            RenderMode::DynamicRendering(DynamicRendering {
                color_attachment_format: format,
                depth_attachment_format: None,
                stencil_attachment_format: None,
            }),
            Options {
                // Two submissions may be in flight (`Gpu`'s slots), and a
                // third recording.
                in_flight_frames: 3,
                srgb_framebuffer: false,
                ..Options::default()
            },
        )
        .map_err(|error| format!("menu renderer: {error}"))?;
        let context = egui::Context::default();
        context.set_pixels_per_point(1.25);
        Ok(Self {
            context,
            renderer,
            device: device.clone(),
            queue,
            command_pool,
            views: HashMap::new(),
            format,
            frees: VecDeque::new(),
            frame: 0,
            started: Instant::now(),
            fps: (Instant::now(), recomp_services::vi::frames_shown(), 0.0),
        })
    }

    /// The swapchain was recreated: views of its old images go.
    pub fn forget_views(&mut self) {
        for (_, view) in self.views.drain() {
            unsafe { self.device.destroy_image_view(view, None) };
        }
    }

    /// Draw the menu and counter over `image` (in the present layout,
    /// `extent` big) into `command`; nothing when neither is shown.
    pub fn draw(
        &mut self,
        command: vk::CommandBuffer,
        image: vk::Image,
        extent: vk::Extent2D,
    ) -> Result<(), String> {
        self.frame += 1;
        let elapsed = self.fps.0.elapsed().as_secs_f32();
        if elapsed >= 0.5 {
            let frames = recomp_services::vi::frames_shown();
            self.fps = (
                Instant::now(),
                frames,
                (frames - self.fps.1) as f32 / elapsed,
            );
        }
        let controls = controls();
        let (menu, show_fps) = (
            controls.menu.load(Ordering::Acquire),
            controls.show_fps.load(Ordering::Acquire),
        );
        while self
            .frees
            .front()
            .is_some_and(|(frame, _)| frame + 3 <= self.frame)
        {
            let (_, ids) = self.frees.pop_front().expect("front exists");
            for id in ids {
                self.renderer
                    .free_texture(id)
                    .map_err(|error| format!("menu textures: {error}"))?;
            }
        }
        if !menu && !show_fps {
            return Ok(());
        }
        let ppp = self.context.pixels_per_point();
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(extent.width as f32, extent.height as f32) / ppp,
            )),
            time: Some(self.started.elapsed().as_secs_f64()),
            events: controls
                .take()
                .into_iter()
                .map(|event| scale_event(event, ppp))
                .collect(),
            ..Default::default()
        };
        let fps = self.fps.2;
        let output = self
            .context
            .run_ui(input, |ui| draw_ui(ui.ctx(), fps, menu, show_fps));
        for (id, deltas) in &output.textures_delta.set {
            for delta in deltas {
                self.renderer
                    .set_texture(self.queue, self.command_pool, *id, delta)
                    .map_err(|error| format!("menu textures: {error}"))?;
            }
        }
        self.frees.push_back((
            self.frame,
            output.textures_delta.free.iter().copied().collect(),
        ));
        let primitives = self
            .context
            .tessellate(output.shapes, output.pixels_per_point);
        let view = self.view(image)?;
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let barrier = |from: vk::ImageLayout, to: vk::ImageLayout| {
            vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
                .old_layout(from)
                .new_layout(to)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(range)
        };
        let attachment = vk::RenderingAttachmentInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE);
        let rendering = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent,
            })
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&attachment));
        unsafe {
            self.device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier(
                    crate::PRESENT_LAYOUT,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                )],
            );
            self.device.cmd_begin_rendering(command, &rendering);
        }
        let drawn = self
            .renderer
            .cmd_draw(command, extent, output.pixels_per_point, &primitives);
        unsafe {
            self.device.cmd_end_rendering(command);
            self.device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier(
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    crate::PRESENT_LAYOUT,
                )],
            );
        }
        drawn.map_err(|error| format!("menu draw: {error}"))
    }

    fn view(&mut self, image: vk::Image) -> Result<vk::ImageView, String> {
        if let Some(view) = self.views.get(&image) {
            return Ok(*view);
        }
        let view = unsafe {
            self.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(self.format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                None,
            )
        }
        .map_err(|error| format!("menu view: {error}"))?;
        self.views.insert(image, view);
        Ok(view)
    }
}

impl Drop for Overlay {
    fn drop(&mut self) {
        self.forget_views();
    }
}

/// Window coordinates (pixels) to egui points.
fn scale_event(event: egui::Event, ppp: f32) -> egui::Event {
    match event {
        egui::Event::PointerMoved(pos) => {
            egui::Event::PointerMoved((pos.to_vec2() / ppp).to_pos2())
        }
        egui::Event::PointerButton {
            pos,
            button,
            pressed,
            modifiers,
        } => egui::Event::PointerButton {
            pos: (pos.to_vec2() / ppp).to_pos2(),
            button,
            pressed,
            modifiers,
        },
        other => other,
    }
}

fn draw_ui(ctx: &egui::Context, fps: f32, menu: bool, show_fps: bool) {
    if show_fps {
        egui::Area::new(egui::Id::new("fps"))
            .anchor(egui::Align2::LEFT_TOP, [8.0, 8.0])
            .interactable(false)
            .show(ctx, |ui| {
                egui::Frame::NONE
                    .fill(egui::Color32::from_black_alpha(170))
                    .inner_margin(6.0)
                    .corner_radius(4.0)
                    .show(ui, |ui| {
                        ui.add(
                            egui::Label::new(
                                egui::RichText::new(format!("{fps:.0} FPS"))
                                    .monospace()
                                    .size(16.0)
                                    .color(egui::Color32::WHITE),
                            )
                            .wrap_mode(egui::TextWrapMode::Extend),
                        );
                    });
            });
    }
    if !menu {
        return;
    }
    let controls = controls();
    egui::Window::new("SwitchRecomp")
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ctx, |ui| {
            let mut fullscreen = controls.fullscreen.load(Ordering::Acquire);
            if ui.checkbox(&mut fullscreen, "Fullscreen (F11)").changed() {
                controls.fullscreen.store(fullscreen, Ordering::Release);
            }
            let mut docked = recomp_services::system::docked();
            if ui
                .checkbox(&mut docked, "Docked mode (1920x1080)")
                .on_hover_text("The game renders for a TV; off is handheld, 1280x720")
                .changed()
            {
                recomp_services::system::set_docked(docked);
            }
            let mut fps_choice = recomp_services::vi::frame_rate();
            let selected = FRAME_RATES
                .iter()
                .find(|(value, _)| *value == fps_choice)
                .map_or_else(
                    || format!("{fps_choice}"),
                    |(_, label)| (*label).to_string(),
                );
            egui::ComboBox::from_label("Frame rate")
                .selected_text(selected)
                .show_ui(ui, |ui| {
                    for (value, label) in FRAME_RATES {
                        if ui.selectable_value(&mut fps_choice, value, label).changed() {
                            recomp_services::vi::set_frame_rate(fps_choice);
                        }
                    }
                });
            let mut resolution = controls.resolution.load(Ordering::Acquire);
            let selected = RESOLUTIONS
                .iter()
                .find(|(value, _)| *value == resolution)
                .map_or_else(
                    || format!("{}x", resolution as f32 / 2.0),
                    |(_, label)| (*label).to_string(),
                );
            egui::ComboBox::from_label("Resolution")
                .selected_text(selected)
                .show_ui(ui, |ui| {
                    for (value, label) in RESOLUTIONS {
                        if ui.selectable_value(&mut resolution, value, label).changed() {
                            controls.resolution.store(resolution, Ordering::Release);
                        }
                    }
                })
                .response
                .on_hover_text(
                    "Renders at a multiple of the game's resolution (docked 1920x1080: 2x is \
                     3840x2160): sharper, on the GPU, at about the same frame rate",
                );
            let mut show = controls.show_fps.load(Ordering::Acquire);
            if ui.checkbox(&mut show, "Show FPS").changed() {
                controls.show_fps.store(show, Ordering::Release);
            }
            ui.separator();
            ui.label("F1 closes this menu");
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn menu_offers_the_native_frame_rates() {
        assert_eq!(FRAME_RATES.map(|(value, _)| value), [0, 60, 120, 180, 240]);
        assert_eq!(
            FRAME_RATES.map(|(_, label)| label),
            [
                "30 (the game's own)",
                "60 (FPS mod)",
                "120 (FPS mod)",
                "180 (FPS mod)",
                "240 (FPS mod)",
            ]
        );
    }
}
