//! `precompile-shaders`: build a GPU-specific Vulkan pipeline cache from the
//! portable SPIR-V manifest, without starting the game.
//!
//! The manifest is the `manifest.tsv` written by
//! `cargo run -p recomp-gfx --example translate_library -- ...`. Each `ok`
//! line names a `.spv` file and its stage. The tool creates one graphics
//! pipeline per shader with the game's descriptor/pipeline layout and a
//! compatible dummy companion stage. The driver compiles the real shader and
//! stores it in `pipeline-cache.bin`; the game then loads that cache.
//!
//! Usage:
//!   shader-precompile <manifest.tsv> <spv-dir> <out/pipeline-cache.bin>

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::slice;

use ash::vk;
use recomp_gfx::maxwell::translate::{RESCALE_BYTES, TEXTURE_BINDING_BASE};
use recomp_gfx::spirv::{self, op, storage};

const UNIFORM_BINDINGS: u32 = 18;
const TEXTURE_BINDINGS: u32 = 32;

fn main() {
    let mut args = std::env::args().skip(1);
    let manifest = PathBuf::from(args.next().expect("manifest.tsv"));
    let spv_dir = PathBuf::from(args.next().expect("SPIR-V directory"));
    let output = PathBuf::from(args.next().expect("output pipeline-cache.bin"));
    let mut start = 0usize;
    let mut count = usize::MAX;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--start" => start = args.next().unwrap().parse().unwrap(),
            "--count" => count = args.next().unwrap().parse().unwrap(),
            _ => {}
        }
    }

    let shaders = read_manifest(&manifest, &spv_dir);
    println!("{} shaders in manifest", shaders.len());
    if shaders.is_empty() {
        panic!("no shaders");
    }

    let entry = unsafe { ash::Entry::load() }.expect("load Vulkan");
    println!("vulkan loaded");
    let instance = create_instance(&entry);
    println!("instance created");
    let (physical, queue_family) = pick_device(&instance);
    println!("physical device picked");
    let device = create_device(&instance, physical, queue_family);
    println!("device created");
    let initial = std::fs::read(&output).unwrap_or_default();
    let cache = unsafe {
        device.create_pipeline_cache(
            &vk::PipelineCacheCreateInfo::default().initial_data(&initial),
            None,
        )
    }
    .expect("pipeline cache");
    println!("cache created");

    let set_layout = create_set_layout(&device);
    println!("set layout created");
    let set_layouts = [set_layout, set_layout];
    let push_constants = [vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        offset: 0,
        size: RESCALE_BYTES,
    }];
    let pipeline_layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&set_layouts)
                .push_constant_ranges(&push_constants),
            None,
        )
    }
    .expect("pipeline layout");
    println!("pipeline layout created");

    let empty_fragment = empty_fragment_spirv();
    println!("dummy shaders generated");
    let empty_fragment_module = unsafe { shader_module(&device, &empty_fragment) };
    let forced_vertex = std::env::var_os("SHADER_TEST_VERTEX")
        .and_then(|path| std::fs::read(path).ok())
        .map(|bytes| {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| u32::from_le_bytes(*chunk))
                .collect::<Vec<u32>>()
        });

    let mut created = 0usize;
    let mut failed = 0usize;
    for (index, shader) in shaders.iter().enumerate().skip(start).take(count) {
        let words = match read_spirv(&spv_dir.join(&shader.file)) {
            Ok(words) => words,
            Err(error) => {
                eprintln!("{}: {error}", shader.file);
                failed += 1;
                continue;
            }
        };
        if !bindings_fit(&words) {
            eprintln!(
                "{}: descriptor binding outside the game layout",
                shader.file
            );
            failed += 1;
            continue;
        }
        let module = unsafe { shader_module(&device, &words) };
        let mut generated_vertex = None;
        let (vertex, fragment, vertex_is_real) = match shader.stage {
            Stage::Vertex => (module, empty_fragment_module, true),
            Stage::Fragment => {
                if let Some(forced) = &forced_vertex {
                    let vertex = unsafe { shader_module(&device, forced) };
                    generated_vertex = Some(vertex);
                    (vertex, module, true)
                } else {
                    let (locations, flat) = fragment_inputs(&words);
                    let generic = generic_vertex_spirv(&locations, &flat);
                    let generic = unsafe { shader_module(&device, &generic) };
                    generated_vertex = Some(generic);
                    (generic, module, false)
                }
            }
        };
        let formats = output_formats(&words);
        let result = create_pipeline(
            &device,
            cache,
            pipeline_layout,
            vertex,
            fragment,
            vertex_is_real,
            formats,
        );
        unsafe {
            device.destroy_shader_module(module, None);
            if let Some(vertex) = generated_vertex {
                device.destroy_shader_module(vertex, None);
            }
        }
        match result {
            Ok(()) => {
                created += 1;
                if created.is_multiple_of(10) {
                    save_cache(&device, cache, &output);
                }
            }
            Err(error) => {
                failed += 1;
                if failed <= 30 {
                    eprintln!("{}: {error}", shader.file);
                }
            }
        }
        if (index + 1) % 100 == 0 || index + 1 == shaders.len() {
            println!(
                "{}/{}: {created} compiled, {failed} failed",
                index + 1,
                shaders.len()
            );
        }
    }

    unsafe {
        device.destroy_shader_module(empty_fragment_module, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_descriptor_set_layout(set_layout, None);
    }
    let data = save_cache(&device, cache, &output);
    println!(
        "{created} shaders compiled, {failed} failed; wrote {} bytes to {}",
        data.len(),
        output.display()
    );
    unsafe {
        device.destroy_pipeline_cache(cache, None);
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    Vertex,
    Fragment,
}

struct Shader {
    file: String,
    stage: Stage,
}

fn read_manifest(path: &Path, dir: &Path) -> Vec<Shader> {
    let text = std::fs::read_to_string(path).expect("read manifest");
    let mut shaders = Vec::new();
    for line in text.lines().skip(1) {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 11 || fields[4] != "ok" {
            continue;
        }
        let stage = match fields[2] {
            "Vertex" => Stage::Vertex,
            "Fragment" => Stage::Fragment,
            _ => continue,
        };
        let file = fields[10].trim().to_string();
        if file.is_empty() || !dir.join(&file).exists() {
            continue;
        }
        shaders.push(Shader { file, stage });
    }
    shaders
}

/// The backend's descriptor set layout: UBOs 0..18 and 32 textures at
/// 32..64. A shader naming anything else would make AMDVLK dereference a
/// missing layout entry, so those draws are skipped by the game too.
fn bindings_fit(words: &[u32]) -> bool {
    let mut at = 5usize;
    while at < words.len() {
        let header = words[at];
        let opcode = (header & 0xffff) as u16;
        let count = (header >> 16) as usize;
        if count == 0 || at + count > words.len() {
            return false;
        }
        if opcode == 71 && count >= 4 {
            // OpDecorate target decoration literal...
            let decoration = words[at + 2];
            let value = words[at + 3];
            if decoration == 33 {
                let fits = value < 18 || (32..64).contains(&value);
                if !fits {
                    return false;
                }
            }
            if decoration == 34 && value > 1 {
                return false;
            }
        }
        at += count;
    }
    true
}

fn read_spirv(path: &Path) -> Result<Vec<u32>, String> {
    let bytes = std::fs::read(path).map_err(|error| error.to_string())?;
    if bytes.len() % 4 != 0 {
        return Err("SPIR-V size is not a multiple of 4".into());
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .copied()
        .map(u32::from_le_bytes)
        .collect())
}

fn create_instance(entry: &ash::Entry) -> ash::Instance {
    let app = vk::ApplicationInfo::default()
        .application_name(c"shader-precompile")
        .api_version(vk::make_api_version(0, 1, 3, 0));
    let info = vk::InstanceCreateInfo::default().application_info(&app);
    unsafe { entry.create_instance(&info, None) }.expect("Vulkan instance")
}

fn pick_device(instance: &ash::Instance) -> (vk::PhysicalDevice, u32) {
    let devices = unsafe { instance.enumerate_physical_devices() }.expect("physical devices");
    for physical in devices {
        let properties = unsafe { instance.get_physical_device_properties(physical) };
        if properties.api_version < vk::make_api_version(0, 1, 3, 0) {
            continue;
        }
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        for (index, family) in families.iter().enumerate() {
            if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                println!(
                    "using {} (Vulkan {}.{}.{})",
                    properties
                        .device_name_as_c_str()
                        .unwrap_or(c"?")
                        .to_string_lossy(),
                    vk::api_version_major(properties.api_version),
                    vk::api_version_minor(properties.api_version),
                    vk::api_version_patch(properties.api_version)
                );
                return (physical, index as u32);
            }
        }
    }
    panic!("no Vulkan 1.3 graphics device");
}

fn create_device(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    queue_family: u32,
) -> ash::Device {
    let priorities = [1.0f32];
    let queues = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family)
        .queue_priorities(&priorities)];
    let mut features = vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true);
    let info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queues)
        .push_next(&mut features);
    unsafe { instance.create_device(physical, &info, None) }.expect("Vulkan device")
}

fn create_set_layout(device: &ash::Device) -> vk::DescriptorSetLayout {
    let mut bindings = Vec::new();
    for binding in 0..UNIFORM_BINDINGS {
        bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::ALL_GRAPHICS),
        );
    }
    for index in 0..TEXTURE_BINDINGS {
        bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(TEXTURE_BINDING_BASE + index)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::ALL_GRAPHICS),
        );
    }
    unsafe {
        device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )
    }
    .expect("descriptor set layout")
}

fn save_cache(device: &ash::Device, cache: vk::PipelineCache, output: &Path) -> Vec<u8> {
    let data = unsafe { device.get_pipeline_cache_data(cache) }.expect("cache data");
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(output, &data).expect("write pipeline cache");
    data
}

unsafe fn shader_module(device: &ash::Device, words: &[u32]) -> vk::ShaderModule {
    device
        .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(words), None)
        .expect("shader module")
}

/// Pipeline color formats must match the fragment outputs' component types;
/// AMDVLK can crash instead of returning an error when an integer output is
/// paired with a float attachment.
fn output_formats(words: &[u32]) -> [vk::Format; 8] {
    let mut formats = [vk::Format::B8G8R8A8_UNORM; 8];
    let mut var_type: HashMap<u32, u32> = HashMap::new();
    let mut pointee: HashMap<u32, u32> = HashMap::new();
    let mut int_signed: HashMap<u32, bool> = HashMap::new();
    let mut float_type: HashMap<u32, bool> = HashMap::new();
    let mut vector_component: HashMap<u32, u32> = HashMap::new();
    let mut location: HashMap<u32, u32> = HashMap::new();
    let mut at = 5usize;
    while at < words.len() {
        let header = words[at];
        let opcode = (header & 0xffff) as u16;
        let count = (header >> 16) as usize;
        if count == 0 || at + count > words.len() {
            break;
        }
        match opcode {
            59 if count >= 4 => {
                var_type.insert(words[at + 2], words[at + 1]);
            }
            32 if count >= 4 => {
                pointee.insert(words[at + 1], words[at + 3]);
            }
            21 if count >= 4 => {
                int_signed.insert(words[at + 1], words[at + 3] != 0);
            }
            22 if count >= 3 => {
                float_type.insert(words[at + 1], true);
            }
            23 if count >= 4 => {
                vector_component.insert(words[at + 1], words[at + 2]);
            }
            71 if count >= 4 && words[at + 2] == 30 => {
                location.insert(words[at + 1], words[at + 3]);
            }
            _ => {}
        }
        at += count;
    }
    for (var, ty) in &var_type {
        let Some(&slot) = location.get(var) else {
            continue;
        };
        if slot >= 8 {
            continue;
        }
        let Some(&base) = pointee.get(ty) else {
            continue;
        };
        let base = vector_component.get(&base).copied().unwrap_or(base);
        if let Some(&signed) = int_signed.get(&base) {
            formats[slot as usize] = if signed {
                vk::Format::R32G32B32A32_SINT
            } else {
                vk::Format::R32G32B32A32_UINT
            };
        } else if float_type.contains_key(&base) {
            formats[slot as usize] = vk::Format::B8G8R8A8_UNORM;
        }
    }
    formats
}

fn create_pipeline(
    device: &ash::Device,
    cache: vk::PipelineCache,
    layout: vk::PipelineLayout,
    vertex: vk::ShaderModule,
    fragment: vk::ShaderModule,
    vertex_is_real: bool,
    formats: [vk::Format; 8],
) -> Result<(), String> {
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment)
            .name(c"main"),
    ];
    let binding = vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(16 * 32)
        .input_rate(vk::VertexInputRate::VERTEX);
    let attributes: Vec<_> = (0..32)
        .map(|location| {
            vk::VertexInputAttributeDescription::default()
                .location(location)
                .binding(0)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .offset(location * 16)
        })
        .collect();
    let vertex_input = if vertex_is_real {
        vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(slice::from_ref(&binding))
            .vertex_attribute_descriptions(&attributes)
    } else {
        vk::PipelineVertexInputStateCreateInfo::default()
    };
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let attachments = [vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(false)
        .color_write_mask(vk::ColorComponentFlags::RGBA); 1];
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(false)
        .depth_write_enable(false)
        .depth_compare_op(vk::CompareOp::ALWAYS);

    let one_format = [formats[0]];
    let mut rendering = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&one_format)
        .depth_attachment_format(vk::Format::UNDEFINED);
    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .color_blend_state(&blend)
        .depth_stencil_state(&depth_stencil)
        .dynamic_state(&dynamic)
        .layout(layout)
        .push_next(&mut rendering);
    match unsafe { device.create_graphics_pipelines(cache, slice::from_ref(&info), None) } {
        Ok(pipelines) => {
            unsafe { device.destroy_pipeline(pipelines[0], None) };
            Ok(())
        }
        Err((_, result)) => Err(result.to_string()),
    }
}

fn empty_fragment_spirv() -> Vec<u32> {
    let mut builder = spirv::Builder::new();
    let main = builder.begin_main();
    builder.entry_point(spirv::model::FRAGMENT, main, &[]);
    builder.execution_mode(main, spirv::mode::ORIGIN_UPPER_LEFT);
    builder.inst_void(op::RETURN, &[]);
    builder.finish()
}

fn generic_vertex_spirv(locations: &[u32], flat: &HashSet<u32>) -> Vec<u32> {
    let mut builder = spirv::Builder::new();
    let float = builder.f32();
    let vec4 = builder.vector(float, 4);
    let zero = builder.const_f32(0.0);
    let one = builder.const_f32(1.0);
    let value = builder.const_composite(vec4, &[zero, zero, zero, one]);
    let main = builder.begin_main();
    let mut interface = Vec::new();
    for location in locations {
        let var = builder.global_variable(vec4, storage::OUTPUT);
        builder.decorate(var, spirv::decoration::LOCATION, &[*location]);
        if flat.contains(location) {
            builder.decorate(var, spirv::decoration::FLAT, &[]);
        }
        builder.store(var, value);
        interface.push(var);
    }
    let position = builder.global_variable(vec4, storage::OUTPUT);
    builder.decorate(
        position,
        spirv::decoration::BUILT_IN,
        &[spirv::built_in::POSITION],
    );
    builder.store(position, value);
    interface.push(position);
    builder.entry_point(spirv::model::VERTEX, main, &interface);
    builder.inst_void(op::RETURN, &[]);
    builder.finish()
}

/// Generic input locations a fragment shader reads, and which of them are
/// flat. The dummy vertex shader emits exactly these varyings, so it never
/// exceeds the component limits of a large real shader.
fn fragment_inputs(words: &[u32]) -> (Vec<u32>, HashSet<u32>) {
    let mut storage_of: HashMap<u32, u32> = HashMap::new();
    let mut location: HashMap<u32, u32> = HashMap::new();
    let mut flat: HashSet<u32> = HashSet::new();
    let mut at = 5usize;
    while at < words.len() {
        let header = words[at];
        let opcode = (header & 0xffff) as u16;
        let count = (header >> 16) as usize;
        if count == 0 || at + count > words.len() {
            break;
        }
        match opcode {
            59 if count >= 4 => {
                storage_of.insert(words[at + 2], words[at + 3]);
            }
            71 if count >= 3 => {
                let target = words[at + 1];
                let decoration = words[at + 2];
                if decoration == 30 {
                    location.insert(target, words[at + 3]);
                } else if decoration == 14 {
                    flat.insert(target);
                }
            }
            _ => {}
        }
        at += count;
    }
    let mut locations = Vec::new();
    let mut flat_locations = HashSet::new();
    for (var, class) in storage_of {
        if class != storage::INPUT {
            continue;
        }
        let Some(slot) = location.get(&var) else {
            continue;
        };
        locations.push(*slot);
        if flat.contains(&var) {
            flat_locations.insert(*slot);
        }
    }
    locations.sort_unstable();
    locations.dedup();
    (locations, flat_locations)
}
