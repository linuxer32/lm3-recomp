//! A small SPIR-V module writer: the types, constants, decorations and
//! instructions the shader translator emits, for a module with one entry
//! point. Types and constants are deduplicated; function-scope variables are
//! collected separately and placed at the top of the entry block, where
//! SPIR-V requires them.
//!
//! The module targets SPIR-V 1.0 (Vulkan 1.0) with the GLSL.std.450 extended
//! instruction set.

use std::collections::{BTreeSet, HashMap};

/// A result id.
pub type Id = u32;

/// Opcodes used by the translator.
pub mod op {
    pub const EXT_INST_IMPORT: u16 = 11;
    pub const EXT_INST: u16 = 12;
    pub const MEMORY_MODEL: u16 = 14;
    pub const ENTRY_POINT: u16 = 15;
    pub const EXECUTION_MODE: u16 = 16;
    pub const CAPABILITY: u16 = 17;
    pub const TYPE_VOID: u16 = 19;
    pub const TYPE_BOOL: u16 = 20;
    pub const TYPE_INT: u16 = 21;
    pub const TYPE_FLOAT: u16 = 22;
    pub const TYPE_VECTOR: u16 = 23;
    pub const TYPE_IMAGE: u16 = 25;
    pub const TYPE_SAMPLED_IMAGE: u16 = 27;
    pub const TYPE_ARRAY: u16 = 28;
    pub const TYPE_STRUCT: u16 = 30;
    pub const TYPE_POINTER: u16 = 32;
    pub const TYPE_FUNCTION: u16 = 33;
    pub const CONSTANT_TRUE: u16 = 41;
    pub const CONSTANT_FALSE: u16 = 42;
    pub const CONSTANT: u16 = 43;
    pub const CONSTANT_COMPOSITE: u16 = 44;
    pub const FUNCTION: u16 = 54;
    pub const FUNCTION_END: u16 = 56;
    pub const VARIABLE: u16 = 59;
    pub const LOAD: u16 = 61;
    pub const STORE: u16 = 62;
    pub const CONTROL_BARRIER: u16 = 224;
    pub const ACCESS_CHAIN: u16 = 65;
    pub const DECORATE: u16 = 71;
    pub const MEMBER_DECORATE: u16 = 72;
    pub const COMPOSITE_CONSTRUCT: u16 = 80;
    pub const COMPOSITE_EXTRACT: u16 = 81;
    pub const IMAGE_SAMPLE_IMPLICIT_LOD: u16 = 87;
    pub const IMAGE_SAMPLE_EXPLICIT_LOD: u16 = 88;
    pub const IMAGE_FETCH: u16 = 95;
    pub const IMAGE_SAMPLE_DREF_IMPLICIT_LOD: u16 = 89;
    pub const IMAGE_SAMPLE_DREF_EXPLICIT_LOD: u16 = 90;
    pub const IMAGE_GATHER: u16 = 96;
    pub const IMAGE_DREF_GATHER: u16 = 97;
    pub const IMAGE: u16 = 100;
    pub const IMAGE_QUERY_SIZE_LOD: u16 = 103;
    pub const IMAGE_QUERY_LEVELS: u16 = 106;
    pub const CONVERT_F_TO_U: u16 = 109;
    pub const CONVERT_F_TO_S: u16 = 110;
    pub const CONVERT_S_TO_F: u16 = 111;
    pub const CONVERT_U_TO_F: u16 = 112;
    pub const BITCAST: u16 = 124;
    pub const S_NEGATE: u16 = 126;
    pub const F_NEGATE: u16 = 127;
    pub const I_ADD: u16 = 128;
    pub const I_MUL: u16 = 132;
    pub const F_ADD: u16 = 129;
    pub const I_SUB: u16 = 130;
    pub const F_MUL: u16 = 133;
    pub const F_SUB: u16 = 131;
    pub const DPDX_FINE: u16 = 210;
    pub const DPDY_FINE: u16 = 211;
    pub const F_DIV: u16 = 136;
    pub const IS_NAN: u16 = 156;
    pub const LOGICAL_EQUAL: u16 = 164;
    pub const LOGICAL_NOT_EQUAL: u16 = 165;
    pub const LOGICAL_OR: u16 = 166;
    pub const LOGICAL_AND: u16 = 167;
    pub const LOGICAL_NOT: u16 = 168;
    pub const SELECT: u16 = 169;
    pub const I_EQUAL: u16 = 170;
    pub const I_NOT_EQUAL: u16 = 171;
    pub const U_GREATER_THAN: u16 = 172;
    pub const S_GREATER_THAN: u16 = 173;
    pub const U_GREATER_THAN_EQUAL: u16 = 174;
    pub const S_GREATER_THAN_EQUAL: u16 = 175;
    pub const U_LESS_THAN: u16 = 176;
    pub const S_LESS_THAN: u16 = 177;
    pub const U_LESS_THAN_EQUAL: u16 = 178;
    pub const S_LESS_THAN_EQUAL: u16 = 179;
    pub const F_ORD_EQUAL: u16 = 180;
    pub const F_UNORD_EQUAL: u16 = 181;
    pub const F_ORD_NOT_EQUAL: u16 = 182;
    pub const F_UNORD_NOT_EQUAL: u16 = 183;
    pub const F_ORD_LESS_THAN: u16 = 184;
    pub const F_UNORD_LESS_THAN: u16 = 185;
    pub const F_ORD_GREATER_THAN: u16 = 186;
    pub const F_UNORD_GREATER_THAN: u16 = 187;
    pub const F_ORD_LESS_THAN_EQUAL: u16 = 188;
    pub const F_UNORD_LESS_THAN_EQUAL: u16 = 189;
    pub const F_ORD_GREATER_THAN_EQUAL: u16 = 190;
    pub const F_UNORD_GREATER_THAN_EQUAL: u16 = 191;
    pub const SHIFT_RIGHT_LOGICAL: u16 = 194;
    pub const SHIFT_RIGHT_ARITHMETIC: u16 = 195;
    pub const SHIFT_LEFT_LOGICAL: u16 = 196;
    pub const BITWISE_OR: u16 = 197;
    pub const BITWISE_XOR: u16 = 198;
    pub const BITWISE_AND: u16 = 199;
    pub const NOT: u16 = 200;
    pub const BIT_FIELD_INSERT: u16 = 201;
    pub const BIT_FIELD_S_EXTRACT: u16 = 202;
    pub const BIT_FIELD_U_EXTRACT: u16 = 203;
    pub const BIT_REVERSE: u16 = 204;
    pub const LOOP_MERGE: u16 = 246;
    pub const SELECTION_MERGE: u16 = 247;
    pub const LABEL: u16 = 248;
    pub const BRANCH: u16 = 249;
    pub const BRANCH_CONDITIONAL: u16 = 250;
    pub const SWITCH: u16 = 251;
    pub const KILL: u16 = 252;
    pub const RETURN: u16 = 253;
    pub const GROUP_NON_UNIFORM_SHUFFLE: u16 = 345;
    pub const GROUP_NON_UNIFORM_QUAD_BROADCAST: u16 = 365;
    pub const GROUP_NON_UNIFORM_QUAD_SWAP: u16 = 366;
}

/// GLSL.std.450 extended instructions used by the translator.
pub mod glsl {
    pub const ROUND_EVEN: u32 = 2;
    pub const TRUNC: u32 = 3;
    pub const F_ABS: u32 = 4;
    pub const FLOOR: u32 = 8;
    pub const CEIL: u32 = 9;
    pub const SIN: u32 = 13;
    pub const COS: u32 = 14;
    pub const EXP2: u32 = 29;
    pub const LOG2: u32 = 30;
    pub const SQRT: u32 = 31;
    pub const INVERSE_SQRT: u32 = 32;
    pub const U_MIN: u32 = 38;
    pub const PACK_HALF_2X16: u32 = 58;
    pub const UNPACK_HALF_2X16: u32 = 62;
    pub const FIND_S_MSB: u32 = 74;
    pub const FIND_U_MSB: u32 = 75;
    pub const S_MIN: u32 = 39;
    pub const U_MAX: u32 = 41;
    pub const S_MAX: u32 = 42;
    pub const F_MIN: u32 = 37;
    pub const F_MAX: u32 = 40;
    pub const F_CLAMP: u32 = 43;
    pub const FMA: u32 = 50;
    pub const N_MIN: u32 = 79;
    pub const N_MAX: u32 = 80;
}

pub mod capability {
    pub const SHADER: u32 = 1;
    pub const IMAGE_GATHER_EXTENDED: u32 = 25;
    pub const CLIP_DISTANCE: u32 = 32;
    pub const IMAGE_QUERY: u32 = 50;
    pub const DERIVATIVE_CONTROL: u32 = 51;
    pub const SAMPLED_1D: u32 = 43;
    pub const SAMPLED_CUBE_ARRAY: u32 = 45;
    pub const GROUP_NON_UNIFORM: u32 = 61;
    pub const GROUP_NON_UNIFORM_SHUFFLE: u32 = 65;
    pub const GROUP_NON_UNIFORM_QUAD: u32 = 68;
}

/// Memory/execution scopes.
pub mod scope {
    pub const WORKGROUP: u32 = 2;
    pub const SUBGROUP: u32 = 3;
}

/// Memory semantics of barriers.
pub mod semantics {
    pub const ACQUIRE_RELEASE: u32 = 0x8;
    pub const WORKGROUP_MEMORY: u32 = 0x100;
    pub const IMAGE_MEMORY: u32 = 0x800;
}

pub mod decoration {
    pub const BLOCK: u32 = 2;
    pub const ARRAY_STRIDE: u32 = 6;
    pub const BUILT_IN: u32 = 11;
    pub const NO_PERSPECTIVE: u32 = 13;
    pub const FLAT: u32 = 14;
    pub const LOCATION: u32 = 30;
    pub const BINDING: u32 = 33;
    pub const DESCRIPTOR_SET: u32 = 34;
    pub const OFFSET: u32 = 35;
}

pub mod built_in {
    pub const POSITION: u32 = 0;
    pub const POINT_SIZE: u32 = 1;
    pub const CLIP_DISTANCE: u32 = 3;
    pub const FRAG_COORD: u32 = 15;
    pub const FRONT_FACING: u32 = 17;
    pub const FRAG_DEPTH: u32 = 22;
    pub const NUM_WORKGROUPS: u32 = 24;
    pub const WORKGROUP_ID: u32 = 26;
    pub const LOCAL_INVOCATION_ID: u32 = 27;
    pub const SUBGROUP_LOCAL_INVOCATION_ID: u32 = 41;
    pub const VERTEX_INDEX: u32 = 42;
    pub const INSTANCE_INDEX: u32 = 43;
}

pub mod storage {
    pub const UNIFORM_CONSTANT: u32 = 0;
    pub const INPUT: u32 = 1;
    pub const UNIFORM: u32 = 2;
    pub const OUTPUT: u32 = 3;
    pub const WORKGROUP: u32 = 4;
    pub const FUNCTION: u32 = 7;
    pub const PUSH_CONSTANT: u32 = 9;
}

pub mod model {
    pub const VERTEX: u32 = 0;
    pub const FRAGMENT: u32 = 4;
    pub const GL_COMPUTE: u32 = 5;
}

pub mod mode {
    pub const ORIGIN_UPPER_LEFT: u32 = 7;
    pub const DEPTH_REPLACING: u32 = 12;
    pub const LOCAL_SIZE: u32 = 17;
}

/// Image dimensions (`OpTypeImage` Dim).
pub mod dim {
    pub const D1: u32 = 0;
    pub const D2: u32 = 1;
    pub const D3: u32 = 2;
    pub const CUBE: u32 = 3;
}

/// Image operand mask bits.
pub mod image_operands {
    pub const BIAS: u32 = 0x1;
    pub const LOD: u32 = 0x2;
    pub const GRAD: u32 = 0x4;
    pub const OFFSET: u32 = 0x10;
}

const MAGIC: u32 = 0x0723_0203;
/// SPIR-V 1.0 (Vulkan 1.0).
pub const VERSION_1_0: u32 = 0x0001_0000;
/// SPIR-V 1.3 (Vulkan 1.1): needed for subgroup (quad) operations.
pub const VERSION_1_3: u32 = 0x0001_0300;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum TypeKey {
    Void,
    Bool,
    Int(u32, bool),
    Float(u32),
    Vector(Id, u32),
    Pointer(u32, Id),
    Function(Id, Vec<Id>),
    Image(Vec<u32>),
    SampledImage(Id),
    Array(Id, Id),
}

/// A module under construction.
#[derive(Debug, Default)]
pub struct Builder {
    next_id: Id,
    version: u32,
    capabilities: BTreeSet<u32>,
    glsl: Option<Id>,
    entry_points: Vec<u32>,
    execution_modes: Vec<u32>,
    annotations: Vec<u32>,
    globals: Vec<u32>,
    locals: Vec<u32>,
    body: Vec<u32>,
    types: HashMap<TypeKey, Id>,
    constants: HashMap<(Id, u32), Id>,
    booleans: [Option<Id>; 2],
    function: Option<(Id, Id, Id, Id)>,
}

/// Append one instruction to `words`.
fn emit(words: &mut Vec<u32>, opcode: u16, operands: &[u32]) {
    words.push(((operands.len() as u32 + 1) << 16) | u32::from(opcode));
    words.extend_from_slice(operands);
}

/// A string literal: UTF-8, nul-terminated, padded to whole words.
fn string_words(text: &str) -> Vec<u32> {
    let mut bytes = text.as_bytes().to_vec();
    bytes.push(0);
    bytes.resize(bytes.len().next_multiple_of(4), 0);
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| u32::from_le_bytes(*chunk))
        .collect()
}

impl Builder {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            version: VERSION_1_0,
            ..Self::default()
        }
    }

    /// Raise the module's SPIR-V version (never lowers it).
    pub fn require_version(&mut self, version: u32) {
        self.version = self.version.max(version);
    }

    pub fn id(&mut self) -> Id {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn capability(&mut self, capability: u32) {
        self.capabilities.insert(capability);
    }

    /// The GLSL.std.450 instruction set, imported on first use.
    pub fn glsl(&mut self) -> Id {
        if let Some(id) = self.glsl {
            return id;
        }
        let id = self.id();
        self.glsl = Some(id);
        id
    }

    fn ty(&mut self, key: TypeKey, opcode: u16, operands: &[u32]) -> Id {
        if let Some(id) = self.types.get(&key) {
            return *id;
        }
        let id = self.id();
        let mut words = vec![id];
        words.extend_from_slice(operands);
        emit(&mut self.globals, opcode, &words);
        self.types.insert(key, id);
        id
    }

    pub fn void(&mut self) -> Id {
        self.ty(TypeKey::Void, op::TYPE_VOID, &[])
    }

    pub fn bool(&mut self) -> Id {
        self.ty(TypeKey::Bool, op::TYPE_BOOL, &[])
    }

    pub fn u32(&mut self) -> Id {
        self.ty(TypeKey::Int(32, false), op::TYPE_INT, &[32, 0])
    }

    pub fn i32(&mut self) -> Id {
        self.ty(TypeKey::Int(32, true), op::TYPE_INT, &[32, 1])
    }

    pub fn f32(&mut self) -> Id {
        self.ty(TypeKey::Float(32), op::TYPE_FLOAT, &[32])
    }

    pub fn vector(&mut self, component: Id, count: u32) -> Id {
        self.ty(
            TypeKey::Vector(component, count),
            op::TYPE_VECTOR,
            &[component, count],
        )
    }

    pub fn pointer(&mut self, storage: u32, pointee: Id) -> Id {
        self.ty(
            TypeKey::Pointer(storage, pointee),
            op::TYPE_POINTER,
            &[storage, pointee],
        )
    }

    pub fn function_type(&mut self, result: Id, parameters: &[Id]) -> Id {
        let mut operands = vec![result];
        operands.extend_from_slice(parameters);
        self.ty(
            TypeKey::Function(result, parameters.to_vec()),
            op::TYPE_FUNCTION,
            &operands,
        )
    }

    /// A sampled float image (`Sampled` = 1, format Unknown).
    pub fn image(&mut self, dim: u32, depth: bool, arrayed: bool) -> Id {
        let sampled = self.f32();
        let operands = [sampled, dim, u32::from(depth), u32::from(arrayed), 0, 1, 0];
        self.ty(TypeKey::Image(operands.to_vec()), op::TYPE_IMAGE, &operands)
    }

    pub fn sampled_image(&mut self, image: Id) -> Id {
        self.ty(
            TypeKey::SampledImage(image),
            op::TYPE_SAMPLED_IMAGE,
            &[image],
        )
    }

    pub fn array(&mut self, element: Id, length: u32) -> Id {
        let length = self.const_u32(length);
        self.ty(
            TypeKey::Array(element, length),
            op::TYPE_ARRAY,
            &[element, length],
        )
    }

    /// A struct type. Not deduplicated: structs carry their own decorations.
    pub fn struct_type(&mut self, members: &[Id]) -> Id {
        let id = self.id();
        let mut operands = vec![id];
        operands.extend_from_slice(members);
        emit(&mut self.globals, op::TYPE_STRUCT, &operands);
        id
    }

    fn constant(&mut self, ty: Id, bits: u32) -> Id {
        if let Some(id) = self.constants.get(&(ty, bits)) {
            return *id;
        }
        let id = self.id();
        emit(&mut self.globals, op::CONSTANT, &[ty, id, bits]);
        self.constants.insert((ty, bits), id);
        id
    }

    pub fn const_u32(&mut self, value: u32) -> Id {
        let ty = self.u32();
        self.constant(ty, value)
    }

    pub fn const_i32(&mut self, value: i32) -> Id {
        let ty = self.i32();
        self.constant(ty, value as u32)
    }

    pub fn const_f32(&mut self, value: f32) -> Id {
        let ty = self.f32();
        self.constant(ty, value.to_bits())
    }

    pub fn const_bool(&mut self, value: bool) -> Id {
        if let Some(id) = self.booleans[usize::from(value)] {
            return id;
        }
        let ty = self.bool();
        let id = self.id();
        let opcode = if value {
            op::CONSTANT_TRUE
        } else {
            op::CONSTANT_FALSE
        };
        emit(&mut self.globals, opcode, &[ty, id]);
        self.booleans[usize::from(value)] = Some(id);
        id
    }

    pub fn const_composite(&mut self, ty: Id, parts: &[Id]) -> Id {
        let id = self.id();
        let mut operands = vec![ty, id];
        operands.extend_from_slice(parts);
        emit(&mut self.globals, op::CONSTANT_COMPOSITE, &operands);
        id
    }

    /// A module-scope variable (`Input`, `Output`, `Uniform`, ...).
    pub fn global_variable(&mut self, pointee: Id, storage: u32) -> Id {
        let pointer = self.pointer(storage, pointee);
        let id = self.id();
        emit(&mut self.globals, op::VARIABLE, &[pointer, id, storage]);
        id
    }

    /// A function-scope variable of the entry function.
    pub fn local_variable(&mut self, pointee: Id) -> Id {
        let pointer = self.pointer(storage::FUNCTION, pointee);
        let id = self.id();
        emit(
            &mut self.locals,
            op::VARIABLE,
            &[pointer, id, storage::FUNCTION],
        );
        id
    }

    pub fn decorate(&mut self, target: Id, decoration: u32, extra: &[u32]) {
        let mut operands = vec![target, decoration];
        operands.extend_from_slice(extra);
        emit(&mut self.annotations, op::DECORATE, &operands);
    }

    pub fn member_decorate(&mut self, target: Id, member: u32, decoration: u32, extra: &[u32]) {
        let mut operands = vec![target, member, decoration];
        operands.extend_from_slice(extra);
        emit(&mut self.annotations, op::MEMBER_DECORATE, &operands);
    }

    pub fn execution_mode(&mut self, function: Id, mode: u32) {
        self.execution_mode_with(function, mode, &[]);
    }

    /// An execution mode with literal operands (`LocalSize x y z`).
    pub fn execution_mode_with(&mut self, function: Id, mode: u32, literals: &[u32]) {
        let mut operands = vec![function, mode];
        operands.extend_from_slice(literals);
        emit(&mut self.execution_modes, op::EXECUTION_MODE, &operands);
    }

    /// Start the entry function. Returns the function id; the body that
    /// follows starts in its entry block.
    pub fn begin_main(&mut self) -> Id {
        let void = self.void();
        let function_type = self.function_type(void, &[]);
        let function = self.id();
        let entry = self.id();
        self.function = Some((function, void, function_type, entry));
        function
    }

    /// Declare the entry point once the interface variables are known.
    pub fn entry_point(&mut self, execution_model: u32, function: Id, interface: &[Id]) {
        let mut operands = vec![execution_model, function];
        operands.extend(string_words("main"));
        operands.extend_from_slice(interface);
        emit(&mut self.entry_points, op::ENTRY_POINT, &operands);
    }

    /// An instruction with a result: returns the new id.
    pub fn inst(&mut self, opcode: u16, result_type: Id, operands: &[u32]) -> Id {
        let id = self.id();
        let mut words = vec![result_type, id];
        words.extend_from_slice(operands);
        emit(&mut self.body, opcode, &words);
        id
    }

    /// An instruction without a result.
    pub fn inst_void(&mut self, opcode: u16, operands: &[u32]) {
        emit(&mut self.body, opcode, operands);
    }

    pub fn label(&mut self, id: Id) {
        emit(&mut self.body, op::LABEL, &[id]);
    }

    pub fn load(&mut self, ty: Id, pointer: Id) -> Id {
        self.inst(op::LOAD, ty, &[pointer])
    }

    pub fn store(&mut self, pointer: Id, value: Id) {
        self.inst_void(op::STORE, &[pointer, value]);
    }

    pub fn bitcast(&mut self, ty: Id, value: Id) -> Id {
        self.inst(op::BITCAST, ty, &[value])
    }

    pub fn ext(&mut self, ty: Id, instruction: u32, arguments: &[Id]) -> Id {
        let set = self.glsl();
        let mut operands = vec![set, instruction];
        operands.extend_from_slice(arguments);
        self.inst(op::EXT_INST, ty, &operands)
    }

    pub fn branch(&mut self, target: Id) {
        self.inst_void(op::BRANCH, &[target]);
    }

    pub fn branch_conditional(&mut self, condition: Id, on_true: Id, on_false: Id) {
        self.inst_void(op::BRANCH_CONDITIONAL, &[condition, on_true, on_false]);
    }

    pub fn selection_merge(&mut self, merge: Id) {
        self.inst_void(op::SELECTION_MERGE, &[merge, 0]);
    }

    pub fn loop_merge(&mut self, merge: Id, continue_target: Id) {
        self.inst_void(op::LOOP_MERGE, &[merge, continue_target, 0]);
    }

    /// `OpSwitch` on a 32-bit selector: `(literal, label)` cases.
    pub fn switch(&mut self, selector: Id, default: Id, cases: &[(u32, Id)]) {
        let mut operands = vec![selector, default];
        for (literal, label) in cases {
            operands.push(*literal);
            operands.push(*label);
        }
        self.inst_void(op::SWITCH, &operands);
    }

    /// Assemble the module.
    pub fn finish(mut self) -> Vec<u32> {
        let mut words = vec![MAGIC, self.version, 0, 0, 0];
        self.capabilities.insert(capability::SHADER);
        for capability in &self.capabilities {
            emit(&mut words, op::CAPABILITY, &[*capability]);
        }
        if let Some(glsl) = self.glsl {
            let mut operands = vec![glsl];
            operands.extend(string_words("GLSL.std.450"));
            emit(&mut words, op::EXT_INST_IMPORT, &operands);
        }
        // Logical addressing, GLSL450 memory model.
        emit(&mut words, op::MEMORY_MODEL, &[0, 1]);
        words.extend_from_slice(&self.entry_points);
        words.extend_from_slice(&self.execution_modes);
        words.extend_from_slice(&self.annotations);
        words.extend_from_slice(&self.globals);
        if let Some((function, void, function_type, entry)) = self.function {
            emit(
                &mut words,
                op::FUNCTION,
                &[void, function, 0, function_type],
            );
            emit(&mut words, op::LABEL, &[entry]);
            words.extend_from_slice(&self.locals);
            words.extend_from_slice(&self.body);
            emit(&mut words, op::FUNCTION_END, &[]);
        }
        words[3] = self.next_id;
        words
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strings_are_nul_terminated_and_padded() {
        assert_eq!(string_words("main"), vec![0x6e69_616d, 0]);
        assert_eq!(string_words("abc"), vec![0x0063_6261]);
    }

    #[test]
    fn types_and_constants_are_shared() {
        let mut builder = Builder::new();
        let a = builder.f32();
        let b = builder.f32();
        assert_eq!(a, b);
        assert_eq!(builder.const_u32(7), builder.const_u32(7));
        assert_ne!(builder.const_u32(7), builder.const_f32(f32::from_bits(7)));
    }

    #[test]
    fn an_empty_fragment_module_has_a_header_and_bound() {
        let mut builder = Builder::new();
        let main = builder.begin_main();
        builder.entry_point(model::FRAGMENT, main, &[]);
        builder.execution_mode(main, mode::ORIGIN_UPPER_LEFT);
        builder.inst_void(op::RETURN, &[]);
        let words = builder.finish();
        assert_eq!(words[0], MAGIC);
        assert_eq!(words[1], VERSION_1_0);
        assert!(words[3] > 4, "bound covers every id");
    }
}
