//! VP9 frames for the console's video decoder (NVDEC).
//!
//! NVDEC decodes only a VP9 frame's tiles: the title's software has already
//! read the frame's headers and hands their values over (a picture info
//! block) along with the frame's probabilities. A host decoder takes whole
//! frames, so each frame's two headers are written again from those values,
//! ahead of its tiles, and libvpx (loaded at run time) decodes the result.
//!
//! The rebuilt frames refresh their frame context and do not adapt it
//! backward (`frame_parallel_decoding_mode`), so libvpx's four contexts hold
//! the probabilities frames ended with, and a frame's compressed header codes
//! the change from one of them to the title's. That reproduces the title's
//! probabilities exactly. It decodes correctly when the stream does not adapt
//! probabilities backward either (LM3's movies set
//! `frame_parallel_decoding_mode` in every frame): the title would adapt them
//! with symbol counts this decoder does not produce.
//!
//! NVDEC is not told which context a frame used, and one context for every
//! frame does not do: streams keep their hidden alternate references in a
//! context of their own, and going back to the other one can need a motion
//! vector probability to return to an even default, which an update cannot
//! code (they are always odd). Each frame takes the context its
//! probabilities are closest to among those every change can be coded
//! from: the one the stream used, whose only differences are the frame's own
//! updates.
//!
//! NVDEC names references by surface; libvpx keeps eight slots. Each output
//! surface takes a slot (the least recently used one), and a frame's
//! references are the slots holding its reference surfaces.
//!
//! Whether a frame is shown is known only from the next frame's flags (a
//! hidden alternate reference lends the frame after it no motion vectors),
//! so a frame is decoded once the next one arrives. Titles decode several
//! frames ahead of the ones VIC converts; a frame VIC wants before its
//! successor came is decoded then, as shown. (Coding the frame after a
//! hidden one error resilient instead does not work: that also clears the
//! reference sign biases, and with them compound prediction.)

use std::ffi::{c_char, c_int, c_long, c_uint, c_void, CStr};

use crate::vp9_tables;

/// Bytes of NVDEC's VP9 picture info block.
pub const PICTURE_BYTES: usize = 0x100;
/// Bytes of NVDEC's VP9 probability table: 1440 of mode and motion vector
/// probabilities, then the coefficients' 4x2x2x6x6 groups of four.
pub const PROBS_BYTES: usize = 1440 + 4 * 2 * 2 * 6 * 6 * 4;

/// Picture info flags.
const KEY_FRAME: u32 = 1;
const ERROR_RESILIENT: u32 = 1 << 3;
const LAST_SHOW_FRAME: u32 = 1 << 4;
const INTRA_ONLY: u32 = 1 << 5;

/// Transform modes.
const ALLOW_32X32: u8 = 3;
const TX_MODE_SELECT: u8 = 4;
/// The interpolation filter chosen per block.
const SWITCHABLE: u8 = 4;
/// Reference modes.
const SINGLE_REFERENCE: u8 = 0;
const COMPOUND_REFERENCE: u8 = 1;
const REFERENCE_MODE_SELECT: u8 = 2;
/// The probability an update flag is coded with.
const UPDATE_PROB: u8 = 252;

/// A decoded frame: 8-bit 4:2:0 planes, tightly packed.
#[derive(Debug, Clone, Default)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

impl Frame {
    pub fn chroma_width(&self) -> u32 {
        self.width.div_ceil(2)
    }
}

/// A segmentation's settings, as the picture info carries them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Segmentation {
    pub enabled: bool,
    pub update_map: bool,
    pub temporal_update: bool,
    pub abs_delta: bool,
    /// Per segment, the features enabled (a bit each).
    pub feature_mask: [u32; 8],
    pub feature_data: [[i16; 4]; 8],
}

/// NVDEC's VP9 picture info (`SetDrvPicSetupOffset`), in the fields the
/// headers need.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Picture {
    pub bitstream_size: u32,
    /// Width and height of the last, golden, alternate and current frames.
    pub sizes: [(u16, u16); 4],
    pub flags: u32,
    /// Per reference frame (intra, last, golden, alternate).
    pub sign_bias: [i8; 4],
    pub filter_level: u8,
    pub sharpness: u8,
    pub base_q: u8,
    pub y_dc_delta: i8,
    pub uv_ac_delta: i8,
    pub uv_dc_delta: i8,
    pub lossless: bool,
    pub tx_mode: u8,
    pub high_precision_mv: bool,
    /// The header's filter literal, or `SWITCHABLE`.
    pub interp_filter: u8,
    pub reference_mode: u8,
    pub log2_tile_cols: u8,
    pub log2_tile_rows: u8,
    pub segmentation: Segmentation,
    pub mode_ref_delta_enabled: bool,
    pub ref_deltas: [i8; 4],
    pub mode_deltas: [i8; 2],
}

impl Picture {
    /// The block at `bytes` (Ryujinx's `Vp9.PictureInfo` layout).
    pub fn parse(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < PICTURE_BYTES {
            return None;
        }
        let u16_at = |at: usize| u16::from_le_bytes([bytes[at], bytes[at + 1]]);
        let u32_at = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        let i8_at = |at: usize| bytes[at] as i8;
        let mut sizes = [(0, 0); 4];
        for (index, size) in sizes.iter_mut().enumerate() {
            let at = 72 + index * 8;
            *size = (u16_at(at), u16_at(at + 2));
        }
        let mut segmentation = Segmentation {
            enabled: bytes[128] != 0,
            update_map: bytes[129] != 0,
            temporal_update: bytes[130] != 0,
            abs_delta: bytes[131] != 0,
            ..Segmentation::default()
        };
        for segment in 0..8 {
            segmentation.feature_mask[segment] = u32_at(132 + segment * 4);
            for feature in 0..4 {
                segmentation.feature_data[segment][feature] =
                    u16_at(164 + (segment * 4 + feature) * 2) as i16;
            }
        }
        Some(Self {
            bitstream_size: u32_at(48),
            sizes,
            flags: u32_at(104),
            sign_bias: [i8_at(108), i8_at(109), i8_at(110), i8_at(111)],
            filter_level: bytes[112],
            sharpness: bytes[113],
            base_q: bytes[114],
            y_dc_delta: i8_at(115),
            uv_ac_delta: i8_at(116),
            uv_dc_delta: i8_at(117),
            lossless: bytes[118] != 0,
            tx_mode: bytes[119],
            high_precision_mv: bytes[120] != 0,
            interp_filter: bytes[121],
            reference_mode: bytes[122],
            log2_tile_cols: bytes[126],
            log2_tile_rows: bytes[127],
            segmentation,
            mode_ref_delta_enabled: bytes[228] != 0,
            ref_deltas: [i8_at(229), i8_at(230), i8_at(231), i8_at(232)],
            mode_deltas: [i8_at(233), i8_at(234)],
        })
    }

    fn key_frame(&self) -> bool {
        self.flags & KEY_FRAME != 0
    }

    fn intra_only(&self) -> bool {
        !self.key_frame() && self.flags & INTRA_ONLY != 0
    }

    fn size(&self) -> (u32, u32) {
        let (width, height) = self.sizes[3];
        (u32::from(width).max(1), u32::from(height).max(1))
    }
}

/// Coefficient probabilities: per transform size, plane type, reference,
/// band, context and node.
type CoefProbs = [[[[[[u8; 3]; 6]; 6]; 2]; 2]; 4];

/// The probabilities a frame context holds (the ones a compressed header
/// can update), plus the segmentation tree probabilities the uncompressed
/// header carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Probs {
    tx8: [[u8; 1]; 2],
    tx16: [[u8; 2]; 2],
    tx32: [[u8; 3]; 2],
    coef: CoefProbs,
    skip: [u8; 3],
    inter_mode: [[u8; 3]; 7],
    interp: [[u8; 2]; 4],
    is_inter: [u8; 4],
    comp_inter: [u8; 5],
    single_ref: [[u8; 2]; 5],
    comp_ref: [u8; 5],
    y_mode: [[u8; 9]; 4],
    partition: [[u8; 3]; 16],
    joints: [u8; 3],
    sign: [u8; 2],
    classes: [[u8; 10]; 2],
    class0: [u8; 2],
    bits: [[u8; 10]; 2],
    class0_fr: [[[u8; 3]; 2]; 2],
    fr: [[u8; 3]; 2],
    class0_hp: [u8; 2],
    hp: [u8; 2],
    seg_tree: [u8; 7],
    seg_pred: [u8; 3],
}

impl Default for Probs {
    /// VP9's defaults (what a key frame resets every context to).
    fn default() -> Self {
        let mut coef = [[[[[[0; 3]; 6]; 6]; 2]; 2]; 4];
        let tables = [
            &vp9_tables::COEF_4X4,
            &vp9_tables::COEF_8X8,
            &vp9_tables::COEF_16X16,
            &vp9_tables::COEF_32X32,
        ];
        for (size, table) in tables.iter().enumerate() {
            for (index, value) in table.iter().enumerate() {
                let (m, l, k, j, i) = (
                    index % 3,
                    index / 3 % 6,
                    index / 18 % 6,
                    index / 108 % 2,
                    index / 216,
                );
                coef[size][i][j][k][l][m] = *value;
            }
        }
        let bits = [136, 140, 148, 160, 176, 192, 224, 234, 234, 240];
        let class0_fr = [[128, 128, 64], [96, 112, 64]];
        Self {
            tx8: [[100], [66]],
            tx16: [[20, 152], [15, 101]],
            tx32: [[3, 136, 37], [5, 52, 13]],
            coef,
            skip: [192, 128, 64],
            inter_mode: [
                [2, 173, 34],
                [7, 145, 85],
                [7, 166, 63],
                [7, 94, 66],
                [8, 64, 46],
                [17, 81, 31],
                [25, 29, 30],
            ],
            interp: [[235, 162], [36, 255], [34, 3], [149, 144]],
            is_inter: [9, 102, 187, 225],
            comp_inter: [239, 183, 119, 96, 41],
            single_ref: [[33, 16], [77, 74], [142, 142], [172, 170], [238, 247]],
            comp_ref: [50, 126, 123, 221, 226],
            y_mode: [
                [65, 32, 18, 144, 162, 194, 41, 51, 98],
                [132, 68, 18, 165, 217, 196, 45, 40, 78],
                [173, 80, 19, 176, 240, 193, 64, 35, 46],
                [221, 135, 38, 194, 248, 121, 96, 85, 29],
            ],
            partition: [
                [199, 122, 141],
                [147, 63, 159],
                [148, 133, 118],
                [121, 104, 114],
                [174, 73, 87],
                [92, 41, 83],
                [82, 99, 50],
                [53, 39, 39],
                [177, 58, 59],
                [68, 26, 63],
                [52, 79, 25],
                [17, 14, 12],
                [222, 34, 30],
                [72, 16, 44],
                [58, 32, 12],
                [10, 7, 6],
            ],
            joints: [32, 64, 96],
            sign: [128, 128],
            classes: [
                [224, 144, 192, 168, 192, 176, 192, 198, 198, 245],
                [216, 128, 176, 160, 176, 176, 192, 198, 198, 208],
            ],
            class0: [216, 208],
            bits: [bits, bits],
            class0_fr: [class0_fr, class0_fr],
            fr: [[64, 96, 64], [64, 96, 64]],
            class0_hp: [160, 160],
            hp: [128, 128],
            seg_tree: [255; 7],
            seg_pred: [255; 3],
        }
    }
}

impl Probs {
    /// NVDEC's probability table (Ryujinx's `Vp9.EntropyProbs` layout:
    /// mode probabilities split into their first eight and their ninth,
    /// and four bytes for each three of the others).
    pub fn parse(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < PROBS_BYTES {
            return None;
        }
        let mut probs = Self::default();
        let at = |offset: usize| bytes[offset];
        for i in 0..2 {
            probs.tx8[i][0] = at(1136 + i);
            for j in 0..2 {
                probs.tx16[i][j] = at(1138 + i * 2 + j);
            }
            for j in 0..3 {
                probs.tx32[i][j] = at(1142 + i * 3 + j);
            }
        }
        for size in 0..4 {
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..6 {
                        for l in 0..6 {
                            for m in 0..3 {
                                let index = ((((size * 2 + i) * 2 + j) * 6 + k) * 6 + l) * 4 + m;
                                probs.coef[size][i][j][k][l][m] = at(1440 + index);
                            }
                        }
                    }
                }
            }
        }
        for i in 0..3 {
            probs.skip[i] = at(1335 + i);
        }
        for i in 0..7 {
            for j in 0..3 {
                probs.inter_mode[i][j] = at(1024 + i * 4 + j);
            }
        }
        for i in 0..4 {
            for j in 0..2 {
                probs.interp[i][j] = at(1322 + i * 2 + j);
            }
            probs.is_inter[i] = at(1052 + i);
            for j in 0..8 {
                probs.y_mode[i][j] = at(1152 + i * 8 + j);
            }
            probs.y_mode[i][8] = at(1148 + i);
        }
        for i in 0..5 {
            probs.comp_inter[i] = at(1330 + i);
            probs.single_ref[i] = [at(1408 + i * 2), at(1409 + i * 2)];
            probs.comp_ref[i] = at(1418 + i);
        }
        for i in 0..16 {
            for j in 0..3 {
                probs.partition[i][j] = at(1248 + i * 4 + j);
            }
        }
        probs.joints = [at(1339), at(1340), at(1341)];
        for i in 0..2 {
            probs.sign[i] = at(1342 + i);
            probs.class0[i] = at(1344 + i);
            for j in 0..3 {
                probs.fr[i][j] = at(1346 + i * 3 + j);
            }
            probs.class0_hp[i] = at(1352 + i);
            probs.hp[i] = at(1354 + i);
            for j in 0..10 {
                probs.classes[i][j] = at(1356 + i * 10 + j);
                probs.bits[i][j] = at(1388 + i * 10 + j);
            }
            for j in 0..2 {
                for k in 0..3 {
                    probs.class0_fr[i][j][k] = at(1376 + (i * 2 + j) * 3 + k);
                }
            }
        }
        for i in 0..7 {
            probs.seg_tree[i] = at(903 + i);
        }
        for i in 0..3 {
            probs.seg_pred[i] = at(910 + i);
        }
        Some(probs)
    }
}

/// Bits written most significant first (the uncompressed header).
#[derive(Debug, Default)]
struct BitWriter {
    bytes: Vec<u8>,
    bits: usize,
}

impl BitWriter {
    fn bit(&mut self, value: bool) {
        if self.bits.is_multiple_of(8) {
            self.bytes.push(0);
        }
        if value {
            *self.bytes.last_mut().expect("pushed") |= 0x80 >> (self.bits % 8);
        }
        self.bits += 1;
    }

    fn literal(&mut self, value: u32, bits: u32) {
        for bit in (0..bits).rev() {
            self.bit(value >> bit & 1 != 0);
        }
    }

    /// A magnitude of `bits` bits, then its sign.
    fn signed(&mut self, value: i32, bits: u32) {
        self.literal(value.unsigned_abs(), bits);
        self.bit(value < 0);
    }
}

/// VP9's boolean (arithmetic) coder, writing (libvpx's `vpx_writer`).
#[derive(Debug)]
struct BoolWriter {
    low: u32,
    range: u32,
    count: i32,
    bytes: Vec<u8>,
}

impl BoolWriter {
    fn new() -> Self {
        let mut writer = Self {
            low: 0,
            range: 255,
            count: -24,
            bytes: Vec::new(),
        };
        // The marker bit a decoder checks is zero.
        writer.write(false, 128);
        writer
    }

    fn write(&mut self, bit: bool, probability: u8) {
        let split = 1 + (((self.range - 1) * u32::from(probability)) >> 8);
        let mut low = self.low;
        let mut range = split;
        if bit {
            low += split;
            range = self.range - split;
        }
        let mut shift = range.leading_zeros() as i32 - 24;
        range <<= shift;
        let mut count = self.count + shift;
        if count >= 0 {
            let offset = shift - count;
            if (low << (offset - 1)) & 0x8000_0000 != 0 {
                // Carry into the bytes written.
                let mut at = self.bytes.len();
                while at > 0 && self.bytes[at - 1] == 0xFF {
                    self.bytes[at - 1] = 0;
                    at -= 1;
                }
                if at > 0 {
                    self.bytes[at - 1] += 1;
                }
            }
            self.bytes.push((low >> (24 - offset)) as u8);
            low <<= offset;
            shift = count;
            low &= 0xFF_FFFF;
            count -= 8;
        }
        low <<= shift;
        self.count = count;
        self.low = low;
        self.range = range;
    }

    fn literal(&mut self, value: u32, bits: u32) {
        for bit in (0..bits).rev() {
            self.write(value >> bit & 1 != 0, 128);
        }
    }

    fn finish(mut self) -> Vec<u8> {
        for _ in 0..32 {
            self.write(false, 128);
        }
        // No ambiguity with a superframe index marker.
        if self.bytes.last().is_some_and(|byte| byte & 0xE0 == 0xC0) {
            self.bytes.push(0);
        }
        self.bytes
    }

    /// A probability's update: none, or the coded difference to `new`.
    fn diff_update(&mut self, old: u8, new: u8) {
        match delta_for(old, new) {
            Some(delta) => {
                self.write(true, UPDATE_PROB);
                self.term_subexp(delta);
            }
            None => self.write(false, UPDATE_PROB),
        }
    }

    /// A motion vector probability's update: seven bits, the low one set.
    fn mv_update(&mut self, old: u8, new: u8) {
        if new != old && new & 1 == 1 {
            self.write(true, UPDATE_PROB);
            self.literal(u32::from(new >> 1), 7);
        } else {
            self.write(false, UPDATE_PROB);
        }
    }

    fn term_subexp(&mut self, word: u32) {
        if word < 16 {
            self.write(false, 128);
            self.literal(word, 4);
        } else if word < 32 {
            self.write(true, 128);
            self.write(false, 128);
            self.literal(word - 16, 4);
        } else if word < 64 {
            self.write(true, 128);
            self.write(true, 128);
            self.write(false, 128);
            self.literal(word - 32, 5);
        } else {
            self.write(true, 128);
            self.write(true, 128);
            self.write(true, 128);
            let value = word - 64;
            if value < 65 {
                self.literal(value, 7);
            } else {
                self.literal(65 + ((value - 65) >> 1), 7);
                self.literal((value - 65) & 1, 1);
            }
        }
    }
}

/// VP9's `inv_map_table`: the 20 coarse steps first, then every other
/// probability in order.
fn inv_map_table() -> &'static [u8; 254] {
    static TABLE: std::sync::OnceLock<[u8; 254]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [0; 254];
        let coarse: Vec<u8> = (0..20).map(|step| 7 + 13 * step).collect();
        let rest = (1..=253u8).filter(|value| !coarse.contains(value));
        for (slot, value) in table.iter_mut().zip(coarse.iter().copied().chain(rest)) {
            *slot = value;
        }
        table
    })
}

/// The probability a decoder makes of `old` and the coded difference
/// `delta` (libvpx's `inv_remap_prob`).
fn inv_remap(delta: u32, old: u8) -> u8 {
    let recenter = |v: i32, m: i32| {
        if v > 2 * m {
            v
        } else if v & 1 == 1 {
            m - ((v + 1) >> 1)
        } else {
            m + (v >> 1)
        }
    };
    let v = i32::from(inv_map_table()[delta as usize]);
    let m = i32::from(old) - 1;
    if (m << 1) <= 255 {
        (1 + recenter(v, m)) as u8
    } else {
        (255 - recenter(v, 255 - 1 - m)) as u8
    }
}

/// The difference to code to turn `old` into `new`, if they differ (a
/// table of every pair, made once).
fn delta_for(old: u8, new: u8) -> Option<u32> {
    static TABLE: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    if old == new || new == 0 || old == 0 {
        return None;
    }
    let table = TABLE.get_or_init(|| {
        // 255 marks a pair no difference reaches.
        let mut table = vec![255u8; 256 * 256];
        for old in 1..=255u8 {
            for delta in 0..254u32 {
                table[usize::from(old) * 256 + usize::from(inv_remap(delta, old))] = delta as u8;
            }
        }
        table
    });
    match table[usize::from(old) * 256 + usize::from(new)] {
        255 => None,
        delta => Some(u32::from(delta)),
    }
}

/// Motion vector probabilities `after` coding could not give the values
/// `target` has (an update makes only odd ones).
fn mv_misses(after: &Probs, target: &Probs, high_precision: bool) -> usize {
    let pairs = [
        (&after.joints[..], &target.joints[..]),
        (&after.sign[..], &target.sign[..]),
        (&after.class0[..], &target.class0[..]),
        (after.classes.as_flattened(), target.classes.as_flattened()),
        (after.bits.as_flattened(), target.bits.as_flattened()),
        (
            after.class0_fr.as_flattened().as_flattened(),
            target.class0_fr.as_flattened().as_flattened(),
        ),
        (after.fr.as_flattened(), target.fr.as_flattened()),
    ];
    let mut misses: usize = pairs
        .iter()
        .map(|(a, b)| a.iter().zip(b.iter()).filter(|(a, b)| a != b).count())
        .sum();
    if high_precision {
        misses += after
            .class0_hp
            .iter()
            .zip(&target.class0_hp)
            .filter(|(a, b)| a != b)
            .count();
        misses += after
            .hp
            .iter()
            .zip(&target.hp)
            .filter(|(a, b)| a != b)
            .count();
    }
    misses
}

/// What the rebuilt headers say beyond the picture info.
struct FrameCoding {
    show: bool,
    error_resilient: bool,
    /// The slot this frame refreshes (all of them for a key frame).
    refresh: u8,
    /// The slots of the last, golden and alternate references.
    references: [u8; 3],
    /// The frame context it decodes with (and refreshes).
    context: u8,
}

/// The uncompressed header of a frame (the VP9 bitstream specification's
/// `uncompressed_header`, profile 0).
fn uncompressed_header(
    picture: &Picture,
    probs: &Probs,
    coding: &FrameCoding,
    compressed_bytes: usize,
) -> Vec<u8> {
    let mut w = BitWriter::default();
    w.literal(2, 2); // frame marker
    w.literal(0, 2); // profile 0
    w.bit(false); // show_existing_frame
    let (width, height) = picture.size();
    let frame_size = |w: &mut BitWriter| {
        w.literal(width - 1, 16);
        w.literal(height - 1, 16);
        w.bit(false); // render size is the frame size
    };
    let sync_code = |w: &mut BitWriter| {
        w.literal(0x49, 8);
        w.literal(0x83, 8);
        w.literal(0x42, 8);
    };
    if picture.key_frame() {
        w.bit(false); // key frame
        w.bit(coding.show);
        w.bit(coding.error_resilient);
        sync_code(&mut w);
        w.literal(1, 3); // BT.601
        w.bit(false); // studio range
        frame_size(&mut w);
    } else {
        w.bit(true);
        w.bit(coding.show);
        w.bit(coding.error_resilient);
        if !coding.show {
            w.bit(picture.intra_only());
        }
        if !coding.error_resilient {
            // Intra-only frames reset every context, as key frames do.
            w.literal(if picture.intra_only() { 3 } else { 0 }, 2);
        }
        if picture.intra_only() {
            sync_code(&mut w);
            w.literal(1 << coding.refresh, 8);
            frame_size(&mut w);
        } else {
            w.literal(1 << coding.refresh, 8);
            for (index, slot) in coding.references.iter().enumerate() {
                w.literal(u32::from(*slot), 3);
                w.bit(picture.sign_bias[index + 1] != 0);
            }
            for _ in 0..3 {
                w.bit(false); // size not taken from a reference
            }
            frame_size(&mut w);
            w.bit(picture.high_precision_mv);
            if picture.interp_filter == SWITCHABLE {
                w.bit(true);
            } else {
                w.bit(false);
                w.literal(u32::from(picture.interp_filter & 3), 2);
            }
        }
    }
    if !coding.error_resilient {
        w.bit(true); // refresh_frame_context
        w.bit(true); // frame_parallel_decoding_mode
    }
    w.literal(u32::from(coding.context), 2); // frame_context_idx

    // Loop filter.
    w.literal(u32::from(picture.filter_level & 63), 6);
    w.literal(u32::from(picture.sharpness & 7), 3);
    w.bit(picture.mode_ref_delta_enabled);
    if picture.mode_ref_delta_enabled {
        w.bit(true); // update every delta
        for delta in picture.ref_deltas {
            w.bit(true);
            w.signed(i32::from(delta), 6);
        }
        for delta in picture.mode_deltas {
            w.bit(true);
            w.signed(i32::from(delta), 6);
        }
    }

    // Quantization.
    w.literal(u32::from(picture.base_q), 8);
    for delta in [picture.y_dc_delta, picture.uv_dc_delta, picture.uv_ac_delta] {
        w.bit(delta != 0);
        if delta != 0 {
            w.signed(i32::from(delta), 4);
        }
    }

    // Segmentation: every feature again, so a reset loses nothing.
    let segmentation = &picture.segmentation;
    w.bit(segmentation.enabled);
    if segmentation.enabled {
        w.bit(segmentation.update_map);
        if segmentation.update_map {
            for probability in probs.seg_tree {
                w.bit(probability != 255);
                if probability != 255 {
                    w.literal(u32::from(probability), 8);
                }
            }
            w.bit(segmentation.temporal_update);
            if segmentation.temporal_update {
                for probability in probs.seg_pred {
                    w.bit(probability != 255);
                    if probability != 255 {
                        w.literal(u32::from(probability), 8);
                    }
                }
            }
        }
        w.bit(true); // update the data
        w.bit(segmentation.abs_delta);
        for segment in 0..8 {
            for (feature, bits) in [8, 6, 2, 0].into_iter().enumerate() {
                let enabled = segmentation.feature_mask[segment] & (1 << feature) != 0;
                w.bit(enabled);
                if enabled {
                    let value = i32::from(segmentation.feature_data[segment][feature]);
                    w.literal(value.unsigned_abs(), bits);
                    if feature < 2 {
                        w.bit(value < 0);
                    }
                }
            }
        }
    }

    // Tiles.
    let (min_cols, max_cols) = tile_cols_range(width);
    let cols = u32::from(picture.log2_tile_cols).clamp(min_cols, max_cols);
    for _ in min_cols..cols {
        w.bit(true);
    }
    if cols < max_cols {
        w.bit(false);
    }
    match picture.log2_tile_rows {
        0 => w.bit(false),
        1 => {
            w.bit(true);
            w.bit(false);
        }
        _ => {
            w.bit(true);
            w.bit(true);
        }
    }
    w.literal(compressed_bytes as u32, 16);
    w.bytes
}

/// The smallest and largest log2 tile column counts a frame `width` wide
/// allows (libvpx's `vp9_get_tile_n_bits`).
fn tile_cols_range(width: u32) -> (u32, u32) {
    let mi_cols = width.div_ceil(8);
    let sb64_cols = mi_cols.div_ceil(8);
    let mut min = 0;
    while (64 << min) < sb64_cols {
        min += 1;
    }
    let mut max = 1;
    while (sb64_cols >> max) >= 4 {
        max += 1;
    }
    (min, (max - 1).max(min))
}

/// The compressed header taking libvpx's context `base` to `target`; returns
/// the bytes and the context libvpx holds after the frame.
fn compressed_header(picture: &Picture, base: &Probs, target: &Probs) -> (Vec<u8>, Probs) {
    let mut w = BoolWriter::new();
    let mut after = base.clone();
    let intra = picture.key_frame() || picture.intra_only();

    // Transform mode and probabilities.
    let tx_mode = if picture.lossless {
        0
    } else {
        let tx_mode = picture.tx_mode.min(TX_MODE_SELECT);
        w.literal(u32::from(tx_mode.min(ALLOW_32X32)), 2);
        if tx_mode >= ALLOW_32X32 {
            w.write(tx_mode == TX_MODE_SELECT, 128);
        }
        tx_mode
    };
    if tx_mode == TX_MODE_SELECT {
        for i in 0..2 {
            w.diff_update(base.tx8[i][0], target.tx8[i][0]);
        }
        for i in 0..2 {
            for j in 0..2 {
                w.diff_update(base.tx16[i][j], target.tx16[i][j]);
            }
        }
        for i in 0..2 {
            for j in 0..3 {
                w.diff_update(base.tx32[i][j], target.tx32[i][j]);
            }
        }
        after.tx8 = target.tx8;
        after.tx16 = target.tx16;
        after.tx32 = target.tx32;
    }

    // Coefficients, per transform size the mode allows.
    let largest = usize::from(tx_mode.min(ALLOW_32X32));
    for size in 0..=largest {
        let changed = base.coef[size] != target.coef[size];
        w.write(changed, 128);
        if !changed {
            continue;
        }
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..6 {
                    let contexts = if k == 0 { 3 } else { 6 };
                    for l in 0..contexts {
                        for m in 0..3 {
                            w.diff_update(
                                base.coef[size][i][j][k][l][m],
                                target.coef[size][i][j][k][l][m],
                            );
                            after.coef[size][i][j][k][l][m] = target.coef[size][i][j][k][l][m];
                        }
                    }
                }
            }
        }
    }

    for i in 0..3 {
        w.diff_update(base.skip[i], target.skip[i]);
    }
    after.skip = target.skip;

    if !intra {
        for i in 0..7 {
            for j in 0..3 {
                w.diff_update(base.inter_mode[i][j], target.inter_mode[i][j]);
            }
        }
        after.inter_mode = target.inter_mode;
        if picture.interp_filter == SWITCHABLE {
            for i in 0..4 {
                for j in 0..2 {
                    w.diff_update(base.interp[i][j], target.interp[i][j]);
                }
            }
            after.interp = target.interp;
        }
        for i in 0..4 {
            w.diff_update(base.is_inter[i], target.is_inter[i]);
        }
        after.is_inter = target.is_inter;

        // Compound prediction needs references on both sides in time.
        // An error resilient frame's reset clears the sign biases.
        let bias = if picture.flags & ERROR_RESILIENT != 0 {
            [0; 4]
        } else {
            picture.sign_bias
        };
        let compound_allowed = bias[2] != bias[1] || bias[3] != bias[1];
        let mode = if compound_allowed {
            picture.reference_mode.min(REFERENCE_MODE_SELECT)
        } else {
            SINGLE_REFERENCE
        };
        if compound_allowed {
            w.write(mode != SINGLE_REFERENCE, 128);
            if mode != SINGLE_REFERENCE {
                w.write(mode == REFERENCE_MODE_SELECT, 128);
            }
        }
        if mode == REFERENCE_MODE_SELECT {
            for i in 0..5 {
                w.diff_update(base.comp_inter[i], target.comp_inter[i]);
            }
            after.comp_inter = target.comp_inter;
        }
        if mode != COMPOUND_REFERENCE {
            for i in 0..5 {
                w.diff_update(base.single_ref[i][0], target.single_ref[i][0]);
                w.diff_update(base.single_ref[i][1], target.single_ref[i][1]);
            }
            after.single_ref = target.single_ref;
        }
        if mode != SINGLE_REFERENCE {
            for i in 0..5 {
                w.diff_update(base.comp_ref[i], target.comp_ref[i]);
            }
            after.comp_ref = target.comp_ref;
        }

        for i in 0..4 {
            for j in 0..9 {
                w.diff_update(base.y_mode[i][j], target.y_mode[i][j]);
            }
        }
        after.y_mode = target.y_mode;
        for i in 0..16 {
            for j in 0..3 {
                w.diff_update(base.partition[i][j], target.partition[i][j]);
            }
        }
        after.partition = target.partition;

        // Motion vectors.
        for j in 0..3 {
            w.mv_update(base.joints[j], target.joints[j]);
        }
        for i in 0..2 {
            w.mv_update(base.sign[i], target.sign[i]);
            for j in 0..10 {
                w.mv_update(base.classes[i][j], target.classes[i][j]);
            }
            w.mv_update(base.class0[i], target.class0[i]);
            for j in 0..10 {
                w.mv_update(base.bits[i][j], target.bits[i][j]);
            }
        }
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..3 {
                    w.mv_update(base.class0_fr[i][j][k], target.class0_fr[i][j][k]);
                }
            }
            for k in 0..3 {
                w.mv_update(base.fr[i][k], target.fr[i][k]);
            }
        }
        if picture.high_precision_mv {
            for i in 0..2 {
                w.mv_update(base.class0_hp[i], target.class0_hp[i]);
                w.mv_update(base.hp[i], target.hp[i]);
            }
        }
        // Only odd motion vector probabilities can be coded.
        let mv = |old: u8, new: u8| if new != old && new & 1 == 1 { new } else { old };
        for j in 0..3 {
            after.joints[j] = mv(base.joints[j], target.joints[j]);
        }
        for i in 0..2 {
            after.sign[i] = mv(base.sign[i], target.sign[i]);
            after.class0[i] = mv(base.class0[i], target.class0[i]);
            for j in 0..10 {
                after.classes[i][j] = mv(base.classes[i][j], target.classes[i][j]);
                after.bits[i][j] = mv(base.bits[i][j], target.bits[i][j]);
            }
            for j in 0..2 {
                for k in 0..3 {
                    after.class0_fr[i][j][k] =
                        mv(base.class0_fr[i][j][k], target.class0_fr[i][j][k]);
                }
            }
            for k in 0..3 {
                after.fr[i][k] = mv(base.fr[i][k], target.fr[i][k]);
            }
            if picture.high_precision_mv {
                after.class0_hp[i] = mv(base.class0_hp[i], target.class0_hp[i]);
                after.hp[i] = mv(base.hp[i], target.hp[i]);
            }
        }
    }
    (w.finish(), after)
}

/// What one NVDEC VP9 decode is given.
pub struct Request {
    pub picture: Picture,
    pub probs: Probs,
    /// The frame's tiles.
    pub bitstream: Vec<u8>,
    /// The engine addresses of the last, golden and alternate reference
    /// surfaces and of the output surface (their luma planes).
    pub surfaces: [u32; 4],
}

/// libvpx's decoder context (`vpx_codec_ctx_t`).
#[repr(C)]
struct CodecContext {
    name: *const c_char,
    iface: *const c_void,
    err: c_int,
    err_detail: *const c_char,
    init_flags: c_long,
    config: *const c_void,
    private: *mut c_void,
}

/// `vpx_codec_dec_cfg_t`.
#[repr(C)]
struct DecoderConfig {
    threads: c_uint,
    width: c_uint,
    height: c_uint,
}

/// `vpx_image_t`.
#[repr(C)]
struct Image {
    format: c_uint,
    color_space: c_uint,
    range: c_uint,
    width: c_uint,
    height: c_uint,
    bit_depth: c_uint,
    display_width: c_uint,
    display_height: c_uint,
    render_width: c_uint,
    render_height: c_uint,
    x_chroma_shift: c_uint,
    y_chroma_shift: c_uint,
    planes: [*mut u8; 4],
    stride: [c_int; 4],
    bps: c_int,
    user_priv: *mut c_void,
    img_data: *mut u8,
    img_data_owner: c_int,
    self_allocd: c_int,
    fb_priv: *mut c_void,
}

/// `VPX_DECODER_ABI_VERSION` of libvpx 1.8 and later.
const VPX_DECODER_ABI_VERSION: c_int = 12;

type Decode =
    unsafe extern "C" fn(*mut CodecContext, *const u8, c_uint, *mut c_void, c_long) -> c_int;
type GetFrame = unsafe extern "C" fn(*mut CodecContext, *mut *const c_void) -> *const Image;
type Destroy = unsafe extern "C" fn(*mut CodecContext) -> c_int;
type ErrorDetail = unsafe extern "C" fn(*const CodecContext) -> *const c_char;

/// libvpx's VP9 decoder, loaded from the host's shared library.
struct Vpx {
    context: Box<CodecContext>,
    decode: Decode,
    get_frame: GetFrame,
    destroy: Destroy,
    error_detail: ErrorDetail,
    _library: libloading::Library,
}

// The context is only used under the host1x lock.
unsafe impl Send for Vpx {}

impl Vpx {
    fn load() -> Result<Self, String> {
        // Windows packages use the MSYS2/vcpkg DLL names; Linux uses sonames.
        // `RECOMP_LIBVPX` overrides both for tests and unusual installs.
        let mut candidates: Vec<std::ffi::OsString> = Vec::new();
        if let Some(path) = std::env::var_os("RECOMP_LIBVPX") {
            candidates.push(path);
        }
        for name in [
            "libvpx-1.dll",
            "libvpx.dll",
            "vpx.dll",
            "libvpx.so",
            "libvpx.so.12",
            "libvpx.so.11",
            "libvpx.so.9",
            "libvpx.so.8",
        ] {
            candidates.push(name.into());
        }
        let library = candidates
            .iter()
            .find_map(|name| unsafe { libloading::Library::new(name) }.ok())
            .ok_or("libvpx was not found")?;
        unsafe {
            let symbol = |name: &[u8]| -> Result<*const c_void, String> {
                library
                    .get::<*const c_void>(name)
                    .map(|symbol| *symbol)
                    .map_err(|error| format!("libvpx: {error}"))
            };
            let iface: unsafe extern "C" fn() -> *const c_void =
                std::mem::transmute(symbol(b"vpx_codec_vp9_dx\0")?);
            let init: unsafe extern "C" fn(
                *mut CodecContext,
                *const c_void,
                *const DecoderConfig,
                c_long,
                c_int,
            ) -> c_int = std::mem::transmute(symbol(b"vpx_codec_dec_init_ver\0")?);
            let decode: Decode = std::mem::transmute(symbol(b"vpx_codec_decode\0")?);
            let get_frame: GetFrame = std::mem::transmute(symbol(b"vpx_codec_get_frame\0")?);
            let destroy: Destroy = std::mem::transmute(symbol(b"vpx_codec_destroy\0")?);
            let error_detail: ErrorDetail =
                std::mem::transmute(symbol(b"vpx_codec_error_detail\0")?);
            let mut context = Box::new(CodecContext {
                name: std::ptr::null(),
                iface: std::ptr::null(),
                err: 0,
                err_detail: std::ptr::null(),
                init_flags: 0,
                config: std::ptr::null(),
                private: std::ptr::null_mut(),
            });
            let config = DecoderConfig {
                threads: 4,
                width: 0,
                height: 0,
            };
            let result = init(&mut *context, iface(), &config, 0, VPX_DECODER_ABI_VERSION);
            if result != 0 {
                return Err(format!("libvpx: decoder init failed ({result})"));
            }
            Ok(Self {
                context,
                decode,
                get_frame,
                destroy,
                error_detail,
                _library: library,
            })
        }
    }

    /// Decode one frame; the picture it shows, if any.
    fn decode(&mut self, frame: &[u8]) -> Result<Option<Frame>, String> {
        unsafe {
            let result = (self.decode)(
                &mut *self.context,
                frame.as_ptr(),
                frame.len() as c_uint,
                std::ptr::null_mut(),
                0,
            );
            if result != 0 {
                let detail = (self.error_detail)(&*self.context);
                let detail = if detail.is_null() {
                    String::new()
                } else {
                    CStr::from_ptr(detail).to_string_lossy().into_owned()
                };
                return Err(format!("libvpx: decode failed ({result}) {detail}"));
            }
            let mut iterator: *const c_void = std::ptr::null();
            let mut shown = None;
            loop {
                let image = (self.get_frame)(&mut *self.context, &mut iterator);
                if image.is_null() {
                    break;
                }
                shown = Some(copy_image(&*image));
            }
            Ok(shown)
        }
    }
}

impl Drop for Vpx {
    fn drop(&mut self) {
        unsafe {
            (self.destroy)(&mut *self.context);
        }
    }
}

/// An 8-bit 4:2:0 image's planes, packed.
unsafe fn copy_image(image: &Image) -> Frame {
    let (width, height) = (image.display_width, image.display_height);
    let plane = |index: usize, width: u32, height: u32| -> Vec<u8> {
        let stride = image.stride[index] as usize;
        let mut bytes = Vec::with_capacity((width * height) as usize);
        for row in 0..height as usize {
            let start = unsafe { image.planes[index].add(row * stride) };
            bytes.extend_from_slice(unsafe { std::slice::from_raw_parts(start, width as usize) });
        }
        bytes
    };
    let (chroma_width, chroma_height) = (width.div_ceil(2), height.div_ceil(2));
    Frame {
        width,
        height,
        y: plane(0, width, height),
        u: plane(1, chroma_width, chroma_height),
        v: plane(2, chroma_width, chroma_height),
    }
}

/// NVDEC's VP9 decoding: the frames rebuilt and decoded so far, and what
/// libvpx holds.
#[derive(Default)]
pub struct Decoder {
    vpx: Option<Vpx>,
    /// Why libvpx could not be loaded, once it failed.
    unavailable: Option<String>,
    /// The probabilities in libvpx's four frame contexts.
    contexts: [Probs; 4],
    /// Per libvpx slot, the surface it holds and when it was last used.
    slots: [Option<(u32, u64)>; 8],
    clock: u64,
    /// The frame waiting for the next one to say whether it is shown.
    queued: Option<Request>,
}

impl Decoder {
    /// Queue `request`'s frame and decode the one queued before it, whose
    /// visibility this one's flags tell: that frame's picture and output
    /// surface, when it is shown. Errors are reported by the caller (once
    /// each).
    pub fn queue(&mut self, request: Request) -> Result<Option<(u32, Frame)>, String> {
        let shown = request.picture.flags & LAST_SHOW_FRAME != 0;
        match self.queued.replace(request) {
            Some(previous) => self.decode(&previous, shown),
            None => Ok(None),
        }
    }

    /// Decode the queued frame now, as shown, if it is the one output to
    /// `surface`.
    pub fn flush(&mut self, surface: u32) -> Result<Option<(u32, Frame)>, String> {
        if !self
            .queued
            .as_ref()
            .is_some_and(|queued| queued.surfaces[3] == surface)
        {
            return Ok(None);
        }
        let queued = self.queued.take().expect("checked");
        self.decode(&queued, true)
    }

    fn decode(&mut self, request: &Request, shown: bool) -> Result<Option<(u32, Frame)>, String> {
        if let Some(reason) = &self.unavailable {
            return Err(reason.clone());
        }
        if self.vpx.is_none() {
            match Vpx::load() {
                Ok(vpx) => self.vpx = Some(vpx),
                Err(reason) => {
                    self.unavailable = Some(reason.clone());
                    return Err(reason);
                }
            }
        }
        let picture = &request.picture;
        let key = picture.key_frame();
        let intra_only = picture.intra_only();
        let error_resilient = picture.flags & ERROR_RESILIENT != 0;
        self.clock += 1;

        let references = request.surfaces[..3]
            .iter()
            .map(|surface| self.slot_of(*surface).unwrap_or(0) as u8)
            .collect::<Vec<u8>>();
        let refresh = if key {
            0
        } else {
            self.assign(request.surfaces[3]) as u8
        };
        if key {
            for slot in &mut self.slots {
                *slot = Some((request.surfaces[3], self.clock));
            }
        }
        // Key frames, intra-only frames (reset here) and error resilient
        // frames reset every context and start from the defaults in context
        // 0. Other frames take the context that needs the fewest changes,
        // among those whose motion vector changes can all be coded.
        let reset = key || intra_only || error_resilient;
        if reset {
            self.contexts = Default::default();
        }
        let candidates = if reset { 0..1 } else { 0..4 };
        let (context, compressed, after) = candidates
            .map(|index| {
                let (compressed, after) =
                    compressed_header(picture, &self.contexts[index], &request.probs);
                let misses = mv_misses(&after, &request.probs, picture.high_precision_mv);
                (index, compressed, after, misses)
            })
            .min_by_key(|(index, compressed, _, misses)| (*misses, compressed.len(), *index))
            .map(|(index, compressed, after, _)| (index, compressed, after))
            .expect("one context at least");
        let coding = FrameCoding {
            // Intra-only frames are never shown directly.
            show: shown && !intra_only,
            error_resilient,
            refresh,
            references: [references[0], references[1], references[2]],
            context: context as u8,
        };
        let mut frame = uncompressed_header(picture, &request.probs, &coding, compressed.len());
        frame.extend_from_slice(&compressed);
        frame.extend_from_slice(&request.bitstream);

        // Error resilient frames do not refresh their context, which their
        // reset left at the defaults.
        if !error_resilient {
            self.contexts[context] = after;
        }
        let picture = self.vpx.as_mut().expect("loaded above").decode(&frame)?;
        Ok(picture.map(|picture| (request.surfaces[3], picture)))
    }

    /// The most recently used slot holding `surface`.
    fn slot_of(&mut self, surface: u32) -> Option<usize> {
        let clock = self.clock;
        let slot = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.is_some_and(|(held, _)| held == surface))
            .max_by_key(|(_, slot)| slot.map(|(_, used)| used))
            .map(|(index, _)| index)?;
        if let Some((_, used)) = &mut self.slots[slot] {
            *used = clock;
        }
        Some(slot)
    }

    /// The slot `surface` is decoded into: the one holding it already, else
    /// the least recently used one.
    fn assign(&mut self, surface: u32) -> usize {
        let slot = self
            .slots
            .iter()
            .position(|slot| slot.is_some_and(|(held, _)| held == surface))
            .unwrap_or_else(|| {
                (0..8)
                    .min_by_key(|index| self.slots[*index].map_or(0, |(_, used)| used))
                    .expect("eight slots")
            });
        self.slots[slot] = Some((surface, self.clock));
        slot
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Windows package ships `libvpx-1.dll`; catch a regression where the
    /// loader only looks for POSIX sonames.
    #[cfg(windows)]
    #[test]
    fn libvpx_loads_on_windows() {
        Vpx::load().expect("load libvpx-1.dll");
    }

    /// libvpx's boolean decoder (`vpx_reader`), to read back what the
    /// writer wrote.
    struct BoolReader<'a> {
        bytes: &'a [u8],
        at: usize,
        value: u64,
        count: i32,
        range: u32,
    }

    impl<'a> BoolReader<'a> {
        fn new(bytes: &'a [u8]) -> Self {
            let mut reader = Self {
                bytes,
                at: 0,
                value: 0,
                count: -8,
                range: 255,
            };
            assert!(!reader.read(128), "marker bit");
            reader
        }

        fn fill(&mut self) {
            let mut shift = 48 - self.count;
            while shift >= 0 {
                let byte = self.bytes.get(self.at).copied().unwrap_or(0);
                self.at += 1;
                self.value |= u64::from(byte) << shift;
                self.count += 8;
                shift -= 8;
            }
        }

        fn read(&mut self, probability: u8) -> bool {
            let p = u32::from(probability);
            let split = (self.range * p + (256 - p)) >> 8;
            if self.count < 0 {
                self.fill();
            }
            let big_split = u64::from(split) << 56;
            let bit = if self.value >= big_split {
                self.range -= split;
                self.value -= big_split;
                true
            } else {
                self.range = split;
                false
            };
            let shift = self.range.leading_zeros() as i32 - 24;
            self.range <<= shift;
            self.value <<= shift;
            self.count -= shift;
            bit
        }

        fn literal(&mut self, bits: u32) -> u32 {
            (0..bits).fold(0, |value, _| value << 1 | u32::from(self.read(128)))
        }

        fn term_subexp(&mut self) -> u32 {
            if !self.read(128) {
                return self.literal(4);
            }
            if !self.read(128) {
                return self.literal(4) + 16;
            }
            if !self.read(128) {
                return self.literal(5) + 32;
            }
            let v = self.literal(7);
            let uniform = if v < 65 {
                v
            } else {
                (v << 1) - 65 + self.literal(1)
            };
            uniform + 64
        }
    }

    #[test]
    fn probability_updates_read_back() {
        let pairs = [
            (128u8, 129u8),
            (1, 255),
            (255, 1),
            (200, 7),
            (7, 200),
            (100, 100),
            (3, 4),
        ];
        let mut writer = BoolWriter::new();
        for (old, new) in pairs {
            writer.diff_update(old, new);
        }
        writer.literal(0x2A5, 10);
        let bytes = writer.finish();
        let mut reader = BoolReader::new(&bytes);
        for (old, new) in pairs {
            let read = if reader.read(UPDATE_PROB) {
                inv_remap(reader.term_subexp(), old)
            } else {
                old
            };
            assert_eq!(read, new, "{old} -> {new}");
        }
        assert_eq!(reader.literal(10), 0x2A5);
    }

    #[test]
    fn every_probability_change_has_a_delta() {
        for old in 1..=255u8 {
            for new in 1..=255u8 {
                if old != new {
                    assert!(delta_for(old, new).is_some(), "{old} -> {new}");
                }
            }
        }
        assert_eq!(inv_map_table()[19], 254);
        assert_eq!(inv_map_table()[20], 1);
    }

    #[test]
    fn tile_columns_follow_the_width() {
        // 1600 pixels are 25 superblocks: one column, at most four.
        assert_eq!(tile_cols_range(1600), (0, 2));
        assert_eq!(tile_cols_range(64), (0, 0));
        assert_eq!(tile_cols_range(4096 + 64), (1, 4));
    }

    #[test]
    fn probability_tables_parse_to_their_last_coefficient() {
        let mut bytes = vec![0u8; PROBS_BYTES];
        assert_eq!(PROBS_BYTES, 3744);
        // 32x32, chroma, inter, band 5, context 5, node 2: the last used byte.
        bytes[PROBS_BYTES - 2] = 77;
        bytes[1335] = 9; // first skip probability
        let probs = Probs::parse(&bytes).expect("whole table");
        assert_eq!(probs.coef[3][1][1][5][5][2], 77);
        assert_eq!(probs.skip[0], 9);
        assert!(Probs::parse(&bytes[..PROBS_BYTES - 1]).is_none());
    }

    #[test]
    fn default_probabilities_are_vp9s() {
        let probs = Probs::default();
        assert_eq!(probs.coef[0][0][0][0][0], [195, 29, 183]);
        assert_eq!(probs.coef[0][0][0][1][0], [31, 107, 169]);
        assert_eq!(probs.coef[3][1][1][5][5][2], vp9_tables::COEF_32X32[431]);
    }

    #[test]
    fn slots_reuse_the_least_recently_used() {
        let mut decoder = Decoder {
            clock: 1,
            slots: [Some((10, 1)); 8],
            ..Decoder::default()
        };
        // A reference keeps its slot fresh; a new surface takes the least
        // recently used one.
        decoder.clock = 2;
        let reference = decoder.slot_of(10).expect("held");
        let next = decoder.assign(11);
        assert_ne!(next, reference);
        assert_eq!(decoder.slot_of(11), Some(next));
        assert_eq!(decoder.slot_of(10), Some(reference));
    }

    /// Bits read most significant first (the uncompressed header).
    struct BitReader<'a> {
        bytes: &'a [u8],
        bit: usize,
    }

    impl BitReader<'_> {
        fn bit(&mut self) -> bool {
            let value = self
                .bytes
                .get(self.bit / 8)
                .is_some_and(|byte| byte >> (7 - self.bit % 8) & 1 != 0);
            self.bit += 1;
            value
        }

        fn literal(&mut self, bits: u32) -> u32 {
            (0..bits).fold(0, |value, _| value << 1 | u32::from(self.bit()))
        }

        fn signed(&mut self, bits: u32) -> i32 {
            let value = self.literal(bits) as i32;
            if self.bit() {
                -value
            } else {
                value
            }
        }
    }

    /// The frames of an IVF file, superframes split.
    fn ivf_frames(file: &[u8]) -> Vec<Vec<u8>> {
        let header = u16::from_le_bytes([file[6], file[7]]) as usize;
        let mut frames = Vec::new();
        let mut at = header;
        while at + 12 <= file.len() {
            let size = u32::from_le_bytes(file[at..at + 4].try_into().unwrap()) as usize;
            let packet = &file[at + 12..at + 12 + size];
            at += 12 + size;
            let marker = packet[size - 1];
            if marker & 0xE0 == 0xC0 {
                let count = (marker & 7) as usize + 1;
                let magnitude = ((marker >> 3) & 3) as usize + 1;
                let index = 2 + magnitude * count;
                if size >= index && packet[size - index] == marker {
                    let mut start = 0;
                    for frame in 0..count {
                        let field = &packet[size - index + 1 + frame * magnitude..];
                        let len = (0..magnitude)
                            .fold(0, |len, byte| len | (field[byte] as usize) << (8 * byte));
                        frames.push(packet[start..start + len].to_vec());
                        start += len;
                    }
                    continue;
                }
            }
            frames.push(packet.to_vec());
        }
        frames
    }

    /// What the title's driver does with a VP9 frame before NVDEC sees it:
    /// read both headers, keep the frame contexts, and hand over the picture
    /// info, the probabilities and the tiles.
    #[derive(Default)]
    struct Driver {
        contexts: [Option<Probs>; 4],
        ref_deltas: [i8; 4],
        mode_deltas: [i8; 2],
        slots: [u32; 8],
        sizes: [(u16, u16); 8],
        next_surface: u32,
        last_shown: bool,
        last_key: bool,
        started: bool,
    }

    impl Driver {
        fn context(&self, index: usize) -> Probs {
            self.contexts[index].clone().unwrap_or_default()
        }

        /// The request for `frame`, or None for a frame that shows an
        /// existing one.
        fn parse(&mut self, frame: &[u8]) -> Option<Request> {
            if !self.started {
                self.ref_deltas = [1, 0, -1, -1];
                self.started = true;
            }
            let mut r = BitReader {
                bytes: frame,
                bit: 0,
            };
            assert_eq!(r.literal(2), 2, "frame marker");
            assert_eq!(r.literal(2), 0, "profile 0");
            if r.bit() {
                return None;
            }
            let key = !r.bit();
            let show = r.bit();
            let error_resilient = r.bit();
            let mut picture = Picture::default();
            let mut intra_only = false;
            let mut reset = 0;
            let mut refresh = 0xFFu32;
            let mut references = [0usize; 3];
            let frame_size = |r: &mut BitReader| {
                let size = ((r.literal(16) + 1) as u16, (r.literal(16) + 1) as u16);
                if r.bit() {
                    r.literal(32);
                }
                size
            };
            if key {
                assert_eq!(r.literal(24), 0x49_8342);
                if r.literal(3) != 7 {
                    r.bit();
                }
                picture.sizes[3] = frame_size(&mut r);
            } else {
                intra_only = if show { false } else { r.bit() };
                reset = if error_resilient { 0 } else { r.literal(2) };
                if intra_only {
                    assert_eq!(r.literal(24), 0x49_8342);
                    refresh = r.literal(8);
                    picture.sizes[3] = frame_size(&mut r);
                } else {
                    refresh = r.literal(8);
                    for (index, reference) in references.iter_mut().enumerate() {
                        *reference = r.literal(3) as usize;
                        picture.sign_bias[index + 1] = r.bit() as i8;
                    }
                    let mut found = None;
                    for reference in references {
                        if r.bit() {
                            found = Some(self.sizes[reference]);
                            break;
                        }
                    }
                    picture.sizes[3] = match found {
                        Some(size) => {
                            if r.bit() {
                                r.literal(32);
                            }
                            size
                        }
                        None => frame_size(&mut r),
                    };
                    picture.high_precision_mv = r.bit();
                    picture.interp_filter = if r.bit() {
                        SWITCHABLE
                    } else {
                        r.literal(2) as u8
                    };
                }
            }
            let (refresh_context, parallel) = if error_resilient {
                (false, true)
            } else {
                (r.bit(), r.bit())
            };
            assert!(parallel, "backward adaptation not emulated");
            let mut context_index = r.literal(2) as usize;
            let mut probs;
            if key || intra_only || error_resilient {
                if key || error_resilient || reset == 3 {
                    self.contexts = Default::default();
                } else if reset == 2 {
                    self.contexts[context_index] = None;
                }
                self.ref_deltas = [1, 0, -1, -1];
                self.mode_deltas = [0, 0];
                picture.sign_bias = [0; 4];
                context_index = 0;
            }
            probs = self.context(context_index);

            picture.filter_level = r.literal(6) as u8;
            picture.sharpness = r.literal(3) as u8;
            picture.mode_ref_delta_enabled = r.bit();
            if picture.mode_ref_delta_enabled && r.bit() {
                for delta in &mut self.ref_deltas {
                    if r.bit() {
                        *delta = r.signed(6) as i8;
                    }
                }
                for delta in &mut self.mode_deltas {
                    if r.bit() {
                        *delta = r.signed(6) as i8;
                    }
                }
            }
            picture.ref_deltas = self.ref_deltas;
            picture.mode_deltas = self.mode_deltas;
            picture.base_q = r.literal(8) as u8;
            let mut deltas = [0i8; 3];
            for delta in &mut deltas {
                if r.bit() {
                    *delta = r.signed(4) as i8;
                }
            }
            picture.y_dc_delta = deltas[0];
            picture.uv_dc_delta = deltas[1];
            picture.uv_ac_delta = deltas[2];
            picture.lossless = picture.base_q == 0 && deltas == [0; 3];
            assert!(!r.bit(), "segmentation not emulated");
            let (min, max) = tile_cols_range(u32::from(picture.sizes[3].0));
            let mut cols = min;
            while cols < max && r.bit() {
                cols += 1;
            }
            picture.log2_tile_cols = cols as u8;
            picture.log2_tile_rows = if r.bit() { 1 + r.bit() as u8 } else { 0 };
            let header_size = r.literal(16) as usize;
            let uncompressed = r.bit.div_ceil(8);

            // The compressed header, applied to the context.
            let mut b = BoolReader::new(&frame[uncompressed..uncompressed + header_size]);
            let update = |b: &mut BoolReader, p: &mut u8| {
                if b.read(UPDATE_PROB) {
                    *p = inv_remap(b.term_subexp(), *p);
                }
            };
            let mv_update = |b: &mut BoolReader, p: &mut u8| {
                if b.read(UPDATE_PROB) {
                    *p = (b.literal(7) << 1 | 1) as u8;
                }
            };
            let tx_mode = if picture.lossless {
                0
            } else {
                let mode = b.literal(2) as u8;
                if mode == ALLOW_32X32 {
                    mode + b.read(128) as u8
                } else {
                    mode
                }
            };
            picture.tx_mode = tx_mode;
            if tx_mode == TX_MODE_SELECT {
                for i in 0..2 {
                    update(&mut b, &mut probs.tx8[i][0]);
                }
                for i in 0..2 {
                    for j in 0..2 {
                        update(&mut b, &mut probs.tx16[i][j]);
                    }
                }
                for i in 0..2 {
                    for j in 0..3 {
                        update(&mut b, &mut probs.tx32[i][j]);
                    }
                }
            }
            for size in 0..=usize::from(tx_mode.min(ALLOW_32X32)) {
                if b.read(128) {
                    for i in 0..2 {
                        for j in 0..2 {
                            for k in 0..6 {
                                for l in 0..if k == 0 { 3 } else { 6 } {
                                    for m in 0..3 {
                                        update(&mut b, &mut probs.coef[size][i][j][k][l][m]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            for i in 0..3 {
                update(&mut b, &mut probs.skip[i]);
            }
            if !key && !intra_only {
                for i in 0..7 {
                    for j in 0..3 {
                        update(&mut b, &mut probs.inter_mode[i][j]);
                    }
                }
                if picture.interp_filter == SWITCHABLE {
                    for i in 0..4 {
                        for j in 0..2 {
                            update(&mut b, &mut probs.interp[i][j]);
                        }
                    }
                }
                for i in 0..4 {
                    update(&mut b, &mut probs.is_inter[i]);
                }
                let bias = picture.sign_bias;
                let mode = if bias[2] != bias[1] || bias[3] != bias[1] {
                    if b.read(128) {
                        if b.read(128) {
                            REFERENCE_MODE_SELECT
                        } else {
                            COMPOUND_REFERENCE
                        }
                    } else {
                        SINGLE_REFERENCE
                    }
                } else {
                    SINGLE_REFERENCE
                };
                picture.reference_mode = mode;
                if mode == REFERENCE_MODE_SELECT {
                    for i in 0..5 {
                        update(&mut b, &mut probs.comp_inter[i]);
                    }
                }
                if mode != COMPOUND_REFERENCE {
                    for i in 0..5 {
                        update(&mut b, &mut probs.single_ref[i][0]);
                        update(&mut b, &mut probs.single_ref[i][1]);
                    }
                }
                if mode != SINGLE_REFERENCE {
                    for i in 0..5 {
                        update(&mut b, &mut probs.comp_ref[i]);
                    }
                }
                for i in 0..4 {
                    for j in 0..9 {
                        update(&mut b, &mut probs.y_mode[i][j]);
                    }
                }
                for i in 0..16 {
                    for j in 0..3 {
                        update(&mut b, &mut probs.partition[i][j]);
                    }
                }
                for j in 0..3 {
                    mv_update(&mut b, &mut probs.joints[j]);
                }
                for i in 0..2 {
                    mv_update(&mut b, &mut probs.sign[i]);
                    for j in 0..10 {
                        mv_update(&mut b, &mut probs.classes[i][j]);
                    }
                    mv_update(&mut b, &mut probs.class0[i]);
                    for j in 0..10 {
                        mv_update(&mut b, &mut probs.bits[i][j]);
                    }
                }
                for i in 0..2 {
                    for j in 0..2 {
                        for k in 0..3 {
                            mv_update(&mut b, &mut probs.class0_fr[i][j][k]);
                        }
                    }
                    for k in 0..3 {
                        mv_update(&mut b, &mut probs.fr[i][k]);
                    }
                }
                if picture.high_precision_mv {
                    for i in 0..2 {
                        mv_update(&mut b, &mut probs.class0_hp[i]);
                        mv_update(&mut b, &mut probs.hp[i]);
                    }
                }
            }
            if refresh_context {
                self.contexts[context_index] = Some(probs.clone());
            }

            // Surfaces: a new one per frame; references by slot.
            self.next_surface += 1;
            let surface = self.next_surface;
            let surfaces = [
                self.slots[references[0]],
                self.slots[references[1]],
                self.slots[references[2]],
                surface,
            ];
            for slot in 0..8 {
                if refresh & (1 << slot) != 0 {
                    self.slots[slot] = surface;
                    self.sizes[slot] = picture.sizes[3];
                }
            }
            picture.flags = u32::from(key)
                | (u32::from(self.last_key) << 1)
                | (u32::from(error_resilient) << 3)
                | (u32::from(self.last_shown) << 4)
                | (u32::from(intra_only) << 5);
            self.last_shown = show;
            self.last_key = key;
            let bitstream = frame[uncompressed + header_size..].to_vec();
            picture.bitstream_size = bitstream.len() as u32;
            Some(Request {
                picture,
                probs,
                bitstream,
                surfaces,
            })
        }
    }

    /// With `VP9_CHECK_IVF=<file>` (a VP9 stream in IVF, profile 0, no
    /// backward adaptation), each frame goes through what NVDEC is given
    /// and the headers rebuilt here, and must decode to libvpx's picture of
    /// the original frame.
    #[test]
    fn rebuilt_frames_decode_like_the_originals() {
        let Some(path) = std::env::var_os("VP9_CHECK_IVF") else {
            return;
        };
        let file = std::fs::read(path).expect("IVF file");
        let frames = ivf_frames(&file);
        let mut reference = Vpx::load().expect("libvpx");
        let mut expected = Vec::new();
        for frame in &frames {
            if let Some(picture) = reference.decode(frame).expect("original decodes") {
                expected.push(picture);
            }
        }
        let mut driver = Driver::default();
        let mut decoder = Decoder::default();
        let mut decoded = Vec::new();
        let mut last_surface = 0;
        for (index, frame) in frames.iter().enumerate() {
            let Some(request) = driver.parse(frame) else {
                continue;
            };
            last_surface = request.surfaces[3];
            match decoder.queue(request) {
                Ok(Some((_, picture))) => decoded.push(picture),
                Ok(None) => {}
                Err(reason) => panic!("frame {} (decoded before it) failed: {reason}", index - 1),
            }
        }
        if let Ok(Some((_, picture))) = decoder.flush(last_surface) {
            decoded.push(picture);
        }
        assert_eq!(decoded.len(), expected.len(), "shown frames");
        for (index, (ours, theirs)) in decoded.iter().zip(&expected).enumerate() {
            let differing = ours.y.iter().zip(&theirs.y).filter(|(a, b)| a != b).count();
            assert_eq!(
                differing, 0,
                "shown frame {index}: {differing} luma bytes differ"
            );
            assert!(
                ours.u == theirs.u && ours.v == theirs.v,
                "shown frame {index}: chroma differs"
            );
        }
        println!(
            "{} frames, {} shown, all identical",
            frames.len(),
            decoded.len()
        );
    }
}
