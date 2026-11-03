//! Texture preparation off the GPU worker thread.
//!
//! Untiling a texture and decoding ASTC (which host GPUs often cannot
//! sample) take milliseconds per texture, and the GPU worker must stay
//! quick. The worker reads the guest bytes and hands them to these threads;
//! the finished texels are uploaded as the next submission begins, or at
//! once for a draw that waits (a bounded time) for its texture. Draws sample
//! a transparent placeholder for textures not ready by then.
//!
//! A large texture with a mip chain also gets a preview job, queued ahead
//! of it (urgent jobs go before the rest), with only its small levels:
//! about 1/64 of the work. Decoding a
//! 1024x1024 ASTC texture and its levels takes 30-50 ms, so a room's worth
//! of new textures (a door opening onto a room seen the first time) kept
//! draws on placeholders for frames, and the room flickered in; the preview
//! is ready within the wait, and draws sample it, briefly blurry, until the
//! whole texture arrives.
//!
//! A draw waiting for a quick texture no worker has started takes the job
//! back and prepares it itself (`Decoder::take`): queued behind a room's
//! big textures, the 32x64x32 color lookup table LM3 grades each frame
//! through missed the wait as a door opened onto a new room, and the
//! frame's final pass graded the image through the transparent
//! placeholder: a black frame.

use std::collections::VecDeque;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};

use crate::bcn::{self, Bcn};
use crate::texture::{self, Layout};

/// One mip level of a texture: where it starts in each layer, its rows of
/// blocks, and the GOBs per block in y it is laid out with.
#[derive(Debug, Clone)]
pub struct Mip {
    pub offset: u64,
    pub row_bytes: u64,
    pub rows: u64,
    pub gobs_in_y: u32,
    pub width: u32,
    pub height: u32,
}

/// A texture's bytes as read from guest memory, and how to turn them into
/// texels.
pub struct Job {
    /// The texture header, which names the texture in the cache.
    pub words: [u32; 8],
    /// Which preparation of the texture this is: one made before the guest
    /// wrote it again is dropped.
    pub generation: u64,
    /// Block-linear bytes, or pitch-linear rows already packed tightly.
    pub source: Arc<[u8]>,
    pub layout: Layout,
    /// The mip levels to prepare (the first one first).
    pub mips: Vec<Mip>,
    /// The texture's level the first of `mips` is: above zero for a preview
    /// of a texture's small levels.
    pub first_level: u32,
    /// How the blocks become texels the device can sample.
    pub decode: Decode,
    /// Array layers (a cube's six faces), each `layer_stride` bytes apart
    /// in `source`.
    pub layers: u32,
    pub layer_stride: u64,
    /// Slices of a 3D texture (1 otherwise) and its blocks' depth in GOBs.
    pub depth: u32,
    pub gobs_in_z: u32,
}

/// How a texture's blocks become texels the device can sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decode {
    /// Uploaded as they are.
    None,
    /// ASTC blocks of this footprint to RGBA8 (host GPUs often cannot
    /// sample ASTC).
    Astc(u32, u32),
    /// BCn blocks to a plain format, on devices without BCn (mobile GPUs).
    Bcn(Bcn),
}

/// Jobs worth doing on the thread that needs them rather than waiting for
/// a worker: about a millisecond at most (an ASTC preview, a lookup table).
const QUICK_DECODED_TEXELS: u64 = 32 * 1024;
const QUICK_BYTES: u64 = 1 << 20;

impl Job {
    /// Whether preparing this is quick: few texels to decode from ASTC or
    /// BCn, or few bytes to untile.
    pub fn quick(&self) -> bool {
        let slices = u64::from(self.layers.max(1)) * u64::from(self.depth.max(1));
        match self.decode {
            Decode::Astc(..) | Decode::Bcn(_) => {
                let texels: u64 = self
                    .mips
                    .iter()
                    .map(|mip| u64::from(mip.width) * u64::from(mip.height))
                    .sum();
                texels * slices <= QUICK_DECODED_TEXELS
            }
            Decode::None => {
                let bytes: u64 = self.mips.iter().map(|mip| mip.row_bytes * mip.rows).sum();
                bytes * slices <= QUICK_BYTES
            }
        }
    }
}

/// A prepared texture: texels tightly packed in its upload format, level
/// by level (every layer of a level together).
pub struct Prepared {
    pub words: [u32; 8],
    pub generation: u64,
    pub texels: Vec<u8>,
    /// Where each level starts in `texels`.
    pub levels: Vec<u64>,
    /// The texture's level the first prepared is (a preview's is above 0).
    pub first_level: u32,
}

/// A few worker threads taking jobs from one queue.
/// Jobs waiting for a worker: urgent ones (previews, and textures without
/// one) before the rest.
#[derive(Default)]
struct Queues {
    urgent: VecDeque<Job>,
    later: VecDeque<Job>,
    closed: bool,
}

pub struct Decoder {
    queues: Arc<(Mutex<Queues>, Condvar)>,
    results: Receiver<Prepared>,
}

impl Decoder {
    pub fn new() -> Self {
        // Bursts of new textures are what decoding has to keep up with; the
        // title keeps about four host threads busy.
        Self::with_workers(
            std::thread::available_parallelism()
                .map_or(2, |count| count.get().saturating_sub(4).clamp(2, 6)),
        )
    }

    fn with_workers(workers: usize) -> Self {
        let queues = Arc::new((Mutex::new(Queues::default()), Condvar::new()));
        let (done, results) = channel();
        for index in 0..workers {
            let queues = Arc::clone(&queues);
            let done: Sender<Prepared> = done.clone();
            std::thread::Builder::new()
                .name(format!("recomp-texture-{index}"))
                .spawn(move || loop {
                    let job = {
                        let (lock, ready) = &*queues;
                        let mut queues =
                            lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                        loop {
                            if let Some(job) = queues
                                .urgent
                                .pop_front()
                                .or_else(|| queues.later.pop_front())
                            {
                                break job;
                            }
                            if queues.closed {
                                return;
                            }
                            queues = ready
                                .wait(queues)
                                .unwrap_or_else(|poisoned| poisoned.into_inner());
                        }
                    };
                    if done.send(prepare(job)).is_err() {
                        return;
                    }
                })
                .expect("texture worker thread");
        }
        Self { queues, results }
    }

    /// Queue `job`; an urgent one goes ahead of every job that is not.
    pub fn submit(&self, job: Job, urgent: bool) {
        let (lock, ready) = &*self.queues;
        let mut queues = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if urgent {
            queues.urgent.push_back(job);
        } else {
            queues.later.push_back(job);
        }
        ready.notify_one();
    }

    /// Take back a quick job for the texture `words` names that no worker
    /// has started: the whole texture if that is quick, else its preview.
    pub fn take(&self, words: [u32; 8]) -> Option<Job> {
        let (lock, _) = &*self.queues;
        let mut queues = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let Queues { urgent, later, .. } = &mut *queues;
        let mut best: Option<(bool, usize, u32)> = None;
        for (is_urgent, queue) in [(true, &*urgent), (false, &*later)] {
            for (index, job) in queue.iter().enumerate() {
                if job.words == words
                    && job.quick()
                    && best.is_none_or(|(_, _, level)| job.first_level < level)
                {
                    best = Some((is_urgent, index, job.first_level));
                }
            }
        }
        let (is_urgent, index, _) = best?;
        if is_urgent {
            urgent.remove(index)
        } else {
            later.remove(index)
        }
    }

    /// The textures prepared since the last call.
    pub fn finished(&self) -> impl Iterator<Item = Prepared> + '_ {
        self.results.try_iter()
    }

    /// The next texture prepared, waiting up to `timeout` for one.
    pub fn wait(&self, timeout: std::time::Duration) -> Option<Prepared> {
        self.results.recv_timeout(timeout).ok()
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        let (lock, ready) = &*self.queues;
        lock.lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .closed = true;
        ready.notify_all();
    }
}

/// Untile (and decode) one texture, level by level.
pub fn prepare(job: Job) -> Prepared {
    let mut texels = Vec::new();
    let mut levels = Vec::new();
    for mip in &job.mips {
        levels.push(texels.len() as u64);
        for layer in 0..u64::from(job.layers.max(1)) {
            let start = ((layer * job.layer_stride + mip.offset) as usize).min(job.source.len());
            let source = &job.source[start..];
            let linear = match job.layout {
                Layout::BlockLinear { .. } => texture::Volume::new(
                    mip.row_bytes,
                    mip.rows,
                    u64::from(mip.gobs_in_y),
                    u64::from(job.gobs_in_z),
                )
                .untile(source, u64::from(job.depth.max(1))),
                Layout::Pitch { .. } => source.to_vec(),
            };
            let slice = (mip.row_bytes * mip.rows).max(1) as usize;
            match job.decode {
                Decode::Astc(block_width, block_height) => {
                    for blocks in linear.chunks(slice) {
                        texels.extend(decode_astc(
                            blocks,
                            mip.width,
                            mip.height,
                            block_width,
                            block_height,
                        ));
                    }
                }
                Decode::Bcn(kind) => {
                    for blocks in linear.chunks(slice) {
                        texels.extend(bcn::decode(kind, blocks, mip.width, mip.height));
                    }
                }
                Decode::None => texels.extend(linear),
            }
        }
    }
    Prepared {
        words: job.words,
        generation: job.generation,
        texels,
        levels,
        first_level: job.first_level,
    }
}

/// ASTC blocks (16 bytes each, rows of blocks packed) to RGBA8 texels.
/// Blocks past the end of `blocks` decode as transparent black.
pub fn decode_astc(
    blocks: &[u8],
    width: u32,
    height: u32,
    block_width: u32,
    block_height: u32,
) -> Vec<u8> {
    let mut texels = vec![0u8; width as usize * height as usize * 4];
    let footprint = astc_decode::Footprint::new(block_width, block_height);
    let _ = astc_decode::astc_decode(blocks, width, height, footprint, |x, y, color| {
        let at = (y as usize * width as usize + x as usize) * 4;
        texels[at..at + 4].copy_from_slice(&color);
    });
    texels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn void_extent_astc_blocks_decode_to_their_color() {
        // A void-extent block: a constant color, here (r, g, b, a) =
        // (0xffff, 0, 0x8000, 0xffff) as UNORM16.
        let mut block = [0u8; 16];
        block[0] = 0xFC;
        block[1] = 0xFD;
        block[2..8].fill(0xFF);
        block[8..10].copy_from_slice(&0xFFFFu16.to_le_bytes());
        block[10..12].copy_from_slice(&0u16.to_le_bytes());
        block[12..14].copy_from_slice(&0x8000u16.to_le_bytes());
        block[14..16].copy_from_slice(&0xFFFFu16.to_le_bytes());
        let texels = decode_astc(&block, 3, 2, 4, 4);
        assert_eq!(texels.len(), 3 * 2 * 4);
        for texel in texels.chunks(4) {
            assert_eq!(texel[0], 0xFF);
            assert_eq!(texel[1], 0);
            assert_eq!(texel[3], 0xFF);
            assert!((0x7F..=0x80).contains(&texel[2]));
        }
    }

    fn job(words: u32, width: u32, height: u32, astc: bool, first_level: u32) -> Job {
        let row_bytes = u64::from(if astc {
            width.div_ceil(4) * 16
        } else {
            width * 4
        });
        let rows = u64::from(if astc { height.div_ceil(4) } else { height });
        Job {
            words: [words; 8],
            generation: 1,
            source: vec![0; (row_bytes * rows) as usize].into(),
            layout: Layout::Pitch {
                stride: row_bytes as u32,
            },
            mips: vec![Mip {
                offset: 0,
                row_bytes,
                rows,
                gobs_in_y: 1,
                width,
                height,
            }],
            first_level,
            decode: if astc {
                Decode::Astc(4, 4)
            } else {
                Decode::None
            },
            layers: 1,
            layer_stride: row_bytes * rows,
            depth: 1,
            gobs_in_z: 1,
        }
    }

    #[test]
    fn small_and_untiled_jobs_are_quick() {
        assert!(job(1, 128, 128, true, 3).quick());
        assert!(!job(1, 1024, 1024, true, 0).quick());
        // LM3's 32x64x32 RGBA8 color lookup table: 256 KiB to untile.
        let table = Job {
            depth: 32,
            ..job(1, 32, 64, false, 0)
        };
        assert!(table.quick());
        assert!(!job(1, 2048, 2048, false, 0).quick());
    }

    #[test]
    fn queued_quick_jobs_are_taken_back() {
        let decoder = Decoder::with_workers(0);
        // A large texture's preview is taken, not the texture itself.
        decoder.submit(job(1, 128, 128, true, 3), true);
        decoder.submit(job(1, 1024, 1024, true, 0), false);
        let taken = decoder.take([1; 8]).expect("the preview is queued");
        assert_eq!(taken.first_level, 3);
        assert!(decoder.take([1; 8]).is_none());
        // A quick texture is taken whole, once.
        decoder.submit(job(2, 32, 32, false, 0), true);
        assert_eq!(decoder.take([2; 8]).map(|job| job.first_level), Some(0));
        assert!(decoder.take([2; 8]).is_none());
        assert!(decoder.take([3; 8]).is_none());
    }

    #[test]
    fn urgent_and_later_jobs_both_come_back() {
        let decoder = Decoder::new();
        let job = |words: u32, first_level: u32| Job {
            words: [words; 8],
            generation: 1,
            source: vec![9; 4].into(),
            layout: Layout::Pitch { stride: 4 },
            mips: vec![Mip {
                offset: 0,
                row_bytes: 4,
                rows: 1,
                gobs_in_y: 1,
                width: 1,
                height: 1,
            }],
            first_level,
            decode: Decode::None,
            layers: 1,
            layer_stride: 4,
            depth: 1,
            gobs_in_z: 1,
        };
        decoder.submit(job(1, 0), false);
        decoder.submit(job(1, 3), true);
        let mut came = Vec::new();
        while came.len() < 2 {
            let prepared = decoder
                .wait(std::time::Duration::from_secs(5))
                .expect("both jobs come back");
            came.push(prepared.first_level);
        }
        came.sort_unstable();
        assert_eq!(came, [0, 3]);
    }

    #[test]
    fn workers_return_prepared_textures() {
        let decoder = Decoder::new();
        decoder.submit(
            Job {
                words: [7; 8],
                generation: 1,
                source: vec![1, 2, 3, 4].into(),
                layout: Layout::Pitch { stride: 4 },
                mips: vec![Mip {
                    offset: 0,
                    row_bytes: 4,
                    rows: 1,
                    gobs_in_y: 1,
                    width: 1,
                    height: 1,
                }],
                first_level: 0,
                decode: Decode::None,
                layers: 1,
                layer_stride: 4,
                depth: 1,
                gobs_in_z: 1,
            },
            true,
        );
        let started = std::time::Instant::now();
        loop {
            if let Some(prepared) = decoder.finished().next() {
                assert_eq!(prepared.words, [7; 8]);
                assert_eq!(prepared.texels, [1, 2, 3, 4]);
                assert_eq!(prepared.levels, [0]);
                break;
            }
            assert!(started.elapsed().as_secs() < 5, "no texture came back");
            std::thread::yield_now();
        }
    }
}
