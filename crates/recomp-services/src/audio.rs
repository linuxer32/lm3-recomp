//! `audren:u` (audio renderer), `audout:u` (PCM output) and the audio
//! device service. Audio out plays on the host's output (`AudioSink`, which
//! the window backend registers); the renderer produces no sound yet: it
//! accepts updates and answers with the status sections the SDK expects,
//! and its system event ticks every 5 ms like the DSP's.

use std::sync::OnceLock;
use std::time::Duration;

use crate::server::{Call, Service};
use crate::services::{new_event, unimplemented};

const SAMPLE_RATE: u32 = 48_000;
const DEVICE_NAME: &[u8] = b"AudioStereoJackOutput";

/// Output status sizes per object (and of the fixed sections).
const MEMORY_POOL_STATUS: u32 = 0x10;
const VOICE_STATUS: u32 = 0x10;
const EFFECT_STATUS: u32 = 0x10;
const SINK_STATUS: u32 = 0x20;
const PERFORMANCE_STATUS: u32 = 0x10;
const BEHAVIOR_STATUS: u32 = 0xB0;
const RENDER_INFO_STATUS: u32 = 0x10;

/// The host's audio output: interleaved 16-bit stereo samples at 48 kHz.
pub trait AudioSink: Send + Sync {
    fn play(&self, samples: &[i16]);
    /// Samples (both channels counted) queued and not heard yet.
    fn queued(&self) -> usize;
    fn clear(&self);
}

static SINK: OnceLock<Box<dyn AudioSink>> = OnceLock::new();

/// Where audio out plays from now on (the first sink registered stays).
pub fn register_audio_sink(sink: Box<dyn AudioSink>) {
    let _ = SINK.set(sink);
}

/// Most audio the host may hold before it is dropped: buffers are handed
/// over as they start playing on audio out's clock, so more means the two
/// clocks drifted apart.
const MAX_QUEUED_SAMPLES: usize = SAMPLE_RATE as usize * 2 / 5;

/// `RECOMP_AUDIO_DUMP=<file>`: also append what plays to `file` (raw
/// 16-bit stereo at 48 kHz) to check the sound without listening.
fn dump(samples: &[i16]) {
    use std::io::Write;
    static FILE: OnceLock<Option<std::sync::Mutex<std::fs::File>>> = OnceLock::new();
    let file = FILE.get_or_init(|| {
        let path = std::env::var_os("RECOMP_AUDIO_DUMP")?;
        std::fs::File::create(path).ok().map(std::sync::Mutex::new)
    });
    if let Some(file) = file {
        let bytes: Vec<u8> = samples
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect();
        let _ = file
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .write_all(&bytes);
    }
}

fn play(samples: &[i16]) {
    dump(samples);
    let Some(sink) = SINK.get() else {
        return;
    };
    if sink.queued() > MAX_QUEUED_SAMPLES {
        sink.clear();
    }
    sink.play(samples);
}

/// An event signalled every `period` by a timer thread: (writable,
/// readable).
fn ticking_event(name: &str, period: Duration) -> u32 {
    let (write, read) = crate::kernel::kernel().create_event();
    let _ = std::thread::Builder::new()
        .name(name.to_string())
        .spawn(move || loop {
            std::thread::sleep(period);
            crate::kernel::kernel().signal_event(write);
        });
    read
}

/// `audren:u` (IAudioRendererManager)
pub struct AudioRendererManager;

/// AudioRendererParameter fields the renderer needs.
#[derive(Debug, Clone, Copy)]
struct RendererParameters {
    sample_rate: u32,
    sample_count: u32,
    mix_buffer_count: u32,
    voices: u32,
    sinks: u32,
    effects: u32,
    revision: u32,
}

impl RendererParameters {
    fn from_call(call: &Call<'_>) -> Self {
        RendererParameters {
            sample_rate: call.arg_u32(0x00),
            sample_count: call.arg_u32(0x04),
            mix_buffer_count: call.arg_u32(0x08),
            voices: call.arg_u32(0x10),
            sinks: call.arg_u32(0x14),
            effects: call.arg_u32(0x18),
            revision: call.arg_u32(0x30),
        }
    }

    /// The N of the "REVN" revision magic.
    fn revision_number(&self) -> u32 {
        (self.revision >> 24).saturating_sub(u32::from(b'0'))
    }

    /// Memory pools: one per effect and four per voice.
    fn memory_pools(&self) -> u32 {
        self.effects + self.voices * 4
    }

    /// The work buffer the SDK must provide: generous and aligned.
    fn work_buffer_size(&self) -> u64 {
        let objects = u64::from(self.voices + self.sinks + self.effects + self.mix_buffer_count);
        (0x10_0000 + objects * 0x4000 + u64::from(self.sample_count) * 0x100)
            .next_multiple_of(0x1000)
    }
}

impl Service for AudioRendererManager {
    fn name(&self) -> String {
        "audren:u".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // OpenAudioRenderer(parameters, transfer memory size, aruid),
            // OpenAudioRendererForManualExecution
            0 | 3 => {
                let parameters = RendererParameters::from_call(call);
                println!("[recomp-ipc] audio renderer opened: {parameters:?}");
                call.push_object(Box::new(AudioRenderer::new(parameters)));
            }
            // GetWorkBufferSize(parameters) -> size
            1 => call.push_u64(RendererParameters::from_call(call).work_buffer_size()),
            // GetAudioDeviceService(aruid),
            // GetAudioDeviceServiceWithRevisionInfo(revision, aruid)
            2 | 4 => call.push_object(Box::new(AudioDevice)),
            _ => return unimplemented(call, "audren:u"),
        }
        Ok(())
    }
}

/// IAudioRenderer
struct AudioRenderer {
    parameters: RendererParameters,
    started: bool,
    system_event: u32,
    frames: u64,
    time_limit: u32,
}

impl AudioRenderer {
    fn new(parameters: RendererParameters) -> Self {
        AudioRenderer {
            parameters,
            started: false,
            system_event: ticking_event("recomp-audren", Duration::from_millis(5)),
            frames: 0,
            time_limit: 100,
        }
    }

    /// The update reply: a header with each section's size, then the
    /// sections (all statuses zero: nothing played, nothing failed).
    fn update_output(&self) -> Vec<u8> {
        let p = &self.parameters;
        let sections = [
            // behaviour (error info), memory pools, voices, voice
            // resources, effects, mixes, sinks, performance, (reserved),
            // render info
            BEHAVIOR_STATUS,
            p.memory_pools() * MEMORY_POOL_STATUS,
            p.voices * VOICE_STATUS,
            0,
            p.effects * EFFECT_STATUS,
            0,
            p.sinks * SINK_STATUS,
            PERFORMANCE_STATUS,
            0,
            // The elapsed frame count is reported from REV5 on.
            if p.revision_number() >= 5 {
                RENDER_INFO_STATUS
            } else {
                0
            },
        ];
        let total = 0x40 + sections.iter().sum::<u32>();
        let mut out = vec![0u8; total as usize];
        out[0..4].copy_from_slice(&p.revision.to_le_bytes());
        for (index, size) in sections.iter().enumerate() {
            let at = 4 + index * 4;
            out[at..at + 4].copy_from_slice(&size.to_le_bytes());
        }
        out[0x3C..0x40].copy_from_slice(&total.to_le_bytes());
        // Sections follow in update order (memory pools, voices, effects,
        // sinks, performance, errors, render info); the elapsed frame count
        // is last.
        if sections[9] != 0 {
            let at = (total - RENDER_INFO_STATUS) as usize;
            out[at..at + 8].copy_from_slice(&self.frames.to_le_bytes());
        }
        out
    }
}

impl Service for AudioRenderer {
    fn name(&self) -> String {
        "audren:IAudioRenderer".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            0 => call.push_u32(self.parameters.sample_rate),
            1 => call.push_u32(self.parameters.sample_count),
            2 => call.push_u32(self.parameters.mix_buffer_count),
            // GetState: 0 started, 1 stopped.
            3 => call.push_u32(u32::from(!self.started)),
            // RequestUpdate(input) -> (output, performance),
            // RequestUpdateAuto
            4 | 10 => {
                self.frames += 1;
                let output = self.update_output();
                call.write_out(0, &output);
                call.write_out(1, &[]);
            }
            5 => self.started = true,
            6 => self.started = false,
            // QuerySystemEvent
            7 => call.push_copy_handle(self.system_event),
            // SetRenderingTimeLimit, GetRenderingTimeLimit
            8 => self.time_limit = call.arg_u32(0),
            9 => call.push_u32(self.time_limit),
            // ExecuteAudioRendererRendering
            11 => {}
            _ => return unimplemented(call, "audren:IAudioRenderer"),
        }
        Ok(())
    }
}

/// IAudioDevice: one stereo output.
struct AudioDevice;

impl AudioDevice {
    fn name_buffer() -> Vec<u8> {
        let mut name = vec![0u8; 0x100];
        name[..DEVICE_NAME.len()].copy_from_slice(DEVICE_NAME);
        name
    }
}

impl Service for AudioDevice {
    fn name(&self) -> String {
        "audren:IAudioDevice".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // ListAudioDeviceName(Auto) -> (names, count)
            0 | 6 => {
                call.write_out(0, &Self::name_buffer());
                call.push_u32(1);
            }
            // SetAudioDeviceOutputVolume(Auto)
            1 | 7 => {}
            // GetAudioDeviceOutputVolume(Auto)
            2 | 8 => call.push_u32(1f32.to_bits()),
            // GetActiveAudioDeviceName(Auto), GetActiveAudioOutputDeviceName
            3 | 10 | 13 => {
                call.write_out(0, &Self::name_buffer());
            }
            // QueryAudioDeviceSystemEvent, ...InputEvent, ...OutputEvent
            4 | 11 | 12 => call.push_copy_handle(new_event()),
            // GetActiveChannelCount
            5 => call.push_u32(2),
            _ => return unimplemented(call, "audren:IAudioDevice"),
        }
        Ok(())
    }
}

/// `audout:u` (IAudioOutManager): one stereo 48 kHz output.
pub struct AudioOutManager;

impl Service for AudioOutManager {
    fn name(&self) -> String {
        "audout:u".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // ListAudioOuts(Auto) -> (names, count)
            0 | 2 => {
                let mut name = vec![0u8; 0x100];
                name[..9].copy_from_slice(b"DeviceOut");
                call.write_out(0, &name);
                call.push_u32(1);
            }
            // OpenAudioOut(Auto)(parameters, aruid, name) ->
            // (sample rate, channels, format, state), IAudioOut, name
            1 | 3 => {
                let mut name = vec![0u8; 0x100];
                name[..9].copy_from_slice(b"DeviceOut");
                call.write_out(0, &name);
                call.push_u32(SAMPLE_RATE);
                call.push_u32(2);
                // PcmInt16, stopped.
                call.push_u32(2);
                call.push_u32(1);
                call.push_object(Box::new(AudioOut::new()));
            }
            _ => return unimplemented(call, "audout:u"),
        }
        Ok(())
    }
}

/// IAudioOut: appended buffers play one after another in real time (2
/// channels of 16-bit samples at 48 kHz; nothing is heard yet) and are
/// released, with the buffer event signalled, when they finish.
///
/// Releasing them at once, as this used to, let LM3's audio thread spin:
/// it polls for free buffers while holding the audio mutex, and the main
/// thread, waiting for that mutex, stalled at the title screen.
struct AudioOut {
    playback: std::sync::Arc<Playback>,
    volume: u32,
    event: (u32, u32),
}

/// The buffers of one audio out and the clock that plays them.
#[derive(Default)]
struct Playback {
    state: std::sync::Mutex<PlaybackState>,
    changed: std::sync::Condvar,
}

#[derive(Default)]
struct PlaybackState {
    started: bool,
    /// Appended, not yet played: tag, length in samples and the samples
    /// (interleaved stereo), handed to the host as the buffer starts.
    queued: std::collections::VecDeque<(u64, u64, Vec<i16>)>,
    /// When the buffer at the head of `queued` finishes playing.
    head_end: Option<std::time::Instant>,
    /// Played, not yet collected by GetReleasedAudioOutBuffers.
    released: Vec<u64>,
    played: u64,
}

impl PlaybackState {
    fn duration(samples: u64) -> Duration {
        Duration::from_nanos(samples * 1_000_000_000 / u64::from(SAMPLE_RATE))
    }

    /// Start the head buffer's clock, if it is not running.
    fn start_head(&mut self, now: std::time::Instant) {
        if self.started && self.head_end.is_none() {
            self.head_end = self.queued.front().map(|(_, samples, pcm)| {
                play(pcm);
                now + Self::duration(*samples)
            });
        }
    }

    /// Release every buffer that has finished by `now`; true if any did.
    fn advance(&mut self, now: std::time::Instant) -> bool {
        let mut any = false;
        while let Some(end) = self.head_end.filter(|end| *end <= now) {
            let Some((tag, samples, _)) = self.queued.pop_front() else {
                self.head_end = None;
                break;
            };
            self.released.push(tag);
            self.played += samples;
            any = true;
            // The next buffer starts where this one ended, unless the clock
            // fell far behind (the host was busy): then it starts now.
            let start = if now.duration_since(end) > Duration::from_millis(100) {
                now
            } else {
                end
            };
            self.head_end = self.queued.front().map(|(_, samples, pcm)| {
                play(pcm);
                start + Self::duration(*samples)
            });
        }
        any
    }
}

/// Collect up to `capacity` released tags: (count, the tags as bytes padded
/// with zeros to the capacity).
fn released_tags(state: &mut PlaybackState, capacity: usize) -> (u32, Vec<u8>) {
    state.advance(std::time::Instant::now());
    let count = capacity.min(state.released.len());
    let mut tags: Vec<u8> = state
        .released
        .drain(..count)
        .flat_map(u64::to_le_bytes)
        .collect();
    tags.resize(capacity * 8, 0);
    (count as u32, tags)
}

impl AudioOut {
    fn new() -> Self {
        let event = crate::kernel::kernel().create_event();
        let playback = std::sync::Arc::new(Playback::default());
        let clock = std::sync::Arc::clone(&playback);
        let write = event.0;
        let _ = std::thread::Builder::new()
            .name("recomp-audout".to_string())
            .spawn(move || {
                let mut state = clock.state.lock().unwrap_or_else(|e| e.into_inner());
                loop {
                    if state.advance(std::time::Instant::now()) {
                        crate::kernel::kernel().signal_event(write);
                    }
                    state = match state.head_end {
                        Some(end) => {
                            let wait = end.saturating_duration_since(std::time::Instant::now());
                            clock
                                .changed
                                .wait_timeout(state, wait)
                                .unwrap_or_else(|e| e.into_inner())
                                .0
                        }
                        None => clock.changed.wait(state).unwrap_or_else(|e| e.into_inner()),
                    };
                }
            });
        AudioOut {
            playback,
            volume: 1f32.to_bits(),
            event,
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, PlaybackState> {
        self.playback
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl Service for AudioOut {
    fn name(&self) -> String {
        "audout:IAudioOut".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // GetAudioOutState: 0 started, 1 stopped.
            0 => call.push_u32(u32::from(!self.state().started)),
            // Start, Stop
            1 | 2 => {
                let mut state = self.state();
                state.started = call.command() == 1;
                if state.started {
                    state.start_head(std::time::Instant::now());
                } else {
                    state.head_end = None;
                }
                self.playback.changed.notify_all();
            }
            // AppendAudioOutBuffer(Auto)(tag, buffer)
            3 | 7 => {
                let tag = call.arg_u64(0);
                let buffer = call.in_buffer(0);
                // AudioOutBuffer: next, sample buffer, capacity, size, offset.
                let field = |at: usize| {
                    buffer
                        .get(at..at + 8)
                        .map_or(0, |b| u64::from_le_bytes(b.try_into().unwrap()))
                };
                let (address, size, offset) = (field(0x8), field(0x18), field(0x20));
                let volume = f32::from_bits(self.volume);
                let pcm: Vec<i16> = call
                    .mem
                    .read_bytes(address.wrapping_add(offset), size.min(1 << 20) as usize)
                    .unwrap_or_default()
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|sample| {
                        let value = f32::from(i16::from_le_bytes(*sample)) * volume;
                        value.clamp(-32768.0, 32767.0) as i16
                    })
                    .collect();
                let mut state = self.state();
                state.queued.push_back((tag, size / 4, pcm));
                state.start_head(std::time::Instant::now());
                self.playback.changed.notify_all();
            }
            // RegisterBufferEvent
            4 => call.push_copy_handle(self.event.1),
            // GetReleasedAudioOutBuffers(Auto) -> (tags, count). The rest of
            // the buffer is zeroed: nn::audio::GetReleasedAudioOutBuffer
            // returns its first tag whatever the count, so a stale tag left
            // there reads as another released buffer (LM3's sink then
            // drained "released" buffers forever, holding the audio lock).
            5 | 8 => {
                let capacity = call.out_buffer(0).map_or(0, |buffer| buffer.size) as usize / 8;
                let (count, tags) = released_tags(&mut self.state(), capacity);
                call.write_out(0, &tags);
                call.push_u32(count);
            }
            // ContainsAudioOutBuffer(tag): appended and not yet collected.
            6 => {
                let tag = call.arg_u64(0);
                let state = self.state();
                let contains = state.queued.iter().any(|(queued, _, _)| *queued == tag)
                    || state.released.contains(&tag);
                call.push_u8(u8::from(contains));
            }
            // GetAudioOutBufferCount
            9 => {
                let count = self.state().queued.len();
                call.push_u32(count as u32);
            }
            // GetAudioOutPlayedSampleCount
            10 => {
                let mut state = self.state();
                state.advance(std::time::Instant::now());
                call.push_u64(state.played);
            }
            // FlushAudioOutBuffers: drop what has not played.
            11 => {
                let mut state = self.state();
                let flushed = !state.queued.is_empty();
                while let Some((tag, _, _)) = state.queued.pop_front() {
                    state.released.push(tag);
                }
                state.head_end = None;
                drop(state);
                if flushed {
                    crate::kernel::kernel().signal_event(self.event.0);
                }
                call.push_u8(u8::from(flushed));
            }
            12 => self.volume = call.arg_u32(0),
            13 => call.push_u32(self.volume),
            _ => return unimplemented(call, "audout:IAudioOut"),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn audio_out_buffers_play_one_after_another() {
        let now = Instant::now();
        let mut state = PlaybackState {
            started: true,
            ..PlaybackState::default()
        };
        // Two 10 ms buffers (480 samples each).
        state.queued.push_back((1, 480, Vec::new()));
        state.queued.push_back((2, 480, Vec::new()));
        state.start_head(now);
        assert!(!state.advance(now + Duration::from_millis(5)));
        assert!(state.advance(now + Duration::from_millis(10)));
        assert_eq!(state.released, [1]);
        assert!(!state.advance(now + Duration::from_millis(15)));
        assert!(state.advance(now + Duration::from_millis(20)));
        assert_eq!(state.released, [1, 2]);
        assert_eq!(state.played, 960);
        assert_eq!(state.head_end, None);
    }

    /// Records what audio out plays.
    struct Recorder(std::sync::Mutex<Vec<i16>>);

    impl AudioSink for Recorder {
        fn play(&self, samples: &[i16]) {
            self.0.lock().unwrap().extend_from_slice(samples);
        }
        fn queued(&self) -> usize {
            0
        }
        fn clear(&self) {}
    }

    #[test]
    fn buffers_reach_the_host_as_they_start_playing() {
        static RECORDED: OnceLock<&'static Recorder> = OnceLock::new();
        let recorder = *RECORDED.get_or_init(|| Box::leak(Box::new(Recorder(Default::default()))));
        struct Shared(&'static Recorder);
        impl AudioSink for Shared {
            fn play(&self, samples: &[i16]) {
                self.0.play(samples);
            }
            fn queued(&self) -> usize {
                0
            }
            fn clear(&self) {}
        }
        register_audio_sink(Box::new(Shared(recorder)));
        let now = Instant::now();
        let mut state = PlaybackState {
            started: true,
            ..PlaybackState::default()
        };
        state.queued.push_back((1, 2, vec![1, -1, 2, -2]));
        state.queued.push_back((2, 1, vec![3, -3]));
        state.start_head(now);
        assert_eq!(*recorder.0.lock().unwrap(), [1, -1, 2, -2]);
        state.advance(now + Duration::from_millis(1));
        assert_eq!(*recorder.0.lock().unwrap(), [1, -1, 2, -2, 3, -3]);
    }

    #[test]
    fn released_tags_are_zero_padded() {
        let mut state = PlaybackState {
            released: vec![0x1234],
            ..PlaybackState::default()
        };
        let (count, tags) = released_tags(&mut state, 2);
        assert_eq!(count, 1);
        assert_eq!(tags.len(), 16);
        assert_eq!(&tags[..8], &0x1234u64.to_le_bytes());
        assert!(tags[8..].iter().all(|byte| *byte == 0));
        // Nothing released: the tag the SDK returns is null.
        let (count, tags) = released_tags(&mut state, 1);
        assert_eq!((count, tags), (0, vec![0; 8]));
    }

    #[test]
    fn stopped_audio_out_does_not_play() {
        let now = Instant::now();
        let mut state = PlaybackState::default();
        state.queued.push_back((1, 480, Vec::new()));
        state.start_head(now);
        assert!(!state.advance(now + Duration::from_secs(1)));
        assert!(state.released.is_empty());
    }
}
