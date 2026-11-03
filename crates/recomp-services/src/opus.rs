//! `hwopus` (IHardwareOpusDecoderManager): Opus decoders. No sound is
//! played yet, so packets decode to silence; what the caller sees of a
//! packet (bytes consumed, samples produced) comes from the packet itself.
//!
//! A hwopus packet is an 8-byte header (big-endian payload size and final
//! range) followed by the payload: one Opus packet (RFC 6716), or for a
//! multistream decoder one self-delimited packet per stream. Every stream
//! covers the same duration, so the first packet's TOC gives the samples.

use crate::server::{Call, Service};
use crate::services::unimplemented;

const HEADER_SIZE: usize = 8;

/// A malformed packet (hwopus module 111; the description is ours).
const RESULT_INVALID_PACKET: u32 = 111 | (1 << 9);

/// Samples per channel in the Opus packet `packet` decoded at
/// `sample_rate`, from its TOC byte (RFC 6716 section 3.1).
pub fn packet_samples(packet: &[u8], sample_rate: u32) -> Option<u32> {
    let toc = *packet.first()?;
    let config = usize::from(toc >> 3);
    // Frame duration in 2.5 ms units: SILK-only, hybrid, CELT-only.
    let units = match config {
        0..=11 => [4, 8, 16, 24][config & 3],
        12..=15 => [4, 8][config & 1],
        _ => [1, 2, 4, 8][config & 3],
    };
    let frames = match toc & 3 {
        0 => 1,
        1 | 2 => 2,
        _ => u32::from(*packet.get(1)? & 0x3F),
    };
    // A packet holds at most 120 ms.
    if frames == 0 || units * frames > 48 {
        return None;
    }
    Some(units * frames * sample_rate / 400)
}

/// Work buffer sizes: what the decoder state needs per channel, rounded up.
fn work_buffer_size(channels: u32) -> u32 {
    0x8000 + channels.clamp(1, 255) * 0x8000
}

/// `hwopus` (IHardwareOpusDecoderManager)
pub struct HardwareOpusDecoderManager;

impl Service for HardwareOpusDecoderManager {
    fn name(&self) -> String {
        "hwopus".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        // Single-stream parameters are arguments (sample rate, channels);
        // multistream ones come in a buffer that starts the same way.
        let multistream = |call: &Call<'_>| {
            let parameters = call.in_buffer(0);
            let field = |at: usize| {
                parameters
                    .get(at..at + 4)
                    .map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()))
            };
            (field(0), field(4))
        };
        match call.command() {
            // OpenHardwareOpusDecoder(Ex)(parameters, work buffer size,
            // work buffer)
            0 | 4 => {
                let decoder = OpusDecoder::new(call.arg_u32(0), call.arg_u32(4));
                call.push_object(Box::new(decoder));
            }
            // GetWorkBufferSize(Ex)(ExEx)(parameters)
            1 | 5 | 8 => call.push_u32(work_buffer_size(call.arg_u32(4))),
            // OpenHardwareOpusDecoderForMultiStream(Ex)
            2 | 6 => {
                let (sample_rate, channels) = multistream(call);
                call.push_object(Box::new(OpusDecoder::new(sample_rate, channels)));
            }
            // GetWorkBufferSizeForMultiStream(Ex)(ExEx)
            3 | 7 | 9 => {
                let (_, channels) = multistream(call);
                call.push_u32(work_buffer_size(channels));
            }
            _ => return unimplemented(call, "hwopus"),
        }
        Ok(())
    }
}

/// IHardwareOpusDecoder
struct OpusDecoder {
    sample_rate: u32,
    channels: u32,
}

impl OpusDecoder {
    fn new(sample_rate: u32, channels: u32) -> Self {
        println!("[recomp-ipc] opus decoder opened: {sample_rate} Hz, {channels} channels");
        OpusDecoder {
            sample_rate,
            channels: channels.clamp(1, 255),
        }
    }

    /// The bytes a packet takes in `input` and the samples per channel it
    /// decodes to.
    fn measure(&self, input: &[u8]) -> Option<(u32, u32)> {
        let size = u32::from_be_bytes(input.get(..4)?.try_into().ok()?);
        let end = HEADER_SIZE.checked_add(size as usize)?;
        let packet = input.get(HEADER_SIZE..end)?;
        Some((end as u32, packet_samples(packet, self.sample_rate)?))
    }
}

impl Service for OpusDecoder {
    fn name(&self) -> String {
        "hwopus:IHardwareOpusDecoder".to_string()
    }

    fn call(&mut self, call: &mut Call<'_>) -> Result<(), u32> {
        match call.command() {
            // DecodeInterleaved(ForMultiStream)(WithPerf)(AndReset)(input)
            // -> (consumed, samples[, decode time]), PCM16 output
            command @ (0 | 2 | 4..=9) => {
                let Some((consumed, samples)) = self.measure(&call.in_buffer(0)) else {
                    return Err(RESULT_INVALID_PACKET);
                };
                let pcm = vec![0u8; (samples * self.channels * 2) as usize];
                call.write_out(0, &pcm);
                call.push_u32(consumed);
                call.push_u32(samples);
                if command >= 4 {
                    call.push_u64(0);
                }
            }
            // SetContext(ForMultiStream)
            1 | 3 => {}
            _ => return unimplemented(call, "hwopus:IHardwareOpusDecoder"),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::packet_samples;

    #[test]
    fn toc_durations() {
        // SILK 60 ms, hybrid 10 ms, CELT 2.5 ms; one frame each.
        assert_eq!(packet_samples(&[3 << 3], 48_000), Some(2880));
        assert_eq!(packet_samples(&[3 << 3], 16_000), Some(960));
        assert_eq!(packet_samples(&[12 << 3], 48_000), Some(480));
        assert_eq!(packet_samples(&[16 << 3], 48_000), Some(120));
        // Two frames (codes 1 and 2), counted frames (code 3).
        assert_eq!(packet_samples(&[(31 << 3) | 1], 48_000), Some(1920));
        assert_eq!(packet_samples(&[(16 << 3) | 3, 5], 48_000), Some(600));
        // Over 120 ms, no frames, no TOC.
        assert_eq!(packet_samples(&[(3 << 3) | 3, 3], 48_000), None);
        assert_eq!(packet_samples(&[(31 << 3) | 3, 0], 48_000), None);
        assert_eq!(packet_samples(&[], 48_000), None);
    }
}
