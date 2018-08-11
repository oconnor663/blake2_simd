#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate core;

#[macro_use]
extern crate arrayref;
extern crate byteorder;

use byteorder::{ByteOrder, LittleEndian};
use core::cmp;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2;
mod portable;

#[cfg(test)]
mod test;

pub const BLOCKBYTES: usize = 128;
pub const OUTBYTES: usize = 64;
pub const KEYBYTES: usize = 64;
pub const SALTBYTES: usize = 16;
pub const PERSONALBYTES: usize = 16;

const IV: [u64; 8] = [
    0x6A09E667F3BCC908,
    0xBB67AE8584CAA73B,
    0x3C6EF372FE94F82B,
    0xA54FF53A5F1D36F1,
    0x510E527FADE682D1,
    0x9B05688C2B3E6C1F,
    0x1F83D9ABFB41BD6B,
    0x5BE0CD19137E2179,
];

// Safety note: The compression interface is unsafe in general, because calling the AVX2
// implementation on a platform that doesn't support AVX2 is undefined behavior. That said, the
// portable implementation is all safe code.
type CompressFn = unsafe fn(&mut StateWords, &Block, count: u128, lastblock: u64);
type Digest = [u8; OUTBYTES];
type StateWords = [u64; 8];
type Block = [u8; BLOCKBYTES];

pub struct State {
    h: StateWords,
    buf: Block,
    buflen: usize,
    count: u128,
    compress_fn: CompressFn,
}

impl State {
    pub fn new() -> Self {
        let mut h = IV;
        // Mask in the digest length.
        h[0] ^= OUTBYTES as u64;
        // Mask in the fanout and depth default parameters.
        h[0] ^= 0x01010000;
        Self {
            h,
            compress_fn: default_compress_impl(),
            buf: [0; BLOCKBYTES],
            buflen: 0,
            count: 0,
        }
    }

    fn fill_buf(&mut self, input: &mut &[u8]) {
        let take = cmp::min(BLOCKBYTES - self.buflen, input.len());
        self.buf[self.buflen..self.buflen + take].copy_from_slice(&input[..take]);
        self.buflen += take;
        self.count += take as u128;
        *input = &input[take..];
    }

    // array_ref triggers unused_unsafe (https://github.com/droundy/arrayref/pull/14)
    #[allow(unused_unsafe)]
    pub fn update(&mut self, mut input: &[u8]) {
        // If we have a partial buffer, try to complete it. If we complete it and there's more
        // input waiting (so we know we don't need to finalize), compress it.
        if self.buflen > 0 {
            self.fill_buf(&mut input);
            if !input.is_empty() {
                unsafe {
                    (self.compress_fn)(&mut self.h, &self.buf, self.count, 0);
                }
                self.buflen = 0;
            }
        }
        // If there's more than a block of input left, compress it directly instead of buffering it.
        while input.len() > BLOCKBYTES {
            self.count += BLOCKBYTES as u128;
            unsafe {
                (self.compress_fn)(&mut self.h, array_ref!(input, 0, BLOCKBYTES), self.count, 0);
            }
            input = &input[BLOCKBYTES..];
        }
        // Buffer any remaining input, to be either compressed or finalized in a subsequent call.
        self.fill_buf(&mut input);
    }

    pub fn finalize(&mut self) -> Digest {
        for i in self.buflen..BLOCKBYTES {
            self.buf[i] = 0;
        }
        unsafe {
            (self.compress_fn)(&mut self.h, &self.buf, self.count, !0);
        }
        let mut out = [0; OUTBYTES];
        LittleEndian::write_u64_into(&self.h, &mut out);
        out
    }
}

#[cfg(feature = "std")]
impl std::io::Write for State {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.update(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[allow(unreachable_code)]
fn default_compress_impl() -> CompressFn {
    // If AVX2 is enabled at the top level for the whole build (using something like
    // RUSTFLAGS="-C target-cpu=native"), return the AVX2 implementation without doing dynamic
    // feature detection. This isn't common, but it's the only way to use AVX2 with no_std, at
    // least until more features get stabilized in the future.
    #[cfg(all(target_feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        return avx2::compress;
    }
    // Do dynamic feature detection at runtime, and use AVX2 if the current CPU supports it. This
    // is what the default build does. Note that no_std doesn't currently support dynamic detection.
    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        if is_x86_feature_detected!("avx2") {
            return avx2::compress;
        }
    }
    // On other platforms (non-x86 or pre-AVX2) use the portable implementation.
    portable::compress
}

pub fn blake2b(input: &[u8]) -> Digest {
    let mut state = State::new();
    state.update(input);
    state.finalize()
}

// This module is pub for internal benchmarks only. Please don't use it.
#[doc(hidden)]
pub mod benchmarks {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub use avx2::compress as compress_avx2;
    pub use portable::compress as compress_portable;

    pub fn force_portable(state: &mut ::State) {
        state.compress_fn = compress_portable;
    }
}
