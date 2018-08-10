#[macro_use]
extern crate arrayref;
extern crate byteorder;

use byteorder::{ByteOrder, LittleEndian};
use std::cmp;

// These modules are only pub for benchmarks. Their API isn't stable.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[doc(hidden)]
pub mod avx2;
#[doc(hidden)]
pub mod portable;

#[cfg(test)]
mod test;

type Digest = [u8; OUTBYTES];

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

pub struct State {
    h: [u64; 8],
    buf: [u8; BLOCKBYTES],
    buflen: usize,
    count: u128,
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

    pub fn update(&mut self, mut input: &[u8]) {
        // If we have a partial buffer, try to complete it. If we complete it and there's more
        // input waiting (so we know we don't need to finalize), compress it.
        if self.buflen > 0 {
            self.fill_buf(&mut input);
            if !input.is_empty() {
                compress(&mut self.h, &self.buf, self.count, 0);
                self.buflen = 0;
            }
        }
        // If there's more than a block of input left, compress it directly instead of buffering it.
        while input.len() > BLOCKBYTES {
            self.count += BLOCKBYTES as u128;
            compress(&mut self.h, array_ref!(input, 0, BLOCKBYTES), self.count, 0);
            input = &input[BLOCKBYTES..];
        }
        // Buffer any remaining input, to be either compressed or finalized in a subsequent call.
        self.fill_buf(&mut input);
    }

    pub fn finalize(&mut self) -> Digest {
        for i in self.buflen..BLOCKBYTES {
            self.buf[i] = 0;
        }
        compress(&mut self.h, &self.buf, self.count, !0);
        let mut out = [0; OUTBYTES];
        LittleEndian::write_u64_into(&self.h, &mut out);
        out
    }
}

fn compress(h: &mut [u64; 8], block: &[u8; BLOCKBYTES], count: u128, lastblock: u64) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2::compress(h, block, count, lastblock) };
        }
    }
    portable::compress(h, block, count, lastblock)
}

pub fn blake2b(input: &[u8]) -> Digest {
    let mut state = State::new();
    state.update(input);
    state.finalize()
}
