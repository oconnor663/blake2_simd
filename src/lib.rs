#[macro_use]
extern crate arrayref;
extern crate byteorder;

use byteorder::{ByteOrder, LittleEndian};
use std::cmp;

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

const SIGMA: [[u8; 16]; 12] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
];

// G is the mixing function, called eight times per round in the compression
// function. V is the 16-word state vector of the compression function, usually
// described as a 4x4 matrix. A, B, C, and D are the mixing indices, set by the
// caller first to the four columns of V, and then to its four diagonals. X and
// Y are words of input, chosen by the caller according to the message
// schedule, SIGMA.
#[inline(always)]
fn g(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
    v[d] = (v[d] ^ v[a]).rotate_right(32);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(24);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(63);
}

#[inline(always)]
fn round(r: usize, m: &[u64; 16], v: &mut [u64; 16]) {
    // Select the message schedule based on the round.
    let s = SIGMA[r];

    // Mix the columns.
    g(v, 0, 4, 8, 12, m[s[0] as usize], m[s[1] as usize]);
    g(v, 1, 5, 9, 13, m[s[2] as usize], m[s[3] as usize]);
    g(v, 2, 6, 10, 14, m[s[4] as usize], m[s[5] as usize]);
    g(v, 3, 7, 11, 15, m[s[6] as usize], m[s[7] as usize]);

    // Mix the rows.
    g(v, 0, 5, 10, 15, m[s[8] as usize], m[s[9] as usize]);
    g(v, 1, 6, 11, 12, m[s[10] as usize], m[s[11] as usize]);
    g(v, 2, 7, 8, 13, m[s[12] as usize], m[s[13] as usize]);
    g(v, 3, 4, 9, 14, m[s[14] as usize], m[s[15] as usize]);
}

// H is the 8-word state vector. `msg` is BLOCKBYTES of input, possibly padded
// with zero bytes in the final block. `count` is the number of bytes fed so
// far, including in this call, though not including padding in the final call.
// `finalize` is set to true only in the final call.
fn compress(h: &mut [u64; 8], msg: &[u8; BLOCKBYTES], count: u128, lastblock: u64) {
    // Initialize the compression state.
    let mut v = [0; 16];
    v[..8].copy_from_slice(h);
    v[8..].copy_from_slice(&IV);
    v[12] ^= count as u64;
    v[13] ^= (count >> 64) as u64;
    v[14] ^= lastblock;
    // v[15] would be the last node flag.

    // Parse the message bytes as ints in little endian order.
    let mut m = [0; 16];
    LittleEndian::read_u64_into(msg, &mut m);

    round(0, &m, &mut v);
    round(1, &m, &mut v);
    round(2, &m, &mut v);
    round(3, &m, &mut v);
    round(4, &m, &mut v);
    round(5, &m, &mut v);
    round(6, &m, &mut v);
    round(7, &m, &mut v);
    round(8, &m, &mut v);
    round(9, &m, &mut v);
    round(10, &m, &mut v);
    round(11, &m, &mut v);

    for i in 0..8 {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

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

pub fn blake2b(input: &[u8]) -> Digest {
    let mut state = State::new();
    state.update(input);
    state.finalize()
}

#[cfg(test)]
mod test {
    extern crate hex;

    use super::*;

    fn eq(h1: &Digest, s2: &str) {
        let s1 = hex::encode(&h1[..]);
        assert_eq!(s1, s2, "hash mismatch");
    }

    #[test]
    fn test_vectors() {
        let io = &[
            (
                &b""[..],
                "786a02f742015903c6c6fd852552d272912f4740e15847618a86e217f71f5419d25e1031afee585313896444934eb04b903a685b1448b755d56f701afe9be2ce",
            ),
            (
                &b"abc"[..],
                "ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d17d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923",
            ),
            (
                &[0; 1000],
                "1ee4e51ecab5210a518f26150e882627ec839967f19d763e1508b12cfefed14858f6a1c9d1f969bc224dc9440f5a6955277e755b9c513f9ba4421c5e50c8d787",
            ),
        ];
        for &(input, output) in io {
            let hash = blake2b(input);
            eq(&hash, output);
        }
    }

    #[test]
    fn test_a_thousand_one_by_one() {
        let mut state = State::new();
        for _ in 0..1000 {
            state.update(&[0]);
        }
        let hash = state.finalize();
        eq(
            &hash,
            "1ee4e51ecab5210a518f26150e882627ec839967f19d763e1508b12cfefed14858f6a1c9d1f969bc224dc9440f5a6955277e755b9c513f9ba4421c5e50c8d787",
        );
    }

    #[test]
    fn test_two_times_five_hundred() {
        let mut state = State::new();
        state.update(&[0; 500]);
        state.update(&[0; 500]);
        let hash = state.finalize();
        eq(
            &hash,
            "1ee4e51ecab5210a518f26150e882627ec839967f19d763e1508b12cfefed14858f6a1c9d1f969bc224dc9440f5a6955277e755b9c513f9ba4421c5e50c8d787",
        );
    }
}
