use byteorder::{ByteOrder, LittleEndian};

use crate::Block;
use crate::StateWords;
use crate::IV;
use crate::SIGMA;

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
pub fn compress(h: &mut StateWords, msg: &Block, count: u128, lastblock: u64, lastnode: u64) {
    // Initialize the compression state.
    let mut v = [
        h[0],
        h[1],
        h[2],
        h[3],
        h[4],
        h[5],
        h[6],
        h[7],
        IV[0],
        IV[1],
        IV[2],
        IV[3],
        IV[4] ^ count as u64,
        IV[5] ^ (count >> 64) as u64,
        IV[6] ^ lastblock,
        IV[7] ^ lastnode,
    ];

    // Parse the message bytes as ints in little endian order.
    let msg_refs = array_refs!(msg, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8);
    let m = [
        LittleEndian::read_u64(msg_refs.0),
        LittleEndian::read_u64(msg_refs.1),
        LittleEndian::read_u64(msg_refs.2),
        LittleEndian::read_u64(msg_refs.3),
        LittleEndian::read_u64(msg_refs.4),
        LittleEndian::read_u64(msg_refs.5),
        LittleEndian::read_u64(msg_refs.6),
        LittleEndian::read_u64(msg_refs.7),
        LittleEndian::read_u64(msg_refs.8),
        LittleEndian::read_u64(msg_refs.9),
        LittleEndian::read_u64(msg_refs.10),
        LittleEndian::read_u64(msg_refs.11),
        LittleEndian::read_u64(msg_refs.12),
        LittleEndian::read_u64(msg_refs.13),
        LittleEndian::read_u64(msg_refs.14),
        LittleEndian::read_u64(msg_refs.15),
    ];

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

    h[0] ^= v[0] ^ v[8];
    h[1] ^= v[1] ^ v[9];
    h[2] ^= v[2] ^ v[10];
    h[3] ^= v[3] ^ v[11];
    h[4] ^= v[4] ^ v[12];
    h[5] ^= v[5] ^ v[13];
    h[6] ^= v[6] ^ v[14];
    h[7] ^= v[7] ^ v[15];
}

pub fn compress4(
    h0: &mut StateWords,
    h1: &mut StateWords,
    h2: &mut StateWords,
    h3: &mut StateWords,
    msg0: &Block,
    msg1: &Block,
    msg2: &Block,
    msg3: &Block,
    count0: u128,
    count1: u128,
    count2: u128,
    count3: u128,
    lastblock0: u64,
    lastblock1: u64,
    lastblock2: u64,
    lastblock3: u64,
    lastnode0: u64,
    lastnode1: u64,
    lastnode2: u64,
    lastnode3: u64,
) {
    compress(h0, msg0, count0, lastblock0, lastnode0);
    compress(h1, msg1, count1, lastblock1, lastnode1);
    compress(h2, msg2, count2, lastblock2, lastnode2);
    compress(h3, msg3, count3, lastblock3, lastnode3);
}
