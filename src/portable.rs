use byteorder::{ByteOrder, LittleEndian};

use super::*;
use guts::u64x2;
use guts::u64x4;

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

pub fn compress2(
    h0: &mut StateWords,
    h1: &mut StateWords,
    msg0: &Block,
    msg1: &Block,
    count0: u128,
    count1: u128,
    lastblock0: u64,
    lastblock1: u64,
    lastnode0: u64,
    lastnode1: u64,
) {
    compress(h0, msg0, count0, lastblock0, lastnode0);
    compress(h1, msg1, count1, lastblock1, lastnode1);
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

pub fn hash4_exact(
    params: &Params,
    input0: &[u8],
    input1: &[u8],
    input2: &[u8],
    input3: &[u8],
) -> [Hash; 4] {
    [
        params.to_state().update(input0).finalize(),
        params.to_state().update(input1).finalize(),
        params.to_state().update(input2).finalize(),
        params.to_state().update(input3).finalize(),
    ]
}

pub fn transpose2(words0: &[u64; 8], words1: &[u64; 8]) -> [u64x2; 8] {
    [
        u64x2([words0[0], words1[0]]),
        u64x2([words0[1], words1[1]]),
        u64x2([words0[2], words1[2]]),
        u64x2([words0[3], words1[3]]),
        u64x2([words0[4], words1[4]]),
        u64x2([words0[5], words1[5]]),
        u64x2([words0[6], words1[6]]),
        u64x2([words0[7], words1[7]]),
    ]
}

pub fn untranspose2(transposed: &[u64x2; 8], out0: &mut [u64; 8], out1: &mut [u64; 8]) {
    *out0 = [
        transposed[0][0],
        transposed[1][0],
        transposed[2][0],
        transposed[3][0],
        transposed[4][0],
        transposed[5][0],
        transposed[6][0],
        transposed[7][0],
    ];
    *out1 = [
        transposed[0][1],
        transposed[1][1],
        transposed[2][1],
        transposed[3][1],
        transposed[4][1],
        transposed[5][1],
        transposed[6][1],
        transposed[7][1],
    ];
}

#[inline(always)]
fn load_from_2(words: &[u64x2; 8], i: usize) -> [u64; 8] {
    [
        words[0][i],
        words[1][i],
        words[2][i],
        words[3][i],
        words[4][i],
        words[5][i],
        words[6][i],
        words[7][i],
    ]
}

#[inline(always)]
fn store_to_2(whole: &mut [u64x2; 8], part: &[u64; 8], i: usize) {
    whole[0][i] = part[0];
    whole[1][i] = part[1];
    whole[2][i] = part[2];
    whole[3][i] = part[3];
    whole[4][i] = part[4];
    whole[5][i] = part[5];
    whole[6][i] = part[6];
    whole[7][i] = part[7];
}

pub fn compress2_transposed(
    h_vecs: &mut [u64x2; 8],
    msg0: &Block,
    msg1: &Block,
    count_low: &u64x2,
    count_high: &u64x2,
    lastblock: &u64x2,
    lastnode: &u64x2,
) {
    let mut state0 = load_from_2(h_vecs, 0);
    let count0 = count_low[0] as u128 + ((count_high[0] as u128) << 64);
    compress(&mut state0, msg0, count0, lastblock[0], lastnode[0]);
    store_to_2(h_vecs, &state0, 0);

    let mut state1 = load_from_2(h_vecs, 1);
    let count1 = count_low[1] as u128 + ((count_high[1] as u128) << 64);
    compress(&mut state1, msg1, count1, lastblock[1], lastnode[1]);
    store_to_2(h_vecs, &state1, 1);
}

pub fn transpose4(
    words0: &[u64; 8],
    words1: &[u64; 8],
    words2: &[u64; 8],
    words3: &[u64; 8],
) -> [u64x4; 8] {
    [
        u64x4([words0[0], words1[0], words2[0], words3[0]]),
        u64x4([words0[1], words1[1], words2[1], words3[1]]),
        u64x4([words0[2], words1[2], words2[2], words3[2]]),
        u64x4([words0[3], words1[3], words2[3], words3[3]]),
        u64x4([words0[4], words1[4], words2[4], words3[4]]),
        u64x4([words0[5], words1[5], words2[5], words3[5]]),
        u64x4([words0[6], words1[6], words2[6], words3[6]]),
        u64x4([words0[7], words1[7], words2[7], words3[7]]),
    ]
}

pub fn untranspose4(
    transposed: &[u64x4; 8],
    out0: &mut [u64; 8],
    out1: &mut [u64; 8],
    out2: &mut [u64; 8],
    out3: &mut [u64; 8],
) {
    *out0 = [
        transposed[0][0],
        transposed[1][0],
        transposed[2][0],
        transposed[3][0],
        transposed[4][0],
        transposed[5][0],
        transposed[6][0],
        transposed[7][0],
    ];
    *out1 = [
        transposed[0][1],
        transposed[1][1],
        transposed[2][1],
        transposed[3][1],
        transposed[4][1],
        transposed[5][1],
        transposed[6][1],
        transposed[7][1],
    ];
    *out2 = [
        transposed[0][2],
        transposed[1][2],
        transposed[2][2],
        transposed[3][2],
        transposed[4][2],
        transposed[5][2],
        transposed[6][2],
        transposed[7][2],
    ];
    *out3 = [
        transposed[0][3],
        transposed[1][3],
        transposed[2][3],
        transposed[3][3],
        transposed[4][3],
        transposed[5][3],
        transposed[6][3],
        transposed[7][3],
    ];
}

#[inline(always)]
fn load_from_4(words: &[u64x4; 8], i: usize) -> [u64; 8] {
    [
        words[0][i],
        words[1][i],
        words[2][i],
        words[3][i],
        words[4][i],
        words[5][i],
        words[6][i],
        words[7][i],
    ]
}

#[inline(always)]
fn store_to_4(whole: &mut [u64x4; 8], part: &[u64; 8], i: usize) {
    whole[0][i] = part[0];
    whole[1][i] = part[1];
    whole[2][i] = part[2];
    whole[3][i] = part[3];
    whole[4][i] = part[4];
    whole[5][i] = part[5];
    whole[6][i] = part[6];
    whole[7][i] = part[7];
}

pub fn compress4_transposed(
    h_vecs: &mut [u64x4; 8],
    msg0: &Block,
    msg1: &Block,
    msg2: &Block,
    msg3: &Block,
    count_low: &u64x4,
    count_high: &u64x4,
    lastblock: &u64x4,
    lastnode: &u64x4,
) {
    let mut state0 = load_from_4(h_vecs, 0);
    let count0 = count_low[0] as u128 + ((count_high[0] as u128) << 64);
    compress(&mut state0, msg0, count0, lastblock[0], lastnode[0]);
    store_to_4(h_vecs, &state0, 0);

    let mut state1 = load_from_4(h_vecs, 1);
    let count1 = count_low[1] as u128 + ((count_high[1] as u128) << 64);
    compress(&mut state1, msg1, count1, lastblock[1], lastnode[1]);
    store_to_4(h_vecs, &state1, 1);

    let mut state2 = load_from_4(h_vecs, 2);
    let count2 = count_low[2] as u128 + ((count_high[2] as u128) << 64);
    compress(&mut state2, msg2, count2, lastblock[2], lastnode[2]);
    store_to_4(h_vecs, &state2, 2);

    let mut state3 = load_from_4(h_vecs, 3);
    let count3 = count_low[3] as u128 + ((count_high[3] as u128) << 64);
    compress(&mut state3, msg3, count3, lastblock[3], lastnode[3]);
    store_to_4(h_vecs, &state3, 3);
}
