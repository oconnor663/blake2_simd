use byteorder::{ByteOrder, LittleEndian};

use super::*;
use crate::guts::{u64_flag, u64x8, Finalize, LastNode};
use core::cmp;

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
pub fn compress(
    block: &[u8; BLOCKBYTES],
    words: &mut u64x8,
    count: u128,
    last_node: LastNode,
    finalize: Finalize,
) {
    // Initialize the compression state.
    let mut v = [
        words[0],
        words[1],
        words[2],
        words[3],
        words[4],
        words[5],
        words[6],
        words[7],
        IV[0],
        IV[1],
        IV[2],
        IV[3],
        IV[4] ^ count as u64,
        IV[5] ^ (count >> 64) as u64,
        IV[6] ^ u64_flag(finalize.yes()),
        IV[7] ^ u64_flag(finalize.yes() && last_node.yes()),
    ];

    // Parse the message bytes as ints in little endian order.
    let msg_refs = array_refs!(block, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8);
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

    words[0] ^= v[0] ^ v[8];
    words[1] ^= v[1] ^ v[9];
    words[2] ^= v[2] ^ v[10];
    words[3] ^= v[3] ^ v[11];
    words[4] ^= v[4] ^ v[12];
    words[5] ^= v[5] ^ v[13];
    words[6] ^= v[6] ^ v[14];
    words[7] ^= v[7] ^ v[15];
}

// Pull a pointer to the final block straight from the input, if there's enough
// input. If there's only a partial block, copy it into the provided buffer,
// and return a pointer to that. Along with that pointer, return the number of
// bytes of real input.
#[inline(always)]
fn final_block<'a>(
    input: &'a [u8],
    offset: usize,
    buffer: &'a mut [u8; BLOCKBYTES],
) -> (&'a [u8; BLOCKBYTES], usize) {
    debug_assert!(offset <= input.len());
    let capped_offset = cmp::min(offset, input.len());
    let remaining = input.len() - capped_offset;
    if remaining >= BLOCKBYTES {
        let block = array_ref!(input, capped_offset, BLOCKBYTES);
        (block, BLOCKBYTES)
    } else {
        // Copy the remaining bytes to the front of the block buffer. The rest
        // is assumed to be initialized to zero.
        buffer[..remaining].copy_from_slice(&input[capped_offset..]);
        (buffer, remaining)
    }
}

pub fn compress1_loop(
    input: &[u8],
    words: &mut u64x8,
    mut count: u128,
    last_node: LastNode,
    finalize: Finalize,
) {
    let mut local_words = *words;
    let mut bulk_end = input.len();
    // Reserve at least one byte for finalization. However, if we're not
    // finalizing, assume that the caller has shaved off a tail or knows more
    // bytes are coming.
    if finalize.yes() {
        bulk_end = bulk_end.saturating_sub(1);
    } else {
        debug_assert_eq!(0, input.len() % BLOCKBYTES);
    }
    bulk_end -= bulk_end % BLOCKBYTES;
    let mut offset = 0;

    while offset < bulk_end {
        let block = array_ref!(input, offset, BLOCKBYTES);
        count = count.wrapping_add(BLOCKBYTES as u128);
        compress(block, &mut local_words, count, last_node, Finalize::No);
        offset += BLOCKBYTES;
    }

    if finalize.yes() {
        let mut buffer = [0; BLOCKBYTES];
        let (block, len) = final_block(input, bulk_end, &mut buffer);
        count = count.wrapping_add(len as u128);
        compress(block, &mut local_words, count, last_node, Finalize::Yes);
        // offset isn't used again
    }

    *words = local_words;
}
