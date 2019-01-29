#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::*;
use crate::guts::u64x2;
use crate::guts::u64x4;
use crate::guts::u64x8;
use core::mem;
use core::ptr;

#[inline(always)]
unsafe fn load_u64x2(a: &u64x2) -> __m128i {
    // u64x2 is fully aligned, so this load is safe.
    _mm_load_si128(a.as_ptr() as *const __m128i)
}

#[inline(always)]
unsafe fn store_u64x2(a: __m128i, dest: &mut u64x2) {
    // u64x2 is fully aligned, so this store is safe.
    _mm_store_si128(dest.as_mut_ptr() as *mut __m128i, a)
}

#[inline(always)]
unsafe fn add(a: __m128i, b: __m128i) -> __m128i {
    _mm_add_epi64(a, b)
}

#[inline(always)]
unsafe fn sub(a: __m128i, b: __m128i) -> __m128i {
    _mm_sub_epi64(a, b)
}

#[inline(always)]
unsafe fn xor(a: __m128i, b: __m128i) -> __m128i {
    _mm_xor_si128(a, b)
}

// Adapted from https://github.com/rust-lang-nursery/stdsimd/pull/479.
macro_rules! _MM_SHUFFLE {
    ($z:expr, $y:expr, $x:expr, $w:expr) => {
        ($z << 6) | ($y << 4) | ($x << 2) | $w
    };
}

#[inline(always)]
unsafe fn rot32(x: __m128i) -> __m128i {
    _mm_shuffle_epi32(x, _MM_SHUFFLE!(2, 3, 0, 1))
}

#[inline(always)]
unsafe fn rot24(x: __m128i) -> __m128i {
    let rotate24 = _mm_setr_epi8(3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10);
    _mm_shuffle_epi8(x, rotate24)
}

#[inline(always)]
unsafe fn rot16(x: __m128i) -> __m128i {
    let rotate16 = _mm_setr_epi8(2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9);
    _mm_shuffle_epi8(x, rotate16)
}

#[inline(always)]
unsafe fn rot63(x: __m128i) -> __m128i {
    _mm_or_si128(_mm_srli_epi64(x, 63), add(x, x))
}

#[inline(always)]
unsafe fn blake2b_round_2x(v: &mut [__m128i; 16], m: &[__m128i; 16], r: usize) {
    v[0] = add(v[0], m[SIGMA[r][0] as usize]);
    v[1] = add(v[1], m[SIGMA[r][2] as usize]);
    v[2] = add(v[2], m[SIGMA[r][4] as usize]);
    v[3] = add(v[3], m[SIGMA[r][6] as usize]);
    v[0] = add(v[0], v[4]);
    v[1] = add(v[1], v[5]);
    v[2] = add(v[2], v[6]);
    v[3] = add(v[3], v[7]);
    v[12] = xor(v[12], v[0]);
    v[13] = xor(v[13], v[1]);
    v[14] = xor(v[14], v[2]);
    v[15] = xor(v[15], v[3]);
    v[12] = rot32(v[12]);
    v[13] = rot32(v[13]);
    v[14] = rot32(v[14]);
    v[15] = rot32(v[15]);
    v[8] = add(v[8], v[12]);
    v[9] = add(v[9], v[13]);
    v[10] = add(v[10], v[14]);
    v[11] = add(v[11], v[15]);
    v[4] = xor(v[4], v[8]);
    v[5] = xor(v[5], v[9]);
    v[6] = xor(v[6], v[10]);
    v[7] = xor(v[7], v[11]);
    v[4] = rot24(v[4]);
    v[5] = rot24(v[5]);
    v[6] = rot24(v[6]);
    v[7] = rot24(v[7]);
    v[0] = add(v[0], m[SIGMA[r][1] as usize]);
    v[1] = add(v[1], m[SIGMA[r][3] as usize]);
    v[2] = add(v[2], m[SIGMA[r][5] as usize]);
    v[3] = add(v[3], m[SIGMA[r][7] as usize]);
    v[0] = add(v[0], v[4]);
    v[1] = add(v[1], v[5]);
    v[2] = add(v[2], v[6]);
    v[3] = add(v[3], v[7]);
    v[12] = xor(v[12], v[0]);
    v[13] = xor(v[13], v[1]);
    v[14] = xor(v[14], v[2]);
    v[15] = xor(v[15], v[3]);
    v[12] = rot16(v[12]);
    v[13] = rot16(v[13]);
    v[14] = rot16(v[14]);
    v[15] = rot16(v[15]);
    v[8] = add(v[8], v[12]);
    v[9] = add(v[9], v[13]);
    v[10] = add(v[10], v[14]);
    v[11] = add(v[11], v[15]);
    v[4] = xor(v[4], v[8]);
    v[5] = xor(v[5], v[9]);
    v[6] = xor(v[6], v[10]);
    v[7] = xor(v[7], v[11]);
    v[4] = rot63(v[4]);
    v[5] = rot63(v[5]);
    v[6] = rot63(v[6]);
    v[7] = rot63(v[7]);

    v[0] = add(v[0], m[SIGMA[r][8] as usize]);
    v[1] = add(v[1], m[SIGMA[r][10] as usize]);
    v[2] = add(v[2], m[SIGMA[r][12] as usize]);
    v[3] = add(v[3], m[SIGMA[r][14] as usize]);
    v[0] = add(v[0], v[5]);
    v[1] = add(v[1], v[6]);
    v[2] = add(v[2], v[7]);
    v[3] = add(v[3], v[4]);
    v[15] = xor(v[15], v[0]);
    v[12] = xor(v[12], v[1]);
    v[13] = xor(v[13], v[2]);
    v[14] = xor(v[14], v[3]);
    v[15] = rot32(v[15]);
    v[12] = rot32(v[12]);
    v[13] = rot32(v[13]);
    v[14] = rot32(v[14]);
    v[10] = add(v[10], v[15]);
    v[11] = add(v[11], v[12]);
    v[8] = add(v[8], v[13]);
    v[9] = add(v[9], v[14]);
    v[5] = xor(v[5], v[10]);
    v[6] = xor(v[6], v[11]);
    v[7] = xor(v[7], v[8]);
    v[4] = xor(v[4], v[9]);
    v[5] = rot24(v[5]);
    v[6] = rot24(v[6]);
    v[7] = rot24(v[7]);
    v[4] = rot24(v[4]);
    v[0] = add(v[0], m[SIGMA[r][9] as usize]);
    v[1] = add(v[1], m[SIGMA[r][11] as usize]);
    v[2] = add(v[2], m[SIGMA[r][13] as usize]);
    v[3] = add(v[3], m[SIGMA[r][15] as usize]);
    v[0] = add(v[0], v[5]);
    v[1] = add(v[1], v[6]);
    v[2] = add(v[2], v[7]);
    v[3] = add(v[3], v[4]);
    v[15] = xor(v[15], v[0]);
    v[12] = xor(v[12], v[1]);
    v[13] = xor(v[13], v[2]);
    v[14] = xor(v[14], v[3]);
    v[15] = rot16(v[15]);
    v[12] = rot16(v[12]);
    v[13] = rot16(v[13]);
    v[14] = rot16(v[14]);
    v[10] = add(v[10], v[15]);
    v[11] = add(v[11], v[12]);
    v[8] = add(v[8], v[13]);
    v[9] = add(v[9], v[14]);
    v[5] = xor(v[5], v[10]);
    v[6] = xor(v[6], v[11]);
    v[7] = xor(v[7], v[8]);
    v[4] = xor(v[4], v[9]);
    v[5] = rot63(v[5]);
    v[6] = rot63(v[6]);
    v[7] = rot63(v[7]);
    v[4] = rot63(v[4]);
}

#[inline(always)]
unsafe fn transpose_message_blocks(msg0: &Block, msg1: &Block) -> [__m128i; 16] {
    [
        _mm_set_epi64x(
            ptr::read_unaligned(msg1.as_ptr().add(0 * 8) as *const i64),
            ptr::read_unaligned(msg0.as_ptr().add(0 * 8) as *const i64),
        ),
        _mm_set_epi64x(
            ptr::read_unaligned(msg1.as_ptr().add(1 * 8) as *const i64),
            ptr::read_unaligned(msg0.as_ptr().add(1 * 8) as *const i64),
        ),
        _mm_set_epi64x(
            ptr::read_unaligned(msg1.as_ptr().add(2 * 8) as *const i64),
            ptr::read_unaligned(msg0.as_ptr().add(2 * 8) as *const i64),
        ),
        _mm_set_epi64x(
            ptr::read_unaligned(msg1.as_ptr().add(3 * 8) as *const i64),
            ptr::read_unaligned(msg0.as_ptr().add(3 * 8) as *const i64),
        ),
        _mm_set_epi64x(
            ptr::read_unaligned(msg1.as_ptr().add(4 * 8) as *const i64),
            ptr::read_unaligned(msg0.as_ptr().add(4 * 8) as *const i64),
        ),
        _mm_set_epi64x(
            ptr::read_unaligned(msg1.as_ptr().add(5 * 8) as *const i64),
            ptr::read_unaligned(msg0.as_ptr().add(5 * 8) as *const i64),
        ),
        _mm_set_epi64x(
            ptr::read_unaligned(msg1.as_ptr().add(6 * 8) as *const i64),
            ptr::read_unaligned(msg0.as_ptr().add(6 * 8) as *const i64),
        ),
        _mm_set_epi64x(
            ptr::read_unaligned(msg1.as_ptr().add(7 * 8) as *const i64),
            ptr::read_unaligned(msg0.as_ptr().add(7 * 8) as *const i64),
        ),
        _mm_set_epi64x(
            ptr::read_unaligned(msg1.as_ptr().add(8 * 8) as *const i64),
            ptr::read_unaligned(msg0.as_ptr().add(8 * 8) as *const i64),
        ),
        _mm_set_epi64x(
            ptr::read_unaligned(msg1.as_ptr().add(9 * 8) as *const i64),
            ptr::read_unaligned(msg0.as_ptr().add(9 * 8) as *const i64),
        ),
        _mm_set_epi64x(
            ptr::read_unaligned(msg1.as_ptr().add(10 * 8) as *const i64),
            ptr::read_unaligned(msg0.as_ptr().add(10 * 8) as *const i64),
        ),
        _mm_set_epi64x(
            ptr::read_unaligned(msg1.as_ptr().add(11 * 8) as *const i64),
            ptr::read_unaligned(msg0.as_ptr().add(11 * 8) as *const i64),
        ),
        _mm_set_epi64x(
            ptr::read_unaligned(msg1.as_ptr().add(12 * 8) as *const i64),
            ptr::read_unaligned(msg0.as_ptr().add(12 * 8) as *const i64),
        ),
        _mm_set_epi64x(
            ptr::read_unaligned(msg1.as_ptr().add(13 * 8) as *const i64),
            ptr::read_unaligned(msg0.as_ptr().add(13 * 8) as *const i64),
        ),
        _mm_set_epi64x(
            ptr::read_unaligned(msg1.as_ptr().add(14 * 8) as *const i64),
            ptr::read_unaligned(msg0.as_ptr().add(14 * 8) as *const i64),
        ),
        _mm_set_epi64x(
            ptr::read_unaligned(msg1.as_ptr().add(15 * 8) as *const i64),
            ptr::read_unaligned(msg0.as_ptr().add(15 * 8) as *const i64),
        ),
    ]
}

#[inline(always)]
unsafe fn compress2_transposed_inline(
    h_vecs: &mut [__m128i; 8],
    msg_vecs: &[__m128i; 16],
    count_low: __m128i,
    count_high: __m128i,
    lastblock: __m128i,
    lastnode: __m128i,
) {
    let mut v = [
        h_vecs[0],
        h_vecs[1],
        h_vecs[2],
        h_vecs[3],
        h_vecs[4],
        h_vecs[5],
        h_vecs[6],
        h_vecs[7],
        _mm_set1_epi64x(IV[0] as i64),
        _mm_set1_epi64x(IV[1] as i64),
        _mm_set1_epi64x(IV[2] as i64),
        _mm_set1_epi64x(IV[3] as i64),
        xor(_mm_set1_epi64x(IV[4] as i64), count_low),
        xor(_mm_set1_epi64x(IV[5] as i64), count_high),
        xor(_mm_set1_epi64x(IV[6] as i64), lastblock),
        xor(_mm_set1_epi64x(IV[7] as i64), lastnode),
    ];

    blake2b_round_2x(&mut v, &msg_vecs, 0);
    blake2b_round_2x(&mut v, &msg_vecs, 1);
    blake2b_round_2x(&mut v, &msg_vecs, 2);
    blake2b_round_2x(&mut v, &msg_vecs, 3);
    blake2b_round_2x(&mut v, &msg_vecs, 4);
    blake2b_round_2x(&mut v, &msg_vecs, 5);
    blake2b_round_2x(&mut v, &msg_vecs, 6);
    blake2b_round_2x(&mut v, &msg_vecs, 7);
    blake2b_round_2x(&mut v, &msg_vecs, 8);
    blake2b_round_2x(&mut v, &msg_vecs, 9);
    blake2b_round_2x(&mut v, &msg_vecs, 10);
    blake2b_round_2x(&mut v, &msg_vecs, 11);

    h_vecs[0] = xor(xor(h_vecs[0], v[0]), v[8]);
    h_vecs[1] = xor(xor(h_vecs[1], v[1]), v[9]);
    h_vecs[2] = xor(xor(h_vecs[2], v[2]), v[10]);
    h_vecs[3] = xor(xor(h_vecs[3], v[3]), v[11]);
    h_vecs[4] = xor(xor(h_vecs[4], v[4]), v[12]);
    h_vecs[5] = xor(xor(h_vecs[5], v[5]), v[13]);
    h_vecs[6] = xor(xor(h_vecs[6], v[6]), v[14]);
    h_vecs[7] = xor(xor(h_vecs[7], v[7]), v[15]);
}

#[target_feature(enable = "sse4.1")]
pub unsafe fn compress2_transposed(
    h_vecs: &mut [u64x2; 8],
    msg0: &Block,
    msg1: &Block,
    count_low: &u64x2,
    count_high: &u64x2,
    lastblock: &u64x2,
    lastnode: &u64x2,
) {
    let m = transpose_message_blocks(msg0, msg1);
    compress2_transposed_inline(
        mem::transmute(h_vecs),
        &m,
        mem::transmute(*count_low),
        mem::transmute(*count_high),
        mem::transmute(*lastblock),
        mem::transmute(*lastnode),
    );
}

#[inline(always)]
unsafe fn load_from_4(words: &[u64x4; 8], i: usize) -> [u64x2; 8] {
    [
        words[0].split()[i],
        words[1].split()[i],
        words[2].split()[i],
        words[3].split()[i],
        words[4].split()[i],
        words[5].split()[i],
        words[6].split()[i],
        words[7].split()[i],
    ]
}

#[inline(always)]
unsafe fn store_to_4(whole: &mut [u64x4; 8], part: &[u64x2; 8], i: usize) {
    whole[0].split_mut()[i] = part[0];
    whole[1].split_mut()[i] = part[1];
    whole[2].split_mut()[i] = part[2];
    whole[3].split_mut()[i] = part[3];
    whole[4].split_mut()[i] = part[4];
    whole[5].split_mut()[i] = part[5];
    whole[6].split_mut()[i] = part[6];
    whole[7].split_mut()[i] = part[7];
}

#[target_feature(enable = "sse4.1")]
pub unsafe fn compress4_transposed(
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
    compress2_transposed(
        &mut state0,
        msg0,
        msg1,
        &count_low.split()[0],
        &count_high.split()[0],
        &lastblock.split()[0],
        &lastnode.split()[0],
    );
    store_to_4(h_vecs, &state0, 0);

    let mut state1 = load_from_4(h_vecs, 1);
    compress2_transposed(
        &mut state1,
        msg2,
        msg3,
        &count_low.split()[1],
        &count_high.split()[1],
        &lastblock.split()[1],
        &lastnode.split()[1],
    );
    store_to_4(h_vecs, &state1, 1);
}

#[inline(always)]
unsafe fn transpose_vecs(a: __m128i, b: __m128i) -> [__m128i; 2] {
    let a_words: [i64; 2] = mem::transmute(a);
    let b_words: [i64; 2] = mem::transmute(b);
    [
        _mm_set_epi64x(b_words[0], a_words[0]),
        _mm_set_epi64x(b_words[1], a_words[1]),
    ]
}

#[target_feature(enable = "sse4.1")]
pub unsafe fn compress2_loop(
    state0: &mut u64x8,
    state1: &mut u64x8,
    input0: &[u8],
    input1: &[u8],
    count_low: &u64x2,
    count_high: &u64x2,
    last_block: &u64x2,
    last_node: &u64x2,
    mut blocks: usize,
    stride: usize,
    buffer_tail: &u64x2,
) {
    // Check the input slice lengths once here. The main loop will do unaligned
    // loads without any further bounds checks.
    assert!(BLOCKBYTES * (stride * (blocks - 1) + 1) <= input0.len());
    assert!(BLOCKBYTES * (stride * (blocks - 1) + 1) <= input1.len());

    // Load all the state words into transposed vectors, where the first vector
    // has the first word of each state, etc. This is the form that 2-way
    // compression operates on, and transposing once at the beginning and once
    // at the end is more efficient that repeating it for each block. Note that
    // these loads are aligned, because u64x2 and u64x8 guarantee alignment.
    let [h0, h1] = transpose_vecs(
        load_u64x2(&state0.split()[0].split()[0]),
        load_u64x2(&state1.split()[0].split()[0]),
    );
    let [h2, h3] = transpose_vecs(
        load_u64x2(&state0.split()[0].split()[1]),
        load_u64x2(&state1.split()[0].split()[1]),
    );
    let [h4, h5] = transpose_vecs(
        load_u64x2(&state0.split()[1].split()[0]),
        load_u64x2(&state1.split()[1].split()[0]),
    );
    let [h6, h7] = transpose_vecs(
        load_u64x2(&state0.split()[1].split()[1]),
        load_u64x2(&state1.split()[1].split()[1]),
    );
    let mut h_vecs = [h0, h1, h2, h3, h4, h5, h6, h7];
    let mut count_low_vec = load_u64x2(count_low);
    let mut count_high_vec = load_u64x2(count_high);
    let mut offset = 0;

    while blocks > 0 {
        // Load all the message words into transposed vectors also. Message
        // loads are unaligned, because these are arbitrary byte pointers from
        // the caller. On modern chips though, there's not much of a
        // performance penalty for unaligned loads.
        let block0 = input0.as_ptr().add(offset) as *const __m128i;
        let block1 = input1.as_ptr().add(offset) as *const __m128i;
        let [m0, m1] = transpose_vecs(
            _mm_loadu_si128(block0.add(0)),
            _mm_loadu_si128(block1.add(0)),
        );
        let [m2, m3] = transpose_vecs(
            _mm_loadu_si128(block0.add(1)),
            _mm_loadu_si128(block1.add(1)),
        );
        let [m4, m5] = transpose_vecs(
            _mm_loadu_si128(block0.add(2)),
            _mm_loadu_si128(block1.add(2)),
        );
        let [m6, m7] = transpose_vecs(
            _mm_loadu_si128(block0.add(3)),
            _mm_loadu_si128(block1.add(3)),
        );
        let [m8, m9] = transpose_vecs(
            _mm_loadu_si128(block0.add(4)),
            _mm_loadu_si128(block1.add(4)),
        );
        let [m10, m11] = transpose_vecs(
            _mm_loadu_si128(block0.add(5)),
            _mm_loadu_si128(block1.add(5)),
        );
        let [m12, m13] = transpose_vecs(
            _mm_loadu_si128(block0.add(6)),
            _mm_loadu_si128(block1.add(6)),
        );
        let [m14, m15] = transpose_vecs(
            _mm_loadu_si128(block0.add(7)),
            _mm_loadu_si128(block1.add(7)),
        );
        let m_vecs = [
            m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15,
        ];

        // Add BLOCKBYTES to the low count bits.
        let old_count_low_vec = count_low_vec;
        count_low_vec = add(count_low_vec, _mm_set1_epi64x(BLOCKBYTES as i64));
        // If this is the last block, subtract the buffer tails.
        if blocks == 1 {
            count_low_vec = sub(count_low_vec, load_u64x2(buffer_tail));
        }
        // Finally if any of the low counts overflowed (after accounting for
        // the buffer tails), increment the corresponding high counts.
        count_high_vec = add(
            count_high_vec,
            _mm_and_si128(
                _mm_cmpgt_epi64(old_count_low_vec, count_low_vec),
                _mm_set1_epi64x(1),
            ),
        );

        // Compressions before the last one always use zero for the
        // finalization flags. The last one will use what the caller supplied,
        // which could also be zero if the input isn't finished.
        let (last_block_vec, last_node_vec) = if blocks == 1 {
            (
                // Again, u64x4 guarantees alignment, so we do an aligned load.
                load_u64x2(last_block),
                load_u64x2(last_node),
            )
        } else {
            (_mm_set1_epi64x(0), _mm_set1_epi64x(0))
        };

        compress2_transposed_inline(
            &mut h_vecs,
            &m_vecs,
            count_low_vec,
            count_high_vec,
            last_block_vec,
            last_node_vec,
        );

        offset += BLOCKBYTES * stride;
        blocks -= 1;
    }

    // Un-transpose the updated state vectors back into the caller's arrays.
    // These are aligned stores.
    let words = transpose_vecs(h_vecs[0], h_vecs[1]);
    store_u64x2(words[0], &mut state0.split_mut()[0].split_mut()[0]);
    store_u64x2(words[1], &mut state1.split_mut()[0].split_mut()[0]);
    let words = transpose_vecs(h_vecs[2], h_vecs[3]);
    store_u64x2(words[0], &mut state0.split_mut()[0].split_mut()[1]);
    store_u64x2(words[1], &mut state1.split_mut()[0].split_mut()[1]);
    let words = transpose_vecs(h_vecs[4], h_vecs[5]);
    store_u64x2(words[0], &mut state0.split_mut()[1].split_mut()[0]);
    store_u64x2(words[1], &mut state1.split_mut()[1].split_mut()[0]);
    let words = transpose_vecs(h_vecs[6], h_vecs[7]);
    store_u64x2(words[0], &mut state0.split_mut()[1].split_mut()[1]);
    store_u64x2(words[1], &mut state1.split_mut()[1].split_mut()[1]);
}
