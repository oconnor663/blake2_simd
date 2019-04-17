#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::*;
use crate::guts::{u64_flag, u64x2, Job, Stride};
use core::mem;

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

// Making this function inline(always) doesn't hose the build time like we see
// in avx2.rs, but we follow the same pattern for consistency.
#[target_feature(enable = "sse4.1")]
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

#[inline(always)]
unsafe fn transpose_vecs(a: __m128i, b: __m128i) -> [__m128i; 2] {
    let a_words: [i64; 2] = mem::transmute(a);
    let b_words: [i64; 2] = mem::transmute(b);
    [
        _mm_set_epi64x(b_words[0], a_words[0]),
        _mm_set_epi64x(b_words[1], a_words[1]),
    ]
}

#[inline(always)]
unsafe fn unsigned_cmpgt_epi64(a: __m128i, b: __m128i) -> __m128i {
    // Because _mm_cmpgt_epi64 operates on signed values, we need to subtract
    // 2^63 from each value before doing the comparison.
    let delta = _mm_set1_epi64x(i64::min_value());
    _mm_cmpgt_epi64(_mm_add_epi64(a, delta), _mm_add_epi64(b, delta))
}

#[inline(always)]
unsafe fn add_carry(lo: &mut __m128i, hi: &mut __m128i, x: __m128i) {
    let old_lo = *lo;
    *lo = _mm_add_epi64(*lo, x);
    let carries = _mm_and_si128(unsigned_cmpgt_epi64(old_lo, *lo), _mm_set1_epi64x(1));
    *hi = _mm_add_epi64(*hi, carries);
}

#[inline(always)]
unsafe fn transpose_state_vecs(jobs: &[Job; 2]) -> [__m128i; 8] {
    // Load all the state words into transposed vectors, where the first vector
    // has the first word of each state, etc. This is the form that 4-way
    // compression operates on, and transposing once at the beginning and once
    // at the end is more efficient that repeating it for each block. Note that
    // these loads are aligned, because u64x4 and u64x8 guarantee alignment.
    let [h0, h1] = transpose_vecs(
        load_u64x2(&jobs[0].words.split()[0].split()[0]),
        load_u64x2(&jobs[1].words.split()[0].split()[0]),
    );
    let [h2, h3] = transpose_vecs(
        load_u64x2(&jobs[0].words.split()[0].split()[1]),
        load_u64x2(&jobs[1].words.split()[0].split()[1]),
    );
    let [h4, h5] = transpose_vecs(
        load_u64x2(&jobs[0].words.split()[1].split()[0]),
        load_u64x2(&jobs[1].words.split()[1].split()[0]),
    );
    let [h6, h7] = transpose_vecs(
        load_u64x2(&jobs[0].words.split()[1].split()[1]),
        load_u64x2(&jobs[1].words.split()[1].split()[1]),
    );
    [h0, h1, h2, h3, h4, h5, h6, h7]
}

#[inline(always)]
unsafe fn untranspose_state_vecs(h_vecs: &[__m128i; 8], jobs: &mut [Job; 2]) {
    // Un-transpose the updated state vectors back into the caller's arrays.
    // These are aligned stores.
    let words = transpose_vecs(h_vecs[0], h_vecs[1]);
    store_u64x2(words[0], &mut jobs[0].words.split_mut()[0].split_mut()[0]);
    store_u64x2(words[1], &mut jobs[1].words.split_mut()[0].split_mut()[0]);
    let words = transpose_vecs(h_vecs[2], h_vecs[3]);
    store_u64x2(words[0], &mut jobs[0].words.split_mut()[0].split_mut()[1]);
    store_u64x2(words[1], &mut jobs[1].words.split_mut()[0].split_mut()[1]);
    let words = transpose_vecs(h_vecs[4], h_vecs[5]);
    store_u64x2(words[0], &mut jobs[0].words.split_mut()[1].split_mut()[0]);
    store_u64x2(words[1], &mut jobs[1].words.split_mut()[1].split_mut()[0]);
    let words = transpose_vecs(h_vecs[6], h_vecs[7]);
    store_u64x2(words[0], &mut jobs[0].words.split_mut()[1].split_mut()[1]);
    store_u64x2(words[1], &mut jobs[1].words.split_mut()[1].split_mut()[1]);
}

#[inline(always)]
unsafe fn transpose_msg_vecs(blocks: [&[u8; BLOCKBYTES]; 2]) -> [__m128i; 16] {
    // These input arrays have no particular alignment, so we use unaligned
    // loads to read from them.
    let ptr0 = blocks[0].as_ptr() as *const __m128i;
    let ptr1 = blocks[1].as_ptr() as *const __m128i;
    let [m0, m1] = transpose_vecs(_mm_loadu_si128(ptr0.add(0)), _mm_loadu_si128(ptr1.add(0)));
    let [m2, m3] = transpose_vecs(_mm_loadu_si128(ptr0.add(1)), _mm_loadu_si128(ptr1.add(1)));
    let [m4, m5] = transpose_vecs(_mm_loadu_si128(ptr0.add(2)), _mm_loadu_si128(ptr1.add(2)));
    let [m6, m7] = transpose_vecs(_mm_loadu_si128(ptr0.add(3)), _mm_loadu_si128(ptr1.add(3)));
    let [m8, m9] = transpose_vecs(_mm_loadu_si128(ptr0.add(4)), _mm_loadu_si128(ptr1.add(4)));
    let [m10, m11] = transpose_vecs(_mm_loadu_si128(ptr0.add(5)), _mm_loadu_si128(ptr1.add(5)));
    let [m12, m13] = transpose_vecs(_mm_loadu_si128(ptr0.add(6)), _mm_loadu_si128(ptr1.add(6)));
    let [m14, m15] = transpose_vecs(_mm_loadu_si128(ptr0.add(7)), _mm_loadu_si128(ptr1.add(7)));
    [
        m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15,
    ]
}

#[inline(always)]
unsafe fn load_counts(jobs: &[Job; 2]) -> (__m128i, __m128i) {
    (
        // There's no _mm_setr_epi64x, so note the arg order.
        _mm_set_epi64x(jobs[1].count as i64, jobs[0].count as i64),
        _mm_set_epi64x((jobs[1].count >> 64) as i64, (jobs[0].count >> 64) as i64),
    )
}

#[inline(always)]
unsafe fn load_flags_vec(flags: [bool; 2]) -> __m128i {
    // There's no _mm_setr_epi64x, so note the arg order.
    _mm_set_epi64x(u64_flag(flags[1]) as i64, u64_flag(flags[0]) as i64)
}

#[inline(always)]
unsafe fn offset_jobs(jobs: &mut [Job; 2], offset: usize) {
    jobs[0].offset(offset);
    jobs[1].offset(offset);
}

#[target_feature(enable = "sse4.1")]
pub unsafe fn compress2_loop(jobs: &mut [Job; 2], stride: Stride) {
    let mut h_vecs = transpose_state_vecs(&jobs);
    let mut offset = 0;
    let (mut counts_lo, mut counts_hi) = load_counts(&jobs);
    let min_len = jobs.iter().map(|job| job.input.len()).min().unwrap();
    let final_block_offset = guts::final_block_offset(min_len, stride);
    let mut buffer0 = [0; BLOCKBYTES];
    let mut buffer1 = [0; BLOCKBYTES];
    let (finblock0, finblock_len0, is_end0) =
        guts::get_block(jobs[0].input, final_block_offset, &mut buffer0, stride);
    let (finblock1, finblock_len1, is_end1) =
        guts::get_block(jobs[1].input, final_block_offset, &mut buffer1, stride);
    let finlastblockvec = load_flags_vec([
        is_end0 && jobs[0].finalize.last_block_flag(),
        is_end1 && jobs[1].finalize.last_block_flag(),
    ]);
    let finlastnodevec = load_flags_vec([
        is_end0 && jobs[0].finalize.last_node_flag(),
        is_end1 && jobs[1].finalize.last_node_flag(),
    ]);
    // There's no _mm_setr_epi64x, so note the arg order.
    let fincountsinc = _mm_set_epi64x(finblock_len1 as i64, finblock_len0 as i64);
    while offset <= final_block_offset {
        let is_final_block = offset == final_block_offset;
        let blocks;
        let last_block_vec;
        let last_node_vec;
        let counts_inc;
        if is_final_block {
            blocks = [finblock0, finblock1];
            last_block_vec = finlastblockvec;
            last_node_vec = finlastnodevec;
            counts_inc = fincountsinc;
        } else {
            // These unsafe pointer casts avoid paying for bounds checks. The
            // final_block_offset math guarantees that these loads are
            // in-bounds.
            let block0 = &*(jobs[0].input.as_ptr().add(offset) as *const [u8; BLOCKBYTES]);
            let block1 = &*(jobs[1].input.as_ptr().add(offset) as *const [u8; BLOCKBYTES]);
            blocks = [block0, block1];
            last_block_vec = _mm_set1_epi64x(0);
            last_node_vec = _mm_set1_epi64x(0);
            counts_inc = _mm_set1_epi64x(BLOCKBYTES as i64);
        }
        add_carry(&mut counts_lo, &mut counts_hi, counts_inc);
        let m_vecs = transpose_msg_vecs(blocks);

        compress2_transposed_inline(
            &mut h_vecs,
            &m_vecs,
            counts_lo,
            counts_hi,
            last_block_vec,
            last_node_vec,
        );

        offset = offset.saturating_add(stride.padded_blockbytes());
    }

    untranspose_state_vecs(&h_vecs, jobs);
    offset_jobs(jobs, offset);
}
