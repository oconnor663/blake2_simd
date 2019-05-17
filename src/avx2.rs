#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::guts::{
    final_block, input_debug_asserts, u64_flag, u64x4, u64x8, Finalize, Job, LastNode, Stride,
};
use crate::{BLOCKBYTES, IV, SIGMA};
use arrayref::array_refs;
use core::cmp;
use core::mem;

#[inline(always)]
unsafe fn load_u64x4(a: &u64x4) -> __m256i {
    // u64x4 is fully aligned, so this load is safe.
    _mm256_load_si256(a.as_ptr() as *const __m256i)
}

#[inline(always)]
unsafe fn store_u64x4(a: __m256i, dest: &mut u64x4) {
    // u64x4 is fully aligned, so this store is safe.
    _mm256_store_si256(dest.as_mut_ptr() as *mut __m256i, a)
}

#[inline(always)]
unsafe fn load_128_unaligned(mem_addr: &[u8; 16]) -> __m128i {
    _mm_loadu_si128(mem_addr.as_ptr() as *const __m128i)
}

#[inline(always)]
unsafe fn add(a: __m256i, b: __m256i) -> __m256i {
    _mm256_add_epi64(a, b)
}

#[inline(always)]
unsafe fn eq(a: __m256i, b: __m256i) -> __m256i {
    _mm256_cmpeq_epi64(a, b)
}

#[inline(always)]
unsafe fn and(a: __m256i, b: __m256i) -> __m256i {
    _mm256_and_si256(a, b)
}

#[inline(always)]
unsafe fn negate_and(a: __m256i, b: __m256i) -> __m256i {
    // Note that "and not" implies the reverse of the actual arg order.
    _mm256_andnot_si256(a, b)
}

#[inline(always)]
unsafe fn xor(a: __m256i, b: __m256i) -> __m256i {
    _mm256_xor_si256(a, b)
}

#[inline(always)]
unsafe fn set1(x: u64) -> __m256i {
    _mm256_set1_epi64x(x as i64)
}

#[inline(always)]
unsafe fn set4(a: u64, b: u64, c: u64, d: u64) -> __m256i {
    _mm256_setr_epi64x(a as i64, b as i64, c as i64, d as i64)
}

// Adapted from https://github.com/rust-lang-nursery/stdsimd/pull/479.
macro_rules! _MM_SHUFFLE {
    ($z:expr, $y:expr, $x:expr, $w:expr) => {
        ($z << 6) | ($y << 4) | ($x << 2) | $w
    };
}

#[inline(always)]
unsafe fn rot32(x: __m256i) -> __m256i {
    _mm256_shuffle_epi32(x, _MM_SHUFFLE!(2, 3, 0, 1))
}

#[inline(always)]
unsafe fn rot24(x: __m256i) -> __m256i {
    let rotate24 = _mm256_setr_epi8(
        3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10, 3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13,
        14, 15, 8, 9, 10,
    );
    _mm256_shuffle_epi8(x, rotate24)
}

#[inline(always)]
unsafe fn rot16(x: __m256i) -> __m256i {
    let rotate16 = _mm256_setr_epi8(
        2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9, 2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12,
        13, 14, 15, 8, 9,
    );
    _mm256_shuffle_epi8(x, rotate16)
}

#[inline(always)]
unsafe fn rot63(x: __m256i) -> __m256i {
    _mm256_or_si256(_mm256_srli_epi64(x, 63), add(x, x))
}

#[inline(always)]
unsafe fn blake2b_g1_v1(
    a: &mut __m256i,
    b: &mut __m256i,
    c: &mut __m256i,
    d: &mut __m256i,
    m: &mut __m256i,
) {
    *a = add(*a, *m);
    *a = add(*a, *b);
    *d = xor(*d, *a);
    *d = rot32(*d);
    *c = add(*c, *d);
    *b = xor(*b, *c);
    *b = rot24(*b);
}

#[inline(always)]
unsafe fn blake2b_g2_v1(
    a: &mut __m256i,
    b: &mut __m256i,
    c: &mut __m256i,
    d: &mut __m256i,
    m: &mut __m256i,
) {
    *a = add(*a, *m);
    *a = add(*a, *b);
    *d = xor(*d, *a);
    *d = rot16(*d);
    *c = add(*c, *d);
    *b = xor(*b, *c);
    *b = rot63(*b);
}

#[inline(always)]
unsafe fn blake2b_diag_v1(_a: &mut __m256i, b: &mut __m256i, c: &mut __m256i, d: &mut __m256i) {
    *d = _mm256_permute4x64_epi64(*d, _MM_SHUFFLE!(2, 1, 0, 3));
    *c = _mm256_permute4x64_epi64(*c, _MM_SHUFFLE!(1, 0, 3, 2));
    *b = _mm256_permute4x64_epi64(*b, _MM_SHUFFLE!(0, 3, 2, 1));
}

#[inline(always)]
unsafe fn blake2b_undiag_v1(_a: &mut __m256i, b: &mut __m256i, c: &mut __m256i, d: &mut __m256i) {
    *d = _mm256_permute4x64_epi64(*d, _MM_SHUFFLE!(0, 3, 2, 1));
    *c = _mm256_permute4x64_epi64(*c, _MM_SHUFFLE!(1, 0, 3, 2));
    *b = _mm256_permute4x64_epi64(*b, _MM_SHUFFLE!(2, 1, 0, 3));
}

#[inline(always)]
unsafe fn compress_block(
    block: &[u8; BLOCKBYTES],
    words: &mut u64x8,
    count: u128,
    last_block: u64,
    last_node: u64,
) {
    let mut a = load_u64x4(&words.split()[0]);
    let mut b = load_u64x4(&words.split()[1]);
    let mut c = load_u64x4(&IV.split()[0]);
    let count_low = count as u64;
    let count_high = (count >> 64) as u64;
    let flags = set4(count_low, count_high, last_block, last_node);
    let mut d = xor(load_u64x4(&IV.split()[1]), flags);

    let msg_chunks = array_refs!(block, 16, 16, 16, 16, 16, 16, 16, 16);
    let m0 = _mm256_broadcastsi128_si256(load_128_unaligned(msg_chunks.0));
    let m1 = _mm256_broadcastsi128_si256(load_128_unaligned(msg_chunks.1));
    let m2 = _mm256_broadcastsi128_si256(load_128_unaligned(msg_chunks.2));
    let m3 = _mm256_broadcastsi128_si256(load_128_unaligned(msg_chunks.3));
    let m4 = _mm256_broadcastsi128_si256(load_128_unaligned(msg_chunks.4));
    let m5 = _mm256_broadcastsi128_si256(load_128_unaligned(msg_chunks.5));
    let m6 = _mm256_broadcastsi128_si256(load_128_unaligned(msg_chunks.6));
    let m7 = _mm256_broadcastsi128_si256(load_128_unaligned(msg_chunks.7));

    let iv0 = a;
    let iv1 = b;
    let mut t0;
    let mut t1;
    let mut b0;

    // round 0
    t0 = _mm256_unpacklo_epi64(m0, m1);
    t1 = _mm256_unpacklo_epi64(m2, m3);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g1_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    t0 = _mm256_unpackhi_epi64(m0, m1);
    t1 = _mm256_unpackhi_epi64(m2, m3);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g2_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    blake2b_diag_v1(&mut a, &mut b, &mut c, &mut d);
    t0 = _mm256_unpacklo_epi64(m4, m5);
    t1 = _mm256_unpacklo_epi64(m6, m7);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g1_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    t0 = _mm256_unpackhi_epi64(m4, m5);
    t1 = _mm256_unpackhi_epi64(m6, m7);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g2_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    blake2b_undiag_v1(&mut a, &mut b, &mut c, &mut d);

    // round 1
    t0 = _mm256_unpacklo_epi64(m7, m2);
    t1 = _mm256_unpackhi_epi64(m4, m6);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g1_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    t0 = _mm256_unpacklo_epi64(m5, m4);
    t1 = _mm256_alignr_epi8(m3, m7, 8);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g2_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    blake2b_diag_v1(&mut a, &mut b, &mut c, &mut d);
    t0 = _mm256_shuffle_epi32(m0, _MM_SHUFFLE!(1, 0, 3, 2));
    t1 = _mm256_unpackhi_epi64(m5, m2);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g1_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    t0 = _mm256_unpacklo_epi64(m6, m1);
    t1 = _mm256_unpackhi_epi64(m3, m1);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g2_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    blake2b_undiag_v1(&mut a, &mut b, &mut c, &mut d);

    // round 2
    t0 = _mm256_alignr_epi8(m6, m5, 8);
    t1 = _mm256_unpackhi_epi64(m2, m7);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g1_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    t0 = _mm256_unpacklo_epi64(m4, m0);
    t1 = _mm256_blend_epi32(m6, m1, 0x33);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g2_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    blake2b_diag_v1(&mut a, &mut b, &mut c, &mut d);
    t0 = _mm256_blend_epi32(m1, m5, 0x33);
    t1 = _mm256_unpackhi_epi64(m3, m4);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g1_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    t0 = _mm256_unpacklo_epi64(m7, m3);
    t1 = _mm256_alignr_epi8(m2, m0, 8);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g2_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    blake2b_undiag_v1(&mut a, &mut b, &mut c, &mut d);

    // round 3
    t0 = _mm256_unpackhi_epi64(m3, m1);
    t1 = _mm256_unpackhi_epi64(m6, m5);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g1_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    t0 = _mm256_unpackhi_epi64(m4, m0);
    t1 = _mm256_unpacklo_epi64(m6, m7);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g2_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    blake2b_diag_v1(&mut a, &mut b, &mut c, &mut d);
    t0 = _mm256_blend_epi32(m2, m1, 0x33);
    t1 = _mm256_blend_epi32(m7, m2, 0x33);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g1_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    t0 = _mm256_unpacklo_epi64(m3, m5);
    t1 = _mm256_unpacklo_epi64(m0, m4);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g2_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    blake2b_undiag_v1(&mut a, &mut b, &mut c, &mut d);

    // round 4
    t0 = _mm256_unpackhi_epi64(m4, m2);
    t1 = _mm256_unpacklo_epi64(m1, m5);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g1_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    t0 = _mm256_blend_epi32(m3, m0, 0x33);
    t1 = _mm256_blend_epi32(m7, m2, 0x33);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g2_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    blake2b_diag_v1(&mut a, &mut b, &mut c, &mut d);
    t0 = _mm256_blend_epi32(m5, m7, 0x33);
    t1 = _mm256_blend_epi32(m1, m3, 0x33);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g1_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    t0 = _mm256_alignr_epi8(m6, m0, 8);
    t1 = _mm256_blend_epi32(m6, m4, 0x33);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g2_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    blake2b_undiag_v1(&mut a, &mut b, &mut c, &mut d);

    // round 5
    t0 = _mm256_unpacklo_epi64(m1, m3);
    t1 = _mm256_unpacklo_epi64(m0, m4);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g1_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    t0 = _mm256_unpacklo_epi64(m6, m5);
    t1 = _mm256_unpackhi_epi64(m5, m1);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g2_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    blake2b_diag_v1(&mut a, &mut b, &mut c, &mut d);
    t0 = _mm256_blend_epi32(m3, m2, 0x33);
    t1 = _mm256_unpackhi_epi64(m7, m0);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g1_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    t0 = _mm256_unpackhi_epi64(m6, m2);
    t1 = _mm256_blend_epi32(m4, m7, 0x33);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g2_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    blake2b_undiag_v1(&mut a, &mut b, &mut c, &mut d);

    // round 6
    t0 = _mm256_blend_epi32(m0, m6, 0x33);
    t1 = _mm256_unpacklo_epi64(m7, m2);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g1_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    t0 = _mm256_unpackhi_epi64(m2, m7);
    t1 = _mm256_alignr_epi8(m5, m6, 8);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g2_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    blake2b_diag_v1(&mut a, &mut b, &mut c, &mut d);
    t0 = _mm256_unpacklo_epi64(m0, m3);
    t1 = _mm256_shuffle_epi32(m4, _MM_SHUFFLE!(1, 0, 3, 2));
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g1_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    t0 = _mm256_unpackhi_epi64(m3, m1);
    t1 = _mm256_blend_epi32(m5, m1, 0x33);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g2_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    blake2b_undiag_v1(&mut a, &mut b, &mut c, &mut d);

    // round 7
    t0 = _mm256_unpackhi_epi64(m6, m3);
    t1 = _mm256_blend_epi32(m1, m6, 0x33);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g1_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    t0 = _mm256_alignr_epi8(m7, m5, 8);
    t1 = _mm256_unpackhi_epi64(m0, m4);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g2_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    blake2b_diag_v1(&mut a, &mut b, &mut c, &mut d);
    t0 = _mm256_unpackhi_epi64(m2, m7);
    t1 = _mm256_unpacklo_epi64(m4, m1);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g1_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    t0 = _mm256_unpacklo_epi64(m0, m2);
    t1 = _mm256_unpacklo_epi64(m3, m5);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g2_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    blake2b_undiag_v1(&mut a, &mut b, &mut c, &mut d);

    // round 8
    t0 = _mm256_unpacklo_epi64(m3, m7);
    t1 = _mm256_alignr_epi8(m0, m5, 8);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g1_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    t0 = _mm256_unpackhi_epi64(m7, m4);
    t1 = _mm256_alignr_epi8(m4, m1, 8);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g2_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    blake2b_diag_v1(&mut a, &mut b, &mut c, &mut d);
    t0 = m6;
    t1 = _mm256_alignr_epi8(m5, m0, 8);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g1_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    t0 = _mm256_blend_epi32(m3, m1, 0x33);
    t1 = m2;
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g2_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    blake2b_undiag_v1(&mut a, &mut b, &mut c, &mut d);

    // round 9
    t0 = _mm256_unpacklo_epi64(m5, m4);
    t1 = _mm256_unpackhi_epi64(m3, m0);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g1_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    t0 = _mm256_unpacklo_epi64(m1, m2);
    t1 = _mm256_blend_epi32(m2, m3, 0x33);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g2_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    blake2b_diag_v1(&mut a, &mut b, &mut c, &mut d);
    t0 = _mm256_unpackhi_epi64(m7, m4);
    t1 = _mm256_unpackhi_epi64(m1, m6);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g1_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    t0 = _mm256_alignr_epi8(m7, m5, 8);
    t1 = _mm256_unpacklo_epi64(m6, m0);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g2_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    blake2b_undiag_v1(&mut a, &mut b, &mut c, &mut d);

    // round 10
    t0 = _mm256_unpacklo_epi64(m0, m1);
    t1 = _mm256_unpacklo_epi64(m2, m3);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g1_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    t0 = _mm256_unpackhi_epi64(m0, m1);
    t1 = _mm256_unpackhi_epi64(m2, m3);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g2_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    blake2b_diag_v1(&mut a, &mut b, &mut c, &mut d);
    t0 = _mm256_unpacklo_epi64(m4, m5);
    t1 = _mm256_unpacklo_epi64(m6, m7);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g1_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    t0 = _mm256_unpackhi_epi64(m4, m5);
    t1 = _mm256_unpackhi_epi64(m6, m7);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g2_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    blake2b_undiag_v1(&mut a, &mut b, &mut c, &mut d);

    // round 11
    t0 = _mm256_unpacklo_epi64(m7, m2);
    t1 = _mm256_unpackhi_epi64(m4, m6);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g1_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    t0 = _mm256_unpacklo_epi64(m5, m4);
    t1 = _mm256_alignr_epi8(m3, m7, 8);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g2_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    blake2b_diag_v1(&mut a, &mut b, &mut c, &mut d);
    t0 = _mm256_shuffle_epi32(m0, _MM_SHUFFLE!(1, 0, 3, 2));
    t1 = _mm256_unpackhi_epi64(m5, m2);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g1_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    t0 = _mm256_unpacklo_epi64(m6, m1);
    t1 = _mm256_unpackhi_epi64(m3, m1);
    b0 = _mm256_blend_epi32(t0, t1, 0xF0);
    blake2b_g2_v1(&mut a, &mut b, &mut c, &mut d, &mut b0);
    blake2b_undiag_v1(&mut a, &mut b, &mut c, &mut d);

    a = xor(a, c);
    b = xor(b, d);
    a = xor(a, iv0);
    b = xor(b, iv1);

    store_u64x4(a, &mut words.split_mut()[0]);
    store_u64x4(b, &mut words.split_mut()[1]);
}

#[target_feature(enable = "avx2")]
pub unsafe fn compress1_loop(
    input: &[u8],
    words: &mut u64x8,
    mut count: u128,
    last_node: LastNode,
    finalize: Finalize,
    stride: Stride,
) {
    input_debug_asserts(input, finalize);

    let mut local_words = *words;

    let mut fin_offset = input.len().saturating_sub(1);
    fin_offset -= fin_offset % stride.padded_blockbytes();
    let mut buf = [0; BLOCKBYTES];
    let (fin_block, fin_len, _) = final_block(input, fin_offset, &mut buf, stride);
    let fin_last_block = u64_flag(finalize.yes());
    let fin_last_node = u64_flag(finalize.yes() && last_node.yes());

    let mut offset = 0;
    loop {
        let block;
        let count_delta;
        let last_block;
        let last_node;
        if offset == fin_offset {
            block = fin_block;
            count_delta = fin_len;
            last_block = fin_last_block;
            last_node = fin_last_node;
        } else {
            // This unsafe cast avoids bounds checks. There's guaranteed to be
            // enough input because `offset < fin_offset`.
            block = &*(input.as_ptr().add(offset) as *const [u8; BLOCKBYTES]);
            count_delta = BLOCKBYTES;
            last_block = u64_flag(false);
            last_node = u64_flag(false);
        };

        count = count.wrapping_add(count_delta as u128);
        compress_block(block, &mut local_words, count, last_block, last_node);

        // Check for termination before bumping the offset, to avoid overflow.
        if offset == fin_offset {
            break;
        }

        offset += stride.padded_blockbytes();
    }

    *words = local_words;
}

// Performance note: Factoring out a G function here doesn't hurt performance,
// unlike in the case of BLAKE2s where it hurts substantially. In fact, on my
// machine, it helps a tiny bit. But the difference it tiny, so I'm going to
// stick to the approach used by https://github.com/sneves/blake2-avx2
// until/unless I can be sure the (tiny) improvement is consistent across
// different Intel microarchitectures. Smaller code size is nice, but a
// divergence between the BLAKE2b and BLAKE2s implementations is less nice.
#[inline(always)]
unsafe fn blake2b_round_4x(v: &mut [__m256i; 16], m: &[__m256i; 16], r: usize) {
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
unsafe fn interleave128(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
    (
        _mm256_permute2x128_si256(a, b, 0x20),
        _mm256_permute2x128_si256(a, b, 0x31),
    )
}

// There are several ways to do a transposition. We could do it naively, with 8 separate
// _mm256_set_epi64x instructions, referencing each of the 64 words explicitly. Or we could copy
// the vecs into contiguous storage and then use gather instructions. This third approach is to use
// a series of unpack instructions to interleave the vectors. In my benchmarks, interleaving is the
// fastest approach. To test this, run `cargo +nightly bench --bench libtest load_4` in the
// https://github.com/oconnor663/bao_experiments repo.
#[inline(always)]
unsafe fn transpose_vecs(
    vec_a: __m256i,
    vec_b: __m256i,
    vec_c: __m256i,
    vec_d: __m256i,
) -> [__m256i; 4] {
    // Interleave 64-bit lates. The low unpack is lanes 00/22 and the high is 11/33.
    let ab_02 = _mm256_unpacklo_epi64(vec_a, vec_b);
    let ab_13 = _mm256_unpackhi_epi64(vec_a, vec_b);
    let cd_02 = _mm256_unpacklo_epi64(vec_c, vec_d);
    let cd_13 = _mm256_unpackhi_epi64(vec_c, vec_d);

    // Interleave 128-bit lanes.
    let (abcd_0, abcd_2) = interleave128(ab_02, cd_02);
    let (abcd_1, abcd_3) = interleave128(ab_13, cd_13);

    [abcd_0, abcd_1, abcd_2, abcd_3]
}

// We'd rather make this a regular function with #[inline(always)], but for
// some reason that blows up compile times by about 10 seconds, at least in
// some cases (BLAKE2b avx2.rs). This macro seems to get the same performance
// result, without the compile time issue.
macro_rules! compress4_transposed {
    (
        $h_vecs:expr,
        $msg_vecs:expr,
        $count_low:expr,
        $count_high:expr,
        $lastblock:expr,
        $lastnode:expr,
    ) => {
        let h_vecs: &mut [__m256i; 8] = $h_vecs;
        let msg_vecs: &[__m256i; 16] = $msg_vecs;
        let count_low: __m256i = $count_low;
        let count_high: __m256i = $count_high;
        let lastblock: __m256i = $lastblock;
        let lastnode: __m256i = $lastnode;

        let mut v = [
            h_vecs[0],
            h_vecs[1],
            h_vecs[2],
            h_vecs[3],
            h_vecs[4],
            h_vecs[5],
            h_vecs[6],
            h_vecs[7],
            set1(IV[0]),
            set1(IV[1]),
            set1(IV[2]),
            set1(IV[3]),
            xor(set1(IV[4]), count_low),
            xor(set1(IV[5]), count_high),
            xor(set1(IV[6]), lastblock),
            xor(set1(IV[7]), lastnode),
        ];

        blake2b_round_4x(&mut v, &msg_vecs, 0);
        blake2b_round_4x(&mut v, &msg_vecs, 1);
        blake2b_round_4x(&mut v, &msg_vecs, 2);
        blake2b_round_4x(&mut v, &msg_vecs, 3);
        blake2b_round_4x(&mut v, &msg_vecs, 4);
        blake2b_round_4x(&mut v, &msg_vecs, 5);
        blake2b_round_4x(&mut v, &msg_vecs, 6);
        blake2b_round_4x(&mut v, &msg_vecs, 7);
        blake2b_round_4x(&mut v, &msg_vecs, 8);
        blake2b_round_4x(&mut v, &msg_vecs, 9);
        blake2b_round_4x(&mut v, &msg_vecs, 10);
        blake2b_round_4x(&mut v, &msg_vecs, 11);

        h_vecs[0] = xor(xor(h_vecs[0], v[0]), v[8]);
        h_vecs[1] = xor(xor(h_vecs[1], v[1]), v[9]);
        h_vecs[2] = xor(xor(h_vecs[2], v[2]), v[10]);
        h_vecs[3] = xor(xor(h_vecs[3], v[3]), v[11]);
        h_vecs[4] = xor(xor(h_vecs[4], v[4]), v[12]);
        h_vecs[5] = xor(xor(h_vecs[5], v[5]), v[13]);
        h_vecs[6] = xor(xor(h_vecs[6], v[6]), v[14]);
        h_vecs[7] = xor(xor(h_vecs[7], v[7]), v[15]);
    };
}

#[inline(always)]
unsafe fn add_to_counts(lo: &mut __m256i, hi: &mut __m256i, delta: __m256i) {
    // If the low counts reach zero, that means they wrapped, unless the delta
    // was also zero.
    *lo = add(*lo, delta);
    let lo_reached_zero = eq(*lo, set1(0));
    let delta_was_zero = eq(delta, set1(0));
    let hi_inc = and(set1(1), negate_and(delta_was_zero, lo_reached_zero));
    *hi = add(*hi, hi_inc);
}

#[inline(always)]
unsafe fn transpose_state_vecs(jobs: &[Job; 4]) -> [__m256i; 8] {
    // Load all the state words into transposed vectors, where the first vector
    // has the first word of each state, etc. This is the form that 4-way
    // compression operates on, and transposing once at the beginning and once
    // at the end is more efficient that repeating it for each block. Note that
    // these loads are aligned, because u64x4 and u64x8 guarantee alignment.
    let [h0, h1, h2, h3] = transpose_vecs(
        load_u64x4(&jobs[0].words.split()[0]),
        load_u64x4(&jobs[1].words.split()[0]),
        load_u64x4(&jobs[2].words.split()[0]),
        load_u64x4(&jobs[3].words.split()[0]),
    );
    let [h4, h5, h6, h7] = transpose_vecs(
        load_u64x4(&jobs[0].words.split()[1]),
        load_u64x4(&jobs[1].words.split()[1]),
        load_u64x4(&jobs[2].words.split()[1]),
        load_u64x4(&jobs[3].words.split()[1]),
    );
    [h0, h1, h2, h3, h4, h5, h6, h7]
}

#[inline(always)]
unsafe fn untranspose_state_vecs(h_vecs: &[__m256i; 8], jobs: &mut [Job; 4]) {
    // Un-transpose the updated state vectors back into the caller's arrays.
    // These are aligned stores.
    let low_words = transpose_vecs(h_vecs[0], h_vecs[1], h_vecs[2], h_vecs[3]);
    store_u64x4(low_words[0], &mut jobs[0].words.split_mut()[0]);
    store_u64x4(low_words[1], &mut jobs[1].words.split_mut()[0]);
    store_u64x4(low_words[2], &mut jobs[2].words.split_mut()[0]);
    store_u64x4(low_words[3], &mut jobs[3].words.split_mut()[0]);
    let high_words = transpose_vecs(h_vecs[4], h_vecs[5], h_vecs[6], h_vecs[7]);
    store_u64x4(high_words[0], &mut jobs[0].words.split_mut()[1]);
    store_u64x4(high_words[1], &mut jobs[1].words.split_mut()[1]);
    store_u64x4(high_words[2], &mut jobs[2].words.split_mut()[1]);
    store_u64x4(high_words[3], &mut jobs[3].words.split_mut()[1]);
}

#[inline(always)]
unsafe fn transpose_msg_vecs(blocks: [*const u8; 4]) -> [__m256i; 16] {
    // These input arrays have no particular alignment, so we use unaligned
    // loads to read from them.
    let ptr0 = blocks[0] as *const __m256i;
    let ptr1 = blocks[1] as *const __m256i;
    let ptr2 = blocks[2] as *const __m256i;
    let ptr3 = blocks[3] as *const __m256i;
    let [m0, m1, m2, m3] = transpose_vecs(
        _mm256_loadu_si256(ptr0.add(0)),
        _mm256_loadu_si256(ptr1.add(0)),
        _mm256_loadu_si256(ptr2.add(0)),
        _mm256_loadu_si256(ptr3.add(0)),
    );
    let [m4, m5, m6, m7] = transpose_vecs(
        _mm256_loadu_si256(ptr0.add(1)),
        _mm256_loadu_si256(ptr1.add(1)),
        _mm256_loadu_si256(ptr2.add(1)),
        _mm256_loadu_si256(ptr3.add(1)),
    );
    let [m8, m9, m10, m11] = transpose_vecs(
        _mm256_loadu_si256(ptr0.add(2)),
        _mm256_loadu_si256(ptr1.add(2)),
        _mm256_loadu_si256(ptr2.add(2)),
        _mm256_loadu_si256(ptr3.add(2)),
    );
    let [m12, m13, m14, m15] = transpose_vecs(
        _mm256_loadu_si256(ptr0.add(3)),
        _mm256_loadu_si256(ptr1.add(3)),
        _mm256_loadu_si256(ptr2.add(3)),
        _mm256_loadu_si256(ptr3.add(3)),
    );
    [
        m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15,
    ]
}

#[inline(always)]
unsafe fn load_counts(jobs: &[Job; 4]) -> (__m256i, __m256i) {
    (
        _mm256_setr_epi64x(
            jobs[0].count as i64,
            jobs[1].count as i64,
            jobs[2].count as i64,
            jobs[3].count as i64,
        ),
        _mm256_setr_epi64x(
            (jobs[0].count >> 64) as i64,
            (jobs[1].count >> 64) as i64,
            (jobs[2].count >> 64) as i64,
            (jobs[3].count >> 64) as i64,
        ),
    )
}

#[inline(always)]
unsafe fn store_counts(jobs: &mut [Job; 4], low: __m256i, high: __m256i) {
    let low_ints: [u64; 4] = mem::transmute(low);
    let high_ints: [u64; 4] = mem::transmute(high);
    for i in 0..jobs.len() {
        jobs[i].count = low_ints[i] as u128 + ((high_ints[i] as u128) << 64);
    }
}

#[inline(always)]
unsafe fn flags_vec(flags: [bool; 4]) -> __m256i {
    set4(
        u64_flag(flags[0]),
        u64_flag(flags[1]),
        u64_flag(flags[2]),
        u64_flag(flags[3]),
    )
}

#[target_feature(enable = "avx2")]
pub unsafe fn compress4_loop(jobs: &mut [Job; 4], finalize: Finalize, stride: Stride) {
    // If we're not finalizing, there can't be a partial block at the end.
    for job in jobs.iter() {
        input_debug_asserts(job.input, finalize);
    }

    let msg_ptrs = [
        jobs[0].input.as_ptr(),
        jobs[1].input.as_ptr(),
        jobs[2].input.as_ptr(),
        jobs[3].input.as_ptr(),
    ];
    let mut h_vecs = transpose_state_vecs(&jobs);
    let (mut counts_lo, mut counts_hi) = load_counts(&jobs);

    // Prepare the final blocks (note, which could be empty if the input is
    // empty). Do all this before entering the main loop.
    let min_len = jobs.iter().map(|job| job.input.len()).min().unwrap();
    let mut fin_offset = min_len.saturating_sub(1);
    fin_offset -= fin_offset % stride.padded_blockbytes();
    // Performance note, making these buffers mem::uninitialized() seems to
    // cause problems in the optimizer.
    let mut buf0: [u8; BLOCKBYTES] = [0; BLOCKBYTES];
    let mut buf1: [u8; BLOCKBYTES] = [0; BLOCKBYTES];
    let mut buf2: [u8; BLOCKBYTES] = [0; BLOCKBYTES];
    let mut buf3: [u8; BLOCKBYTES] = [0; BLOCKBYTES];
    let (block0, len0, finalize0) = final_block(jobs[0].input, fin_offset, &mut buf0, stride);
    let (block1, len1, finalize1) = final_block(jobs[1].input, fin_offset, &mut buf1, stride);
    let (block2, len2, finalize2) = final_block(jobs[2].input, fin_offset, &mut buf2, stride);
    let (block3, len3, finalize3) = final_block(jobs[3].input, fin_offset, &mut buf3, stride);
    let fin_blocks = [
        block0.as_ptr(),
        block1.as_ptr(),
        block2.as_ptr(),
        block3.as_ptr(),
    ];
    let fin_counts_delta = set4(len0 as u64, len1 as u64, len2 as u64, len3 as u64);
    let fin_last_block;
    let fin_last_node;
    if finalize.yes() {
        fin_last_block = flags_vec([finalize0, finalize1, finalize2, finalize3]);
        fin_last_node = flags_vec([
            finalize0 && jobs[0].last_node.yes(),
            finalize1 && jobs[1].last_node.yes(),
            finalize2 && jobs[2].last_node.yes(),
            finalize3 && jobs[3].last_node.yes(),
        ]);
    } else {
        fin_last_block = set1(0);
        fin_last_node = set1(0);
    }

    // The main loop.
    let mut offset = 0;
    loop {
        let blocks;
        let counts_delta;
        let last_block;
        let last_node;
        if offset == fin_offset {
            blocks = fin_blocks;
            counts_delta = fin_counts_delta;
            last_block = fin_last_block;
            last_node = fin_last_node;
        } else {
            blocks = [
                msg_ptrs[0].add(offset),
                msg_ptrs[1].add(offset),
                msg_ptrs[2].add(offset),
                msg_ptrs[3].add(offset),
            ];
            counts_delta = set1(BLOCKBYTES as u64);
            last_block = set1(0);
            last_node = set1(0);
        };

        let m_vecs = transpose_msg_vecs(blocks);
        add_to_counts(&mut counts_lo, &mut counts_hi, counts_delta);
        compress4_transposed!(
            &mut h_vecs,
            &m_vecs,
            counts_lo,
            counts_hi,
            last_block,
            last_node,
        );

        // Check for termination before bumping the offset, to avoid overflow.
        if offset == fin_offset {
            break;
        }

        offset += stride.padded_blockbytes();
    }

    // Write out the results.
    untranspose_state_vecs(&h_vecs, &mut *jobs);
    store_counts(&mut *jobs, counts_lo, counts_hi);
    let max_consumed = offset.saturating_add(stride.padded_blockbytes());
    for job in jobs.iter_mut() {
        let consumed = cmp::min(max_consumed, job.input.len());
        job.input = &job.input[consumed..];
    }
}
