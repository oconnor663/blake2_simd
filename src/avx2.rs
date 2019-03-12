#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::*;
use crate::guts;
use crate::guts::{u64_flag, u64x4, u64x8};

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
unsafe fn sub(a: __m256i, b: __m256i) -> __m256i {
    _mm256_sub_epi64(a, b)
}

#[inline(always)]
unsafe fn xor(a: __m256i, b: __m256i) -> __m256i {
    _mm256_xor_si256(a, b)
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
unsafe fn compress_block(h: &mut u64x8, msg: &Block, count: u128, lastblock: u64, lastnode: u64) {
    let mut a = load_u64x4(&h.split()[0]);
    let mut b = load_u64x4(&h.split()[1]);
    let mut c = load_u64x4(&IV.split()[0]);
    let count_low = count as i64;
    let count_high = (count >> 64) as i64;
    let flags = _mm256_set_epi64x(lastnode as i64, lastblock as i64, count_high, count_low);
    let mut d = xor(load_u64x4(&IV.split()[1]), flags);

    let msg_chunks = array_refs!(msg, 16, 16, 16, 16, 16, 16, 16, 16);
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

    store_u64x4(a, &mut h.split_mut()[0]);
    store_u64x4(b, &mut h.split_mut()[1]);
}

#[target_feature(enable = "avx2")]
pub unsafe fn compress1_loop(
    state: &mut u64x8,
    input: &[u8],
    mut count: u128,
    last_block: u64,
    last_node: u64,
    mut blocks: usize,
    stride: usize,
    buffer_tail: usize,
) {
    // Check the input length once to avoid bounds checks in the loop below.
    assert!(BLOCKBYTES * (stride * (blocks - 1) + 1) <= input.len());

    let mut offset = 0;
    while blocks > 0 {
        // An unchecked pointer deref, guarded by the assert above.
        let block = &*(input.as_ptr().add(offset) as *const Block);

        count = count.wrapping_add(BLOCKBYTES as u128);
        if blocks == 1 {
            count = count.wrapping_sub(buffer_tail as u128);
        }
        let (maybe_last_block, maybe_last_node) = if blocks == 1 {
            (last_block, last_node)
        } else {
            (0, 0)
        };

        compress_block(state, block, count, maybe_last_block, maybe_last_node);

        offset += stride * BLOCKBYTES;
        blocks -= 1;
    }
}

#[target_feature(enable = "avx2")]
pub unsafe fn compress1_loop_b(
    state: &mut u64x8,
    input: &[u8],
    count: &mut u128,
    last_block: bool,
    last_node: bool,
    parallel_stride: bool,
) {
    if !last_block {
        debug_assert!(!last_node);
        debug_assert_eq!(0, input.len() % BLOCKBYTES);
    }
    let mut offset = 0;
    let final_block_offset = guts::final_block_offset(input.len(), parallel_stride);
    let mut buffer = [0; BLOCKBYTES];
    let (finblock, finblock_len) = guts::get_block(input, final_block_offset, &mut buffer);
    let mut local_count = *count;
    while offset <= final_block_offset {
        let is_final_block = offset == final_block_offset;
        let block;
        if is_final_block {
            block = finblock;
            local_count = local_count.wrapping_add(finblock_len as u128);
        } else {
            // This is an unsafe pointer cast to avoid bounds checks. The count
            // returned by loop_iterations() guarantees that this read is
            // in-bounds.
            block = &*(input.as_ptr().add(offset) as *const [u8; BLOCKBYTES]);
            local_count = local_count.wrapping_add(BLOCKBYTES as u128);
        }
        compress_block(
            state,
            block,
            local_count,
            u64_flag(is_final_block && last_block),
            u64_flag(is_final_block && last_node),
        );
        offset += guts::padded_blockbytes(parallel_stride);
    }
    *count = local_count;
}

#[inline(always)]
unsafe fn load_256_from_u64(x: u64) -> __m256i {
    _mm256_set1_epi64x(x as i64)
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

#[inline(always)]
unsafe fn compress4_transposed_inline(
    h_vecs: &mut [__m256i; 8],
    msg_vecs: &[__m256i; 16],
    count_low: __m256i,
    count_high: __m256i,
    lastblock: __m256i,
    lastnode: __m256i,
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
        load_256_from_u64(IV[0]),
        load_256_from_u64(IV[1]),
        load_256_from_u64(IV[2]),
        load_256_from_u64(IV[3]),
        xor(load_256_from_u64(IV[4]), count_low),
        xor(load_256_from_u64(IV[5]), count_high),
        xor(load_256_from_u64(IV[6]), lastblock),
        xor(load_256_from_u64(IV[7]), lastnode),
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
}

#[inline(always)]
unsafe fn unsigned_cmpgt_epi64(a: __m256i, b: __m256i) -> __m256i {
    // Because _mm256_cmpgt_epi64 operates on signed values, we need to
    // subtract 2^63 from each value before doing the comparison.
    let delta = _mm256_set1_epi64x(i64::min_value());
    _mm256_cmpgt_epi64(_mm256_add_epi64(a, delta), _mm256_add_epi64(b, delta))
}

#[target_feature(enable = "avx2")]
pub unsafe fn compress4_loop(
    state0: &mut u64x8,
    state1: &mut u64x8,
    state2: &mut u64x8,
    state3: &mut u64x8,
    input0: &[u8],
    input1: &[u8],
    input2: &[u8],
    input3: &[u8],
    count_low: &u64x4,
    count_high: &u64x4,
    last_block: &u64x4,
    last_node: &u64x4,
    mut blocks: usize,
    stride: usize,
    buffer_tail: &u64x4,
) {
    // Check the input slice lengths once here. The main loop will do unaligned
    // loads without any further bounds checks.
    assert!(BLOCKBYTES * (stride * (blocks - 1) + 1) <= input0.len());
    assert!(BLOCKBYTES * (stride * (blocks - 1) + 1) <= input1.len());
    assert!(BLOCKBYTES * (stride * (blocks - 1) + 1) <= input2.len());
    assert!(BLOCKBYTES * (stride * (blocks - 1) + 1) <= input3.len());

    // Load all the state words into transposed vectors, where the first vector
    // has the first word of each state, etc. This is the form that 4-way
    // compression operates on, and transposing once at the beginning and once
    // at the end is more efficient that repeating it for each block. Note that
    // these loads are aligned, because u64x4 and u64x8 guarantee alignment.
    let [h0, h1, h2, h3] = transpose_vecs(
        load_u64x4(&state0.split()[0]),
        load_u64x4(&state1.split()[0]),
        load_u64x4(&state2.split()[0]),
        load_u64x4(&state3.split()[0]),
    );
    let [h4, h5, h6, h7] = transpose_vecs(
        load_u64x4(&state0.split()[1]),
        load_u64x4(&state1.split()[1]),
        load_u64x4(&state2.split()[1]),
        load_u64x4(&state3.split()[1]),
    );
    let mut h_vecs = [h0, h1, h2, h3, h4, h5, h6, h7];
    let mut count_low_vec = load_u64x4(count_low);
    let mut count_high_vec = load_u64x4(count_high);
    let mut offset = 0;

    while blocks > 0 {
        // Load all the message words into transposed vectors also. Message
        // loads are unaligned, because these are arbitrary byte pointers from
        // the caller. On modern chips though, there's not much of a
        // performance penalty for unaligned loads.
        let block0 = input0.as_ptr().add(offset) as *const __m256i;
        let block1 = input1.as_ptr().add(offset) as *const __m256i;
        let block2 = input2.as_ptr().add(offset) as *const __m256i;
        let block3 = input3.as_ptr().add(offset) as *const __m256i;
        let [m0, m1, m2, m3] = transpose_vecs(
            _mm256_loadu_si256(block0.add(0)),
            _mm256_loadu_si256(block1.add(0)),
            _mm256_loadu_si256(block2.add(0)),
            _mm256_loadu_si256(block3.add(0)),
        );
        let [m4, m5, m6, m7] = transpose_vecs(
            _mm256_loadu_si256(block0.add(1)),
            _mm256_loadu_si256(block1.add(1)),
            _mm256_loadu_si256(block2.add(1)),
            _mm256_loadu_si256(block3.add(1)),
        );
        let [m8, m9, m10, m11] = transpose_vecs(
            _mm256_loadu_si256(block0.add(2)),
            _mm256_loadu_si256(block1.add(2)),
            _mm256_loadu_si256(block2.add(2)),
            _mm256_loadu_si256(block3.add(2)),
        );
        let [m12, m13, m14, m15] = transpose_vecs(
            _mm256_loadu_si256(block0.add(3)),
            _mm256_loadu_si256(block1.add(3)),
            _mm256_loadu_si256(block2.add(3)),
            _mm256_loadu_si256(block3.add(3)),
        );
        let m_vecs = [
            m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15,
        ];

        // Add BLOCKBYTES to the low count bits.
        let old_count_low_vec = count_low_vec;
        count_low_vec = add(count_low_vec, _mm256_set1_epi64x(BLOCKBYTES as i64));
        // If this is the last block, subtract the buffer tails.
        if blocks == 1 {
            count_low_vec = sub(count_low_vec, load_u64x4(buffer_tail));
        }
        // Finally if any of the low counts overflowed (after accounting for
        // the buffer tails), increment the corresponding high counts.
        count_high_vec = add(
            count_high_vec,
            _mm256_and_si256(
                unsigned_cmpgt_epi64(old_count_low_vec, count_low_vec),
                _mm256_set1_epi64x(1),
            ),
        );

        // Compressions before the last one always use zero for the
        // finalization flags. The last one will use what the caller supplied,
        // which could also be zero if the input isn't finished.
        let (last_block_vec, last_node_vec) = if blocks == 1 {
            (load_u64x4(last_block), load_u64x4(last_node))
        } else {
            (_mm256_set1_epi64x(0), _mm256_set1_epi64x(0))
        };

        compress4_transposed_inline(
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
    let low_words = transpose_vecs(h_vecs[0], h_vecs[1], h_vecs[2], h_vecs[3]);
    store_u64x4(low_words[0], &mut state0.split_mut()[0]);
    store_u64x4(low_words[1], &mut state1.split_mut()[0]);
    store_u64x4(low_words[2], &mut state2.split_mut()[0]);
    store_u64x4(low_words[3], &mut state3.split_mut()[0]);
    let high_words = transpose_vecs(h_vecs[4], h_vecs[5], h_vecs[6], h_vecs[7]);
    store_u64x4(high_words[0], &mut state0.split_mut()[1]);
    store_u64x4(high_words[1], &mut state1.split_mut()[1]);
    store_u64x4(high_words[2], &mut state2.split_mut()[1]);
    store_u64x4(high_words[3], &mut state3.split_mut()[1]);
}

#[inline(always)]
unsafe fn transpose_state_vecs(states: &[&mut u64x8; 4]) -> [__m256i; 8] {
    // Load all the state words into transposed vectors, where the first vector
    // has the first word of each state, etc. This is the form that 4-way
    // compression operates on, and transposing once at the beginning and once
    // at the end is more efficient that repeating it for each block. Note that
    // these loads are aligned, because u64x4 and u64x8 guarantee alignment.
    let [h0, h1, h2, h3] = transpose_vecs(
        load_u64x4(&states[0].split()[0]),
        load_u64x4(&states[1].split()[0]),
        load_u64x4(&states[2].split()[0]),
        load_u64x4(&states[3].split()[0]),
    );
    let [h4, h5, h6, h7] = transpose_vecs(
        load_u64x4(&states[0].split()[1]),
        load_u64x4(&states[1].split()[1]),
        load_u64x4(&states[2].split()[1]),
        load_u64x4(&states[3].split()[1]),
    );
    [h0, h1, h2, h3, h4, h5, h6, h7]
}

#[inline(always)]
unsafe fn untranspose_state_vecs(h_vecs: &[__m256i; 8], states: [&mut u64x8; 4]) {
    // Un-transpose the updated state vectors back into the caller's arrays.
    // These are aligned stores.
    let low_words = transpose_vecs(h_vecs[0], h_vecs[1], h_vecs[2], h_vecs[3]);
    store_u64x4(low_words[0], &mut states[0].split_mut()[0]);
    store_u64x4(low_words[1], &mut states[1].split_mut()[0]);
    store_u64x4(low_words[2], &mut states[2].split_mut()[0]);
    store_u64x4(low_words[3], &mut states[3].split_mut()[0]);
    let high_words = transpose_vecs(h_vecs[4], h_vecs[5], h_vecs[6], h_vecs[7]);
    store_u64x4(high_words[0], &mut states[0].split_mut()[1]);
    store_u64x4(high_words[1], &mut states[1].split_mut()[1]);
    store_u64x4(high_words[2], &mut states[2].split_mut()[1]);
    store_u64x4(high_words[3], &mut states[3].split_mut()[1]);
}

#[inline(always)]
unsafe fn transpose_msg_vecs(
    block0: &[u8; BLOCKBYTES],
    block1: &[u8; BLOCKBYTES],
    block2: &[u8; BLOCKBYTES],
    block3: &[u8; BLOCKBYTES],
) -> [__m256i; 16] {
    // These input arrays have no particular alignment, so we use unaligned
    // loads to read from them.
    let ptr0 = block0.as_ptr() as *const __m256i;
    let ptr1 = block1.as_ptr() as *const __m256i;
    let ptr2 = block2.as_ptr() as *const __m256i;
    let ptr3 = block3.as_ptr() as *const __m256i;
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
unsafe fn load_counts_low(count0: u128, count1: u128, count2: u128, count3: u128) -> __m256i {
    _mm256_setr_epi64x(count0 as i64, count1 as i64, count2 as i64, count3 as i64)
}

#[inline(always)]
unsafe fn load_counts_high(count0: u128, count1: u128, count2: u128, count3: u128) -> __m256i {
    _mm256_setr_epi64x(
        (count0 >> 64) as i64,
        (count1 >> 64) as i64,
        (count2 >> 64) as i64,
        (count3 >> 64) as i64,
    )
}

#[inline(always)]
unsafe fn load_flags_vec(flags: [bool; 4]) -> __m256i {
    _mm256_setr_epi64x(
        u64_flag(flags[0]) as i64,
        u64_flag(flags[1]) as i64,
        u64_flag(flags[2]) as i64,
        u64_flag(flags[3]) as i64,
    )
}

#[target_feature(enable = "avx2")]
pub unsafe fn compress4_loop_b(
    states: [&mut u64x8; 4],
    inputs: [&[u8]; 4],
    counts: [&mut u128; 4],
    last_block: [bool; 4],
    last_node: [bool; 4],
    parallel_stride: bool,
) -> usize {
    for i in 0..inputs.len() {
        if !last_block[i] {
            debug_assert!(!last_node[i]);
            debug_assert_eq!(0, inputs[i].len() % BLOCKBYTES);
        }
    }
    let mut h_vecs = transpose_state_vecs(&states);
    let mut offset = 0;
    let mut count0 = *counts[0];
    let mut count1 = *counts[1];
    let mut count2 = *counts[2];
    let mut count3 = *counts[3];
    let min_len = inputs.iter().map(|i| i.len()).min().unwrap();
    let final_block_offset = guts::final_block_offset(min_len, parallel_stride);
    let mut buffer0 = [0; BLOCKBYTES];
    let mut buffer1 = [0; BLOCKBYTES];
    let mut buffer2 = [0; BLOCKBYTES];
    let mut buffer3 = [0; BLOCKBYTES];
    let (finblock0, finblock_len0) = guts::get_block(inputs[0], final_block_offset, &mut buffer0);
    let (finblock1, finblock_len1) = guts::get_block(inputs[1], final_block_offset, &mut buffer1);
    let (finblock2, finblock_len2) = guts::get_block(inputs[2], final_block_offset, &mut buffer2);
    let (finblock3, finblock_len3) = guts::get_block(inputs[3], final_block_offset, &mut buffer3);
    let finlastblockvec = load_flags_vec(last_block);
    let finlastnodevec = load_flags_vec(last_node);
    let zerovec = _mm256_set1_epi64x(0);
    while offset <= final_block_offset {
        let is_final_block = offset == final_block_offset;
        let block0;
        let block1;
        let block2;
        let block3;
        let last_block_vec;
        let last_node_vec;
        if is_final_block {
            block0 = finblock0;
            block1 = finblock1;
            block2 = finblock2;
            block3 = finblock3;
            last_block_vec = finlastblockvec;
            last_node_vec = finlastnodevec;
            count0 = count0.wrapping_add(finblock_len0 as u128);
            count1 = count1.wrapping_add(finblock_len1 as u128);
            count2 = count2.wrapping_add(finblock_len2 as u128);
            count3 = count3.wrapping_add(finblock_len3 as u128);
        } else {
            // These unsafe pointer casts avoid paying for bounds checks. The
            // final_block_offset math guarantees that these loads are
            // in-bounds.
            block0 = &*(inputs[0].as_ptr().add(offset) as *const [u8; BLOCKBYTES]);
            block1 = &*(inputs[1].as_ptr().add(offset) as *const [u8; BLOCKBYTES]);
            block2 = &*(inputs[2].as_ptr().add(offset) as *const [u8; BLOCKBYTES]);
            block3 = &*(inputs[3].as_ptr().add(offset) as *const [u8; BLOCKBYTES]);
            last_block_vec = zerovec;
            last_node_vec = zerovec;
            count0 = count0.wrapping_add(BLOCKBYTES as u128);
            count1 = count1.wrapping_add(BLOCKBYTES as u128);
            count2 = count2.wrapping_add(BLOCKBYTES as u128);
            count3 = count3.wrapping_add(BLOCKBYTES as u128);
        }
        let m_vecs = transpose_msg_vecs(block0, block1, block2, block3);
        let counts_low = load_counts_low(count0, count1, count2, count3);
        let counts_high = load_counts_high(count0, count1, count2, count3);

        compress4_transposed_inline(
            &mut h_vecs,
            &m_vecs,
            counts_low,
            counts_high,
            last_block_vec,
            last_node_vec,
        );

        offset = offset.saturating_add(guts::padded_blockbytes(parallel_stride));
    }

    *counts[0] = count0;
    *counts[1] = count1;
    *counts[2] = count2;
    *counts[3] = count3;
    untranspose_state_vecs(&h_vecs, states);
    offset
}
