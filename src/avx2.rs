use std::arch::x86_64::*;

use BLOCKBYTES;
use IV;

#[inline(always)]
unsafe fn load_256_unaligned(mem_addr: &[u64; 4]) -> __m256i {
    _mm256_loadu_si256(mem_addr.as_ptr() as *const __m256i)
}

#[inline(always)]
unsafe fn store_256_unaligned(mem_addr: &mut [u64; 4], a: __m256i) {
    _mm256_storeu_si256(mem_addr.as_mut_ptr() as *mut __m256i, a);
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

// array_ref triggers unused_unsafe (https://github.com/droundy/arrayref/pull/14)
#[allow(unused_unsafe)]
#[target_feature(enable = "avx2")]
pub unsafe fn compress(h: &mut [u64; 8], msg: &[u8; BLOCKBYTES], count: u128, lastblock: u64) {
    unsafe {
        let mut a = load_256_unaligned(array_ref!(h, 0, 4));
        let mut b = load_256_unaligned(array_ref!(h, 4, 4));
        let mut c = load_256_unaligned(array_ref!(IV, 0, 4));
        let count_low = count as i64;
        let count_high = (count >> 64) as i64;
        let lastnode = 0;
        let mut d = xor(
            load_256_unaligned(array_ref!(IV, 4, 4)),
            _mm256_set_epi64x(lastnode, lastblock as i64, count_high, count_low),
        );
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

        store_256_unaligned(array_mut_ref!(h, 0, 4), a);
        store_256_unaligned(array_mut_ref!(h, 4, 4), b);
    }
}
