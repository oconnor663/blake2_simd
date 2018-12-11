#![feature(test)]

extern crate blake2b_simd;
#[cfg(feature = "libsodium-ffi")]
extern crate libsodium_ffi;
#[cfg(feature = "openssl")]
extern crate openssl;
extern crate test;

use blake2b_simd::*;
use test::Bencher;

const BLOCK: &[u8; BLOCKBYTES] = &[0; BLOCKBYTES];
const MB: &[u8; 1_000_000] = &[0; 1_000_000];

#[bench]
fn bench_blake2b_avx2_one_block(b: &mut Bencher) {
    b.bytes = BLOCK.len() as u64;
    b.iter(|| blake2b(BLOCK));
}

#[bench]
fn bench_blake2b_avx2_one_mb(b: &mut Bencher) {
    b.bytes = MB.len() as u64;
    b.iter(|| blake2b(MB));
}

#[bench]
fn bench_blake2b_portable_one_block(b: &mut Bencher) {
    b.bytes = BLOCK.len() as u64;
    b.iter(|| {
        let mut state = State::new();
        benchmarks::force_portable(&mut state);
        state.update(BLOCK);
        state.finalize()
    });
}

#[bench]
fn bench_blake2b_portable_one_mb(b: &mut Bencher) {
    b.bytes = MB.len() as u64;
    b.iter(|| {
        let mut state = State::new();
        benchmarks::force_portable(&mut state);
        state.update(MB);
        state.finalize()
    });
}

#[bench]
fn bench_blake2bp_one_mb(b: &mut Bencher) {
    b.bytes = MB.len() as u64;
    b.iter(|| blake2bp::blake2bp(MB));
}

#[bench]
fn bench_blake2b_update4_one_block(b: &mut Bencher) {
    b.bytes = 4 * BLOCK.len() as u64;
    b.iter(|| {
        let mut state0 = State::new();
        let mut state1 = State::new();
        let mut state2 = State::new();
        let mut state3 = State::new();
        update4(
            &mut state0,
            &mut state1,
            &mut state2,
            &mut state3,
            BLOCK,
            BLOCK,
            BLOCK,
            BLOCK,
        );
        finalize4(&mut state0, &mut state1, &mut state2, &mut state3)
    });
}

#[bench]
fn bench_blake2b_update4_one_mb(b: &mut Bencher) {
    b.bytes = 4 * MB.len() as u64;
    b.iter(|| {
        let mut state0 = State::new();
        let mut state1 = State::new();
        let mut state2 = State::new();
        let mut state3 = State::new();
        update4(
            &mut state0,
            &mut state1,
            &mut state2,
            &mut state3,
            MB,
            MB,
            MB,
            MB,
        );
        finalize4(&mut state0, &mut state1, &mut state2, &mut state3)
    });
}

#[bench]
fn bench_blake2b_hash4_one_block(b: &mut Bencher) {
    b.bytes = 4 * BLOCKBYTES as u64;
    let block0 = [0xf0; BLOCKBYTES];
    let block1 = [0xf1; BLOCKBYTES];
    let block2 = [0xf2; BLOCKBYTES];
    let block3 = [0xf3; BLOCKBYTES];
    let params = Params::new();
    b.iter(|| hash4_exact(&params, &block0, &block1, &block2, &block3));
}

#[bench]
fn bench_blake2b_hash4_4096(b: &mut Bencher) {
    const CHUNK: usize = 4096;
    b.bytes = 4 * CHUNK as u64;
    let chunk0 = [0xf0; CHUNK];
    let chunk1 = [0xf1; CHUNK];
    let chunk2 = [0xf2; CHUNK];
    let chunk3 = [0xf3; CHUNK];
    let params = Params::new();
    b.iter(|| hash4_exact(&params, &chunk0, &chunk1, &chunk2, &chunk3));
}

#[bench]
fn bench_blake2b_hash4_one_mb(b: &mut Bencher) {
    const MB: usize = 1 << 20;
    b.bytes = 4 * MB as u64;
    let mb0 = [0xf0; MB];
    let mb1 = [0xf1; MB];
    let mb2 = [0xf2; MB];
    let mb3 = [0xf3; MB];
    let params = Params::new();
    b.iter(|| hash4_exact(&params, &mb0, &mb1, &mb2, &mb3));
}

#[cfg(feature = "libsodium-ffi")]
#[bench]
fn bench_libsodium_one_mb(b: &mut Bencher) {
    b.bytes = MB.len() as u64;
    let mut out = [0; 64];
    unsafe {
        let init_ret = libsodium_ffi::sodium_init();
        assert_eq!(0, init_ret);
    }
    b.iter(|| unsafe {
        libsodium_ffi::crypto_generichash(
            out.as_mut_ptr(),
            out.len(),
            MB.as_ptr(),
            MB.len() as u64,
            std::ptr::null(),
            0,
        );
    });
}

#[cfg(feature = "openssl")]
#[bench]
fn bench_openssl_md5_one_mb(b: &mut Bencher) {
    b.bytes = MB.len() as u64;
    b.iter(|| openssl::hash::hash(openssl::hash::MessageDigest::md5(), MB));
}

#[cfg(feature = "openssl")]
#[bench]
fn bench_openssl_sha1_one_mb(b: &mut Bencher) {
    b.bytes = MB.len() as u64;
    b.iter(|| openssl::hash::hash(openssl::hash::MessageDigest::sha1(), MB));
}

#[cfg(feature = "openssl")]
#[bench]
fn bench_openssl_sha512_one_mb(b: &mut Bencher) {
    b.bytes = MB.len() as u64;
    b.iter(|| openssl::hash::hash(openssl::hash::MessageDigest::sha512(), MB));
}

#[bench]
fn bench_guts_compress1_portable(b: &mut Bencher) {
    b.bytes = BLOCK.len() as u64;
    let portable = guts::Implementation::portable();
    let mut state = [1; 8];
    b.iter(|| {
        portable.compress(&mut state, BLOCK, 0, 0, 0);
    });
    test::black_box(&mut state);
}

#[bench]
fn bench_guts_compress1_avx2(b: &mut Bencher) {
    b.bytes = BLOCK.len() as u64;
    let avx2 = if let Some(imp) = guts::Implementation::avx2_if_supported() {
        imp
    } else {
        return;
    };
    let mut state = [1; 8];
    b.iter(|| {
        avx2.compress(&mut state, BLOCK, 0, 0, 0);
    });
    test::black_box(&mut state);
}

#[bench]
fn bench_guts_compress2_portable(b: &mut Bencher) {
    b.bytes = BLOCK.len() as u64 * 2;
    let portable = guts::Implementation::portable();
    let mut state = [guts::u64x2([1, 1]); 8];
    let count_low = guts::u64x2([1, 1]);
    let count_high = guts::u64x2([1, 1]);
    let lastblock = guts::u64x2([1, 1]);
    let lastnode = guts::u64x2([1, 1]);
    b.iter(|| {
        portable.compress2(
            &mut state,
            BLOCK,
            BLOCK,
            &count_low,
            &count_high,
            &lastblock,
            &lastnode,
        );
    });
    test::black_box(&mut state);
}

#[bench]
fn bench_guts_compress2_sse41(b: &mut Bencher) {
    b.bytes = BLOCK.len() as u64 * 2;
    let sse41 = if let Some(imp) = guts::Implementation::sse41_if_supported() {
        imp
    } else {
        return;
    };
    let mut state = [guts::u64x2([1, 1]); 8];
    let count_low = guts::u64x2([1, 1]);
    let count_high = guts::u64x2([1, 1]);
    let lastblock = guts::u64x2([1, 1]);
    let lastnode = guts::u64x2([1, 1]);
    b.iter(|| {
        sse41.compress2(
            &mut state,
            BLOCK,
            BLOCK,
            &count_low,
            &count_high,
            &lastblock,
            &lastnode,
        );
    });
    test::black_box(&mut state);
}

#[bench]
fn bench_guts_transpose2_portable(b: &mut Bencher) {
    let portable = guts::Implementation::portable();
    let state0 = [1; 8];
    let state1 = [2; 8];
    b.iter(|| portable.transpose2(&state0, &state1));
}

#[bench]
fn bench_guts_untranspose2_portable(b: &mut Bencher) {
    let portable = guts::Implementation::portable();
    let mut state0 = [1; 8];
    let mut state1 = [2; 8];
    let transposed = portable.transpose2(&state0, &state1);
    b.iter(|| {
        portable.untranspose2(&transposed, &mut state0, &mut state1);
        test::black_box(&mut state0);
        test::black_box(&mut state1);
    });
}

#[bench]
fn bench_guts_compress4_portable(b: &mut Bencher) {
    b.bytes = BLOCK.len() as u64 * 4;
    let portable = guts::Implementation::portable();
    let mut state = [guts::u64x4([1, 1, 1, 1]); 8];
    let count_low = guts::u64x4([1, 1, 1, 1]);
    let count_high = guts::u64x4([1, 1, 1, 1]);
    let lastblock = guts::u64x4([1, 1, 1, 1]);
    let lastnode = guts::u64x4([1, 1, 1, 1]);
    b.iter(|| {
        portable.compress4(
            &mut state,
            BLOCK,
            BLOCK,
            BLOCK,
            BLOCK,
            &count_low,
            &count_high,
            &lastblock,
            &lastnode,
        );
    });
    test::black_box(&mut state);
}

#[bench]
fn bench_guts_compress4_sse41(b: &mut Bencher) {
    b.bytes = BLOCK.len() as u64 * 4;
    let sse41 = if let Some(imp) = guts::Implementation::sse41_if_supported() {
        imp
    } else {
        return;
    };
    let mut state = [guts::u64x4([1, 1, 1, 1]); 8];
    let count_low = guts::u64x4([1, 1, 1, 1]);
    let count_high = guts::u64x4([1, 1, 1, 1]);
    let lastblock = guts::u64x4([1, 1, 1, 1]);
    let lastnode = guts::u64x4([1, 1, 1, 1]);
    b.iter(|| {
        sse41.compress4(
            &mut state,
            BLOCK,
            BLOCK,
            BLOCK,
            BLOCK,
            &count_low,
            &count_high,
            &lastblock,
            &lastnode,
        );
    });
    test::black_box(&mut state);
}

#[bench]
fn bench_guts_compress4_avx2(b: &mut Bencher) {
    b.bytes = BLOCK.len() as u64 * 4;
    let avx2 = if let Some(imp) = guts::Implementation::avx2_if_supported() {
        imp
    } else {
        return;
    };
    let mut state = [guts::u64x4([1, 1, 1, 1]); 8];
    let count_low = guts::u64x4([1, 1, 1, 1]);
    let count_high = guts::u64x4([1, 1, 1, 1]);
    let lastblock = guts::u64x4([1, 1, 1, 1]);
    let lastnode = guts::u64x4([1, 1, 1, 1]);
    b.iter(|| {
        avx2.compress4(
            &mut state,
            BLOCK,
            BLOCK,
            BLOCK,
            BLOCK,
            &count_low,
            &count_high,
            &lastblock,
            &lastnode,
        );
    });
    test::black_box(&mut state);
}

#[bench]
fn bench_guts_transpose4_portable(b: &mut Bencher) {
    let portable = guts::Implementation::portable();
    let state0 = [1; 8];
    let state1 = [2; 8];
    let state2 = [3; 8];
    let state3 = [4; 8];
    b.iter(|| portable.transpose4(&state0, &state1, &state2, &state3));
}

#[bench]
fn bench_guts_transpose4_avx2(b: &mut Bencher) {
    let avx2 = if let Some(imp) = guts::Implementation::avx2_if_supported() {
        imp
    } else {
        return;
    };
    let state0 = [1; 8];
    let state1 = [2; 8];
    let state2 = [3; 8];
    let state3 = [4; 8];
    b.iter(|| avx2.transpose4(&state0, &state1, &state2, &state3));
}

#[bench]
fn bench_guts_untranspose4_portable(b: &mut Bencher) {
    let portable = guts::Implementation::portable();
    let mut state0 = [1; 8];
    let mut state1 = [2; 8];
    let mut state2 = [3; 8];
    let mut state3 = [4; 8];
    let transposed = portable.transpose4(&state0, &state1, &state2, &state3);
    b.iter(|| {
        portable.untranspose4(
            &transposed,
            &mut state0,
            &mut state1,
            &mut state2,
            &mut state3,
        );
        test::black_box(&mut state0);
        test::black_box(&mut state1);
        test::black_box(&mut state2);
        test::black_box(&mut state3);
    });
}

#[bench]
fn bench_guts_untranspose4_avx2(b: &mut Bencher) {
    let avx2 = if let Some(imp) = guts::Implementation::avx2_if_supported() {
        imp
    } else {
        return;
    };
    let mut state0 = [1; 8];
    let mut state1 = [2; 8];
    let mut state2 = [3; 8];
    let mut state3 = [4; 8];
    let transposed = avx2.transpose4(&state0, &state1, &state2, &state3);
    b.iter(|| {
        avx2.untranspose4(
            &transposed,
            &mut state0,
            &mut state1,
            &mut state2,
            &mut state3,
        );
        test::black_box(&mut state0);
        test::black_box(&mut state1);
        test::black_box(&mut state2);
        test::black_box(&mut state3);
    });
}
