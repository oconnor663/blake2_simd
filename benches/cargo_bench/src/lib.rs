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
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_blake2b_avx2_compress(b: &mut Bencher) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    b.bytes = BLOCK.len() as u64;
    let mut h = [0; 8];
    b.iter(|| unsafe { benchmarks::compress_avx2(&mut h, BLOCK, 0, 0, 0) });
}

#[bench]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_blake2b_avx2_compress4(b: &mut Bencher) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    b.bytes = BLOCK.len() as u64 * 4;
    let mut h1 = [0; 8];
    let mut h2 = [0; 8];
    let mut h3 = [0; 8];
    let mut h4 = [0; 8];
    b.iter(|| unsafe {
        benchmarks::compress4_avx2(
            &mut h1, &mut h2, &mut h3, &mut h4, &BLOCK, BLOCK, BLOCK, BLOCK, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
        );
    });
}

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
fn bench_blake2b_portable_compress(b: &mut Bencher) {
    b.bytes = BLOCK.len() as u64;
    let mut h = [0; 8];
    b.iter(|| benchmarks::compress_portable(&mut h, BLOCK, 0, 0, 0));
}

#[bench]
fn bench_blake2b_portable_compress4(b: &mut Bencher) {
    b.bytes = BLOCK.len() as u64 * 4;
    let mut h1 = [0; 8];
    let mut h2 = [0; 8];
    let mut h3 = [0; 8];
    let mut h4 = [0; 8];
    b.iter(|| {
        benchmarks::compress4_portable(
            &mut h1, &mut h2, &mut h3, &mut h4, &BLOCK, BLOCK, BLOCK, BLOCK, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
        );
    });
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
