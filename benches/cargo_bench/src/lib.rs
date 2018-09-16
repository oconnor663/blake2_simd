#![feature(test)]

extern crate blake2b_simd;
#[cfg(feature = "libsodium-ffi")]
extern crate libsodium_ffi;
#[cfg(feature = "openssl")]
extern crate openssl;
extern crate rayon;
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
    // BLAKE2bp requires exactly 4 threads, and this benchmark performs best
    // when we set that number explicitly. The b2sum binary also sets it.
    rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build_global()
        .unwrap();
    b.bytes = MB.len() as u64;
    b.iter(|| blake2bp(MB, OUTBYTES));
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
