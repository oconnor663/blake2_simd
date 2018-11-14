#![feature(test)]

extern crate blake2b_simd;
#[cfg(feature = "libsodium-ffi")]
extern crate libsodium_ffi;
#[cfg(feature = "openssl")]
extern crate openssl;
extern crate test;

use blake2b_simd::*;
use std::mem;
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
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_blake2b_avx2_compress4_transposed(b: &mut Bencher) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    b.bytes = BLOCK.len() as u64 * 4;
    unsafe {
        let mut h_vecs = mem::zeroed();
        let msg0 = [1; BLOCKBYTES];
        let msg1 = [2; BLOCKBYTES];
        let msg2 = [3; BLOCKBYTES];
        let msg3 = [4; BLOCKBYTES];
        let count_low = mem::zeroed();
        let count_high = mem::zeroed();
        let lastblock = mem::zeroed();
        let lastnode = mem::zeroed();
        b.iter(|| {
            benchmarks::compress4_transposed_avx2(
                &mut h_vecs,
                &msg0,
                &msg1,
                &msg2,
                &msg3,
                count_low,
                count_high,
                lastblock,
                lastnode,
            );
            test::black_box(&mut h_vecs);
        });
    }
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
