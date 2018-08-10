#![feature(test)]

extern crate blake2b_simd;
extern crate test;

use test::Bencher;

#[bench]
fn blake2b_100bytes(b: &mut Bencher) {
    b.bytes = 100;
    b.iter(|| blake2b_simd::blake2b(&[0; 100]));
}

#[bench]
fn blake2b_1kb(b: &mut Bencher) {
    b.bytes = 1000;
    b.iter(|| blake2b_simd::blake2b(&[0; 1_000]));
}

#[bench]
fn blake2b_1mb(b: &mut Bencher) {
    b.bytes = 1_000_000;
    b.iter(|| blake2b_simd::blake2b(&[0; 1_000_000]));
}

#[bench]
fn blake2b_compress_portable(b: &mut Bencher) {
    let input = &[0; blake2b_simd::BLOCKBYTES];
    b.bytes = input.len() as u64;
    let mut h = [0; 8];
    b.iter(|| blake2b_simd::portable::compress(&mut h, input, 0, 0));
}

#[bench]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn blake2b_compress_avx2(b: &mut Bencher) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let input = &[0; blake2b_simd::BLOCKBYTES];
    b.bytes = input.len() as u64;
    let mut h = [0; 8];
    b.iter(|| unsafe { blake2b_simd::avx2::compress(&mut h, input, 0, 0) });
}
