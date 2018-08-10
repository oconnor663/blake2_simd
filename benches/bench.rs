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
fn blake2b_compress(b: &mut Bencher) {
    b.bytes = blake2b_simd::BLOCKBYTES as u64;
    let mut state = blake2b_simd::State::new();
    b.iter(|| state._bench_compress(&[0; blake2b_simd::BLOCKBYTES]));
}

#[bench]
fn blake2b_compress_simd(b: &mut Bencher) {
    b.bytes = blake2b_simd::BLOCKBYTES as u64;
    let mut state = blake2b_simd::State::new();
    b.iter(|| state._bench_compress_simd(&[0; blake2b_simd::BLOCKBYTES]));
}
