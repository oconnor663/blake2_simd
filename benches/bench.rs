#![feature(test)]

extern crate blake2b_simd;
extern crate test;

use test::Bencher;

#[bench]
fn blake2b_100bytes(b: &mut Bencher) {
    b.iter(|| blake2b_simd::blake2b(&[0; 100]));
}

#[bench]
fn blake2b_1kb(b: &mut Bencher) {
    b.iter(|| blake2b_simd::blake2b(&[0; 1_000]));
}

#[bench]
fn blake2b_1mb(b: &mut Bencher) {
    b.iter(|| blake2b_simd::blake2b(&[0; 1_000_000]));
}
