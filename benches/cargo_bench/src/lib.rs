#![feature(test)]

extern crate blake2b_simd;
#[cfg(feature = "libsodium-ffi")]
extern crate libsodium_ffi;
#[cfg(feature = "openssl")]
extern crate openssl;
extern crate test;

use blake2b_simd::*;
use test::Bencher;

const MB: usize = 1 << 20;

fn make_input(b: &mut Bencher, len: usize) -> Vec<u8> {
    // Fill the vec with something other than zero, to avoid optimizations
    // using zeroed memory pages.
    b.bytes += len as u64;
    vec![0b01010101; len]
}

#[bench]
fn bench_blake2b_avx2_one_block(b: &mut Bencher) {
    let input = make_input(b, BLOCKBYTES);
    b.iter(|| blake2b(&input));
}

#[bench]
fn bench_blake2b_avx2_one_mb(b: &mut Bencher) {
    let input = make_input(b, MB);
    b.iter(|| blake2b(&input));
}

#[bench]
fn bench_blake2b_portable_one_block(b: &mut Bencher) {
    let input = make_input(b, BLOCKBYTES);
    b.iter(|| {
        let mut state = State::new();
        benchmarks::force_portable(&mut state);
        state.update(&input);
        state.finalize()
    });
}

#[bench]
fn bench_blake2b_portable_one_mb(b: &mut Bencher) {
    let input = make_input(b, MB);
    b.iter(|| {
        let mut state = State::new();
        benchmarks::force_portable(&mut state);
        state.update(&input);
        state.finalize()
    });
}

#[bench]
fn bench_blake2bp_one_mb(b: &mut Bencher) {
    let input = make_input(b, MB);
    b.iter(|| blake2bp::blake2bp(&input));
}

#[cfg(feature = "libsodium-ffi")]
#[bench]
fn bench_libsodium_one_mb(b: &mut Bencher) {
    let input = make_input(b, MB);
    let mut out = [0; 64];
    unsafe {
        let init_ret = libsodium_ffi::sodium_init();
        assert_eq!(0, init_ret);
    }
    b.iter(|| unsafe {
        libsodium_ffi::crypto_generichash(
            out.as_mut_ptr(),
            out.len(),
            input.as_ptr(),
            input.len() as u64,
            std::ptr::null(),
            0,
        );
    });
}

#[cfg(feature = "openssl")]
#[bench]
fn bench_openssl_md5_one_mb(b: &mut Bencher) {
    let input = make_input(b, MB);
    b.iter(|| openssl::hash::hash(openssl::hash::MessageDigest::md5(), &input));
}

#[cfg(feature = "openssl")]
#[bench]
fn bench_openssl_sha1_one_mb(b: &mut Bencher) {
    let input = make_input(b, MB);
    b.iter(|| openssl::hash::hash(openssl::hash::MessageDigest::sha1(), &input));
}

#[cfg(feature = "openssl")]
#[bench]
fn bench_openssl_sha512_one_mb(b: &mut Bencher) {
    let input = make_input(b, MB);
    b.iter(|| openssl::hash::hash(openssl::hash::MessageDigest::sha512(), &input));
}
#[bench]
fn bench_blake2b_update_many_4x_block(b: &mut Bencher) {
    let mut states = [State::new(), State::new(), State::new(), State::new()];
    let inputs = [
        make_input(b, BLOCKBYTES),
        make_input(b, BLOCKBYTES),
        make_input(b, BLOCKBYTES),
        make_input(b, BLOCKBYTES),
    ];
    b.iter(|| {
        many::update_many(states.iter_mut().zip(inputs.iter().map(|v| v.as_slice())));
    });
    test::black_box(&states);
}

#[bench]
fn bench_blake2b_update_many_4x_4096(b: &mut Bencher) {
    let mut states = [State::new(), State::new(), State::new(), State::new()];
    let inputs = [
        make_input(b, 4096),
        make_input(b, 4096),
        make_input(b, 4096),
        make_input(b, 4096),
    ];
    b.iter(|| {
        many::update_many(states.iter_mut().zip(inputs.iter().map(|v| v.as_slice())));
    });
    test::black_box(&states);
}

#[bench]
fn bench_blake2b_update_many_4x_1mb(b: &mut Bencher) {
    let mut states = [State::new(), State::new(), State::new(), State::new()];
    let inputs = [
        make_input(b, MB),
        make_input(b, MB),
        make_input(b, MB),
        make_input(b, MB),
    ];
    b.iter(|| {
        many::update_many(states.iter_mut().zip(inputs.iter().map(|v| v.as_slice())));
    });
    test::black_box(&states);
}
