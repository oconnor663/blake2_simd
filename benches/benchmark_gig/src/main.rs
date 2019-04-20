//! To squeeze the absolute most out of this benchmark, with optimizations specific to the current
//! machine, try this:
//!
//!     RUSTFLAGS="-C target-cpu=native -C target-feature=-avx2" cargo +nightly run --release --bin benchmark_gig
//!
//! Note that we're *disabling* AVX2 with target-feature. I find that when it's enabled, the
//! portable implementation ends up *much* slower. Our AVX2 compress function will compile with
//! AVX2 regardless, because of its local annotations. Also the nightly compiler seems to produce
//! faster code than stable.

extern crate blake2b_simd;

use rand::RngCore;
use std::cmp::Ordering::{Greater, Less};
use std::time::Instant;

const INPUT_LEN: usize = 1_000_000_000;
const RUNS: usize = 10;

type HashFn = fn(input: &[u8]);

fn print(nanos: u128, message: &str) {
    // (bits / ns) = (GB / sec)
    let rate = INPUT_LEN as f64 / nanos as f64;
    let secs = nanos as f64 / 1e9;
    println!("{:.06}s ({:.06} GB/s) {}", secs, rate, message);
}

fn run(input: &[u8], hash_fn: HashFn, name: &str, rates: &mut Vec<(String, f64)>) {
    println!("{}", name);
    let mut total = 0;
    for i in 0..RUNS {
        let before = Instant::now();
        hash_fn(input);
        let after = Instant::now();
        let diff = (after - before).as_nanos();
        if i == 0 {
            // Skip the first run, because things like OpenSSL can do
            // initialization.
            print(diff, "(ignored)");
        } else {
            print(diff, "");
            total += diff;
        }
    }
    // Note that GB/s is the same unit as bytes/ns.
    let rate = input.len() as f64 / (total as f64 / (RUNS - 1) as f64);
    rates.push((name.to_string(), rate));
}

fn hash_portable(input: &[u8]) {
    let mut state = blake2b_simd::State::new();
    blake2b_simd::benchmarks::force_portable(&mut state);
    state.update(input);
    state.finalize();
}

fn hash_avx2(input: &[u8]) {
    blake2b_simd::blake2b(input);
}

fn hash_many(input: &[u8]) {
    let params = blake2b_simd::Params::new();
    let quarter = input.len() / 4;
    let input0 = &input[0 * quarter..][..quarter];
    let input1 = &input[1 * quarter..][..quarter];
    let input2 = &input[2 * quarter..][..quarter];
    let input3 = &input[3 * quarter..][..quarter];
    let mut jobs = [
        blake2b_simd::many::HashManyJob::new(&params, input0),
        blake2b_simd::many::HashManyJob::new(&params, input1),
        blake2b_simd::many::HashManyJob::new(&params, input2),
        blake2b_simd::many::HashManyJob::new(&params, input3),
    ];
    blake2b_simd::many::hash_many(&mut jobs);
}

fn hash_blake2bp(input: &[u8]) {
    blake2b_simd::blake2bp::blake2bp(input);
}

fn libsodium(input: &[u8]) {
    let mut out = [0; 64];
    unsafe {
        let init_ret = libsodium_ffi::sodium_init();
        assert!(init_ret != -1);
    }
    unsafe {
        libsodium_ffi::crypto_generichash(
            out.as_mut_ptr(),
            out.len(),
            input.as_ptr(),
            input.len() as u64,
            std::ptr::null(),
            0,
        );
    };
}

fn openssl_sha1(input: &[u8]) {
    openssl::hash::hash(openssl::hash::MessageDigest::sha1(), &input).unwrap();
}

fn openssl_sha512(input: &[u8]) {
    openssl::hash::hash(openssl::hash::MessageDigest::sha512(), &input).unwrap();
}

fn main() {
    let mut input = vec![0; INPUT_LEN];
    rand::thread_rng().fill_bytes(&mut input);
    let mut rates = Vec::new();

    // Benchmark the portable implementation.
    run(&input, hash_portable, "blake2b_simd portable", &mut rates);

    // Benchmark the AVX2 implementation.
    run(&input, hash_avx2, "blake2b_simd AVX2", &mut rates);

    // Benchmark the 4-way AVX2 implementation.
    run(&input, hash_many, "blake2b_simd hash_many", &mut rates);

    // Benchmark BLAKE2bp.
    run(&input, hash_blake2bp, "blake2b_simd BLAKE2bp", &mut rates);

    // Benchmark BLAKE2bp.
    run(&input, libsodium, "libsodium BLAKE2b", &mut rates);

    // Benchmark BLAKE2bp.
    run(&input, openssl_sha1, "OpenSSL SHA-1", &mut rates);

    // Benchmark BLAKE2bp.
    run(&input, openssl_sha512, "OpenSSL SHA-512", &mut rates);

    // Sort by highest rate first.
    rates.sort_by(|a, b| if a.1 > b.1 { Less } else { Greater });

    let max_name_len = rates.iter().map(|pair| pair.0.len()).max().unwrap();
    println!();
    for (name, rate) in rates {
        println!("{0:1$} {2:.3}", name, max_name_len, rate);
    }
}
