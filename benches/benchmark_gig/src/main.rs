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
use std::time::Instant;

const INPUT_LEN: usize = 1_000_000_000;
const RUNS: usize = 10;

type HashFn = fn(input: &[u8]);

fn run(input: &[u8], hash_fn: HashFn) -> f64 {
    let mut total = 0;
    for i in 0..RUNS {
        let before = Instant::now();
        hash_fn(input);
        let after = Instant::now();
        let diff = (after - before).as_nanos();
        if i == 0 {
            // Skip the first run, because things like OpenSSL can do
            // initialization.
        } else {
            // println!("t {:.3}", INPUT_LEN as f64 / diff as f64);
            total += diff;
        }
    }
    // Note that GB/s is the same unit as bytes/ns.
    let rate = input.len() as f64 / (total as f64 / (RUNS - 1) as f64);
    rate
}

fn hash_blake2bp(input: &[u8]) {
    blake2b_simd::benchmarks::simple_blake2bp(input);
}

fn main() {
    let offset: usize = std::env::args().nth(1).unwrap().parse().unwrap();
    let mut input = vec![0; INPUT_LEN + offset + 1];

    rand::thread_rng().fill_bytes(&mut input);
    let input = &input[offset..];

    // Benchmark BLAKE2bp.
    let rate = run(&input, hash_blake2bp);

    println!("{:.3}", rate);
}
