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

use std::time::{Duration, Instant};

const NS_PER_SEC: u64 = 1_000_000_000;
const INPUT_LEN: usize = 1_000_000_000;
const RUNS: usize = 10;

type HashFn = fn(input: &[u8]);

fn print(d: Duration, message: &str) {
    let nanos: u64 = NS_PER_SEC * d.as_secs() + d.subsec_nanos() as u64;
    let secs: f64 = nanos as f64 / NS_PER_SEC as f64;
    // (bits / ns) = (GB / sec)
    let rate: f64 = INPUT_LEN as f64 / nanos as f64;
    println!("{:.06}s ({:.06} GB/s) {}", secs, rate, message);
}

fn run(input: &[u8], hash_fn: HashFn) {
    let mut fastest = Duration::from_secs(u64::max_value());
    let mut total = Duration::from_secs(0);
    for i in 0..RUNS {
        let before = Instant::now();
        hash_fn(input);
        let after = Instant::now();
        let diff = after - before;
        if i == 0 {
            // Skip the first run, because it pays fixed costs like zeroing memory.
            print(diff, "(ignored)");
        } else {
            print(diff, "");
            total += diff;
            if diff < fastest {
                fastest = diff;
            }
        }
    }
    let average = total / (RUNS - 1) as u32;
    println!("-----");
    print(average, "average");
    print(fastest, "fastest");
    println!("-----");
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

fn hash_update4(input: &[u8]) {
    let mut state0 = blake2b_simd::State::new();
    let mut state1 = blake2b_simd::State::new();
    let mut state2 = blake2b_simd::State::new();
    let mut state3 = blake2b_simd::State::new();
    let quarter = input.len() / 4;
    let input0 = &input[0 * quarter..][..quarter];
    let input1 = &input[1 * quarter..][..quarter];
    let input2 = &input[2 * quarter..][..quarter];
    let input3 = &input[3 * quarter..][..quarter];
    blake2b_simd::update4(
        &mut state0,
        &mut state1,
        &mut state2,
        &mut state3,
        input0,
        input1,
        input2,
        input3,
    );
    blake2b_simd::finalize4(&mut state0, &mut state1, &mut state2, &mut state3);
}

fn hash_blake2bp(input: &[u8]) {
    blake2b_simd::blake2bp::blake2bp(input);
}

fn main() {
    let input = vec![0; INPUT_LEN];

    // Benchmark the portable implementation.
    println!("run #1, the portable implementation");
    run(&input, hash_portable);

    // Benchmark the AVX2 implementation.
    println!("run #2, the AVX2 implementation");
    run(&input, hash_avx2);

    // Benchmark the 4-way AVX2 implementation.
    println!("run #3, the 4-way AVX2 implementation");
    run(&input, hash_update4);

    // Benchmark BLAKE2bp.
    println!("run #4, BLAKE2bp");
    run(&input, hash_blake2bp);
}
