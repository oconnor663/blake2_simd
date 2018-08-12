extern crate blake2b_simd;

use std::time::{Duration, Instant};

const NS_PER_SEC: u64 = 1_000_000_000;
const INPUT_LEN: usize = 1_000_000_000;
const RUNS: usize = 10;

#[inline(never)]
fn hash(input: &[u8], force_portable: bool) -> blake2b_simd::Hash {
    let mut state = blake2b_simd::State::new();
    if force_portable {
        blake2b_simd::benchmarks::force_portable(&mut state);
    }
    state.update(&input);
    state.finalize()
}

fn print(d: Duration, message: &str) {
    let nanos: u64 = NS_PER_SEC * d.as_secs() + d.subsec_nanos() as u64;
    let secs: f64 = nanos as f64 / NS_PER_SEC as f64;
    // (ns / sec) / (ns / GB) = GB / sec
    let rate: f64 = NS_PER_SEC as f64 / nanos as f64;
    println!("{:.06}s ({:.06} GB/s) {}", secs, rate, message);
}

fn run(input: &[u8], force_portable: bool) {
    let mut fastest = Duration::from_secs(u64::max_value());
    let mut total = Duration::from_secs(0);
    for i in 0..RUNS {
        let before = Instant::now();
        hash(input, force_portable);
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

fn main() {
    let input = vec![0; INPUT_LEN];

    // First benchmark with the portable implementation.
    println!("run #1, the portable implementation");
    run(&input, true);

    // Then benchmark with the AVX2 implementation.
    println!("run #1, the AVX2 implementation (presumably)");
    run(&input, false);
}
