extern crate blake2b_simd;

use std::time::{Duration, Instant};

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

fn print(d: Duration, name: &str) {
    println!("{}.{:06}s {}", d.as_secs(), d.subsec_micros(), name);
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
            print(diff, "hash (ignored)");
        } else {
            print(diff, "hash");
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
