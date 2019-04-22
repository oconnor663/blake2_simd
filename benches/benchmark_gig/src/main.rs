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
use std::cmp::Ordering;
use std::time::Instant;

const INPUT_LEN: usize = 1_000_000_000;
const RUNS: usize = 3;
const WORKERS: usize = 5;

type HashFn = fn(input: &[u8]);

fn float_ordering(a: &f32, b: &f32) -> Ordering {
    if a < b {
        Ordering::Less
    } else {
        Ordering::Greater
    }
}

fn run(input: &[u8], hash_fn: HashFn) -> f32 {
    // One throwaway run.
    hash_fn(input);

    let mut rates = Vec::new();
    for _ in 0..RUNS {
        let before = Instant::now();
        hash_fn(input);
        let after = Instant::now();
        let diff = (after - before).as_nanos();
        // Note that GB/s is the same unit as bytes/ns.
        let rate = input.len() as f32 / diff as f32;
        rates.push(rate);
    }
    rates.sort_unstable_by(float_ordering);
    // dbg!(&rates);
    rates[rates.len() / 2]
}

#[inline(never)]
fn hash_blake2bp(input: &[u8]) {
    blake2b_simd::benchmarks::simple_blake2bp(input);
}

fn get_offset() -> usize {
    std::env::args()
        .nth(1)
        .expect("need one arg")
        .parse()
        .expect("arg parsing")
}

fn worker() {
    let offset = get_offset();
    let mut input_vec = vec![0; INPUT_LEN + offset + 1];
    rand::thread_rng().fill_bytes(&mut input_vec);
    let input = &input_vec[offset..];

    // Benchmark BLAKE2bp.
    let rate = run(&input, hash_blake2bp);

    println!("{:.10}", rate);
}

fn main() {
    if std::env::var("BENCH_WORKER").is_ok() {
        worker();
        return;
    }
    std::env::set_var("BENCH_WORKER", "1");
    let offset = get_offset();
    let mut rates = Vec::new();
    for _ in 0..WORKERS {
        let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
        cmd.arg(offset.to_string());
        cmd.stdout(std::process::Stdio::piped());
        let output = cmd.spawn().unwrap().wait_with_output().unwrap();
        let rate: f32 = std::str::from_utf8(&output.stdout)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        rates.push(rate);
    }
    rates.sort_unstable_by(float_ordering);
    // dbg!(&rates);
    println!("{:.10}", rates[rates.len() / 2]);
}
