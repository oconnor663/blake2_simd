//! There is some random noise associated with benchmarking which is "sticky"
//! to the process running the benchmark, for example ASLR can affect memory
//! locality. To account for this and make benchmarks more stable, we run
//! benchmarks in multiple workers processes and average their results.
//!
//! References:
//! - https://blog.phusion.nl/2017/07/13/understanding-your-benchmarks-and-easy-tips-for-fixing-them/
//! - https://lwn.net/Articles/725114/

extern crate blake2b_simd;

use rand::RngCore;
use std::cmp::Ordering::{Greater, Less};
use std::env;
use std::io::prelude::*;
use std::process;
use std::ptr;
use std::str;
use std::time::Instant;

// The amount of input to give each worker is determined by the amount of time
// we want each benchmark to run in total. Defaults to 1 sec, but this is
// overridable with the MS_PER_BENCH env var.
const DEFAULT_MS_PER_BENCH: u128 = 1000;

// Copy the strategy used by the Python `perf` tool: 20 workers with 3 runs
// each (plus a warmup run).
const WORKERS: usize = 20;
const RUNS_PER_WORKER: usize = 3;

const CALIBRATION_INPUT_LEN: usize = 1 << 20;

static ALGOS: &[(&str, HashFn)] = &[
    ("blake2b_simd portable", hash_portable),
    ("blake2b_simd AVX2", hash_avx2),
    ("blake2b_simd hash_many", hash_many),
    ("blake2b_simd BLAKE2bp", hash_blake2bp),
    ("libsodium BLAKE2b", libsodium),
    ("OpenSSL SHA-1", openssl_sha1),
    ("OpenSSL SHA-512", openssl_sha512),
];

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
            ptr::null(),
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

type HashFn = fn(input: &[u8]);

fn make_input(len: usize) -> Vec<u8> {
    let mut input = vec![0; len];
    rand::thread_rng().fill_bytes(&mut input);
    input
}

fn time_ns<F: FnOnce()>(f: F) -> u128 {
    let before = Instant::now();
    f();
    let after = Instant::now();
    (after - before).as_nanos()
}

fn get_hash_fn(name: &str) -> HashFn {
    let mut hash_fn = None;
    for &(algo_name, f) in ALGOS {
        if name == algo_name {
            hash_fn = Some(f);
            break;
        }
    }
    hash_fn.expect(&format!("no such algo: {}", name))
}

// Note that bytes/nanosecond and GB/second are the same unit.
fn rate_f32(ns: u128, input_len: usize) -> f32 {
    input_len as f32 / ns as f32
}

fn worker(algo: &str) {
    let hash_fn = get_hash_fn(algo);
    let input_len: usize = env::var("WORKER_LEN").unwrap().parse().unwrap();
    let mut input = vec![0; input_len];
    rand::thread_rng().fill_bytes(&mut input);

    // Do a dummy run to warm up.
    hash_fn(&input);

    let mut total_ns = 0;
    for _ in 0..RUNS_PER_WORKER {
        let before = Instant::now();
        hash_fn(&input);
        let after = Instant::now();
        let ns = (after - before).as_nanos();
        // eprintln!("run throughput: {}", rate_f32(ns, input_len));
        total_ns += ns;
    }
    println!("{}", total_ns);
}

fn ns_per_run() -> u128 {
    let ms_per_bench: u128 = if let Some(secs) = env::var_os("MS_PER_BENCH") {
        let ms_str = secs.to_str().expect("unicode");
        if let Ok(ms) = ms_str.trim().parse() {
            ms
        } else {
            panic!("invalid int {:?}", ms_str);
        }
    } else {
        DEFAULT_MS_PER_BENCH
    };
    let ns_per_bench = 1_000_000 * ms_per_bench;
    let total_runs_with_warmup = WORKERS * (RUNS_PER_WORKER + 1);
    ns_per_bench / total_runs_with_warmup as u128
}

fn main() {
    if let Ok(name) = env::var("BENCH_ALGO") {
        worker(&name);
        return;
    }

    println!("Units are GB/s. Set bench time with MS_PER_BENCH (default one second).");
    let mut throughputs = Vec::new();
    for &(algo_name, _) in ALGOS {
        print!("{}: ", algo_name);
        std::io::stdout().flush().unwrap();

        // Test the speed of the hash function on a small input (1 MB), and use
        // that to figure out how input to give each worker. Note that it's
        // important to do this in the main process, because doing it
        // individually in each worker would give more input to slower workers.
        let test_input = make_input(CALIBRATION_INPUT_LEN);
        let hash_fn = get_hash_fn(algo_name);
        hash_fn(&test_input); // warm-up calibration run
        let test_ns = time_ns(|| hash_fn(&test_input));

        // Given the test time found above, compute the worker input length.
        let worker_len = (CALIBRATION_INPUT_LEN as u128 * ns_per_run() / test_ns) as usize;

        // Fire off all the workers in series and collect their reported times.
        let mut total_ns = 0;
        for _ in 0..WORKERS {
            env::set_var("BENCH_ALGO", algo_name);
            env::set_var("WORKER_LEN", worker_len.to_string());
            let mut cmd = process::Command::new(env::current_exe().unwrap());
            cmd.stdout(process::Stdio::piped());
            let child = cmd.spawn().unwrap();
            let output = child.wait_with_output().unwrap();
            let ns: u128 = str::from_utf8(&output.stdout)
                .unwrap()
                .trim()
                .parse()
                .unwrap();
            // eprintln!(
            //     "worker throughput: {}",
            //     rate_f32(ns, RUNS_PER_WORKER * worker_len)
            // );
            total_ns += ns;
        }
        let throughput = rate_f32(total_ns, WORKERS * RUNS_PER_WORKER * worker_len);
        // eprintln!("final throughput: {}", throughput);
        throughputs.push((throughput, algo_name));
        println!("{:.3}", throughput);
    }

    // Sort by the fastest rate.
    throughputs.sort_by(|t1, t2| if t1.0 > t2.0 { Less } else { Greater });

    let max_name_len = ALGOS.iter().map(|(name, _)| name.len()).max().unwrap();
    println!("\nIn order:");
    for &(throughput, name) in &throughputs {
        println!("{0:1$} {2:.3}", name, max_name_len, throughput);
    }
}
