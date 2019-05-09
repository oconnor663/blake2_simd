//! There is some random noise associated with benchmarking which is "sticky"
//! to the process running the benchmark, for example ASLR can affect memory
//! locality. To account for this and make benchmarks more stable, we run
//! benchmarks in multiple workers processes and average their results.
//!
//! References:
//! - https://blog.phusion.nl/2017/07/13/understanding-your-benchmarks-and-easy-tips-for-fixing-them/
//! - https://lwn.net/Articles/725114/

extern crate blake2b_simd;

use rand::seq::SliceRandom;
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
    ("blake2b_simd many", hash_many),
    ("blake2b_simd BLAKE2bp", hash_blake2bp),
    ("blake2b_avx2_neves BLAKE2bp", hash_neves_blake2bp),
    ("libsodium BLAKE2b", libsodium),
    ("OpenSSL SHA-1", openssl_sha1),
    ("OpenSSL SHA-512", openssl_sha512),
];

fn hash_portable(input: &mut RandomInput) {
    let mut state = blake2b_simd::State::new();
    blake2b_simd::benchmarks::force_portable(&mut state);
    state.update(input.get());
    state.finalize();
}

fn hash_avx2(input: &mut RandomInput) {
    blake2b_simd::blake2b(input.get());
}

fn hash_many(input: &mut RandomInput) {
    let mut input_slices = arrayvec::ArrayVec::<[&[u8]; blake2b_simd::many::MAX_DEGREE]>::new();
    for _ in 0..blake2b_simd::many::degree() {
        input_slices.push(&[]);
    }
    input.get_n(&mut input_slices);

    let params = blake2b_simd::Params::new();
    let mut jobs = arrayvec::ArrayVec::<[_; blake2b_simd::many::MAX_DEGREE]>::new();
    for &input_slice in &input_slices {
        let job = blake2b_simd::many::HashManyJob::new(&params, input_slice);
        jobs.push(job);
    }

    blake2b_simd::many::hash_many(&mut jobs);
}

fn hash_blake2bp(input: &mut RandomInput) {
    blake2b_simd::blake2bp::blake2bp(input.get());
}

fn hash_neves_blake2bp(input: &mut RandomInput) {
    blake2_avx2_neves::blake2bp(input.get());
}

fn libsodium(input: &mut RandomInput) {
    let mut out = [0; 64];
    unsafe {
        let init_ret = libsodium_ffi::sodium_init();
        assert!(init_ret != -1);
    }
    let input_slice = input.get();
    unsafe {
        libsodium_ffi::crypto_generichash(
            out.as_mut_ptr(),
            out.len(),
            input_slice.as_ptr(),
            input_slice.len() as u64,
            ptr::null(),
            0,
        );
    };
}

fn openssl_sha1(input: &mut RandomInput) {
    openssl::hash::hash(openssl::hash::MessageDigest::sha1(), input.get()).unwrap();
}

fn openssl_sha512(input: &mut RandomInput) {
    openssl::hash::hash(openssl::hash::MessageDigest::sha512(), input.get()).unwrap();
}

type HashFn = fn(input: &mut RandomInput);

struct VecIter<T> {
    vec: Vec<T>,
    i: usize,
}

impl<T: Clone> VecIter<T> {
    fn next(&mut self) -> T {
        let item = self.vec[self.i].clone();
        self.i += 1;
        if self.i >= self.vec.len() {
            self.i = 0;
        }
        item
    }
}

// This struct randomizes two things:
// 1. The actual bytes of input.
// 2. The page offset the input starts at.
struct RandomInput {
    buf: Vec<u8>,
    len: usize,
    offsets: VecIter<usize>,
}

impl RandomInput {
    fn new(len: usize) -> Self {
        let page_size: usize = page_size::get();
        let mut buf = vec![0u8; len + page_size];
        rand::thread_rng().fill_bytes(&mut buf);
        let mut offsets: Vec<usize> = (0..page_size).collect();
        offsets.shuffle(&mut rand::thread_rng());
        Self {
            buf,
            len,
            offsets: VecIter { vec: offsets, i: 0 },
        }
    }

    fn get(&mut self) -> &[u8] {
        &self.buf[self.offsets.next()..][..self.len]
    }

    // for hash_many
    #[inline]
    fn get_n<'a>(&'a mut self, out: &mut [&'a [u8]]) {
        let n = out.len();
        let slice_len = self.len / n;
        let mut pos = 0;
        for slice in out {
            pos += self.offsets.next() / n;
            *slice = &self.buf[pos..][..slice_len];
            pos += slice_len;
        }
    }
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

fn ns_per_worker() -> u128 {
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
    ns_per_bench / WORKERS as u128
}

fn worker(algo: &str) {
    let hash_fn = get_hash_fn(algo);
    let input_len: usize = env::var("WORKER_LEN").unwrap().parse().unwrap();
    let mut input = RandomInput::new(input_len);

    // Do a dummy run to warm up.
    hash_fn(&mut input);

    let mut total_ns = 0;
    for _ in 0..RUNS_PER_WORKER {
        let ns = time_ns(|| hash_fn(&mut input));
        // eprintln!("run throughput: {}", rate_f32(ns, input_len));
        total_ns += ns;
    }
    println!("{}", total_ns);
}

fn run_algo(algo_name: &str) -> f32 {
    // Test the speed of the hash function on a small input (1 MB), and use
    // that to figure out how input to give each worker. Note that it's
    // important to do this in the main process, because doing it
    // individually in each worker would give more input to slower workers.
    let hash_fn = get_hash_fn(algo_name);
    let test_ns = time_ns(|| {
        let mut test_input = RandomInput::new(CALIBRATION_INPUT_LEN);
        hash_fn(&mut test_input); // dummy warm up run
        for _ in 0..RUNS_PER_WORKER {
            hash_fn(&mut test_input);
        }
    });

    // Given the test time found above, compute the worker input length.
    let worker_len = (CALIBRATION_INPUT_LEN as u128 * ns_per_worker() / test_ns) as usize;

    // Fire off all the workers in series and collect their reported times.
    let mut times = Vec::new();
    let mut _total_ns = 0;
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
        times.push(ns);
        _total_ns += ns;
    }
    times.sort();
    let median_time = times[times.len() / 2];
    let throughput = rate_f32(median_time, RUNS_PER_WORKER * worker_len);

    // an alternative that uses the average instead of the minimum
    // let throughput = rate_f32(total_ns, WORKERS * RUNS_PER_WORKER * worker_len);

    // eprintln!("final throughput: {}", throughput);
    throughput
}

fn main() {
    if let Ok(name) = env::var("BENCH_ALGO") {
        worker(&name);
        return;
    }

    // If a positional argument is given, it should be a substring of exactly
    // one algorithm name. In that case, run just that algorithm, and print the
    // result with no other formatting.
    if let Some(arg) = std::env::args().nth(1) {
        let matches: Vec<&str> = ALGOS
            .iter()
            .map(|&(name, _)| name)
            .filter(|name| name.contains(arg.as_str()))
            .collect();
        if matches.is_empty() {
            panic!("no match");
        }
        if matches.len() > 1 {
            panic!("too many matches {:?}", &matches);
        }
        let algo_name = matches[0];
        let throughput = run_algo(algo_name);
        println!("{:.3}", throughput);
        return;
    }

    // Otherwise run all the benchmarks and print them sorted at the end.
    println!("Units are GB/s. Set bench time with MS_PER_BENCH (default one second).");
    let mut throughputs = Vec::new();
    for &(algo_name, _) in ALGOS {
        print!("{}: ", algo_name);
        std::io::stdout().flush().unwrap();

        let throughput = run_algo(algo_name);
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
