#![feature(test)]

extern crate amd64_timer;
extern crate blake2b_simd;
extern crate openssl;
extern crate test;

const TOTAL_BYTES_PER_TYPE: usize = 1 << 30; // 1 gigabyte

fn compression_fn() -> (u64, usize) {
    const SIZE: usize = 128;
    let iterations = TOTAL_BYTES_PER_TYPE / SIZE;
    let mut total_ticks = 0;
    for _ in 0..iterations {
        let start = amd64_timer::ticks_modern();
        let input = &[0; 128];
        let mut h = [0; 8];
        unsafe {
            blake2b_simd::benchmarks::compress_avx2(&mut h, input, 0, 0, 0);
        }
        test::black_box(&h);
        let end = amd64_timer::ticks_modern();
        total_ticks += end - start;
    }
    (total_ticks, iterations * SIZE)
}

fn hash_one_block() -> (u64, usize) {
    const SIZE: usize = 128;
    let iterations = TOTAL_BYTES_PER_TYPE / SIZE;
    let mut total_ticks = 0;
    for _ in 0..iterations {
        let start = amd64_timer::ticks_modern();
        test::black_box(&blake2b_simd::blake2b(&[0; SIZE]));
        let end = amd64_timer::ticks_modern();
        total_ticks += end - start;
    }
    (total_ticks, iterations * SIZE)
}

fn hash_one_mb() -> (u64, usize) {
    const SIZE: usize = 1_000_000;
    let iterations = TOTAL_BYTES_PER_TYPE / SIZE;
    let mut total_ticks = 0;
    for _ in 0..iterations {
        let start = amd64_timer::ticks_modern();
        test::black_box(&blake2b_simd::blake2b(&[0; SIZE]));
        let end = amd64_timer::ticks_modern();
        total_ticks += end - start;
    }
    (total_ticks, iterations * SIZE)
}

fn hash_one_mb_in_chunks() -> (u64, usize) {
    const SIZE: usize = 1 << 20;
    const CHUNK_SIZE: usize = 1024;
    let iterations = TOTAL_BYTES_PER_TYPE / SIZE;
    let mut total_ticks = 0;
    for _ in 0..iterations {
        let start = amd64_timer::ticks_modern();
        let mut state = blake2b_simd::State::new();
        for _ in 0..(SIZE / CHUNK_SIZE) {
            state.update(&[0; CHUNK_SIZE]);
        }
        test::black_box(&state.finalize());
        let end = amd64_timer::ticks_modern();
        total_ticks += end - start;
    }
    (total_ticks, iterations * SIZE)
}

fn hash_one_mb_blake2bp() -> (u64, usize) {
    const SIZE: usize = 1_000_000;
    let iterations = TOTAL_BYTES_PER_TYPE / SIZE;
    let mut total_ticks = 0;
    for _ in 0..iterations {
        let start = amd64_timer::ticks_modern();
        test::black_box(&blake2b_simd::blake2bp::blake2bp(&[0; SIZE]));
        let end = amd64_timer::ticks_modern();
        total_ticks += end - start;
    }
    (total_ticks, iterations * SIZE)
}

fn hash_one_mb_sha1() -> (u64, usize) {
    const SIZE: usize = 1_000_000;
    let iterations = TOTAL_BYTES_PER_TYPE / SIZE;
    let mut total_ticks = 0;
    for _ in 0..iterations {
        let start = amd64_timer::ticks_modern();
        test::black_box(&openssl::hash::hash(
            openssl::hash::MessageDigest::sha1(),
            &[0; SIZE],
        ));
        let end = amd64_timer::ticks_modern();
        total_ticks += end - start;
    }
    (total_ticks, iterations * SIZE)
}

fn hash_one_mb_sha512() -> (u64, usize) {
    const SIZE: usize = 1_000_000;
    let iterations = TOTAL_BYTES_PER_TYPE / SIZE;
    let mut total_ticks = 0;
    for _ in 0..iterations {
        let start = amd64_timer::ticks_modern();
        test::black_box(&openssl::hash::hash(
            openssl::hash::MessageDigest::sha512(),
            &[0; SIZE],
        ));
        let end = amd64_timer::ticks_modern();
        total_ticks += end - start;
    }
    (total_ticks, iterations * SIZE)
}

fn main() {
    assert!(is_x86_feature_detected!("avx2"));
    let cases: &[(&str, fn() -> (u64, usize))] = &[
        ("compress", compression_fn),
        ("one block", hash_one_block),
        ("one mb", hash_one_mb),
        ("one mb chunks", hash_one_mb_in_chunks),
        ("one mb blake2bp", hash_one_mb_blake2bp),
        ("one mb sha1", hash_one_mb_sha1),
        ("one mb sha512", hash_one_mb_sha512),
    ];

    for &(name, f) in cases.iter() {
        // Warmup loop.
        f();
        // Loop for real.
        let (total_cycles, total_bytes) = f();
        println!(
            "{:15} {:.3}",
            name,
            total_cycles as f64 / total_bytes as f64
        );
    }
}
