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
    let input = &[0; 128];
    let mut h = [0; 8];
    for _ in 0..iterations {
        let start = amd64_timer::ticks_modern();
        unsafe {
            blake2b_simd::benchmarks::compress_avx2(&mut h, input, 0, 0, 0);
        }
        let end = amd64_timer::ticks_modern();
        total_ticks += end - start;
    }
    (total_ticks, iterations * SIZE)
}

fn compression_fn_4x() -> (u64, usize) {
    const SIZE: usize = 4 * 128;
    let iterations = TOTAL_BYTES_PER_TYPE / SIZE;
    let mut total_ticks = 0;
    let msg0 = &[0; 128];
    let msg1 = &[0; 128];
    let msg2 = &[0; 128];
    let msg3 = &[0; 128];
    let h0 = &mut [0; 8];
    let h1 = &mut [0; 8];
    let h2 = &mut [0; 8];
    let h3 = &mut [0; 8];
    for _ in 0..iterations {
        let start = amd64_timer::ticks_modern();
        unsafe {
            blake2b_simd::benchmarks::compress_4x_avx2(
                h0, h1, h2, h3, msg0, msg1, msg2, msg3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            );
        }
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

fn hash_one_mb_update4() -> (u64, usize) {
    const SIZE: usize = 1_000_000;
    let iterations = TOTAL_BYTES_PER_TYPE / SIZE;
    let mut total_ticks = 0;
    for _ in 0..iterations {
        let start = amd64_timer::ticks_modern();
        let mut state0 = blake2b_simd::State::new();
        let mut state1 = blake2b_simd::State::new();
        let mut state2 = blake2b_simd::State::new();
        let mut state3 = blake2b_simd::State::new();
        blake2b_simd::update4(
            &mut state0,
            &mut state1,
            &mut state2,
            &mut state3,
            &[0; SIZE / 4],
            &[0; SIZE / 4],
            &[0; SIZE / 4],
            &[0; SIZE / 4],
        );
        test::black_box(&blake2b_simd::finalize4(
            &mut state0,
            &mut state1,
            &mut state2,
            &mut state3,
        ));
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
        ("compress4x", compression_fn_4x),
        ("one block", hash_one_block),
        ("one mb", hash_one_mb),
        ("one mb chunks", hash_one_mb_in_chunks),
        ("one mb blake2bp", hash_one_mb_blake2bp),
        ("one mb update4", hash_one_mb_update4),
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
