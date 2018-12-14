#![feature(test)]

extern crate amd64_timer;
extern crate blake2b_simd;
extern crate openssl;
extern crate test;

const TOTAL_BYTES_PER_TYPE: usize = 1 << 30; // 1 gigabyte

fn blake2b_compression_avx2() -> (u64, usize) {
    let avx2 = if let Some(avx2) = blake2b_simd::guts::Implementation::avx2_if_supported() {
        avx2
    } else {
        panic!("No AVX2 support.");
    };
    const SIZE: usize = 128;
    let iterations = TOTAL_BYTES_PER_TYPE / SIZE;
    let mut total_ticks = 0;
    let input = &[1; 128];
    let mut h = [1; 8];
    for _ in 0..iterations {
        let start = amd64_timer::ticks_modern();
        avx2.compress(&mut h, input, 0, 0, 0);
        let end = amd64_timer::ticks_modern();
        total_ticks += end - start;
    }
    (total_ticks, iterations * SIZE)
}

fn blake2b_compression_portable() -> (u64, usize) {
    let portable = blake2b_simd::guts::Implementation::portable();
    const SIZE: usize = 128;
    let iterations = TOTAL_BYTES_PER_TYPE / SIZE;
    let mut total_ticks = 0;
    let input = &[1; 128];
    let mut h = [1; 8];
    for _ in 0..iterations {
        let start = amd64_timer::ticks_modern();
        portable.compress(&mut h, input, 0, 0, 0);
        let end = amd64_timer::ticks_modern();
        total_ticks += end - start;
    }
    (total_ticks, iterations * SIZE)
}

fn blake2b_compression_4x() -> (u64, usize) {
    let avx2 = if let Some(avx2) = blake2b_simd::guts::Implementation::avx2_if_supported() {
        avx2
    } else {
        panic!("No AVX2 support.");
    };
    const SIZE: usize = 4 * 128;
    let iterations = TOTAL_BYTES_PER_TYPE / SIZE;
    let mut total_ticks = 0;
    let msg0 = &[1; 128];
    let msg1 = &[1; 128];
    let msg2 = &[1; 128];
    let msg3 = &[1; 128];
    let mut state = [blake2b_simd::guts::u64x4([1; 4]); 8];
    let count_low = blake2b_simd::guts::u64x4([1; 4]);
    let count_high = blake2b_simd::guts::u64x4([1; 4]);
    let lastblock = blake2b_simd::guts::u64x4([1; 4]);
    let lastnode = blake2b_simd::guts::u64x4([1; 4]);
    for _ in 0..iterations {
        let start = amd64_timer::ticks_modern();
        avx2.compress4(
            &mut state,
            msg0,
            msg1,
            msg2,
            msg3,
            &count_low,
            &count_high,
            &lastblock,
            &lastnode,
        );
        let end = amd64_timer::ticks_modern();
        total_ticks += end - start;
    }
    (total_ticks, iterations * SIZE)
}

fn blake2b_one_mb() -> (u64, usize) {
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

fn blake2bp_one_mb() -> (u64, usize) {
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

fn blake2b_update4_one_mb() -> (u64, usize) {
    const SIZE: usize = 4_000_000;
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

fn sha1_openssl_one_mb() -> (u64, usize) {
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

fn sha512_openssl_one_mb() -> (u64, usize) {
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
        ("BLAKE2b portable compression", blake2b_compression_portable),
        ("BLAKE2b AVX2 compression", blake2b_compression_avx2),
        ("BLAKE2b 4-way AVX2 compression", blake2b_compression_4x),
        ("BLAKE2b 1 MB", blake2b_one_mb),
        ("BLAKE2bp 1 MB", blake2bp_one_mb),
        ("BLAKE2b update4 1 MB", blake2b_update4_one_mb),
        ("SHA1 OpenSSL 1 MB", sha1_openssl_one_mb),
        ("SHA512 OpenSSL 1 MB", sha512_openssl_one_mb),
    ];

    for &(name, f) in cases.iter() {
        // Warmup loop.
        f();
        // Loop for real.
        let (total_cycles, total_bytes) = f();
        println!(
            "{0:1$}  {2:.3}",
            name,
            cases.iter().map(|case| case.0.len()).max().unwrap(),
            total_cycles as f64 / total_bytes as f64
        );
    }
}
