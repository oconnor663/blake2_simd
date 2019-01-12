#![feature(test)]

extern crate blake2b_simd;
#[cfg(feature = "libsodium-ffi")]
extern crate libsodium_ffi;
#[cfg(feature = "openssl")]
extern crate openssl;
extern crate test;

use blake2b_simd::*;
use test::Bencher;

const BLOCK: &[u8; BLOCKBYTES] = &[0; BLOCKBYTES];
const MB: usize = 1 << 20;

fn make_input(b: &mut Bencher, len: usize) -> Vec<u8> {
    // Fill the vec with something other than zero, to avoid optimizations
    // using zeroed memory pages.
    b.bytes += len as u64;
    vec![0b01010101; len]
}

#[bench]
fn bench_blake2b_avx2_one_block(b: &mut Bencher) {
    let input = make_input(b, BLOCKBYTES);
    b.iter(|| blake2b(&input));
}

#[bench]
fn bench_blake2b_avx2_one_mb(b: &mut Bencher) {
    let input = make_input(b, MB);
    b.iter(|| blake2b(&input));
}

#[bench]
fn bench_blake2b_portable_one_block(b: &mut Bencher) {
    let input = make_input(b, BLOCKBYTES);
    b.iter(|| {
        let mut state = State::new();
        benchmarks::force_portable(&mut state);
        state.update(&input);
        state.finalize()
    });
}

#[bench]
fn bench_blake2b_portable_one_mb(b: &mut Bencher) {
    let input = make_input(b, MB);
    b.iter(|| {
        let mut state = State::new();
        benchmarks::force_portable(&mut state);
        state.update(&input);
        state.finalize()
    });
}

#[bench]
fn bench_blake2bp_one_mb(b: &mut Bencher) {
    let input = make_input(b, MB);
    b.iter(|| blake2bp::blake2bp(&input));
}

#[bench]
fn bench_blake2b_update4_one_block(b: &mut Bencher) {
    let input0 = make_input(b, BLOCKBYTES);
    let input1 = make_input(b, BLOCKBYTES);
    let input2 = make_input(b, BLOCKBYTES);
    let input3 = make_input(b, BLOCKBYTES);
    b.iter(|| {
        let mut state0 = State::new();
        let mut state1 = State::new();
        let mut state2 = State::new();
        let mut state3 = State::new();
        update4(
            &mut state0,
            &mut state1,
            &mut state2,
            &mut state3,
            &input0,
            &input1,
            &input2,
            &input3,
        );
        finalize4(&mut state0, &mut state1, &mut state2, &mut state3)
    });
}

#[bench]
fn bench_blake2b_update4_one_mb(b: &mut Bencher) {
    let len = MB / 4;
    let input0 = make_input(b, len);
    let input1 = make_input(b, len);
    let input2 = make_input(b, len);
    let input3 = make_input(b, len);
    b.iter(|| {
        let mut state0 = State::new();
        let mut state1 = State::new();
        let mut state2 = State::new();
        let mut state3 = State::new();
        update4(
            &mut state0,
            &mut state1,
            &mut state2,
            &mut state3,
            &input0,
            &input1,
            &input2,
            &input3,
        );
        finalize4(&mut state0, &mut state1, &mut state2, &mut state3)
    });
}

#[cfg(feature = "libsodium-ffi")]
#[bench]
fn bench_libsodium_one_mb(b: &mut Bencher) {
    let input = make_input(b, MB);
    let mut out = [0; 64];
    unsafe {
        let init_ret = libsodium_ffi::sodium_init();
        assert_eq!(0, init_ret);
    }
    b.iter(|| unsafe {
        libsodium_ffi::crypto_generichash(
            out.as_mut_ptr(),
            out.len(),
            input.as_ptr(),
            input.len() as u64,
            std::ptr::null(),
            0,
        );
    });
}

#[cfg(feature = "openssl")]
#[bench]
fn bench_openssl_md5_one_mb(b: &mut Bencher) {
    let input = make_input(b, MB);
    b.iter(|| openssl::hash::hash(openssl::hash::MessageDigest::md5(), &input));
}

#[cfg(feature = "openssl")]
#[bench]
fn bench_openssl_sha1_one_mb(b: &mut Bencher) {
    let input = make_input(b, MB);
    b.iter(|| openssl::hash::hash(openssl::hash::MessageDigest::sha1(), &input));
}

#[cfg(feature = "openssl")]
#[bench]
fn bench_openssl_sha512_one_mb(b: &mut Bencher) {
    let input = make_input(b, MB);
    b.iter(|| openssl::hash::hash(openssl::hash::MessageDigest::sha512(), &input));
}

#[bench]
fn bench_guts_compress1_portable(b: &mut Bencher) {
    b.bytes = BLOCK.len() as u64;
    let portable = guts::Implementation::portable();
    let mut state = guts::u64x8([1; 8]);
    b.iter(|| {
        portable.compress(&mut state, BLOCK, 0, 0, 0);
    });
    test::black_box(&mut state);
}

#[bench]
fn bench_guts_compress1_avx2(b: &mut Bencher) {
    b.bytes = BLOCK.len() as u64;
    let avx2 = if let Some(imp) = guts::Implementation::avx2_if_supported() {
        imp
    } else {
        return;
    };
    let mut state = guts::u64x8([1; 8]);
    b.iter(|| {
        avx2.compress(&mut state, BLOCK, 0, 0, 0);
    });
    test::black_box(&mut state);
}

#[bench]
fn bench_guts_compress2_portable(b: &mut Bencher) {
    b.bytes = BLOCK.len() as u64 * 2;
    let portable = guts::Implementation::portable();
    let mut state = [guts::u64x2([1, 1]); 8];
    let count_low = guts::u64x2([1, 1]);
    let count_high = guts::u64x2([1, 1]);
    let lastblock = guts::u64x2([1, 1]);
    let lastnode = guts::u64x2([1, 1]);
    b.iter(|| {
        portable.compress2(
            &mut state,
            BLOCK,
            BLOCK,
            &count_low,
            &count_high,
            &lastblock,
            &lastnode,
        );
    });
    test::black_box(&mut state);
}

#[bench]
fn bench_guts_compress2_sse41(b: &mut Bencher) {
    b.bytes = BLOCK.len() as u64 * 2;
    let sse41 = if let Some(imp) = guts::Implementation::sse41_if_supported() {
        imp
    } else {
        return;
    };
    let mut state = [guts::u64x2([1, 1]); 8];
    let count_low = guts::u64x2([1, 1]);
    let count_high = guts::u64x2([1, 1]);
    let lastblock = guts::u64x2([1, 1]);
    let lastnode = guts::u64x2([1, 1]);
    b.iter(|| {
        sse41.compress2(
            &mut state,
            BLOCK,
            BLOCK,
            &count_low,
            &count_high,
            &lastblock,
            &lastnode,
        );
    });
    test::black_box(&mut state);
}

#[bench]
fn bench_guts_transpose2_portable(b: &mut Bencher) {
    let portable = guts::Implementation::portable();
    let state0 = [1; 8];
    let state1 = [2; 8];
    b.iter(|| portable.transpose2(&state0, &state1));
}

#[bench]
fn bench_guts_untranspose2_portable(b: &mut Bencher) {
    let portable = guts::Implementation::portable();
    let mut state0 = [1; 8];
    let mut state1 = [2; 8];
    let transposed = portable.transpose2(&state0, &state1);
    b.iter(|| {
        portable.untranspose2(&transposed, &mut state0, &mut state1);
        test::black_box(&mut state0);
        test::black_box(&mut state1);
    });
}

#[bench]
fn bench_guts_compress4_portable(b: &mut Bencher) {
    b.bytes = BLOCK.len() as u64 * 4;
    let portable = guts::Implementation::portable();
    let mut state = [guts::u64x4([1, 1, 1, 1]); 8];
    let count_low = guts::u64x4([1, 1, 1, 1]);
    let count_high = guts::u64x4([1, 1, 1, 1]);
    let lastblock = guts::u64x4([1, 1, 1, 1]);
    let lastnode = guts::u64x4([1, 1, 1, 1]);
    b.iter(|| {
        portable.compress4(
            &mut state,
            BLOCK,
            BLOCK,
            BLOCK,
            BLOCK,
            &count_low,
            &count_high,
            &lastblock,
            &lastnode,
        );
    });
    test::black_box(&mut state);
}

#[bench]
fn bench_guts_compress4_sse41(b: &mut Bencher) {
    b.bytes = BLOCK.len() as u64 * 4;
    let sse41 = if let Some(imp) = guts::Implementation::sse41_if_supported() {
        imp
    } else {
        return;
    };
    let mut state = [guts::u64x4([1, 1, 1, 1]); 8];
    let count_low = guts::u64x4([1, 1, 1, 1]);
    let count_high = guts::u64x4([1, 1, 1, 1]);
    let lastblock = guts::u64x4([1, 1, 1, 1]);
    let lastnode = guts::u64x4([1, 1, 1, 1]);
    b.iter(|| {
        sse41.compress4(
            &mut state,
            BLOCK,
            BLOCK,
            BLOCK,
            BLOCK,
            &count_low,
            &count_high,
            &lastblock,
            &lastnode,
        );
    });
    test::black_box(&mut state);
}

#[bench]
fn bench_guts_compress4_avx2(b: &mut Bencher) {
    b.bytes = BLOCK.len() as u64 * 4;
    let avx2 = if let Some(imp) = guts::Implementation::avx2_if_supported() {
        imp
    } else {
        return;
    };
    let mut state = [guts::u64x4([1, 1, 1, 1]); 8];
    let count_low = guts::u64x4([1, 1, 1, 1]);
    let count_high = guts::u64x4([1, 1, 1, 1]);
    let lastblock = guts::u64x4([1, 1, 1, 1]);
    let lastnode = guts::u64x4([1, 1, 1, 1]);
    b.iter(|| {
        avx2.compress4(
            &mut state,
            BLOCK,
            BLOCK,
            BLOCK,
            BLOCK,
            &count_low,
            &count_high,
            &lastblock,
            &lastnode,
        );
    });
    test::black_box(&mut state);
}

#[bench]
fn bench_guts_transpose4_portable(b: &mut Bencher) {
    let portable = guts::Implementation::portable();
    let state0 = [1; 8];
    let state1 = [2; 8];
    let state2 = [3; 8];
    let state3 = [4; 8];
    b.iter(|| portable.transpose4(&state0, &state1, &state2, &state3));
}

#[bench]
fn bench_guts_transpose4_avx2(b: &mut Bencher) {
    let avx2 = if let Some(imp) = guts::Implementation::avx2_if_supported() {
        imp
    } else {
        return;
    };
    let state0 = [1; 8];
    let state1 = [2; 8];
    let state2 = [3; 8];
    let state3 = [4; 8];
    b.iter(|| avx2.transpose4(&state0, &state1, &state2, &state3));
}

#[bench]
fn bench_guts_untranspose4_portable(b: &mut Bencher) {
    let portable = guts::Implementation::portable();
    let mut state0 = [1; 8];
    let mut state1 = [2; 8];
    let mut state2 = [3; 8];
    let mut state3 = [4; 8];
    let transposed = portable.transpose4(&state0, &state1, &state2, &state3);
    b.iter(|| {
        portable.untranspose4(
            &transposed,
            &mut state0,
            &mut state1,
            &mut state2,
            &mut state3,
        );
        test::black_box(&mut state0);
        test::black_box(&mut state1);
        test::black_box(&mut state2);
        test::black_box(&mut state3);
    });
}

#[bench]
fn bench_guts_untranspose4_avx2(b: &mut Bencher) {
    let avx2 = if let Some(imp) = guts::Implementation::avx2_if_supported() {
        imp
    } else {
        return;
    };
    let mut state0 = [1; 8];
    let mut state1 = [2; 8];
    let mut state2 = [3; 8];
    let mut state3 = [4; 8];
    let transposed = avx2.transpose4(&state0, &state1, &state2, &state3);
    b.iter(|| {
        avx2.untranspose4(
            &transposed,
            &mut state0,
            &mut state1,
            &mut state2,
            &mut state3,
        );
        test::black_box(&mut state0);
        test::black_box(&mut state1);
        test::black_box(&mut state2);
        test::black_box(&mut state3);
    });
}

#[bench]
fn bench_compress4_loop_avx2_one_block(b: &mut Bencher) {
    if guts::Implementation::avx2_if_supported().is_none() {
        return;
    }
    let input0 = make_input(b, BLOCKBYTES);
    let input1 = make_input(b, BLOCKBYTES);
    let input2 = make_input(b, BLOCKBYTES);
    let input3 = make_input(b, BLOCKBYTES);
    let count_low = guts::u64x4([0; 4]);
    let count_high = guts::u64x4([0; 4]);
    let last_block = guts::u64x4([0; 4]);
    let last_node = guts::u64x4([0; 4]);
    let buffer_tail = guts::u64x4([0; 4]);
    b.iter(|| unsafe {
        let mut state0 = guts::u64x8([1; 8]);
        let mut state1 = guts::u64x8([2; 8]);
        let mut state2 = guts::u64x8([3; 8]);
        let mut state3 = guts::u64x8([4; 8]);
        benchmarks::compress4_loop_avx2(
            &mut state0,
            &mut state1,
            &mut state2,
            &mut state3,
            &input0,
            &input1,
            &input2,
            &input3,
            &count_low,
            &count_high,
            &last_block,
            &last_node,
            1,
            1,
            &buffer_tail,
        );
        test::black_box(&mut state0);
        test::black_box(&mut state1);
        test::black_box(&mut state2);
        test::black_box(&mut state3);
    });
}

#[bench]
fn bench_compress4_loop_avx2_one_mb(b: &mut Bencher) {
    if guts::Implementation::avx2_if_supported().is_none() {
        return;
    }
    let len = (1 << 20) / 4;
    let input0 = make_input(b, len);
    let input1 = make_input(b, len);
    let input2 = make_input(b, len);
    let input3 = make_input(b, len);
    let count_low = guts::u64x4([0; 4]);
    let count_high = guts::u64x4([0; 4]);
    let last_block = guts::u64x4([0; 4]);
    let last_node = guts::u64x4([0; 4]);
    let buffer_tail = guts::u64x4([0; 4]);
    b.iter(|| unsafe {
        let mut state0 = guts::u64x8([1; 8]);
        let mut state1 = guts::u64x8([2; 8]);
        let mut state2 = guts::u64x8([3; 8]);
        let mut state3 = guts::u64x8([4; 8]);
        benchmarks::compress4_loop_avx2(
            &mut state0,
            &mut state1,
            &mut state2,
            &mut state3,
            &input0,
            &input1,
            &input2,
            &input3,
            &count_low,
            &count_high,
            &last_block,
            &last_node,
            len / BLOCKBYTES,
            1,
            &buffer_tail,
        );
        test::black_box(&mut state0);
        test::black_box(&mut state1);
        test::black_box(&mut state2);
        test::black_box(&mut state3);
    });
}

#[bench]
fn bench_compress4_loop_avx2_one_mb_striped(b: &mut Bencher) {
    if guts::Implementation::avx2_if_supported().is_none() {
        return;
    }
    let input = make_input(b, MB);
    let count_low = guts::u64x4([0; 4]);
    let count_high = guts::u64x4([0; 4]);
    let last_block = guts::u64x4([0; 4]);
    let last_node = guts::u64x4([0; 4]);
    let buffer_tail = guts::u64x4([0; 4]);
    b.iter(|| unsafe {
        let mut state0 = guts::u64x8([1; 8]);
        let mut state1 = guts::u64x8([2; 8]);
        let mut state2 = guts::u64x8([3; 8]);
        let mut state3 = guts::u64x8([4; 8]);
        benchmarks::compress4_loop_avx2(
            &mut state0,
            &mut state1,
            &mut state2,
            &mut state3,
            &input[0 * BLOCKBYTES..],
            &input[1 * BLOCKBYTES..],
            &input[2 * BLOCKBYTES..],
            &input[3 * BLOCKBYTES..],
            &count_low,
            &count_high,
            &last_block,
            &last_node,
            MB / (BLOCKBYTES * 4),
            4,
            &buffer_tail,
        );
        test::black_box(&mut state0);
        test::black_box(&mut state1);
        test::black_box(&mut state2);
        test::black_box(&mut state3);
    });
}
