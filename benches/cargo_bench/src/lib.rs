#![feature(test)]

extern crate blake2b_simd;
#[cfg(feature = "libsodium-ffi")]
extern crate libsodium_ffi;
#[cfg(feature = "openssl")]
extern crate openssl;
extern crate test;

use blake2b_simd::*;
use test::Bencher;

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
fn bench_compress4_loop_avx2_one_block_b(b: &mut Bencher) {
    if guts::Implementation::avx2_if_supported().is_none() {
        return;
    }
    let input0 = make_input(b, BLOCKBYTES);
    let input1 = make_input(b, BLOCKBYTES);
    let input2 = make_input(b, BLOCKBYTES);
    let input3 = make_input(b, BLOCKBYTES);
    let last_block = [true; 4];
    let last_node = [true; 4];
    b.iter(|| unsafe {
        let mut state0 = guts::u64x8([1; 8]);
        let mut state1 = guts::u64x8([2; 8]);
        let mut state2 = guts::u64x8([3; 8]);
        let mut state3 = guts::u64x8([4; 8]);
        let mut count0 = 0;
        let mut count1 = 0;
        let mut count2 = 0;
        let mut count3 = 0;
        benchmarks::compress4_loop_avx2_b(
            [&mut state0, &mut state1, &mut state2, &mut state3],
            [&input0, &input1, &input2, &input3],
            [&mut count0, &mut count1, &mut count2, &mut count3],
            last_block,
            last_node,
            1,
            1,
        );
        test::black_box(&mut state0);
        test::black_box(&mut state1);
        test::black_box(&mut state2);
        test::black_box(&mut state3);
    });
}

#[bench]
fn bench_compress4_loop_avx2_one_mb_b(b: &mut Bencher) {
    if guts::Implementation::avx2_if_supported().is_none() {
        return;
    }
    let len = (1 << 20) / 4;
    let input0 = make_input(b, len);
    let input1 = make_input(b, len);
    let input2 = make_input(b, len);
    let input3 = make_input(b, len);
    let last_block = [true; 4];
    let last_node = [true; 4];
    b.iter(|| unsafe {
        let mut state0 = guts::u64x8([1; 8]);
        let mut state1 = guts::u64x8([2; 8]);
        let mut state2 = guts::u64x8([3; 8]);
        let mut state3 = guts::u64x8([4; 8]);
        let mut count0 = 0;
        let mut count1 = 0;
        let mut count2 = 0;
        let mut count3 = 0;
        benchmarks::compress4_loop_avx2_b(
            [&mut state0, &mut state1, &mut state2, &mut state3],
            [&input0, &input1, &input2, &input3],
            [&mut count0, &mut count1, &mut count2, &mut count3],
            last_block,
            last_node,
            len / BLOCKBYTES,
            1,
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
fn bench_hash_many_4_blocks(b: &mut Bencher) {
    let params = [Params::new(), Params::new(), Params::new(), Params::new()];
    let input = make_input(b, 4 * BLOCKBYTES);
    let inputs = [
        &input[0 * BLOCKBYTES..][..BLOCKBYTES],
        &input[1 * BLOCKBYTES..][..BLOCKBYTES],
        &input[2 * BLOCKBYTES..][..BLOCKBYTES],
        &input[3 * BLOCKBYTES..][..BLOCKBYTES],
    ];
    b.iter(|| {
        let mut outputs = [Hash::empty(), Hash::empty(), Hash::empty(), Hash::empty()];
        hash_many(&inputs[..], &mut outputs[..], &params[..]);
        test::black_box(&mut outputs);
    });
}

#[bench]
fn bench_hash_many_4_mb(b: &mut Bencher) {
    let params = [Params::new(), Params::new(), Params::new(), Params::new()];
    let len = 1 << 20;
    let input0 = make_input(b, len);
    let input1 = make_input(b, len);
    let input2 = make_input(b, len);
    let input3 = make_input(b, len);
    let inputs = [&input0[..], &input1[..], &input2[..], &input3[..]];
    b.iter(|| {
        let mut outputs = [Hash::empty(), Hash::empty(), Hash::empty(), Hash::empty()];
        hash_many(&inputs[..], &mut outputs[..], &params[..]);
        test::black_box(&mut outputs);
    });
}
