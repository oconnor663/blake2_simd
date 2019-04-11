#![feature(test)]

extern crate blake2b_simd;
#[cfg(feature = "libsodium-ffi")]
extern crate libsodium_ffi;
#[cfg(feature = "openssl")]
extern crate openssl;
extern crate test;

use blake2b_simd::guts::{Finalize, Job, Stride};
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
fn bench_compress4_loop_avx2_one_block(b: &mut Bencher) {
    if guts::Implementation::avx2_if_supported().is_none() {
        return;
    }
    let input0 = make_input(b, BLOCKBYTES);
    let input1 = make_input(b, BLOCKBYTES);
    let input2 = make_input(b, BLOCKBYTES);
    let input3 = make_input(b, BLOCKBYTES);
    let mut words0 = guts::u64x8([1; 8]);
    let mut words1 = guts::u64x8([2; 8]);
    let mut words2 = guts::u64x8([3; 8]);
    let mut words3 = guts::u64x8([4; 8]);
    b.iter(|| unsafe {
        let mut jobs = [
            Job::new(&mut words0, 0, &input0, Finalize::YesOrdinary),
            Job::new(&mut words1, 0, &input1, Finalize::YesOrdinary),
            Job::new(&mut words2, 0, &input2, Finalize::YesOrdinary),
            Job::new(&mut words3, 0, &input3, Finalize::YesOrdinary),
        ];
        benchmarks::compress4_loop_avx2(&mut jobs, Stride::Normal);
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
    let mut words0 = guts::u64x8([1; 8]);
    let mut words1 = guts::u64x8([2; 8]);
    let mut words2 = guts::u64x8([3; 8]);
    let mut words3 = guts::u64x8([4; 8]);
    b.iter(|| unsafe {
        let mut jobs = [
            Job::new(&mut words0, 0, &input0, Finalize::YesOrdinary),
            Job::new(&mut words1, 0, &input1, Finalize::YesOrdinary),
            Job::new(&mut words2, 0, &input2, Finalize::YesOrdinary),
            Job::new(&mut words3, 0, &input3, Finalize::YesOrdinary),
        ];
        benchmarks::compress4_loop_avx2(&mut jobs, Stride::Normal);
    });
}

#[bench]
fn bench_compress2_loop_avx2_one_mb(b: &mut Bencher) {
    if guts::Implementation::avx2_if_supported().is_none() {
        return;
    }
    let len = (1 << 20) / 2;
    let input0 = make_input(b, len);
    let input1 = make_input(b, len);
    let mut words0 = guts::u64x8([1; 8]);
    let mut words1 = guts::u64x8([2; 8]);
    b.iter(|| unsafe {
        let mut jobs = [
            Job::new(&mut words0, 0, &input0, Finalize::YesOrdinary),
            Job::new(&mut words1, 0, &input1, Finalize::YesOrdinary),
        ];
        benchmarks::compress2_loop_sse41(&mut jobs, Stride::Normal);
    });
}

#[bench]
fn bench_compress1_loop_avx2_one_mb(b: &mut Bencher) {
    if guts::Implementation::avx2_if_supported().is_none() {
        return;
    }
    let len = 1 << 20;
    let input0 = make_input(b, len);
    let mut words0 = guts::u64x8([1; 8]);
    b.iter(|| unsafe {
        let job = Job::new(&mut words0, 0, &input0, Finalize::YesOrdinary);
        benchmarks::compress1_loop_avx2(job, Stride::Normal);
    });
}

#[bench]
fn bench_hash_many_4x_block(b: &mut Bencher) {
    let params = Params::new();
    let inputs = [
        make_input(b, BLOCKBYTES),
        make_input(b, BLOCKBYTES),
        make_input(b, BLOCKBYTES),
        make_input(b, BLOCKBYTES),
    ];
    b.iter(|| {
        let mut jobs = [
            hash_many::HashManyJob::new(&params, &inputs[0]),
            hash_many::HashManyJob::new(&params, &inputs[1]),
            hash_many::HashManyJob::new(&params, &inputs[2]),
            hash_many::HashManyJob::new(&params, &inputs[3]),
        ];
        hash_many::hash_many(jobs.iter_mut());
        [
            jobs[0].to_hash(),
            jobs[1].to_hash(),
            jobs[2].to_hash(),
            jobs[3].to_hash(),
        ]
    });
}

#[bench]
fn bench_hash_many_4x_4096(b: &mut Bencher) {
    let params = Params::new();
    let inputs = [
        make_input(b, 4096),
        make_input(b, 4096),
        make_input(b, 4096),
        make_input(b, 4096),
    ];
    b.iter(|| {
        let mut jobs = [
            hash_many::HashManyJob::new(&params, &inputs[0]),
            hash_many::HashManyJob::new(&params, &inputs[1]),
            hash_many::HashManyJob::new(&params, &inputs[2]),
            hash_many::HashManyJob::new(&params, &inputs[3]),
        ];
        hash_many::hash_many(jobs.iter_mut());
        [
            jobs[0].to_hash(),
            jobs[1].to_hash(),
            jobs[2].to_hash(),
            jobs[3].to_hash(),
        ]
    });
}

#[bench]
fn bench_hash_many_4x_1mb(b: &mut Bencher) {
    let params = Params::new();
    let inputs = [
        make_input(b, MB),
        make_input(b, MB),
        make_input(b, MB),
        make_input(b, MB),
    ];
    b.iter(|| {
        let mut jobs = [
            hash_many::HashManyJob::new(&params, &inputs[0]),
            hash_many::HashManyJob::new(&params, &inputs[1]),
            hash_many::HashManyJob::new(&params, &inputs[2]),
            hash_many::HashManyJob::new(&params, &inputs[3]),
        ];
        hash_many::hash_many(jobs.iter_mut());
        [
            jobs[0].to_hash(),
            jobs[1].to_hash(),
            jobs[2].to_hash(),
            jobs[3].to_hash(),
        ]
    });
}
