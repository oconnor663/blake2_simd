#![feature(test)]

extern crate test;

use blake2b_simd::*;
use rand::RngCore;
use test::Bencher;

const MB: usize = 1 << 20;

// Do two special things to the input:
// 1. Fill it with random bytes. This probably isn't strictly necessary, but
//    it's important that it's written to with *something*, because I believe
//    asking the allocator for pages of all zeroes hits special copy-on-write
//    optimizations in the kernel that we want to avoid.
// 2. Make sure that the input is aligned to a page boundary. WORKER_OFFSET is
//    interpreted relative to that alignment, so an offset of zero means the
//    input starts exactly at a the page boundary. This seems especially
//    important for the performance of BLAKE2bp, though I'm not sure exactly
//    why. Exact page alignment seems to give reasonably good performance
//    compared to most other offsets (though not strictly the best, see
//    https://github.com/oconnor663/blake2b_simd/issues/8), and it seems like a
//    "fair" benchmarking point.
fn make_input<'a>(b: &mut Bencher, vec: &'a mut Vec<u8>, len: usize) -> &'a [u8] {
    b.bytes += len as u64;
    let page_size: usize = page_size::get();
    *vec = vec![0; len + page_size];
    rand::thread_rng().fill_bytes(vec);
    let allocated_offset = vec.as_ptr() as usize % page_size;
    let next_page_start = page_size - allocated_offset;
    let target_offset = next_page_start;
    &vec[target_offset..][..len]
}

#[bench]
fn bench_blake2b_avx2_one_block(b: &mut Bencher) {
    let mut vec = Vec::new();
    let input = make_input(b, &mut vec, BLOCKBYTES);
    b.iter(|| blake2b(&input));
}

#[bench]
fn bench_blake2b_avx2_one_mb(b: &mut Bencher) {
    let mut vec = Vec::new();
    let input = make_input(b, &mut vec, MB);
    b.iter(|| blake2b(&input));
}

#[bench]
fn bench_blake2b_portable_one_block(b: &mut Bencher) {
    let mut vec = Vec::new();
    let input = make_input(b, &mut vec, BLOCKBYTES);
    b.iter(|| {
        let mut state = State::new();
        benchmarks::force_portable(&mut state);
        state.update(&input);
        state.finalize()
    });
}

#[bench]
fn bench_blake2b_portable_one_mb(b: &mut Bencher) {
    let mut vec = Vec::new();
    let input = make_input(b, &mut vec, MB);
    b.iter(|| {
        let mut state = State::new();
        benchmarks::force_portable(&mut state);
        state.update(&input);
        state.finalize()
    });
}

#[bench]
fn bench_blake2bp_one_mb(b: &mut Bencher) {
    let mut vec = Vec::new();
    let input = make_input(b, &mut vec, MB);
    b.iter(|| blake2bp::blake2bp(&input));
}

#[bench]
fn bench_blake2b_update_many_4x_4096(b: &mut Bencher) {
    let mut vec0 = Vec::new();
    let mut vec1 = Vec::new();
    let mut vec2 = Vec::new();
    let mut vec3 = Vec::new();
    let inputs = [
        make_input(b, &mut vec0, 4096),
        make_input(b, &mut vec1, 4096),
        make_input(b, &mut vec2, 4096),
        make_input(b, &mut vec3, 4096),
    ];
    b.iter(|| {
        let mut states = [State::new(), State::new(), State::new(), State::new()];
        many::update_many(states.iter_mut().zip(inputs.iter()));
        [
            states[0].finalize(),
            states[1].finalize(),
            states[2].finalize(),
            states[3].finalize(),
        ]
    });
}

#[bench]
fn bench_blake2b_update_many_4x_1mb(b: &mut Bencher) {
    let mut vec0 = Vec::new();
    let mut vec1 = Vec::new();
    let mut vec2 = Vec::new();
    let mut vec3 = Vec::new();
    let inputs = [
        make_input(b, &mut vec0, MB),
        make_input(b, &mut vec1, MB),
        make_input(b, &mut vec2, MB),
        make_input(b, &mut vec3, MB),
    ];
    b.iter(|| {
        let mut states = [State::new(), State::new(), State::new(), State::new()];
        many::update_many(states.iter_mut().zip(inputs.iter()));
        [
            states[0].finalize(),
            states[1].finalize(),
            states[2].finalize(),
            states[3].finalize(),
        ]
    });
}

#[bench]
fn bench_blake2b_hash_many_4x_4096(b: &mut Bencher) {
    let mut vec0 = Vec::new();
    let mut vec1 = Vec::new();
    let mut vec2 = Vec::new();
    let mut vec3 = Vec::new();
    let inputs = [
        make_input(b, &mut vec0, 4096),
        make_input(b, &mut vec1, 4096),
        make_input(b, &mut vec2, 4096),
        make_input(b, &mut vec3, 4096),
    ];
    let params = Params::new();
    b.iter(|| {
        let mut jobs = [
            many::HashManyJob::new(&params, &inputs[0]),
            many::HashManyJob::new(&params, &inputs[1]),
            many::HashManyJob::new(&params, &inputs[2]),
            many::HashManyJob::new(&params, &inputs[3]),
        ];
        many::hash_many(jobs.iter_mut());
        [
            jobs[0].to_hash(),
            jobs[1].to_hash(),
            jobs[2].to_hash(),
            jobs[3].to_hash(),
        ]
    });
}

#[bench]
fn bench_blake2b_hash_many_4x_1mb(b: &mut Bencher) {
    let mut vec0 = Vec::new();
    let mut vec1 = Vec::new();
    let mut vec2 = Vec::new();
    let mut vec3 = Vec::new();
    let inputs = [
        make_input(b, &mut vec0, MB),
        make_input(b, &mut vec1, MB),
        make_input(b, &mut vec2, MB),
        make_input(b, &mut vec3, MB),
    ];
    let params = Params::new();
    b.iter(|| {
        let mut jobs = [
            many::HashManyJob::new(&params, &inputs[0]),
            many::HashManyJob::new(&params, &inputs[1]),
            many::HashManyJob::new(&params, &inputs[2]),
            many::HashManyJob::new(&params, &inputs[3]),
        ];
        many::hash_many(jobs.iter_mut());
        [
            jobs[0].to_hash(),
            jobs[1].to_hash(),
            jobs[2].to_hash(),
            jobs[3].to_hash(),
        ]
    });
}

#[bench]
fn bench_neves_blake2b_avx2(b: &mut Bencher) {
    let mut vec = Vec::new();
    let input = make_input(b, &mut vec, MB);
    b.iter(|| blake2_avx2_neves::blake2b(&input));
}

#[bench]
fn bench_neves_blake2bp_avx2(b: &mut Bencher) {
    let mut vec = Vec::new();
    let input = make_input(b, &mut vec, MB);
    b.iter(|| blake2_avx2_neves::blake2bp(&input));
}

#[bench]
fn bench_neves_blake2sp_avx2(b: &mut Bencher) {
    let mut vec = Vec::new();
    let input = make_input(b, &mut vec, MB);
    b.iter(|| blake2_avx2_neves::blake2sp(&input));
}

#[cfg(feature = "libsodium-ffi")]
#[bench]
fn bench_libsodium_one_mb(b: &mut Bencher) {
    let mut vec = Vec::new();
    let input = make_input(b, &mut vec, MB);
    let mut out = [0; 64];
    unsafe {
        let init_ret = libsodium_ffi::sodium_init();
        assert!(init_ret != -1);
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
    let mut vec = Vec::new();
    let input = make_input(b, &mut vec, MB);
    b.iter(|| openssl::hash::hash(openssl::hash::MessageDigest::md5(), &input));
}

#[cfg(feature = "openssl")]
#[bench]
fn bench_openssl_sha1_one_mb(b: &mut Bencher) {
    let mut vec = Vec::new();
    let input = make_input(b, &mut vec, MB);
    b.iter(|| openssl::hash::hash(openssl::hash::MessageDigest::sha1(), &input));
}

#[cfg(feature = "openssl")]
#[bench]
fn bench_openssl_sha512_one_mb(b: &mut Bencher) {
    let mut vec = Vec::new();
    let input = make_input(b, &mut vec, MB);
    b.iter(|| openssl::hash::hash(openssl::hash::MessageDigest::sha512(), &input));
}
