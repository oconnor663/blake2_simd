#![feature(test)]

extern crate test;

use blake2b_simd::*;
use rand::seq::SliceRandom;
use rand::RngCore;
use test::Bencher;

const MB: usize = 1 << 20;

// This struct randomizes two things:
// 1. The actual bytes of input.
// 2. The page offset the input starts at.
struct RandomInput {
    buf: Vec<u8>,
    len: usize,
    offsets: Vec<usize>,
    offset_index: usize,
}

impl RandomInput {
    fn new(b: &mut Bencher, len: usize) -> Self {
        b.bytes += len as u64;
        let page_size: usize = page_size::get();
        let mut buf = vec![0u8; len + page_size];
        rand::thread_rng().fill_bytes(&mut buf);
        let mut offsets: Vec<usize> = (0..page_size).collect();
        offsets.shuffle(&mut rand::thread_rng());
        Self {
            buf,
            len,
            offsets,
            offset_index: 0,
        }
    }

    fn get(&mut self) -> &[u8] {
        let offset = self.offsets[self.offset_index];
        self.offset_index += 1;
        if self.offset_index >= self.offsets.len() {
            self.offset_index = 0;
        }
        &self.buf[offset..][..self.len]
    }
}

#[bench]
fn bench_blake2b_avx2_one_block(b: &mut Bencher) {
    let mut input = RandomInput::new(b, BLOCKBYTES);
    b.iter(|| blake2b(input.get()));
}

#[bench]
fn bench_blake2b_avx2_one_mb(b: &mut Bencher) {
    let mut input = RandomInput::new(b, MB);
    b.iter(|| blake2b(input.get()));
}

#[bench]
fn bench_blake2b_portable_one_block(b: &mut Bencher) {
    let mut input = RandomInput::new(b, BLOCKBYTES);
    b.iter(|| {
        let mut state = State::new();
        benchmarks::force_portable(&mut state);
        state.update(input.get());
        state.finalize()
    });
}

#[bench]
fn bench_blake2b_portable_one_mb(b: &mut Bencher) {
    let mut input = RandomInput::new(b, MB);
    b.iter(|| {
        let mut state = State::new();
        benchmarks::force_portable(&mut state);
        state.update(input.get());
        state.finalize()
    });
}

#[bench]
fn bench_blake2bp_one_mb(b: &mut Bencher) {
    let mut input = RandomInput::new(b, MB);
    b.iter(|| blake2bp::blake2bp(input.get()));
}

#[bench]
fn bench_blake2b_update_many_4x_4096(b: &mut Bencher) {
    let mut inputs = [
        RandomInput::new(b, 4096),
        RandomInput::new(b, 4096),
        RandomInput::new(b, 4096),
        RandomInput::new(b, 4096),
    ];
    b.iter(|| {
        let mut states = [State::new(), State::new(), State::new(), State::new()];
        let inputs_iter = inputs.iter_mut().map(|input| input.get());
        many::update_many(states.iter_mut().zip(inputs_iter));
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
    let mut inputs = [
        RandomInput::new(b, MB),
        RandomInput::new(b, MB),
        RandomInput::new(b, MB),
        RandomInput::new(b, MB),
    ];
    b.iter(|| {
        let mut states = [State::new(), State::new(), State::new(), State::new()];
        let inputs_iter = inputs.iter_mut().map(|input| input.get());
        many::update_many(states.iter_mut().zip(inputs_iter));
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
    let mut input0 = RandomInput::new(b, 4096);
    let mut input1 = RandomInput::new(b, 4096);
    let mut input2 = RandomInput::new(b, 4096);
    let mut input3 = RandomInput::new(b, 4096);
    let params = Params::new();
    b.iter(|| {
        let mut jobs = [
            many::HashManyJob::new(&params, input0.get()),
            many::HashManyJob::new(&params, input1.get()),
            many::HashManyJob::new(&params, input2.get()),
            many::HashManyJob::new(&params, input3.get()),
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
    let mut input0 = RandomInput::new(b, MB);
    let mut input1 = RandomInput::new(b, MB);
    let mut input2 = RandomInput::new(b, MB);
    let mut input3 = RandomInput::new(b, MB);
    let params = Params::new();
    b.iter(|| {
        let mut jobs = [
            many::HashManyJob::new(&params, input0.get()),
            many::HashManyJob::new(&params, input1.get()),
            many::HashManyJob::new(&params, input2.get()),
            many::HashManyJob::new(&params, input3.get()),
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

#[cfg(feature = "blake2_avx2_sneves")]
#[bench]
fn bench_sneves_blake2b_avx2(b: &mut Bencher) {
    let mut input = RandomInput::new(b, MB);
    b.iter(|| blake2_avx2_sneves::blake2b(input.get()));
}

#[cfg(feature = "blake2_avx2_sneves")]
#[bench]
fn bench_sneves_blake2bp_avx2(b: &mut Bencher) {
    let mut input = RandomInput::new(b, MB);
    b.iter(|| blake2_avx2_sneves::blake2bp(input.get()));
}

#[cfg(feature = "blake2_avx2_sneves")]
#[bench]
fn bench_sneves_blake2sp_avx2(b: &mut Bencher) {
    let mut input = RandomInput::new(b, MB);
    b.iter(|| blake2_avx2_sneves::blake2sp(input.get()));
}

#[cfg(feature = "libsodium-ffi")]
#[bench]
fn bench_libsodium_one_mb(b: &mut Bencher) {
    let mut input = RandomInput::new(b, MB);
    let mut out = [0; 64];
    unsafe {
        let init_ret = libsodium_ffi::sodium_init();
        assert!(init_ret != -1);
    }
    b.iter(|| unsafe {
        let input_slice = input.get();
        libsodium_ffi::crypto_generichash(
            out.as_mut_ptr(),
            out.len(),
            input_slice.as_ptr(),
            input_slice.len() as u64,
            std::ptr::null(),
            0,
        );
    });
}

#[cfg(feature = "openssl")]
#[bench]
fn bench_openssl_md5_one_mb(b: &mut Bencher) {
    let mut input = RandomInput::new(b, MB);
    b.iter(|| openssl::hash::hash(openssl::hash::MessageDigest::md5(), input.get()));
}

#[cfg(feature = "openssl")]
#[bench]
fn bench_openssl_sha1_one_mb(b: &mut Bencher) {
    let mut input = RandomInput::new(b, MB);
    b.iter(|| openssl::hash::hash(openssl::hash::MessageDigest::sha1(), input.get()));
}

#[cfg(feature = "openssl")]
#[bench]
fn bench_openssl_sha512_one_mb(b: &mut Bencher) {
    let mut input = RandomInput::new(b, MB);
    b.iter(|| openssl::hash::hash(openssl::hash::MessageDigest::sha512(), input.get()));
}
