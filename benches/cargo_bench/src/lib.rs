#![feature(test)]

extern crate test;

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
    let mut input = RandomInput::new(b, blake2b_simd::BLOCKBYTES);
    b.iter(|| blake2b_simd::blake2b(input.get()));
}

#[bench]
fn bench_blake2s_avx2_one_block(b: &mut Bencher) {
    let mut input = RandomInput::new(b, blake2s_simd::BLOCKBYTES);
    b.iter(|| blake2s_simd::blake2s(input.get()));
}

#[bench]
fn bench_blake2b_avx2_one_mb(b: &mut Bencher) {
    let mut input = RandomInput::new(b, MB);
    b.iter(|| blake2b_simd::blake2b(input.get()));
}

#[bench]
fn bench_blake2s_avx2_one_mb(b: &mut Bencher) {
    let mut input = RandomInput::new(b, MB);
    b.iter(|| blake2s_simd::blake2s(input.get()));
}

#[bench]
fn bench_blake2b_portable_one_block(b: &mut Bencher) {
    let mut input = RandomInput::new(b, blake2b_simd::BLOCKBYTES);
    b.iter(|| {
        let mut state = blake2b_simd::State::new();
        // TODO: Put an implementation parameter on the params object, so we
        // don't have to pay the copying overhead of update.
        blake2b_simd::benchmarks::force_portable(&mut state);
        state.update(input.get());
        state.finalize()
    });
}

#[bench]
fn bench_blake2s_portable_one_block(b: &mut Bencher) {
    let mut input = RandomInput::new(b, blake2s_simd::BLOCKBYTES);
    b.iter(|| {
        let mut state = blake2s_simd::State::new();
        blake2s_simd::benchmarks::force_portable(&mut state);
        state.update(input.get());
        state.finalize()
    });
}

#[bench]
fn bench_blake2b_portable_one_mb(b: &mut Bencher) {
    let mut input = RandomInput::new(b, MB);
    b.iter(|| {
        let mut state = blake2b_simd::State::new();
        blake2b_simd::benchmarks::force_portable(&mut state);
        state.update(input.get());
        state.finalize()
    });
}

#[bench]
fn bench_blake2s_portable_one_mb(b: &mut Bencher) {
    let mut input = RandomInput::new(b, MB);
    b.iter(|| {
        let mut state = blake2s_simd::State::new();
        blake2s_simd::benchmarks::force_portable(&mut state);
        state.update(input.get());
        state.finalize()
    });
}

#[bench]
fn bench_blake2bp_one_mb(b: &mut Bencher) {
    let mut input = RandomInput::new(b, MB);
    b.iter(|| blake2b_simd::blake2bp::blake2bp(input.get()));
}

#[bench]
fn bench_blake2sp_one_mb(b: &mut Bencher) {
    let mut input = RandomInput::new(b, MB);
    b.iter(|| blake2s_simd::blake2sp::blake2sp(input.get()));
}

#[bench]
fn bench_blake2b_hash_many_2x_1mb(b: &mut Bencher) {
    let mut input0 = RandomInput::new(b, MB);
    let mut input1 = RandomInput::new(b, MB);
    let params = blake2b_simd::Params::new();
    b.iter(|| {
        let mut jobs = [
            blake2b_simd::many::HashManyJob::new(&params, input0.get()),
            blake2b_simd::many::HashManyJob::new(&params, input1.get()),
        ];
        blake2b_simd::many::hash_many(jobs.iter_mut());
        [jobs[0].to_hash(), jobs[1].to_hash()]
    });
}

#[bench]
fn bench_blake2b_hash_many_4x_1mb(b: &mut Bencher) {
    let mut input0 = RandomInput::new(b, MB);
    let mut input1 = RandomInput::new(b, MB);
    let mut input2 = RandomInput::new(b, MB);
    let mut input3 = RandomInput::new(b, MB);
    let params = blake2b_simd::Params::new();
    b.iter(|| {
        let mut jobs = [
            blake2b_simd::many::HashManyJob::new(&params, input0.get()),
            blake2b_simd::many::HashManyJob::new(&params, input1.get()),
            blake2b_simd::many::HashManyJob::new(&params, input2.get()),
            blake2b_simd::many::HashManyJob::new(&params, input3.get()),
        ];
        blake2b_simd::many::hash_many(jobs.iter_mut());
        [
            jobs[0].to_hash(),
            jobs[1].to_hash(),
            jobs[2].to_hash(),
            jobs[3].to_hash(),
        ]
    });
}

#[bench]
fn bench_blake2s_hash_many_4x_1mb(b: &mut Bencher) {
    let mut input0 = RandomInput::new(b, MB);
    let mut input1 = RandomInput::new(b, MB);
    let mut input2 = RandomInput::new(b, MB);
    let mut input3 = RandomInput::new(b, MB);
    let params = blake2s_simd::Params::new();
    b.iter(|| {
        let mut jobs = [
            blake2s_simd::many::HashManyJob::new(&params, input0.get()),
            blake2s_simd::many::HashManyJob::new(&params, input1.get()),
            blake2s_simd::many::HashManyJob::new(&params, input2.get()),
            blake2s_simd::many::HashManyJob::new(&params, input3.get()),
        ];
        blake2s_simd::many::hash_many(jobs.iter_mut());
        [
            jobs[0].to_hash(),
            jobs[1].to_hash(),
            jobs[2].to_hash(),
            jobs[3].to_hash(),
        ]
    });
}

#[bench]
fn bench_blake2s_hash_many_8x_1mb(b: &mut Bencher) {
    let mut input0 = RandomInput::new(b, MB);
    let mut input1 = RandomInput::new(b, MB);
    let mut input2 = RandomInput::new(b, MB);
    let mut input3 = RandomInput::new(b, MB);
    let mut input4 = RandomInput::new(b, MB);
    let mut input5 = RandomInput::new(b, MB);
    let mut input6 = RandomInput::new(b, MB);
    let mut input7 = RandomInput::new(b, MB);
    let params = blake2s_simd::Params::new();
    b.iter(|| {
        let mut jobs = [
            blake2s_simd::many::HashManyJob::new(&params, input0.get()),
            blake2s_simd::many::HashManyJob::new(&params, input1.get()),
            blake2s_simd::many::HashManyJob::new(&params, input2.get()),
            blake2s_simd::many::HashManyJob::new(&params, input3.get()),
            blake2s_simd::many::HashManyJob::new(&params, input4.get()),
            blake2s_simd::many::HashManyJob::new(&params, input5.get()),
            blake2s_simd::many::HashManyJob::new(&params, input6.get()),
            blake2s_simd::many::HashManyJob::new(&params, input7.get()),
        ];
        blake2s_simd::many::hash_many(jobs.iter_mut());
        [
            jobs[0].to_hash(),
            jobs[1].to_hash(),
            jobs[2].to_hash(),
            jobs[3].to_hash(),
            jobs[4].to_hash(),
            jobs[5].to_hash(),
            jobs[6].to_hash(),
            jobs[7].to_hash(),
        ]
    });
}

// Note for comparison: The blake2-avx2-sneves C code is currently compiled
// with -mavx2 but *not* with -march=native. Upstream uses -march=native, but
// -mavx2 is closer to how blake2b_simd is compiled, and it makes the benchmark
// more apples-to-apples. However, since the C code was tuned with
// -march=native in mind, it's possible this switcharoo makes the comparison
// unfair in other ways. I haven't asked the author yet.
#[cfg(feature = "blake2-avx2-sneves")]
#[bench]
fn bench_sneves_blake2b_avx2(b: &mut Bencher) {
    let mut input = RandomInput::new(b, MB);
    b.iter(|| blake2_avx2_sneves::blake2b(input.get()));
}

#[cfg(feature = "blake2-avx2-sneves")]
#[bench]
fn bench_sneves_blake2bp_avx2(b: &mut Bencher) {
    let mut input = RandomInput::new(b, MB);
    b.iter(|| blake2_avx2_sneves::blake2bp(input.get()));
}

#[cfg(feature = "blake2-avx2-sneves")]
#[bench]
fn bench_sneves_blake2sp_avx2(b: &mut Bencher) {
    let mut input = RandomInput::new(b, MB);
    b.iter(|| blake2_avx2_sneves::blake2sp(input.get()));
}

// Note for comparison: Unlike the blake2-avx2-sneves C code above, the
// KangarooTwelve C code *is* compiled with -march=native. Their build system
// is more involved than above, and I don't want to muck around with it.
// Current benchmarks are almost exactly on par with blake2b_simd, maybe just a
// hair faster, which is a surprising coincidence. However, with the equivalent
// flag RUSTFLAGS="-C target-cpu=native", blake2b_simd pulls ahead.
#[cfg(feature = "kangarootwelve")]
#[bench]
fn bench_kangarootwelve(b: &mut Bencher) {
    let mut input = RandomInput::new(b, MB);
    b.iter(|| kangarootwelve::kangarootwelve(input.get()));
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
