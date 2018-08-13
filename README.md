# blake2b_simd [![Build Status](https://travis-ci.org/oconnor663/blake2b_simd.svg?branch=master)](https://travis-ci.org/oconnor663/blake2b_simd) [![Build status](https://ci.appveyor.com/api/projects/status/yq6eo0b6ong2l0xj/branch/master?svg=true)](https://ci.appveyor.com/project/oconnor663/blake2b-simd/branch/master) [![docs.rs](https://docs.rs/blake2b_simd/badge.svg)](https://docs.rs/blake2b_simd)

[Repo](https://github.com/oconnor663/blake2b_simd) — [Docs](https://docs.rs/blake2b_simd) — [Crate](https://crates.io/crates/blake2b_simd)

An implementation of the BLAKE2b hash with:

- 100% stable Rust.
- A fast AVX2 implementation ported from [libsodium](https://github.com/jedisct1/libsodium).
- A portable, safe implementation for other platforms.
- Dynamic CPU feature detection. Binaries for x86 include the AVX2 implementation by default
  and call it if the processor supports it at runtime.
- `no_std` support. `std` is on by default, for feature detection and `std::io::Write`.
- All the features from the [the BLAKE2 spec](https://blake2.net/blake2.pdf), like adjustable
  length, keying, and associated data for tree hashing.

## Example

```rust
use blake2b_simd::{blake2b, Params};

assert!(blake2b(b"foo") == blake2b(b"foo"));
assert!(blake2b(b"foo") != blake2b(b"bar"));

let hash = Params::new()
    .hash_length(16)
    .key(b"The Magic Words are Squeamish Ossifrage")
    .personal(b"L. P. Waterhouse")
    .to_state()
    .update(b"foo")
    .update(b"bar")
    .update(b"baz")
    .finalize();
assert_eq!("ee8ff4e9be887297cf79348dc35dab56", &hash.to_hex());
```

## Performance

The AVX2 implementation in this crate is ported from the C implementation in libsodium. That
implementation was [originally written](https://github.com/sneves/blake2-avx2) by Samuel Neves
and [integrated into libsodium](https://github.com/jedisct1/libsodium/commit/0131a720826045e476e6dd6a8e7a1991f1d941aa)
by Frank Denis. All credit for performance goes to those authors.

The `benchmark_gig` binary in this crate allocates a gigabyte (10⁹) array and repeatedly hashes
it to measure throughput. A similar C program, `benches/bench_libsodium.c`, does the same thing
using libsodium's implementation of BLAKE2b. Here are the results from my laptop:

- Intel Core i5-8250U, Arch Linux, kernel version 4.17.13
- libsodium version 1.0.16, gcc 8.2.0, `gcc -O3 -lsodium benches/bench_libsodium.c` (via the
  helper script `benches/bench_libsodium.sh`)
- rustc 1.30.0-nightly (73c78734b 2018-08-05), `cargo +nightly run --release --bin benchmark_gig`

```table
               ╭────────────┬────────────╮
               │ portable   │ AVX2       │
╭──────────────┼────────────┼────────────┤
│ blake2b_simd │ 0.771 GB/s │ 1.005 GB/s │
│ libsodium    │ 0.743 GB/s │ 0.939 GB/s │
╰──────────────┴────────────┴────────────╯
```

The `b2sum` sub-crate is a clone of the `b2sum` utility from coreutils. The
`benches/bench_b2sum.py` script runs it against several coreutils hashes, on a 10 MB file of
random data. Here are the results from my laptop:

```table
╭───────────────────────────┬────────────╮
│ blake2b_simd b2sum --mmap │ 0.676 GB/s │
│ blake2b_simd b2sum        │ 0.649 GB/s │
│ coreutils sha1sum         │ 0.628 GB/s │
│ coreutils b2sum           │ 0.536 GB/s │
│ coreutils md5sum          │ 0.476 GB/s │
│ coreutils sha512sum       │ 0.464 GB/s │
╰───────────────────────────┴────────────╯
```
