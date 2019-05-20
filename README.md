# blake2b_simd [![Build Status](https://travis-ci.org/oconnor663/blake2b_simd.svg?branch=master)](https://travis-ci.org/oconnor663/blake2b_simd) [![docs.rs](https://docs.rs/blake2b_simd/badge.svg)](https://docs.rs/blake2b_simd)

[Repo](https://github.com/oconnor663/blake2b_simd) —
[Docs](https://docs.rs/blake2b_simd) —
[Crate](https://crates.io/crates/blake2b_simd)

An implementation of the BLAKE2b hash with:

- 100% stable Rust.
- A SIMD implementation based on Samuel Neves' [`blake2-avx2`]. This implementation is very
  fast. See the Performance section below.
- A portable, safe implementation for other platforms.
- Dynamic CPU feature detection. Binaries for x86 include SIMD implementations by default and
  use the fastest implementation the processor supports.
- All the features from the [the BLAKE2 spec], like adjustable length, keying, and associated
  data for tree hashing.
- A clone of the Coreutils `b2sum` command line utility, provided as a sub-crate. `b2sum`
  includes command line flags for all the BLAKE2 associated data features.
- `no_std` support. The `std` Cargo feature is on by default, for CPU feature detection and
  for implementing `std::io::Write`.
- The SIMD-friendly [BLAKE2bp] variant. This implementation is single-threaded, but it's faster
  than BLAKE2b, because it uses AVX2 more efficiently. It's available on the command line as
  `b2sum --blake2bp`.
- Support for computing multiple BLAKE2b hashes in parallel. See [`many::hash_many`] and
  [`many::update_many`]. These interfaces match the efficiency of BLAKE2bp but produce BLAKE2b
  hashes. They're a building block for the [Bao project].

## Example

```rust
use blake2b_simd::{blake2b, Params};

let expected = "ca002330e69d3e6b84a46a56a6533fd79d51d97a3bb7cad6c2ff43b354185d6d\
                c1e723fb3db4ae0737e120378424c714bb982d9dc5bbd7a0ab318240ddd18f8d";
let hash = blake2b(b"foo");
assert_eq!(expected, &hash.to_hex());

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

An example using the included `b2sum` command line utility:

```bash
$ cd b2sum
$ cargo build --release
    Finished release [optimized] target(s) in 0.04s
$ echo hi | ./target/release/b2sum --length 256
de9543b2ae1b2b87434a730727db17f5ac8b8c020b84a5cb8c5fbcc1423443ba  -
```

## Performance

To run small benchmarks yourself, run `cargo +nightly bench`. That suite
includes some comparison benchmarks with OpenSSL and libsodium. If you
don't have those installed on your machine, you can run
`cargo +nightly bench --no-default-features --features std` instead.

The `benches/bench_multiprocess` sub-crate runs various hash functions on
long inputs in memory and tries to average over many sources of
variability. Here are the results from my laptop for `cargo run --release`:

- Intel Core i5-8250U, Arch Linux, kernel version 5.0.13
- libsodium version 1.0.17
- OpenSSL version 1.1.1.b
- rustc 1.34.1

```table
╭─────────────────────────┬────────────╮
│ blake2b_simd many::hash │ 2.223 GB/s │
│ blake2b_simd BLAKE2bp   │ 2.211 GB/s │
│ sneves BLAKE2bp         │ 2.143 GB/s │
│ blake2b_simd BLAKE2b    │ 1.008 GB/s │
│ OpenSSL SHA-1           │ 0.971 GB/s │
│ sneves BLAKE2b          │ 0.949 GB/s │
│ libsodium BLAKE2b       │ 0.940 GB/s │
│ blake2b_simd portable   │ 0.812 GB/s │
│ OpenSSL SHA-512         │ 0.666 GB/s │
╰─────────────────────────┴────────────╯
```

The `benches/bench_b2sum.py` script benchmarks `b2sum` against several
Coreutils hashes, on a 1 GB file of random data. Here are the results from
my laptop:

```table
╭───────────────────────────────┬────────────╮
│ blake2b_simd b2sum --blake2bp │ 1.517 GB/s │
│ blake2b_simd b2sum            │ 0.820 GB/s │
│ coreutils sha1sum             │ 0.805 GB/s │
│ coreutils b2sum               │ 0.668 GB/s │
│ coreutils md5sum              │ 0.595 GB/s │
│ coreutils sha512sum           │ 0.593 GB/s │
╰───────────────────────────────┴────────────╯
```

[libsodium]: https://github.com/jedisct1/libsodium
[the BLAKE2 spec]: https://blake2.net/blake2.pdf
[`blake2-avx2`]: https://github.com/sneves/blake2-avx2
[included in libsodium]: https://github.com/jedisct1/libsodium/commit/0131a720826045e476e6dd6a8e7a1991f1d941aa
[BLAKE2bp]: https://docs.rs/blake2b_simd/latest/blake2b_simd/blake2bp/index.html
[`many::hash_many`]: https://docs.rs/blake2b_simd/latest/blake2b_simd/many/fn.hash_many.html
[`many::update_many`]: https://docs.rs/blake2b_simd/latest/blake2b_simd/many/fn.update_many.html
[Bao project]: https://github.com/oconnor663/bao
