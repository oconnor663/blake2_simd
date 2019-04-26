# blake2b_simd [![Build Status](https://travis-ci.org/oconnor663/blake2b_simd.svg?branch=master)](https://travis-ci.org/oconnor663/blake2b_simd) [![docs.rs](https://docs.rs/blake2b_simd/badge.svg)](https://docs.rs/blake2b_simd)

[Repo](https://github.com/oconnor663/blake2b_simd) —
[Docs](https://docs.rs/blake2b_simd) —
[Crate](https://crates.io/crates/blake2b_simd)

An implementation of the BLAKE2b hash with:

- 100% stable Rust.
- An AVX2 implementation ported from [Samuel Neves' implementation]. This implementation is
  faster than any hash function provided by OpenSSL. See the Performance section below.
- A portable, safe implementation for other platforms.
- Dynamic CPU feature detection. Binaries for x86 include the AVX2 implementation by default
  and call it if the processor supports it at runtime.
- All the features from the [the BLAKE2 spec], like adjustable length, keying, and associated
  data for tree hashing.
- A clone of the Coreutils `b2sum` command line utility, provided as a sub-crate. `b2sum`
  includes command line flags for all the BLAKE2 associated data features.
- `no_std` support. The `std` Cargo feature is on by default, for CPU feature detection and
  for implementing `std::io::Write`.
- An implementation of the parallel [BLAKE2bp] variant. This implementation is single-threaded,
  but it's twice as fast as BLAKE2b, because it uses AVX2 more efficiently. It's available on
  the command line as `b2sum --blake2bp`.
- Support for computing multiple BLAKE2b hashes in parallel, matching the throughput of
  BLAKE2bp. See [`update4`] and [`finalize4`].

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

The AVX2 implementation in this crate is a port of [Samuel Neves' implementation], which is
also [included in libsodium]. Most of the credit for performance goes to him. To run small
benchmarks yourself, first install OpenSSL and libsodium on your machine, then:

```sh
cd benches/cargo_bench
# Use --no-default-features if you're missing OpenSSL or libsodium.
cargo +nightly bench
```

The `benches/benchmark_gig` sub-crate allocates a 1 GB array and repeatedly hashes it to
measure throughput. A similar C program, `benches/bench_libsodium.c`, does the same thing using
libsodium's implementation of BLAKE2b. Here are the results from my laptop:

- Intel Core i5-8250U, Arch Linux, kernel version 4.18.16
- libsodium version 1.0.16, gcc 8.2.1, `gcc -O3 -lsodium benches/bench_libsodium.c` (via the
  helper script `benches/bench_libsodium.sh`)
- rustc 1.31.0-nightly (f99911a4a 2018-10-23), `cargo +nightly run --release`

```table
╭───────────────────────┬────────────╮
│ blake2b_simd BLAKE2bp │ 2.069 GB/s │
│ blake2b_simd update4  │ 2.057 GB/s │
│ blake2b_simd AVX2     │ 1.005 GB/s │
│ libsodium AVX2        │ 0.939 GB/s │
│ blake2b_simd portable │ 0.771 GB/s │
│ libsodium portable    │ 0.743 GB/s │
╰───────────────────────┴────────────╯
```

The `benches/bench_b2sum.py` script benchmarks `b2sum` against several Coreutils hashes, on a
1 GB file of random data. Here are the results from my laptop:

```table
╭───────────────────────────────┬────────────╮
│ blake2b_simd b2sum --blake2bp │ 1.423 GB/s │
│ blake2b_simd b2sum            │ 0.810 GB/s │
│ coreutils sha1sum             │ 0.802 GB/s │
│ coreutils b2sum               │ 0.660 GB/s │
│ coreutils md5sum              │ 0.600 GB/s │
│ coreutils sha512sum           │ 0.593 GB/s │
╰───────────────────────────────┴────────────╯
```

The `benches/count_cycles` sub-crate (`cargo +nightly run --release`) measures a long message
throughput of 1.8 cycles per byte for BLAKE2b, and 0.9 cycles per byte for BLAKE2bp and
[`update4`].

[libsodium]: https://github.com/jedisct1/libsodium
[the BLAKE2 spec]: https://blake2.net/blake2.pdf
[Samuel Neves' implementation]: https://github.com/sneves/blake2-avx2
[included in libsodium]: https://github.com/jedisct1/libsodium/commit/0131a720826045e476e6dd6a8e7a1991f1d941aa
[BLAKE2bp]: https://docs.rs/blake2b_simd/latest/blake2b_simd/blake2bp/index.html
[`update4`]: https://docs.rs/blake2b_simd/latest/blake2b_simd/fn.update4.html
[`finalize4`]: https://docs.rs/blake2b_simd/latest/blake2b_simd/fn.finalize4.html
