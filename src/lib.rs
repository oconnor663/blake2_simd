//! [Repo](https://github.com/oconnor663/blake2b_simd) —
//! [Docs](https://docs.rs/blake2b_simd) —
//! [Crate](https://crates.io/crates/blake2b_simd)
//!
//! An implementation of the BLAKE2b hash with:
//!
//! - 100% stable Rust.
//! - An AVX2 implementation ported from [Samuel Neves' implementation]. This implementation is
//!   faster than any hash function provided by OpenSSL. See the Performance section below.
//! - A portable, safe implementation for other platforms.
//! - Dynamic CPU feature detection. Binaries for x86 include the AVX2 implementation by default
//!   and call it if the processor supports it at runtime.
//! - All the features from the [the BLAKE2 spec], like adjustable length, keying, and associated
//!   data for tree hashing.
//! - A clone of the Coreutils `b2sum` command line utility, provided as a sub-crate. `b2sum`
//!   includes command line flags for all the BLAKE2 associated data features.
//! - `no_std` support. The `std` Cargo feature is on by default, for CPU feature detection and
//!   for implementing `std::io::Write`.
//! - An implementation of the parallel [BLAKE2bp] variant. This implementation is single-threaded,
//!   but it's twice as fast as BLAKE2b, because it uses AVX2 more efficiently. It's available on
//!   the command line as `b2sum --blake2bp`.
//! - Support for computing multiple BLAKE2b hashes in parallel, matching the throughput of
//!   BLAKE2bp. See [`update4`] and [`finalize4`].
//!
//! # Example
//!
//! ```
//! use blake2b_simd::{blake2b, Params};
//!
//! let expected = "ca002330e69d3e6b84a46a56a6533fd79d51d97a3bb7cad6c2ff43b354185d6d\
//!                 c1e723fb3db4ae0737e120378424c714bb982d9dc5bbd7a0ab318240ddd18f8d";
//! let hash = blake2b(b"foo");
//! assert_eq!(expected, &hash.to_hex());
//!
//! let hash = Params::new()
//!     .hash_length(16)
//!     .key(b"The Magic Words are Squeamish Ossifrage")
//!     .personal(b"L. P. Waterhouse")
//!     .to_state()
//!     .update(b"foo")
//!     .update(b"bar")
//!     .update(b"baz")
//!     .finalize();
//! assert_eq!("ee8ff4e9be887297cf79348dc35dab56", &hash.to_hex());
//! ```
//!
//! An example using the included `b2sum` command line utility:
//!
//! ```bash
//! $ cd b2sum
//! $ cargo build --release
//!     Finished release [optimized] target(s) in 0.04s
//! $ echo hi | ./target/release/b2sum --length 256
//! de9543b2ae1b2b87434a730727db17f5ac8b8c020b84a5cb8c5fbcc1423443ba  -
//! ```
//!
//! # Performance
//!
//! The AVX2 implementation in this crate is a port of [Samuel Neves' implementation], which is
//! also [included in libsodium]. Most of the credit for performance goes to him. To run small
//! benchmarks yourself, first install OpenSSL and libsodium on your machine, then:
//!
//! ```sh
//! cd benches/cargo_bench
//! # Use --no-default-features if you're missing OpenSSL or libsodium.
//! cargo +nightly bench
//! ```
//!
//! The `benches/benchmark_gig` sub-crate allocates a 1 GB array and repeatedly hashes it to
//! measure throughput. A similar C program, `benches/bench_libsodium.c`, does the same thing using
//! libsodium's implementation of BLAKE2b. Here are the results from my laptop:
//!
//! - Intel Core i5-8250U, Arch Linux, kernel version 4.18.16
//! - libsodium version 1.0.16, gcc 8.2.1, `gcc -O3 -lsodium benches/bench_libsodium.c` (via the
//!   helper script `benches/bench_libsodium.sh`)
//! - rustc 1.31.0-nightly (f99911a4a 2018-10-23), `cargo +nightly run --release`
//!
//! ```table
//! ╭───────────────────────┬────────────╮
//! │ blake2b_simd BLAKE2bp │ 2.069 GB/s │
//! │ blake2b_simd update4  │ 2.057 GB/s │
//! │ blake2b_simd AVX2     │ 1.005 GB/s │
//! │ libsodium AVX2        │ 0.939 GB/s │
//! │ blake2b_simd portable │ 0.771 GB/s │
//! │ libsodium portable    │ 0.743 GB/s │
//! ╰───────────────────────┴────────────╯
//! ```
//!
//! The `benches/bench_b2sum.py` script benchmarks `b2sum` against several Coreutils hashes, on a
//! 1 GB file of random data. Here are the results from my laptop:
//!
//! ```table
//! ╭───────────────────────────────┬────────────╮
//! │ blake2b_simd b2sum --blake2bp │ 1.423 GB/s │
//! │ blake2b_simd b2sum            │ 0.810 GB/s │
//! │ coreutils sha1sum             │ 0.802 GB/s │
//! │ coreutils b2sum               │ 0.660 GB/s │
//! │ coreutils md5sum              │ 0.600 GB/s │
//! │ coreutils sha512sum           │ 0.593 GB/s │
//! ╰───────────────────────────────┴────────────╯
//! ```
//!
//! The `benches/count_cycles` sub-crate (`cargo +nightly run --release`) measures a long message
//! throughput of 1.8 cycles per byte for BLAKE2b, and 0.9 cycles per byte for BLAKE2bp and
//! [`update4`].
//!
//! [libsodium]: https://github.com/jedisct1/libsodium
//! [the BLAKE2 spec]: https://blake2.net/blake2.pdf
//! [Samuel Neves' implementation]: https://github.com/sneves/blake2-avx2
//! [included in libsodium]: https://github.com/jedisct1/libsodium/commit/0131a720826045e476e6dd6a8e7a1991f1d941aa
//! [BLAKE2bp]: https://docs.rs/blake2b_simd/latest/blake2b_simd/blake2bp/index.html
//! [`update4`]: https://docs.rs/blake2b_simd/latest/blake2b_simd/fn.update4.html
//! [`finalize4`]: https://docs.rs/blake2b_simd/latest/blake2b_simd/fn.finalize4.html
// Note that the links above wind up in README.md, so they need to be absolute.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate core;

#[macro_use]
extern crate arrayref;
extern crate arrayvec;
extern crate byteorder;
extern crate constant_time_eq;

use byteorder::{ByteOrder, LittleEndian};
use core::cmp;
use core::fmt;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2;
mod portable;

pub mod blake2bp;

#[cfg(test)]
mod test;

/// The max hash length.
pub const OUTBYTES: usize = 64;
/// The max key length.
pub const KEYBYTES: usize = 64;
/// The max salt length.
pub const SALTBYTES: usize = 16;
/// The max personalization length.
pub const PERSONALBYTES: usize = 16;
/// The number input bytes passed to each call to the compression function. Small benchmarks need
/// to use an even multiple of `BLOCKBYTES`, or else their apparent throughput will be low.
pub const BLOCKBYTES: usize = 128;

const IV: [u64; 8] = [
    0x6A09E667F3BCC908,
    0xBB67AE8584CAA73B,
    0x3C6EF372FE94F82B,
    0xA54FF53A5F1D36F1,
    0x510E527FADE682D1,
    0x9B05688C2B3E6C1F,
    0x1F83D9ABFB41BD6B,
    0x5BE0CD19137E2179,
];

const SIGMA: [[u8; 16]; 12] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
];

// Safety note: The compression interface is unsafe in general, even though the portable
// implementation is safe, because calling the AVX2 implementation on a platform that doesn't
// support AVX2 is undefined behavior.
type CompressFn = unsafe fn(&mut StateWords, &Block, count: u128, lastblock: u64, lastnode: u64);
type Compress4Fn = unsafe fn(
    state0: &mut StateWords,
    state1: &mut StateWords,
    state2: &mut StateWords,
    state3: &mut StateWords,
    block0: &Block,
    block1: &Block,
    block2: &Block,
    block3: &Block,
    count0: u128,
    count1: u128,
    count2: u128,
    count3: u128,
    lastblock0: u64,
    lastblock1: u64,
    lastblock2: u64,
    lastblock3: u64,
    lastnode0: u64,
    lastnode1: u64,
    lastnode2: u64,
    lastnode3: u64,
);
type StateWords = [u64; 8];
type Block = [u8; BLOCKBYTES];
type HexString = arrayvec::ArrayString<[u8; 2 * OUTBYTES]>;

/// Compute the BLAKE2b hash of a slice of bytes, using default parameters.
///
/// # Example
///
/// ```
/// # use blake2b_simd::{blake2b, Params};
/// let expected = "ca002330e69d3e6b84a46a56a6533fd79d51d97a3bb7cad6c2ff43b354185d6d\
///                 c1e723fb3db4ae0737e120378424c714bb982d9dc5bbd7a0ab318240ddd18f8d";
/// let hash = blake2b(b"foo");
/// assert_eq!(&hash.to_hex(), expected);
/// ```
pub fn blake2b(input: &[u8]) -> Hash {
    State::new().update(input).finalize()
}

/// A parameter builder for `State` that exposes all the various BLAKE2 features.
///
/// Apart from `hash_length`, which controls the length of the final `Hash`, all of these
/// parameters are just associated data that gets mixed with the input. For all the details, see
/// [the BLAKE2 spec](https://blake2.net/blake2.pdf).
///
/// Several of the parameters have a valid range defined in the spec and documented below. Trying
/// to set an invalid parameter will panic.
///
/// # Example
///
/// ```
/// # use blake2b_simd::Params;
/// let mut state = Params::new().hash_length(32).to_state();
/// ```
#[derive(Clone)]
pub struct Params {
    hash_length: u8,
    key_length: u8,
    key: [u8; KEYBYTES],
    salt: [u8; SALTBYTES],
    personal: [u8; PERSONALBYTES],
    fanout: u8,
    max_depth: u8,
    max_leaf_length: u32,
    node_offset: u64,
    node_depth: u8,
    inner_hash_length: u8,
    last_node: bool,
}

impl Params {
    /// Equivalent to `Params::default()`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct a `State` object based on these parameters.
    pub fn to_state(&self) -> State {
        State::with_params(self)
    }

    /// Set the length of the final hash in bytes, from 1 to `OUTBYTES` (64). Apart from
    /// controlling the length of the final `Hash`, this is also associated data, and changing it
    /// will result in a totally different hash.
    ///
    /// Note that the `b2sum` command line utility expects the `--length` flag in bits rather than
    /// bytes, for compatibility with the `b2sum` implementation in coreutils. The BLAKE2 standard
    /// defines the parameter as a count of bytes, however, and this method follows the standard.
    pub fn hash_length(&mut self, length: usize) -> &mut Self {
        assert!(
            1 <= length && length <= OUTBYTES,
            "Bad hash length: {}",
            length
        );
        self.hash_length = length as u8;
        self
    }

    /// Use a secret key, so that BLAKE2b acts as a MAC. The maximum key length is `KEYBYTES` (64).
    /// An empty key is equivalent to having no key at all.
    pub fn key(&mut self, key: &[u8]) -> &mut Self {
        assert!(key.len() <= KEYBYTES, "Bad key length: {}", key.len());
        self.key_length = key.len() as u8;
        self.key = [0; KEYBYTES];
        self.key[..key.len()].copy_from_slice(key);
        self
    }

    /// At most `SALTBYTES` (16). Shorter salts are padded with null bytes. An empty salt is
    /// equivalent to having no salt at all.
    pub fn salt(&mut self, salt: &[u8]) -> &mut Self {
        assert!(salt.len() <= SALTBYTES, "Bad salt length: {}", salt.len());
        self.salt = [0; SALTBYTES];
        self.salt[..salt.len()].copy_from_slice(salt);
        self
    }

    /// At most `PERSONALBYTES` (16). Shorter personalizations are padded with null bytes. An empty
    /// personalization is equivalent to having no personalization at all.
    pub fn personal(&mut self, personalization: &[u8]) -> &mut Self {
        assert!(
            personalization.len() <= PERSONALBYTES,
            "Bad personalization length: {}",
            personalization.len()
        );
        self.personal = [0; PERSONALBYTES];
        self.personal[..personalization.len()].copy_from_slice(personalization);
        self
    }

    /// From 0 (meaning unlimited) to 255. The default is 1 (meaning sequential).
    pub fn fanout(&mut self, fanout: u8) -> &mut Self {
        self.fanout = fanout;
        self
    }

    /// From 1 (the default, meaning sequential) to 255 (meaning unlimited).
    pub fn max_depth(&mut self, depth: u8) -> &mut Self {
        assert!(depth != 0, "Bad max depth: {}", depth);
        self.max_depth = depth;
        self
    }

    /// From 0 (the default, meaning unlimited or sequential) to `2^32 - 1`.
    pub fn max_leaf_length(&mut self, length: u32) -> &mut Self {
        self.max_leaf_length = length;
        self
    }

    /// From 0 (the default, meaning first, leftmost, leaf, or sequential) to `2^64 - 1`.
    pub fn node_offset(&mut self, offset: u64) -> &mut Self {
        self.node_offset = offset;
        self
    }

    /// From 0 (the default, meaning leaf or sequential) to 255.
    pub fn node_depth(&mut self, depth: u8) -> &mut Self {
        self.node_depth = depth;
        self
    }

    /// From 0 (the default, meaning sequential) to `OUTBYTES` (64).
    ///
    /// Note that the `b2sum` command line utility expects the `--inner-hash-length` flag in bits
    /// rather than bytes, to stay consistent with `--length`. The BLAKE2 standard defines the
    /// parameter as a count of bytes, however, and this method follows the standard.
    pub fn inner_hash_length(&mut self, length: usize) -> &mut Self {
        assert!(length <= OUTBYTES, "Bad inner hash length: {}", length);
        self.inner_hash_length = length as u8;
        self
    }

    /// Indicates the rightmost node in a row. This can also be changed on the `State` object
    /// itself, potentially after hashing has begun. See [`State::set_last_node`].
    ///
    /// [`State::set_last_node`]: struct.State.html#method.set_last_node
    pub fn last_node(&mut self, last_node: bool) -> &mut Self {
        self.last_node = last_node;
        self
    }
}

impl Default for Params {
    fn default() -> Self {
        Self {
            hash_length: OUTBYTES as u8,
            key_length: 0,
            key: [0; KEYBYTES],
            salt: [0; SALTBYTES],
            personal: [0; PERSONALBYTES],
            // NOTE: fanout and max_depth don't default to zero!
            fanout: 1,
            max_depth: 1,
            max_leaf_length: 0,
            node_offset: 0,
            node_depth: 0,
            inner_hash_length: 0,
            last_node: false,
        }
    }
}

impl fmt::Debug for Params {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Params {{ hash_length: {}, key_length: {}, salt: {:?}, personal: {:?}, fanout: {}, \
             max_depth: {}, max_leaf_length: {}, node_offset: {}, node_depth: {}, \
             inner_hash_length: {}, last_node: {} }}",
            self.hash_length,
            // NB: Don't print the key itself. Debug shouldn't leak secrets.
            self.key_length,
            &self.salt,
            &self.personal,
            self.fanout,
            self.max_depth,
            self.max_leaf_length,
            self.node_offset,
            self.node_depth,
            self.inner_hash_length,
            self.last_node,
        )
    }
}

/// An incremental hasher for BLAKE2b.
///
/// # Example
///
/// ```
/// use blake2b_simd::{State, blake2b};
///
/// let mut state = blake2b_simd::State::new();
///
/// state.update(b"foo");
/// assert_eq!(blake2b(b"foo"), state.finalize());
///
/// state.update(b"bar");
/// assert_eq!(blake2b(b"foobar"), state.finalize());
/// ```
#[derive(Clone)]
pub struct State {
    h: StateWords,
    buf: Block,
    buflen: u8,
    count: u128,
    compress_fn: CompressFn,
    last_node: bool,
    hash_length: u8,
}

impl State {
    /// Equivalent to `State::default()` or `Params::default().to_state()`.
    pub fn new() -> Self {
        Self::with_params(&Params::default())
    }

    fn with_params(params: &Params) -> Self {
        let (salt_left, salt_right) = array_refs!(&params.salt, 8, 8);
        let (personal_left, personal_right) = array_refs!(&params.personal, 8, 8);
        let mut state = Self {
            h: [
                IV[0]
                    ^ params.hash_length as u64
                    ^ (params.key_length as u64) << 8
                    ^ (params.fanout as u64) << 16
                    ^ (params.max_depth as u64) << 24
                    ^ (params.max_leaf_length as u64) << 32,
                IV[1] ^ params.node_offset,
                IV[2] ^ params.node_depth as u64 ^ (params.inner_hash_length as u64) << 8,
                IV[3],
                IV[4] ^ LittleEndian::read_u64(salt_left),
                IV[5] ^ LittleEndian::read_u64(salt_right),
                IV[6] ^ LittleEndian::read_u64(personal_left),
                IV[7] ^ LittleEndian::read_u64(personal_right),
            ],
            compress_fn: default_compress_impl().0,
            buf: [0; BLOCKBYTES],
            buflen: 0,
            count: 0,
            last_node: params.last_node,
            hash_length: params.hash_length,
        };
        if params.key_length > 0 {
            let mut key_block = [0; BLOCKBYTES];
            key_block[..KEYBYTES].copy_from_slice(&params.key);
            state.update(&key_block);
        }
        state
    }

    fn fill_buf(&mut self, input: &mut &[u8]) {
        let take = cmp::min(BLOCKBYTES - self.buflen as usize, input.len());
        self.buf[self.buflen as usize..self.buflen as usize + take].copy_from_slice(&input[..take]);
        self.buflen += take as u8;
        self.count += take as u128;
        *input = &input[take..];
    }

    // If the state already has some input in its buffer, try to fill the buffer and perform a
    // compression. However, only do the compression if there's more input coming, otherwise it
    // will give the wrong hash it the caller finalizes immediately after.
    fn compress_buffer_if_possible(&mut self, input: &mut &[u8]) {
        if self.buflen > 0 {
            self.fill_buf(input);
            if !input.is_empty() {
                unsafe {
                    (self.compress_fn)(&mut self.h, &self.buf, self.count, 0, 0);
                }
                self.buflen = 0;
            }
        }
    }

    /// Add input to the hash. You can call `update` any number of times.
    pub fn update(&mut self, mut input: &[u8]) -> &mut Self {
        // If we have a partial buffer, try to complete it.
        self.compress_buffer_if_possible(&mut input);
        // While there's more than a block of input left (which also means we cleared the buffer
        // above), compress blocks directly without copying.
        while input.len() > BLOCKBYTES {
            self.count += BLOCKBYTES as u128;
            let block = array_ref!(input, 0, BLOCKBYTES);
            unsafe {
                (self.compress_fn)(&mut self.h, block, self.count, 0, 0);
            }
            input = &input[BLOCKBYTES..];
        }
        // Buffer any remaining input, to be either compressed or finalized in a subsequent call.
        // Note that this represents some copying overhead, which in theory we could avoid in
        // all-at-once setting. A function hardcoded for exactly BLOCKSIZE input bytes is about 10%
        // faster than using this implementation for the same input. But non-multiple sizes still
        // require copying, and the savings disappear into the noise for any larger multiple. Any
        // caller so concerned with performance that they're shaping their hash inputs down to the
        // single byte, should just call the compression function directly.
        self.fill_buf(&mut input);
        self
    }

    /// Finalize the state and return a `Hash`. This method is idempotent, and calling it multiple
    /// times will give the same result. It's also possible to `update` with more input in between.
    pub fn finalize(&mut self) -> Hash {
        for i in self.buflen as usize..BLOCKBYTES {
            self.buf[i] = 0;
        }
        let last_node = if self.last_node { !0 } else { 0 };
        let mut h_copy = self.h;
        unsafe {
            (self.compress_fn)(&mut h_copy, &self.buf, self.count, !0, last_node);
        }
        Hash {
            bytes: state_words_to_bytes(&h_copy),
            len: self.hash_length,
        }
    }

    /// Set a flag indicating that this is the last node of its level in a tree hash. This is
    /// equivalent to [`Params::last_node`], except that it can be set at any time before calling
    /// `finalize`. That allows callers to begin hashing a node without knowing ahead of time
    /// whether it's the last in its level. For more details about the intended use of this flag
    /// [the BLAKE2 spec].
    ///
    /// [`Params::last_node`]: struct.Params.html#method.last_node
    /// [the BLAKE2 spec]: https://blake2.net/blake2.pdf
    pub fn set_last_node(&mut self, last_node: bool) -> &mut Self {
        self.last_node = last_node;
        self
    }

    /// Return the total number of bytes input so far.
    pub fn count(&self) -> u128 {
        self.count
    }
}

fn state_words_to_bytes(state_words: &StateWords) -> [u8; OUTBYTES] {
    let mut bytes = [0; OUTBYTES];
    {
        let refs = mut_array_refs!(&mut bytes, 8, 8, 8, 8, 8, 8, 8, 8);
        LittleEndian::write_u64(refs.0, state_words[0]);
        LittleEndian::write_u64(refs.1, state_words[1]);
        LittleEndian::write_u64(refs.2, state_words[2]);
        LittleEndian::write_u64(refs.3, state_words[3]);
        LittleEndian::write_u64(refs.4, state_words[4]);
        LittleEndian::write_u64(refs.5, state_words[5]);
        LittleEndian::write_u64(refs.6, state_words[6]);
        LittleEndian::write_u64(refs.7, state_words[7]);
    }
    bytes
}

#[cfg(feature = "std")]
impl std::io::Write for State {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.update(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl fmt::Debug for State {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // NB: Don't print the words. Leaking them would allow length extension.
        write!(
            f,
            "State {{ count: {}, hash_length: {}, last_node: {} }}",
            self.count, self.hash_length, self.last_node,
        )
    }
}

impl Default for State {
    fn default() -> Self {
        Self::with_params(&Params::default())
    }
}

/// A finalized BLAKE2 hash, with constant-time equality.
#[derive(Clone, Copy)]
pub struct Hash {
    bytes: [u8; OUTBYTES],
    len: u8,
}

impl Hash {
    /// Convert the hash to a byte slice. Note that if you're using BLAKE2b as a MAC, you need
    /// constant time equality, which `&[u8]` doesn't provide.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len as usize]
    }

    /// Convert the hash to a lowercase hexadecimal
    /// [`ArrayString`](https://docs.rs/arrayvec/0.4/arrayvec/struct.ArrayString.html).
    pub fn to_hex(&self) -> HexString {
        bytes_to_hex(self.as_bytes())
    }
}

fn bytes_to_hex(bytes: &[u8]) -> HexString {
    let mut s = arrayvec::ArrayString::new();
    let table = b"0123456789abcdef";
    for &b in bytes {
        s.push(table[(b >> 4) as usize] as char);
        s.push(table[(b & 0xf) as usize] as char);
    }
    s
}

/// This implementation is constant time, if the two hashes are the same length.
impl PartialEq for Hash {
    fn eq(&self, other: &Hash) -> bool {
        constant_time_eq::constant_time_eq(&self.as_bytes(), &other.as_bytes())
    }
}

/// This implementation is constant time, if the slice is the same length as the hash.
impl PartialEq<[u8]> for Hash {
    fn eq(&self, other: &[u8]) -> bool {
        constant_time_eq::constant_time_eq(&self.as_bytes(), other)
    }
}

impl Eq for Hash {}

impl AsRef<[u8]> for Hash {
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl fmt::Debug for Hash {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Hash(0x{})", self.to_hex())
    }
}

// Safety: The unsafe blocks above rely on this function to never return avx2::compress except on
// platforms where it's safe to call.
#[allow(unreachable_code)]
fn default_compress_impl() -> (CompressFn, Compress4Fn) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // If AVX2 is enabled at the top level for the whole build (using something like
        // RUSTFLAGS="-C target-cpu=native"), return the AVX2 implementation without doing dynamic
        // feature detection. This isn't common, but it's the only way to use AVX2 with no_std, at
        // least until more features get stabilized in the future.
        #[cfg(target_feature = "avx2")]
        {
            return (avx2::compress, avx2::compress4);
        }
        // Do dynamic feature detection at runtime, and use AVX2 if the current CPU supports it.
        // This is what the default build does. Note that no_std doesn't currently support dynamic
        // detection.
        #[cfg(feature = "std")]
        {
            if is_x86_feature_detected!("avx2") {
                return (avx2::compress, avx2::compress4);
            }
        }
    }
    // On other platforms (non-x86 or pre-AVX2) use the portable implementation.
    (portable::compress, portable::compress4)
}

/// Update four `State` objects at the same time.
///
/// This implementation isn't multithreaded. Rather, it uses AVX2 (if available) to hash the four
/// inputs in parallel on a single thread, which is more efficient than hashing them one at a time.
/// It uses the same underlying machinery as BLAKE2bp, and like BLAKE2bp is has about twice the
/// overall throughput of regular BLAKE2b.
///
/// Note that you can benefit from this implementation even if you're already using multiple
/// threads. If you have enough separate inputs, hashing four of them per thread raises the
/// throughput of each thread. With many threads in practice, this seems to be about a 50% increase
/// rather than the 100% increase we see in single thread benchmarks, possibly because of
/// interactions with Turbo Boost and Hyper-Threading on Intel processors.
///
/// `update4` can only operate in parallel as long as all four inputs still have bytes left. Once
/// one of the inputs is exhausted, it falls back to regular serial hashing for the rest. To get
/// the best throughput, use inputs that are roughly the same length.
///
/// Unlike BLAKE2bp, which is specifically designed to have four lanes, parallel BLAKE2b isn't tied
/// to any particular number of lanes. When the AVX-512 instruction set becomes more widespread,
/// for example, we could add an `update8` implementation to take full advantage of it. We could
/// also add an SSE-based `update2` implementation to support older machines.
///
/// # Example
///
/// ```
/// use blake2b_simd::{blake2b, finalize4, update4, State};
///
/// let mut state0 = State::new();
/// let mut state1 = State::new();
/// let mut state2 = State::new();
/// let mut state3 = State::new();
///
/// update4(
///     &mut state0,
///     &mut state1,
///     &mut state2,
///     &mut state3,
///     b"foo",
///     b"bar",
///     b"baz",
///     b"bing",
/// );
///
/// let parallel_hashes = finalize4(&mut state0, &mut state1, &mut state2, &mut state3);
///
/// let serial_hashes = [
///     blake2b(b"foo"),
///     blake2b(b"bar"),
///     blake2b(b"baz"),
///     blake2b(b"bing"),
/// ];
/// assert_eq!(serial_hashes, parallel_hashes);
/// ```
///
/// [`update`]: struct.State.html#method.update
pub fn update4(
    state0: &mut State,
    state1: &mut State,
    state2: &mut State,
    state3: &mut State,
    mut input0: &[u8],
    mut input1: &[u8],
    mut input2: &[u8],
    mut input3: &[u8],
) {
    // First we need to make sure all the buffers are clear.
    state0.compress_buffer_if_possible(&mut input0);
    state1.compress_buffer_if_possible(&mut input1);
    state2.compress_buffer_if_possible(&mut input2);
    state3.compress_buffer_if_possible(&mut input3);
    // Now, as long as all of the states have more than a block of input coming (so that we know we
    // don't need to finalize any of them), compress in parallel directly into their state words.
    let (_, compress4_fn) = default_compress_impl();
    while input0.len() > BLOCKBYTES
        && input1.len() > BLOCKBYTES
        && input2.len() > BLOCKBYTES
        && input3.len() > BLOCKBYTES
    {
        state0.count += BLOCKBYTES as u128;
        state1.count += BLOCKBYTES as u128;
        state2.count += BLOCKBYTES as u128;
        state3.count += BLOCKBYTES as u128;
        unsafe {
            compress4_fn(
                &mut state0.h,
                &mut state1.h,
                &mut state2.h,
                &mut state3.h,
                array_ref!(input0, 0, BLOCKBYTES),
                array_ref!(input1, 0, BLOCKBYTES),
                array_ref!(input2, 0, BLOCKBYTES),
                array_ref!(input3, 0, BLOCKBYTES),
                state0.count as u128,
                state1.count as u128,
                state2.count as u128,
                state3.count as u128,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            );
        }
        input0 = &input0[BLOCKBYTES..];
        input1 = &input1[BLOCKBYTES..];
        input2 = &input2[BLOCKBYTES..];
        input3 = &input3[BLOCKBYTES..];
    }
    // Finally, if there's any remaining input, add it into the state the usual way. Note that if
    // one of the inputs is short, this could actually be more work than the loop above. The caller
    // should hopefully arrange for that not to happen.
    state0.update(input0);
    state1.update(input1);
    state2.update(input2);
    state3.update(input3);
}

/// Finalize four `State` objects at the same time.
///
/// This is the counterpart to [`update4`]. Like the regular [`finalize`], this is idempotent.
/// Calling it multiple times on the same states will produce the same output, and it's possible to
/// add more input in between calls.
///
/// # Example
///
/// ```
/// use blake2b_simd::{blake2b, finalize4, update4, State};
///
/// let mut state0 = State::new();
/// let mut state1 = State::new();
/// let mut state2 = State::new();
/// let mut state3 = State::new();
///
/// update4(
///     &mut state0,
///     &mut state1,
///     &mut state2,
///     &mut state3,
///     b"foo",
///     b"bar",
///     b"baz",
///     b"bing",
/// );
///
/// let parallel_hashes = finalize4(&mut state0, &mut state1, &mut state2, &mut state3);
///
/// let serial_hashes = [
///     blake2b(b"foo"),
///     blake2b(b"bar"),
///     blake2b(b"baz"),
///     blake2b(b"bing"),
/// ];
/// assert_eq!(serial_hashes, parallel_hashes);
/// ```
///
/// [`update4`]: fn.update4.html
/// [`finalize`]: struct.State.html#method.finalize
pub fn finalize4(
    state0: &mut State,
    state1: &mut State,
    state2: &mut State,
    state3: &mut State,
) -> [Hash; 4] {
    // Zero out the buffer tails, which might contain bytes from previous blocks.
    for i in state0.buflen as usize..BLOCKBYTES {
        state0.buf[i] = 0;
    }
    for i in state1.buflen as usize..BLOCKBYTES {
        state1.buf[i] = 0;
    }
    for i in state2.buflen as usize..BLOCKBYTES {
        state2.buf[i] = 0;
    }
    for i in state3.buflen as usize..BLOCKBYTES {
        state3.buf[i] = 0;
    }
    // Translate the last node flag of each state into the u64 that BLAKE2 uses.
    let last_node0: u64 = if state0.last_node { !0 } else { 0 };
    let last_node1: u64 = if state1.last_node { !0 } else { 0 };
    let last_node2: u64 = if state2.last_node { !0 } else { 0 };
    let last_node3: u64 = if state3.last_node { !0 } else { 0 };
    // Make copies of all the state words. This step is what makes finalize idempotent.
    let mut h_copy0 = state0.h;
    let mut h_copy1 = state1.h;
    let mut h_copy2 = state2.h;
    let mut h_copy3 = state3.h;
    // Do the final parallel compression step.
    let (_, compress4_fn) = default_compress_impl();
    unsafe {
        compress4_fn(
            &mut h_copy0,
            &mut h_copy1,
            &mut h_copy2,
            &mut h_copy3,
            &state0.buf,
            &state1.buf,
            &state2.buf,
            &state3.buf,
            state0.count as u128,
            state1.count as u128,
            state2.count as u128,
            state3.count as u128,
            !0,
            !0,
            !0,
            !0,
            last_node0,
            last_node1,
            last_node2,
            last_node3,
        );
    }
    // Extract the resulting hashes.
    [
        Hash {
            bytes: state_words_to_bytes(&h_copy0),
            len: state0.hash_length,
        },
        Hash {
            bytes: state_words_to_bytes(&h_copy1),
            len: state1.hash_length,
        },
        Hash {
            bytes: state_words_to_bytes(&h_copy2),
            len: state2.hash_length,
        },
        Hash {
            bytes: state_words_to_bytes(&h_copy3),
            len: state3.hash_length,
        },
    ]
}

// This module is pub for internal benchmarks only. Please don't use it.
#[doc(hidden)]
pub mod benchmarks {
    pub use crate::portable::compress as compress_portable;
    pub use crate::portable::compress4 as compress4_portable;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub use crate::avx2::compress as compress_avx2;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub use crate::avx2::compress4 as compress4_avx2;

    // Safety: The portable implementation should be safe to call on any platform.
    pub fn force_portable(state: &mut crate::State) {
        state.compress_fn = compress_portable;
    }
    pub fn force_portable_blake2bp(state: &mut crate::blake2bp::State) {
        crate::blake2bp::force_portable(state);
    }
}
