//! [Repo](https://github.com/oconnor663/blake2b_simd) —
//! [Docs](https://docs.rs/blake2b_simd) —
//! [Crate](https://crates.io/crates/blake2b_simd)
//!
//! An implementation of the BLAKE2b hash with:
//!
//! - 100% stable Rust.
//! - An AVX2 implementation ported from [libsodium](https://github.com/jedisct1/libsodium). This
//!   implementation is faster than libsodium's, and faster than any hash function provided by
//!   OpenSSL. See the Performance section below.
//! - A portable, safe implementation for other platforms.
//! - Dynamic CPU feature detection. Binaries for x86 include the AVX2 implementation by default
//!   and call it if the processor supports it at runtime.
//! - All the features from the [the BLAKE2 spec](https://blake2.net/blake2.pdf), like adjustable
//!   length, keying, and associated data for tree hashing.
//! - A clone of the Coreutils `b2sum` command line utility, provided as a sub-crate. `b2sum`
//!   includes command line flags for all the BLAKE2 associated data features.
//! - `no_std` support. The `std` Cargo feature is on by default, for CPU feature detection and
//!   for implementing `std::io::Write`.
//! - An implementation of the multithreaded BLAKE2bp variant. Enable it with `blake2bp` Cargo
//!   feature.
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
//! The AVX2 implementation in this crate is ported from the C implementation in libsodium. That
//! implementation was [originally written](https://github.com/sneves/blake2-avx2) by Samuel Neves
//! and [integrated into libsodium](https://github.com/jedisct1/libsodium/commit/0131a720826045e476e6dd6a8e7a1991f1d941aa)
//! by Frank Denis. All credit for performance goes to those authors.
//!
//! To run small benchmarks yourself, first install OpenSSL and libsodium on your machine, then:
//!
//! ```sh
//! cd benches/cargo_bench
//! # Use --no-default-features if you're missing OpenSSL or libsodium.
//! cargo +nightly bench
//! ```
//!
//! The `benches/benchmark_gig` sub-crate allocates a gigabyte (10⁹) array and repeatedly hashes it
//! to measure throughput. A similar C program, `benches/bench_libsodium.c`, does the same thing
//! using libsodium's implementation of BLAKE2b. Here are the results from my laptop:
//!
//! - Intel Core i5-8250U, Arch Linux, kernel version 4.17.13
//! - libsodium version 1.0.16, gcc 8.2.0, `gcc -O3 -lsodium benches/bench_libsodium.c` (via the
//!   helper script `benches/bench_libsodium.sh`)
//! - rustc 1.30.0-nightly (73c78734b 2018-08-05), `cargo +nightly run --release`
//!
//! ```table
//!                ╭────────────┬────────────╮
//!                │ portable   │ AVX2       │
//! ╭──────────────┼────────────┼────────────┤
//! │ blake2b_simd │ 0.771 GB/s │ 1.005 GB/s │
//! │ libsodium    │ 0.743 GB/s │ 0.939 GB/s │
//! ╰──────────────┴────────────┴────────────╯
//! ```
//!
//! The `benches/bench_b2sum.py` script benchmarks `b2sum` against several Coreutils hashes, on a
//! 10 MB file of random data. Here are the results from my laptop:
//!
//! ```table
//! ╭───────────────────────────┬────────────╮
//! │ blake2b_simd b2sum --mmap │ 0.676 GB/s │
//! │ blake2b_simd b2sum        │ 0.649 GB/s │
//! │ coreutils sha1sum         │ 0.628 GB/s │
//! │ coreutils b2sum           │ 0.536 GB/s │
//! │ coreutils md5sum          │ 0.476 GB/s │
//! │ coreutils sha512sum       │ 0.464 GB/s │
//! ╰───────────────────────────┴────────────╯
//! ```
//!
//! The `benches/count_cycles` sub-crate (`cargo +nightly run --release`) measures a peak
//! throughput of 1.8 cycles per byte.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate core;

#[macro_use]
extern crate arrayref;
extern crate arrayvec;
extern crate byteorder;
extern crate constant_time_eq;

use arrayvec::ArrayString;
use byteorder::{ByteOrder, LittleEndian};
use core::cmp;
use core::fmt;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2;
mod portable;

#[cfg(feature = "blake2bp")]
mod blake2bp;
#[cfg(feature = "blake2bp")]
pub use blake2bp::blake2bp;

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

// Safety note: The compression interface is unsafe in general, even though the portable
// implementation is safe, because calling the AVX2 implementation on a platform that doesn't
// support AVX2 is undefined behavior.
type CompressFn = unsafe fn(&mut StateWords, &Block, count: u128, lastblock: u64, lastnode: u64);
type StateWords = [u64; 8];
type Block = [u8; BLOCKBYTES];

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
    pub(crate) hash_length: u8, // visible to blake2bp
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

    /// Set the length of the final hash, from 1 to `OUTBYTES` (64). Apart from controlling the
    /// length of the final `Hash`, this is also associated data, and changing it will result in a
    /// totally different hash.
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
             max_depth: {}, max_leaf_length: {}, node_offset: {}, node_depth: {}, inner_hash_length: {} }}",
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
                IV[4] ^ LittleEndian::read_u64(&params.salt[..8]),
                IV[5] ^ LittleEndian::read_u64(&params.salt[8..]),
                IV[6] ^ LittleEndian::read_u64(&params.personal[..8]),
                IV[7] ^ LittleEndian::read_u64(&params.personal[8..]),
            ],
            compress_fn: default_compress_impl(),
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

    /// Add input to the hash. You can call `update` any number of times.
    pub fn update(&mut self, mut input: &[u8]) -> &mut Self {
        // If we have a partial buffer, try to complete it. If we complete it and there's more
        // input waiting (so we know we don't need to finalize), compress it.
        if self.buflen > 0 {
            self.fill_buf(&mut input);
            if !input.is_empty() {
                unsafe {
                    (self.compress_fn)(&mut self.h, &self.buf, self.count, 0, 0);
                }
                self.buflen = 0;
            }
        }
        // While there's more than a block of input left, compress blocks directly without copying.
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

    fn fill_buf(&mut self, input: &mut &[u8]) {
        let take = cmp::min(BLOCKBYTES - self.buflen as usize, input.len());
        self.buf[self.buflen as usize..self.buflen as usize + take].copy_from_slice(&input[..take]);
        self.buflen += take as u8;
        self.count += take as u128;
        *input = &input[take..];
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
        let mut hash = Hash {
            bytes: [0; OUTBYTES],
            len: self.hash_length,
        };
        LittleEndian::write_u64_into(&h_copy, &mut hash.bytes);
        hash
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
    pub fn to_hex(&self) -> ArrayString<[u8; 2 * OUTBYTES]> {
        let mut s = ArrayString::new();
        let table = b"0123456789abcdef";
        for &b in self.as_bytes() {
            s.push(table[(b >> 4) as usize] as char);
            s.push(table[(b & 0xf) as usize] as char);
        }
        s
    }
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
fn default_compress_impl() -> CompressFn {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // If AVX2 is enabled at the top level for the whole build (using something like
        // RUSTFLAGS="-C target-cpu=native"), return the AVX2 implementation without doing dynamic
        // feature detection. This isn't common, but it's the only way to use AVX2 with no_std, at
        // least until more features get stabilized in the future.
        #[cfg(target_feature = "avx2")]
        {
            return avx2::compress;
        }
        // Do dynamic feature detection at runtime, and use AVX2 if the current CPU supports it.
        // This is what the default build does. Note that no_std doesn't currently support dynamic
        // detection.
        #[cfg(feature = "std")]
        {
            if is_x86_feature_detected!("avx2") {
                return avx2::compress;
            }
        }
    }
    // On other platforms (non-x86 or pre-AVX2) use the portable implementation.
    portable::compress
}

// This module is pub for internal benchmarks only. Please don't use it.
#[doc(hidden)]
pub mod benchmarks {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub use avx2::compress as compress_avx2;
    pub use portable::compress as compress_portable;

    // Safety: The portable implementation should be safe to call on any platform.
    pub fn force_portable(state: &mut ::State) {
        state.compress_fn = compress_portable;
    }
}
