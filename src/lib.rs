//! An implementation of the BLAKE2b hash with:
//!
//! - 100% stable Rust.
//! - A fast AVX2 implementation ported from [libsodium](https://github.com/jedisct1/libsodium).
//! - A portable, safe implementation for other platforms.
//! - Dynamic CPU feature detection. All x86 binaries include the AVX2 implementation and use it on
//!   platforms that support it.
//! - `no_std` support. `std` is on by default, for feature detection and `std::io::Write`.
//! - All the features from the [the BLAKE2 spec](https://blake2.net/blake2.pdf), like adjustable
//!   length, keying, and associated data for tree hashing.
//!
//! # Example
//!
//! ```
//! let mut params = blake2b_simd::Params::default();
//! params.hash_length(16);
//! params.key(b"The Magic Words are Squeamish Ossifrage");
//! params.personal(b"L. P. Waterhouse");
//! let mut state = blake2b_simd::State::with_params(&params);
//! state.update(b"foo");
//! state.update(b"bar");
//! state.update(b"baz");
//! let hash = state.finalize();
//! assert_eq!("ee8ff4e9be887297cf79348dc35dab56", &hash.hex());
//! ```

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

#[cfg(test)]
mod test;

const BLOCKBYTES: usize = 128;
/// The max hash length.
pub const OUTBYTES: usize = 64;
/// The max key length.
pub const KEYBYTES: usize = 64;
/// The max salt length.
pub const SALTBYTES: usize = 16;
/// The max personalization length.
pub const PERSONALBYTES: usize = 16;

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
pub fn blake2b(input: &[u8]) -> Hash {
    let mut state = State::new();
    state.update(input);
    state.finalize()
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
/// let mut params = blake2b_simd::Params::default();
/// params.hash_length(32);
/// let mut state = blake2b_simd::State::with_params(&params);
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
}

impl Params {
    fn words(&self) -> StateWords {
        let mut words = [0; 8];
        words[0] ^= self.hash_length as u64;
        words[0] ^= (self.key_length as u64) << 8;
        words[0] ^= (self.fanout as u64) << 16;
        words[0] ^= (self.max_depth as u64) << 24;
        words[0] ^= (self.max_leaf_length as u64) << 32;
        words[1] ^= self.node_offset;
        words[2] ^= self.node_depth as u64;
        words[2] ^= (self.inner_hash_length as u64) << 8;
        // The last half of word 2 and all of word 3 are reserved.
        words[4] ^= LittleEndian::read_u64(&self.salt[..8]);
        words[5] ^= LittleEndian::read_u64(&self.salt[8..]);
        words[6] ^= LittleEndian::read_u64(&self.personal[..8]);
        words[7] ^= LittleEndian::read_u64(&self.personal[8..]);
        words
    }

    /// Set the length of the final hash, from 1 to `OUTBYTES` (64). Apart from controlling the
    /// length of the final `Hash`, this is also associated data, and changing it will result in a
    /// totally different hash.
    pub fn hash_length(&mut self, length: usize) {
        assert!(
            1 <= length && length <= OUTBYTES,
            "Bad hash length: {}",
            length
        );
        self.hash_length = length as u8;
    }

    /// Use a secret key, so that BLAKE2b acts as a MAC. The maximum key length is `KEYBYTES` (64).
    /// An empty key is equivalent to having no key at all.
    pub fn key(&mut self, key: &[u8]) {
        assert!(key.len() <= KEYBYTES, "Bad key length: {}", key.len());
        self.key_length = key.len() as u8;
        self.key[..key.len()].copy_from_slice(key);
    }

    /// At most `SALTBYTES` (16). Shorter salts are padded with null bytes. An empty salt is
    /// equivalent to having no salt at all.
    pub fn salt(&mut self, salt: &[u8]) {
        assert!(salt.len() <= SALTBYTES, "Bad salt length: {}", salt.len());
        self.salt = [0; SALTBYTES];
        self.salt[..salt.len()].copy_from_slice(salt);
    }

    /// At most `PERSONALBYTES` (16). Shorter personalizations are padded with null bytes. An empty
    /// personalization is equivalent to having no personalization at all.
    pub fn personal(&mut self, personalization: &[u8]) {
        assert!(
            personalization.len() <= PERSONALBYTES,
            "Bad personalization length: {}",
            personalization.len()
        );
        self.personal = [0; PERSONALBYTES];
        self.personal[..personalization.len()].copy_from_slice(personalization);
    }

    /// From 0 (meaning unlimited) to 255. The default is 1 (meaning sequential).
    pub fn fanout(&mut self, fanout: u8) {
        self.fanout = fanout;
    }

    /// From 1 (the default, meaning sequential) to 255 (meaning unlimited).
    pub fn max_depth(&mut self, depth: u8) {
        assert!(depth != 0, "Bad max depth: {}", depth);
        self.max_depth = depth;
    }

    /// From 0 (the default, meaning unlimited or sequential) to `2^32 - 1`.
    pub fn max_leaf_length(&mut self, length: u32) {
        self.max_leaf_length = length;
    }

    /// From 0 (the default, meaning first, leftmost, leaf, or sequential) to `2^64 - 1`.
    pub fn node_offset(&mut self, offset: u64) {
        self.node_offset = offset;
    }

    /// From 0 (the default, meaning leaf or sequential) to 255.
    pub fn node_depth(&mut self, depth: u8) {
        self.node_depth = depth;
    }

    /// From 0 (the default, meaning sequential) to `OUTBYTES` (64).
    pub fn inner_hash_length(&mut self, length: usize) {
        assert!(length <= OUTBYTES, "Bad inner hash length: {}", length);
        self.inner_hash_length = length as u8;
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
/// let mut state = blake2b_simd::State::new();
/// state.update(b"foo");
/// let hash1 = state.finalize();
/// state.update(b"bar");
/// let hash2 = state.finalize();
/// assert!(hash1 != hash2);
/// ```
#[derive(Clone)]
pub struct State {
    h: StateWords,
    buf: Block,
    buflen: usize,
    count: u128,
    compress_fn: CompressFn,
    last_node: bool,
    hash_length: u8,
}

impl State {
    /// Construct a new `State` with default parameters.
    pub fn new() -> Self {
        Self::with_params(&Params::default())
    }

    /// Construct a new `State` with from a given `Params` object. This lets the caller customize
    /// the length of the final `Hash` and add other associated data.
    pub fn with_params(params: &Params) -> Self {
        let param_words = params.words();
        let mut state = Self {
            h: [
                IV[0] ^ param_words[0],
                IV[1] ^ param_words[1],
                IV[2] ^ param_words[2],
                IV[3] ^ param_words[3],
                IV[4] ^ param_words[4],
                IV[5] ^ param_words[5],
                IV[6] ^ param_words[6],
                IV[7] ^ param_words[7],
            ],
            compress_fn: default_compress_impl(),
            buf: [0; BLOCKBYTES],
            buflen: 0,
            count: 0,
            last_node: false,
            hash_length: params.hash_length,
        };
        if params.key_length > 0 {
            let mut key_block = [0; BLOCKBYTES];
            key_block[..KEYBYTES].copy_from_slice(&params.key);
            state.update(&key_block);
        }
        state
    }

    /// Set a flag indicating that this is the last node of its level in a tree hash. This is
    /// associated data like the other features in the `Params` object, except that it can be set
    /// at any time before calling `finalize`. That allows callers to begin hashing a node without
    /// knowing ahead of time whether it's the last in its level. For more details about the
    /// intended use of this flag [the BLAKE2 spec](https://blake2.net/blake2.pdf).
    pub fn set_last_node(&mut self, val: bool) {
        self.last_node = val;
    }

    fn fill_buf(&mut self, input: &mut &[u8]) {
        let take = cmp::min(BLOCKBYTES - self.buflen, input.len());
        self.buf[self.buflen..self.buflen + take].copy_from_slice(&input[..take]);
        self.buflen += take;
        self.count += take as u128;
        *input = &input[take..];
    }

    /// Add input to the hash. You can call `update` any number of times.
    // array_ref triggers unused_unsafe (https://github.com/droundy/arrayref/pull/14)
    #[allow(unused_unsafe)]
    pub fn update(&mut self, mut input: &[u8]) {
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
        // If there's more than a block of input left, compress it directly instead of buffering it.
        while input.len() > BLOCKBYTES {
            self.count += BLOCKBYTES as u128;
            let block = array_ref!(input, 0, BLOCKBYTES);
            unsafe {
                (self.compress_fn)(&mut self.h, block, self.count, 0, 0);
            }
            input = &input[BLOCKBYTES..];
        }
        // Buffer any remaining input, to be either compressed or finalized in a subsequent call.
        self.fill_buf(&mut input);
    }

    /// Finalize the state and return a `Hash`. This method is idempotent, and calling it multiple
    /// times will give the same result. It's also possible to `update` with more input in between.
    pub fn finalize(&mut self) -> Hash {
        for i in self.buflen..BLOCKBYTES {
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

#[allow(unreachable_code)]
fn default_compress_impl() -> CompressFn {
    // If AVX2 is enabled at the top level for the whole build (using something like
    // RUSTFLAGS="-C target-cpu=native"), return the AVX2 implementation without doing dynamic
    // feature detection. This isn't common, but it's the only way to use AVX2 with no_std, at
    // least until more features get stabilized in the future.
    #[cfg(all(target_feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        return avx2::compress;
    }
    // Do dynamic feature detection at runtime, and use AVX2 if the current CPU supports it. This
    // is what the default build does. Note that no_std doesn't currently support dynamic detection.
    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        if is_x86_feature_detected!("avx2") {
            return avx2::compress;
        }
    }
    // On other platforms (non-x86 or pre-AVX2) use the portable implementation.
    portable::compress
}

/// A finalized BLAKE2 hash, with constant-time equality.
#[derive(Clone, Copy)]
pub struct Hash {
    bytes: [u8; OUTBYTES],
    len: u8,
}

impl Hash {
    /// Get the hash as a slice of bytes. Note that slices don't provide constant-time equality
    /// checks, so avoid this method if you're using BLAKE2b as a MAC.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes[..self.len as usize]
    }

    /// Convert the hash to a lowercase hexadecimal
    /// [`ArrayString`](https://docs.rs/arrayvec/0.4/arrayvec/struct.ArrayString.html).
    pub fn hex(&self) -> ArrayString<[u8; 2 * OUTBYTES]> {
        let mut s = ArrayString::new();
        let table = b"0123456789abcdef";
        for &b in self.bytes() {
            s.push(table[(b >> 4) as usize] as char);
            s.push(table[(b & 0xf) as usize] as char);
        }
        s
    }
}

/// This implementation is constant time, if the two hashes are the same length.
impl PartialEq for Hash {
    fn eq(&self, other: &Hash) -> bool {
        constant_time_eq::constant_time_eq(&self.bytes(), &other.bytes())
    }
}

/// This implementation is constant time, if the slice is the same length as the hash.
impl PartialEq<[u8]> for Hash {
    fn eq(&self, other: &[u8]) -> bool {
        constant_time_eq::constant_time_eq(&self.bytes(), other)
    }
}

impl Eq for Hash {}

impl AsRef<[u8]> for Hash {
    fn as_ref(&self) -> &[u8] {
        self.bytes()
    }
}

impl fmt::Debug for Hash {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Hash(0x{})", self.hex())
    }
}

// This module is pub for internal benchmarks only. Please don't use it.
#[doc(hidden)]
pub mod benchmarks {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub use avx2::compress as compress_avx2;
    pub use portable::compress as compress_portable;

    pub fn force_portable(state: &mut ::State) {
        state.compress_fn = compress_portable;
    }
}
