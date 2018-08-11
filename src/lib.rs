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

pub const BLOCKBYTES: usize = 128;
pub const OUTBYTES: usize = 64;
pub const KEYBYTES: usize = 64;
pub const SALTBYTES: usize = 16;
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

// Safety note: The compression interface is unsafe in general, because calling the AVX2
// implementation on a platform that doesn't support AVX2 is undefined behavior. That said, the
// portable implementation is all safe code.
type CompressFn = unsafe fn(&mut StateWords, &Block, count: u128, lastblock: u64, lastnode: u64);
type StateWords = [u64; 8];
type Block = [u8; BLOCKBYTES];

pub fn blake2b(input: &[u8]) -> Digest {
    let mut state = State::new();
    state.update(input);
    state.finalize()
}

#[derive(Clone)]
pub struct Params {
    digest_length: u8,
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
    pub fn words(&self) -> StateWords {
        let mut words = IV;
        words[0] ^= self.digest_length as u64;
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

    pub fn key_block(&self) -> Option<[u8; BLOCKBYTES]> {
        if self.key_length > 0 {
            let mut block = [0; BLOCKBYTES];
            block[..KEYBYTES].copy_from_slice(&self.key);
            Some(block)
        } else {
            None
        }
    }

    pub fn digest_length(&self) -> u8 {
        self.digest_length
    }

    /// Set the length of the final hash. This is associated data too, so changing the length will
    /// give a totally different hash. The maximum digest length is `OUTBYTES` (64).
    pub fn set_digest_length(&mut self, length: usize) {
        assert!(
            1 <= length && length <= OUTBYTES,
            "Bad digest length: {}",
            length
        );
        self.digest_length = length as u8;
    }

    /// Use a secret key, so that BLAKE2b acts as a MAC. The maximum key length is `KEYBYTES` (64).
    /// An empty key is equivalent to having no key at all.
    pub fn set_key(&mut self, key: &[u8]) {
        assert!(key.len() <= KEYBYTES, "Bad key length: {}", key.len());
        self.key_length = key.len() as u8;
        self.key[..key.len()].copy_from_slice(key);
    }

    /// At most `SALTBYTES` (16). Shorter salts are padded with null bytes. An empty salt is
    /// equivalent to having no salt at all.
    pub fn set_salt(&mut self, salt: &[u8]) {
        assert!(salt.len() <= SALTBYTES, "Bad salt length: {}", salt.len());
        self.salt = [0; SALTBYTES];
        self.salt[..salt.len()].copy_from_slice(salt);
    }

    /// At most `PERSONALBYTES` (16). Shorter personalizations are padded with null bytes. An empty
    /// personalization is equivalent to having no personalization at all.
    pub fn set_personal(&mut self, personalization: &[u8]) {
        assert!(
            personalization.len() <= PERSONALBYTES,
            "Bad personalization length: {}",
            personalization.len()
        );
        self.personal = [0; PERSONALBYTES];
        self.personal[..personalization.len()].copy_from_slice(personalization);
    }

    /// From 0 (meaning unlimited) to 255. The default is 1 (meaning sequential).
    pub fn set_fanout(&mut self, fanout: u8) {
        self.fanout = fanout;
    }

    /// From 1 (the default, meaning sequential) to 255 (meaning unlimited).
    pub fn set_max_depth(&mut self, depth: u8) {
        assert!(depth != 0, "Bad max depth: {}", depth);
        self.max_depth = depth;
    }

    /// From 0 (the default, meaning unlimited or sequential) to `2^32 - 1`.
    pub fn set_max_leaf_length(&mut self, length: u32) {
        self.max_leaf_length = length;
    }

    /// From 0 (the default, meaning first, leftmost, leaf, or sequential) to `2^64 - 1`.
    pub fn set_node_offset(&mut self, offset: u64) {
        self.node_offset = offset;
    }

    /// From 0 (the default, meaning leaf or sequential) to 255.
    pub fn set_node_depth(&mut self, depth: u8) {
        self.node_depth = depth;
    }

    /// From 0 (the default, meaning sequential) to `OUTBYTES` (64).
    pub fn set_inner_hash_length(&mut self, length: usize) {
        assert!(length <= OUTBYTES, "Bad inner hash length: {}", length);
        self.inner_hash_length = length as u8;
    }
}

impl Default for Params {
    fn default() -> Self {
        Self {
            digest_length: OUTBYTES as u8,
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

#[derive(Clone)]
pub struct State {
    h: StateWords,
    buf: Block,
    buflen: usize,
    count: u128,
    compress_fn: CompressFn,
    last_node: bool,
    digest_length: u8,
}

impl State {
    pub fn new() -> Self {
        Self::with_params(&Params::default())
    }

    pub fn with_params(params: &Params) -> Self {
        let mut state = Self {
            h: params.words(),
            compress_fn: default_compress_impl(),
            buf: [0; BLOCKBYTES],
            buflen: 0,
            count: 0,
            last_node: false,
            digest_length: params.digest_length(),
        };
        if let Some(key_block) = params.key_block() {
            state.update(&key_block);
        }
        state
    }

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

    /// Finish the final hashing step, and return a `Digest`. Calling this multiple times will give
    /// the same answer. It's also allowed to `update` with more input after calling this.
    pub fn finalize(&mut self) -> Digest {
        for i in self.buflen..BLOCKBYTES {
            self.buf[i] = 0;
        }
        let last_node = if self.last_node { !0 } else { 0 };
        let mut h_copy = self.h;
        unsafe {
            (self.compress_fn)(&mut h_copy, &self.buf, self.count, !0, last_node);
        }
        let mut digest = Digest {
            bytes: [0; OUTBYTES],
            len: self.digest_length,
        };
        LittleEndian::write_u64_into(&h_copy, &mut digest.bytes);
        digest
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

/// A finalized BLAKE2 hash.
///
/// `Digest` supports constant-time equality checks, for cases where BLAKE2 is being used as a MAC.
#[derive(Clone)]
pub struct Digest {
    bytes: [u8; OUTBYTES],
    len: u8,
}

impl Digest {
    pub fn bytes(&self) -> &[u8] {
        &self.bytes[..self.len as usize]
    }

    /// Convert the digest to a lowercase hexadecimal `String`. Requires `std`.
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

/// This implementation is constant time, if the two digests are the same
/// length.
impl PartialEq for Digest {
    fn eq(&self, other: &Digest) -> bool {
        constant_time_eq::constant_time_eq(&self.bytes(), &other.bytes())
    }
}

impl Eq for Digest {}

impl AsRef<[u8]> for Digest {
    fn as_ref(&self) -> &[u8] {
        self.bytes()
    }
}

impl fmt::Debug for Digest {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Digest({})", self.hex())
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
