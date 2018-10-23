//! An implementation of BLAKE2bp, a variant of BLAKE2b that takes advantage of the parallelism of
//! modern processors.
//!
//! The AVX2 implementation of BLAKE2bp is about twice as fast that of BLAKE2b, because it's able
//! to use AVX2's vector operations more efficiently. However, note that it's a different hash
//! function, and it gives a different hash from BLAKE2b for the same input.
//!
//! # Example
//!
//! ```
//! use blake2b_simd::blake2bp;
//!
//! let hash = blake2bp::Params::new()
//!     .hash_length(16)
//!     .key(b"The Magic Words are Squeamish Ossifrage")
//!     .to_state()
//!     .update(b"foo")
//!     .update(b"bar")
//!     .update(b"baz")
//!     .finalize();
//! assert_eq!("e69c7d2c42a5ac14948772231c68c552", &hash.to_hex());
//! ```

use core::cmp;
use core::fmt;
use Compress4xFn;
use Hash;
use State as Blake2bState;
use BLOCKBYTES;
use KEYBYTES;
use OUTBYTES;

#[cfg(feature = "std")]
use std;

/// Compute the BLAKE2bp hash of a slice of bytes, using default parameters.
///
/// # Example
///
/// ```
/// # use blake2b_simd::blake2bp::blake2bp;
/// let expected = "8ca9ccee7946afcb686fe7556628b5ba1bf9a691da37ca58cd049354d99f3704\
///                 2c007427e5f219b9ab5063707ec6823872dee413ee014b4d02f2ebb6abb5f643";
/// let hash = blake2bp(b"foo");
/// assert_eq!(expected, &hash.to_hex());
/// ```
pub fn blake2bp(input: &[u8]) -> Hash {
    State::new().update(input).finalize()
}

/// A parameter builder for BLAKE2bp, just like the [`Params`](../struct.Params.html) type for
/// BLAKE2b.
///
/// This builder only supports configuring the hash length and a secret key. This matches the
/// options provided by the [reference
/// implementation](https://github.com/BLAKE2/BLAKE2/blob/320c325437539ae91091ce62efec1913cd8093c2/ref/blake2.h#L162-L165).
///
/// # Example
///
/// ```
/// use blake2b_simd::blake2bp;
/// let mut state = blake2bp::Params::new().hash_length(32).to_state();
/// ```
#[derive(Clone)]
pub struct Params {
    hash_length: u8,
    key_length: u8,
    key: [u8; KEYBYTES],
}

impl Params {
    /// Equivalent to `Params::default()`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct a BLAKE2bp `State` object based on these parameters.
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

    /// Use a secret key, so that BLAKE2bp acts as a MAC. The maximum key length is `KEYBYTES`
    /// (64). An empty key is equivalent to having no key at all.
    pub fn key(&mut self, key: &[u8]) -> &mut Self {
        assert!(key.len() <= KEYBYTES, "Bad key length: {}", key.len());
        self.key_length = key.len() as u8;
        self.key = [0; KEYBYTES];
        self.key[..key.len()].copy_from_slice(key);
        self
    }
}

impl Default for Params {
    fn default() -> Self {
        Self {
            hash_length: OUTBYTES as u8,
            key_length: 0,
            key: [0; KEYBYTES],
        }
    }
}

impl fmt::Debug for Params {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Params {{ hash_length: {}, key_length: {} }}",
            self.hash_length,
            // NB: Don't print the key itself. Debug shouldn't leak secrets.
            self.key_length,
        )
    }
}

/// An incremental hasher for BLAKE2bp, just like the [`State`](../struct.State.html) type for
/// BLAKE2b.
///
/// # Example
///
/// ```
/// use blake2b_simd::blake2bp;
///
/// let mut state = blake2bp::State::new();
/// state.update(b"foo");
/// state.update(b"bar");
/// let hash = state.finalize();
///
/// let expected = "e654427b6ef02949471712263e59071abbb6aa94855674c1daeed6cfaf127c33\
///                 dfa3205f7f7f71e4f0673d25fa82a368488911f446bccd323af3ab03f53e56e5";
/// assert_eq!(expected, &hash.to_hex());
/// ```
#[derive(Clone)]
pub struct State {
    leaf0: Blake2bState,
    leaf1: Blake2bState,
    leaf2: Blake2bState,
    leaf3: Blake2bState,
    root: Blake2bState,
    // Note that this buffer is twice as large as what compress4x needs. That guarantees that we
    // have enough input when we compress to know we don't need to finalize any of the leaves.
    buf: [u8; 8 * BLOCKBYTES],
    buflen: u16,
    // This count isn't used for hashing, only for self.count().
    count: u128,
    pub(crate) compress_4x_fn: Compress4xFn,
}

impl State {
    /// Equivalent to `State::default()` or `Params::default().to_state()`.
    pub fn new() -> Self {
        Self::with_params(&Params::default())
    }

    // TODO: There are a couple places in this function where we reach into the BLAKE2b State
    // object and manually overwrite its fields. This is unfortunate, and it means you can't
    // actually build BLAKE2bp out of the BLAKE2b public interface. (You can make it work for the
    // basic default-length-no-key case, but you can't implement either of those parameters
    // correctly.) It might be nice to talk to the designers about whether this is the intended
    // state of affairs.
    fn with_params(params: &Params) -> Self {
        let mut base_params = ::Params::new();
        base_params
            .hash_length(params.hash_length as usize)
            .key(&params.key[..params.key_length as usize])
            .fanout(4)
            .max_depth(2)
            .max_leaf_length(0)
            // Note that inner_hash_length is always OUTBYTES, regardless of the hash_length
            // parameter. This isn't documented in the spec, but it matches the behavior of the
            // reference implementation: https://github.com/BLAKE2/BLAKE2/blob/320c325437539ae91091ce62efec1913cd8093c2/ref/blake2bp-ref.c#L55
            .inner_hash_length(OUTBYTES);
        let leaf_state = |worker_index| {
            let mut state = base_params
                .clone()
                .node_offset(worker_index)
                .node_depth(0)
                .last_node(worker_index == 3)
                .to_state();
            // Force the output length to be OUTBYTES, matching the inner_hash_length parameter.
            // Note that the regular hash_length parameter still contributes associated data to
            // these instances.
            state.hash_length = OUTBYTES as u8;
            state
        };
        let mut root_state = base_params
            .clone()
            .node_offset(0)
            .node_depth(1)
            .last_node(true)
            .to_state();
        // Clear the keybytes from the root state buffer. Only the leaf nodes will hash the actual
        // key bytes, though the key length still contributes associated data to the root node.
        // Again this isn't documented in the spec, but it matches the behavior of the reference
        // implementation: https://github.com/BLAKE2/BLAKE2/blob/320c325437539ae91091ce62efec1913cd8093c2/ref/blake2bp-ref.c#L128
        // This particular behavior (though not the inner hash length behavior above) is also
        // corroborated by the official test vectors; see tests/vector_tests.rs.
        root_state.buflen = 0;
        root_state.count = 0;
        Self {
            leaf0: leaf_state(0),
            leaf1: leaf_state(1),
            leaf2: leaf_state(2),
            leaf3: leaf_state(3),
            root: root_state,
            buf: [0; 8 * BLOCKBYTES],
            buflen: 0,
            count: 0,
            compress_4x_fn: ::default_compress_impl().1,
        }
    }

    fn fill_buf(&mut self, input: &mut &[u8]) {
        let take = cmp::min(self.buf.len() - self.buflen as usize, input.len());
        self.buf[self.buflen as usize..self.buflen as usize + take].copy_from_slice(&input[..take]);
        self.buflen += take as u16;
        self.count += take as u128;
        *input = &input[take..];
    }

    fn compress_4x(
        input: &[u8; 4 * BLOCKBYTES],
        leaf0: &mut Blake2bState,
        leaf1: &mut Blake2bState,
        leaf2: &mut Blake2bState,
        leaf3: &mut Blake2bState,
        compress_4x_fn: Compress4xFn,
    ) {
        // Note that this is reaching into the underlying state objects, so it assumes they don't
        // get input through their normal update() interface. Also we can only call this when we're
        // sure there's more input coming.
        debug_assert_eq!(0, leaf0.buflen);
        debug_assert_eq!(0, leaf1.buflen);
        debug_assert_eq!(0, leaf2.buflen);
        debug_assert_eq!(0, leaf3.buflen);
        debug_assert_eq!(leaf0.count, leaf1.count);
        debug_assert_eq!(leaf0.count, leaf2.count);
        debug_assert_eq!(leaf0.count, leaf3.count);
        leaf0.count += BLOCKBYTES as u128;
        leaf1.count += BLOCKBYTES as u128;
        leaf2.count += BLOCKBYTES as u128;
        leaf3.count += BLOCKBYTES as u128;
        let count = leaf0.count;
        let msg_refs = array_refs!(input, BLOCKBYTES, BLOCKBYTES, BLOCKBYTES, BLOCKBYTES);
        unsafe {
            (compress_4x_fn)(
                &mut leaf0.h,
                &mut leaf1.h,
                &mut leaf2.h,
                &mut leaf3.h,
                msg_refs.0,
                msg_refs.1,
                msg_refs.2,
                msg_refs.3,
                count,
                0,
                0,
            );
        }
    }

    /// Add input to the hash. You can call `update` any number of times.
    pub fn update(&mut self, mut input: &[u8]) -> &mut Self {
        // If we have a partial buffer, try to complete it. If we complete it and there's more
        // input waiting, we need to compress to make more room. However, because we need to be
        // sure that *none* of the leaves would need to be finalized as part of this round of
        // compression, we need to buffer more than we would for BLAKE2b.
        if self.buflen > 0 {
            self.fill_buf(&mut input);
            if !input.is_empty() {
                // The buffer is large enough for two compressions. If it's full and there's more
                // input coming, always do at least the first compression, on the left half of the
                // buffer.
                Self::compress_4x(
                    array_ref!(self.buf, 0, 4 * BLOCKBYTES),
                    &mut self.leaf0,
                    &mut self.leaf1,
                    &mut self.leaf2,
                    &mut self.leaf3,
                    self.compress_4x_fn,
                );
                self.buflen -= BLOCKBYTES as u16;
                // Now, if there's enough input still coming that all four leaves are going to get
                // more, we can do the second compression and clear the buffer. Otherwise, we have
                // to shift the remainder of the buffer to the left (and we know in this case the
                // direct-from-memory loop will get skipped too).
                if input.len() > 3 * BLOCKBYTES {
                    Self::compress_4x(
                        array_ref!(self.buf, 4 * BLOCKBYTES, 4 * BLOCKBYTES),
                        &mut self.leaf0,
                        &mut self.leaf1,
                        &mut self.leaf2,
                        &mut self.leaf3,
                        self.compress_4x_fn,
                    );
                    self.buflen = 0;
                } else {
                    let (left, right) = self.buf.split_at_mut(4 * BLOCKBYTES);
                    left[..self.buflen as usize].copy_from_slice(&right[..self.buflen as usize]);
                }
            }
        }

        // While there are more than 7 input blocks coming, then we know that we can perform a
        // compression and still have more input coming for each leaf. (We also know that the
        // buffer must have been emptied above.)
        while input.len() > 7 * BLOCKBYTES {
            self.count += 4 * BLOCKBYTES as u128;
            let block = array_ref!(input, 0, 4 * BLOCKBYTES);
            Self::compress_4x(
                block,
                &mut self.leaf0,
                &mut self.leaf1,
                &mut self.leaf2,
                &mut self.leaf3,
                self.compress_4x_fn,
            );
            input = &input[4 * BLOCKBYTES..];
        }

        // Buffer any remaining input, to be either compressed or finalized in a subsequent call.
        self.fill_buf(&mut input);
        debug_assert_eq!(0, input.len());
        self
    }

    /// Finalize the state and return a `Hash`. This method is idempotent, and calling it multiple
    /// times will give the same result. It's also possible to `update` with more input in between.
    pub fn finalize(&mut self) -> Hash {
        let mut leaf0 = self.leaf0.clone();
        let mut leaf1 = self.leaf1.clone();
        let mut leaf2 = self.leaf2.clone();
        let mut leaf3 = self.leaf3.clone();
        let chunks = array_refs!(
            &self.buf, BLOCKBYTES, BLOCKBYTES, BLOCKBYTES, BLOCKBYTES, BLOCKBYTES, BLOCKBYTES,
            BLOCKBYTES, BLOCKBYTES
        );
        let mut buflen = self.buflen as usize;
        leaf0.update(&chunks.0[..cmp::min(buflen, BLOCKBYTES)]);
        buflen = buflen.saturating_sub(BLOCKBYTES);
        leaf1.update(&chunks.1[..cmp::min(buflen, BLOCKBYTES)]);
        buflen = buflen.saturating_sub(BLOCKBYTES);
        leaf2.update(&chunks.2[..cmp::min(buflen, BLOCKBYTES)]);
        buflen = buflen.saturating_sub(BLOCKBYTES);
        leaf3.update(&chunks.3[..cmp::min(buflen, BLOCKBYTES)]);
        buflen = buflen.saturating_sub(BLOCKBYTES);
        leaf0.update(&chunks.4[..cmp::min(buflen, BLOCKBYTES)]);
        buflen = buflen.saturating_sub(BLOCKBYTES);
        leaf1.update(&chunks.5[..cmp::min(buflen, BLOCKBYTES)]);
        buflen = buflen.saturating_sub(BLOCKBYTES);
        leaf2.update(&chunks.6[..cmp::min(buflen, BLOCKBYTES)]);
        buflen = buflen.saturating_sub(BLOCKBYTES);
        leaf3.update(&chunks.7[..cmp::min(buflen, BLOCKBYTES)]);
        let mut root = self.root.clone();
        root.update(leaf0.finalize().as_bytes());
        root.update(leaf1.finalize().as_bytes());
        root.update(leaf2.finalize().as_bytes());
        root.update(leaf3.finalize().as_bytes());
        root.finalize()
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
        write!(
            f,
            "State {{ count: {}, root: {:?}, leaf0: {:?}, leaf1: {:?}, \
             leaf2: {:?}, leaf3: {:?} }}",
            self.count, self.root, self.leaf0, self.leaf1, self.leaf2, self.leaf3
        )
    }
}

impl Default for State {
    fn default() -> Self {
        Self::with_params(&Params::default())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_blake2bp() {
        // From https://raw.githubusercontent.com/BLAKE2/BLAKE2/master/testvectors/blake2-kat.json.
        let vectors: &[(&[u8], &str)] = &[
            // Note that memory mapping doesn't work on zero-length input.
            (
                b"\x00",
                "a139280e72757b723e6473d5be59f36e9d50fc5cd7d4585cbc09804895a36c52\
                 1242fb2789f85cb9e35491f31d4a6952f9d8e097aef94fa1ca0b12525721f03d",
            ),
            (
                b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\
                  \x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\
                  \x20\x21\x22\x23\x24\x25\x26\x27\x28\x29\x2a\x2b\x2c\x2d\x2e\x2f\
                  \x30\x31\x32\x33\x34\x35\x36\x37\x38\x39\x3a\x3b\x3c\x3d\x3e\x3f\
                  \x40\x41\x42\x43\x44\x45\x46\x47\x48\x49\x4a\x4b\x4c\x4d\x4e\x4f\
                  \x50\x51\x52\x53\x54\x55\x56\x57\x58\x59\x5a\x5b\x5c\x5d\x5e\x5f\
                  \x60\x61\x62\x63\x64\x65\x66\x67\x68\x69\x6a\x6b\x6c\x6d\x6e\x6f\
                  \x70\x71\x72\x73\x74\x75\x76\x77\x78\x79\x7a\x7b\x7c\x7d\x7e\x7f\
                  \x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x8b\x8c\x8d\x8e\x8f\
                  \x90\x91\x92\x93\x94\x95\x96\x97\x98\x99\x9a\x9b\x9c\x9d\x9e\x9f\
                  \xa0\xa1\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xab\xac\xad\xae\xaf\
                  \xb0\xb1\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xbb\xbc\xbd\xbe\xbf\
                  \xc0\xc1\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xcb\xcc\xcd\xce\xcf\
                  \xd0\xd1\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xdb\xdc\xdd\xde\xdf\
                  \xe0\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xeb\xec\xed\xee\xef\
                  \xf0\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xfb\xfc\xfd\xfe",
                "3f35c45d24fcfb4acca651076c08000e279ebbff37a1333ce19fd577202dbd24\
                 b58c514e36dd9ba64af4d78eea4e2dd13bc18d798887dd971376bcae0087e17e",
            ),
        ];

        for &(input, expected) in vectors {
            let found = blake2bp(input);
            assert_eq!(&*expected, &*found.to_hex());
        }
    }
}
