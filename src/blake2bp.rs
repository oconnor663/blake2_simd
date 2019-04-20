//! BLAKE2bp, a variant of BLAKE2b that uses SIMD more efficiently.
//!
//! The AVX2 implementation of BLAKE2bp is about twice as fast that of BLAKE2b.
//! However, note that it's a different hash function, and it gives a different
//! hash from BLAKE2b for the same input.
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

use crate::guts::{self, Finalize, Stride};
use crate::many;
use crate::Hash;
use crate::Params as Blake2bParams;
use crate::BLOCKBYTES;
use crate::KEYBYTES;
use crate::OUTBYTES;
use byteorder::{ByteOrder, LittleEndian};
use core::cmp;
use core::fmt;

#[cfg(feature = "std")]
use std;

pub(crate) const DEGREE: usize = 4;

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
    leaf_words: [guts::u64x8; DEGREE],
    root_words: guts::u64x8,
    // Note that this buffer is twice as large as what compress4 needs. That guarantees that we
    // have enough input when we compress to know we don't need to finalize any of the leaves.
    buf: [u8; 8 * BLOCKBYTES],
    buf_len: u16,
    // Note that this is the *per-leaf* count.
    count: u128,
    hash_length: u8,
    implementation: guts::Implementation,
}

impl State {
    /// Equivalent to `State::default()` or `Params::default().to_state()`.
    pub fn new() -> Self {
        Self::with_params(&Params::default())
    }

    fn with_params(params: &Params) -> Self {
        let implementation = guts::Implementation::detect();
        let mut base_params = Blake2bParams::new();
        base_params
            .hash_length(params.hash_length as usize)
            .key(&params.key[..params.key_length as usize])
            .fanout(DEGREE as u8)
            .max_depth(2)
            .max_leaf_length(0)
            // Note that inner_hash_length is always OUTBYTES, regardless of the hash_length
            // parameter. This isn't documented in the spec, but it matches the behavior of the
            // reference implementation: https://github.com/BLAKE2/BLAKE2/blob/320c325437539ae91091ce62efec1913cd8093c2/ref/blake2bp-ref.c#L55
            .inner_hash_length(OUTBYTES);
        let leaf_words = |worker_index| {
            base_params
                .clone()
                .node_offset(worker_index)
                .node_depth(0)
                // Note that setting the last_node flag here has no effect,
                // because it isn't included in the state words.
                .to_state_words()
        };
        let leaf_words = [leaf_words(0), leaf_words(1), leaf_words(2), leaf_words(3)];
        let root_words = base_params
            .clone()
            .node_offset(0)
            .node_depth(1)
            // Note that setting the last_node flag here has no effect, because
            // it isn't included in the state words.
            .to_state_words();

        // If a key is set, initalize the buffer to contain the key bytes. Note
        // that only the leaves hash key bytes. The root doesn't, even though
        // the key length it still set in its parameters. Again this isn't
        // documented in the spec, but it matches the behavior of the reference
        // implementation:
        // https://github.com/BLAKE2/BLAKE2/blob/320c325437539ae91091ce62efec1913cd8093c2/ref/blake2bp-ref.c#L128
        // This particular behavior (though not the inner hash length behavior
        // above) is also corroborated by the official test vectors; see
        // tests/vector_tests.rs.
        let mut buf = [0; 2 * DEGREE * BLOCKBYTES];
        let mut buf_len = 0;
        if params.key_length > 0 {
            for i in 0..DEGREE {
                let keybytes = &params.key[..params.key_length as usize];
                buf[i * BLOCKBYTES..][..keybytes.len()].copy_from_slice(keybytes);
                buf_len = BLOCKBYTES * DEGREE;
            }
        }

        Self {
            leaf_words,
            root_words,
            buf,
            buf_len: buf_len as u16,
            count: 0, // count gets updated in self.compress()
            hash_length: params.hash_length,
            implementation,
        }
    }

    fn fill_buf(&mut self, input: &mut &[u8]) {
        let take = cmp::min(self.buf.len() - self.buf_len as usize, input.len());
        self.buf[self.buf_len as usize..][..take].copy_from_slice(&input[..take]);
        self.buf_len += take as u16;
        *input = &input[take..];
    }

    fn compress_to_leaves(
        leaves: &mut [guts::u64x8; DEGREE],
        input: &[u8],
        count: &mut u128,
        implementation: guts::Implementation,
    ) {
        // The input is assumed to be the same number of complete blocks for
        // each of the leaves.
        debug_assert!(input.len() > 0);
        debug_assert_eq!(0, input.len() % (DEGREE * BLOCKBYTES));

        let jobs = leaves.iter_mut().enumerate().map(|(i, words)| {
            guts::Job::new(words, *count, &input[i * BLOCKBYTES..], Finalize::NotYet)
        });
        many::compress_many(jobs, implementation, Stride::Parallel);
        // Note that count is the bytes input *per-leaf*.
        *count = count.wrapping_add((input.len() / DEGREE) as u128);
    }

    /// Add input to the hash. You can call `update` any number of times.
    pub fn update(&mut self, mut input: &[u8]) -> &mut Self {
        // If we have a partial buffer, try to complete it. If we complete it and there's more
        // input waiting, we need to compress to make more room. However, because we need to be
        // sure that *none* of the leaves would need to be finalized as part of this round of
        // compression, we need to buffer more than we would for BLAKE2b.
        if self.buf_len > 0 {
            self.fill_buf(&mut input);
            // The buffer is large enough for two compressions. If we've filled
            // the buffer and there's still more input coming, then we have to
            // do at least one compression. If there's enough input still
            // coming that all the leaves are guaranteed to get more, do both
            // compressions in the buffer. Otherwise, do just one and shift the
            // back half of the buffer to the front.
            if !input.is_empty() {
                if input.len() > (DEGREE - 1) * BLOCKBYTES {
                    // Enough input coming to do both compressions.
                    Self::compress_to_leaves(
                        &mut self.leaf_words,
                        &self.buf,
                        &mut self.count,
                        self.implementation,
                    );
                    self.buf_len = 0;
                } else {
                    // Only enough input coming for one compression.
                    Self::compress_to_leaves(
                        &mut self.leaf_words,
                        &self.buf[..DEGREE * BLOCKBYTES],
                        &mut self.count,
                        self.implementation,
                    );
                    self.buf_len = (DEGREE * BLOCKBYTES) as u16;
                    let (buf_front, buf_back) = self.buf.split_at_mut(DEGREE * BLOCKBYTES);
                    buf_front.copy_from_slice(buf_back);
                }
            }
        }

        // Now we directly compress as much input as possible, without copying
        // it into the buffer. We need to make sure we buffer at least one byte
        // for each of the leaves, so that we know we don't need to finalize
        // them.
        let needed_tail = (DEGREE - 1) * BLOCKBYTES + 1;
        let mut bulk_bytes = input.len().saturating_sub(needed_tail);
        bulk_bytes -= bulk_bytes % (DEGREE * BLOCKBYTES);
        if bulk_bytes > 0 {
            Self::compress_to_leaves(
                &mut self.leaf_words,
                &input[..bulk_bytes],
                &mut self.count,
                self.implementation,
            );
            input = &input[bulk_bytes..];
        }

        // Buffer any remaining input, to be either compressed or finalized in
        // a subsequent call.
        self.fill_buf(&mut input);
        debug_assert_eq!(0, input.len());
        self
    }

    /// Finalize the state and return a `Hash`. This method is idempotent, and calling it multiple
    /// times will give the same result. It's also possible to `update` with more input in between.
    pub fn finalize(&self) -> Hash {
        let buf_len = self.buf_len as usize;
        let mut leaves_copy = self.leaf_words;

        // Figure out how many group compressions we're going to do. If the
        // buffer has two blocks of input for each leaf, we'll compress the
        // full buffer, and all of the jobs will be finalizing. If it has less,
        // we'll do only one, and some of them might not finalize. Note that
        // the last leaf always finalizes here, and it sets the last node flag.
        let two_compressions = buf_len > (2 * DEGREE - 1) * BLOCKBYTES;
        let leaf_not_yet = |leaf_index| {
            if two_compressions {
                false
            } else {
                buf_len > (DEGREE + leaf_index) * BLOCKBYTES
            }
        };

        // Map the group compression jobs.
        let jobs = leaves_copy
            .iter_mut()
            .enumerate()
            .map(|(leaf_index, leaf_words)| {
                let input_start = cmp::min(leaf_index * BLOCKBYTES, buf_len);
                let input_end = if two_compressions {
                    // Compress the whole buffer.
                    buf_len
                } else {
                    // Only compress the first half, if we even have that much.
                    cmp::min(buf_len, DEGREE * BLOCKBYTES)
                };
                let input = &self.buf[input_start..input_end];
                let finalize = if leaf_index == DEGREE - 1 {
                    Finalize::YesLastNode
                } else if leaf_not_yet(leaf_index) {
                    Finalize::NotYet
                } else {
                    Finalize::YesOrdinary
                };
                guts::Job::new(leaf_words, self.count, input, finalize)
            });

        // Run the group compression jobs.
        many::compress_many(jobs, self.implementation, Stride::Parallel);

        // We just finished all the batch compressions we could. Some of the
        // leaves (though note, never the last one) might have one more block
        // left to finalize them.
        for leaf_index in 0..DEGREE - 1 {
            if leaf_not_yet(leaf_index) {
                let block_start = (DEGREE + leaf_index) * BLOCKBYTES;
                debug_assert!(buf_len > block_start);
                let block_len = cmp::min(BLOCKBYTES, buf_len - block_start);
                let job = guts::Job::new(
                    &mut leaves_copy[leaf_index],
                    self.count.wrapping_add(BLOCKBYTES as u128),
                    &self.buf[block_start..][..block_len],
                    Finalize::YesOrdinary,
                );
                // Note that compress1_loop is always available in all
                // implementations, so we can call it directly without checking
                // anything. Also stride doesn't really matter here, because
                // this is just one block.
                self.implementation.compress1_loop(job, Stride::Parallel);
            }
        }

        // Compress each of the four finalized hashes into the root words as
        // input, using two compressions. Note that even if a future version of
        // this implementation supports the hash_length parameter and sets it
        // as associated data for all nodes, this step must still use the
        // untruncated output of each leaf. Note also that, as mentioned above,
        // the root node doesn't hash any key bytes.
        let mut root_words_copy = self.root_words;
        let mut block = [0; DEGREE * OUTBYTES];
        debug_assert_eq!(block.len(), DEGREE * BLOCKBYTES / 2);
        for leaf_index in 0..DEGREE {
            LittleEndian::write_u64_into(
                &leaves_copy[leaf_index][..],
                &mut block[leaf_index * OUTBYTES..][..OUTBYTES],
            );
        }
        // Again compress1_loop is always available, but here we have two
        // blocks so stride matters.
        let job = guts::Job::new(&mut root_words_copy, 0, &block, Finalize::YesLastNode);
        self.implementation.compress1_loop(job, Stride::Normal);
        Hash {
            bytes: crate::state_words_to_bytes(&root_words_copy),
            len: self.hash_length,
        }
    }

    /// Return the total number of bytes input so far.
    pub fn count(&self) -> u128 {
        // Remember that self.count is *per-leaf*.
        self.count
            .wrapping_mul(DEGREE as u128)
            .wrapping_add(self.buf_len as u128)
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
            "State {{ count: {}, hash_length: {} }}",
            self.count(),
            self.hash_length,
        )
    }
}

impl Default for State {
    fn default() -> Self {
        Self::with_params(&Params::default())
    }
}

pub(crate) fn force_portable(state: &mut State) {
    state.implementation = guts::Implementation::portable();
}

#[cfg(test)]
pub(crate) mod test {
    use super::*;
    use crate::paint_test_input;

    // This is a simple reference implementation without the complicated buffering or parameter
    // support of the real implementation. We need this because the official test vectors don't
    // include any inputs large enough to exercise all the branches in the buffering logic.
    fn blake2bp_reference(input: &[u8]) -> Hash {
        let mut leaves = [
            Blake2bParams::new()
                .fanout(DEGREE as u8)
                .max_depth(2)
                .node_offset(0)
                .inner_hash_length(OUTBYTES)
                .to_state(),
            Blake2bParams::new()
                .fanout(DEGREE as u8)
                .max_depth(2)
                .node_offset(1)
                .inner_hash_length(OUTBYTES)
                .to_state(),
            Blake2bParams::new()
                .fanout(DEGREE as u8)
                .max_depth(2)
                .node_offset(2)
                .inner_hash_length(OUTBYTES)
                .to_state(),
            Blake2bParams::new()
                .fanout(DEGREE as u8)
                .max_depth(2)
                .node_offset(3)
                .inner_hash_length(OUTBYTES)
                .last_node(true)
                .to_state(),
        ];
        for (i, chunk) in input.chunks(BLOCKBYTES).enumerate() {
            leaves[i % DEGREE].update(chunk);
        }
        let mut root = Blake2bParams::new()
            .fanout(DEGREE as u8)
            .max_depth(2)
            .node_depth(1)
            .inner_hash_length(OUTBYTES)
            .last_node(true)
            .to_state();
        for leaf in &mut leaves {
            root.update(leaf.finalize().as_bytes());
        }
        root.finalize()
    }

    #[test]
    fn test_against_reference() {
        let mut buf = [0; 21 * BLOCKBYTES];
        paint_test_input(&mut buf);
        // - 8 blocks is just enought to fill the double buffer.
        // - 9 blocks triggers the "perform one compression on the double buffer" case.
        // - 11 blocks is the largest input where only one compression may be performed, on the
        //   first half of the buffer, because there's not enough input to avoid needing to
        //   finalize the second half.
        // - 12 blocks triggers the "perform both compressions in the double buffer" case.
        // - 15 blocks is the largest input where, after compressing 8 blocks from the buffer,
        //   there's not enough input to hash directly from memory.
        // - 16 blocks triggers "after emptying the buffer, hash directly from memory".
        for num_blocks in 0..=20 {
            for &extra in &[0, 1, BLOCKBYTES - 1] {
                // First hash the input all at once, as a sanity check.
                let input = &buf[..num_blocks * BLOCKBYTES + extra];
                let expected = blake2bp_reference(&input);
                let found = blake2bp(&input);
                assert_eq!(expected, found);

                // Then, do it again, but buffer 1 byte of input first. That causes the buffering
                // branch to trigger.
                let mut state = State::new();
                let maybe_one = cmp::min(1, input.len());
                state.update(&input[..maybe_one]);
                assert_eq!(maybe_one as u128, state.count());
                // Do a throwaway finalize here to check for idempotency.
                state.finalize();
                state.update(&input[maybe_one..]);
                assert_eq!(input.len() as u128, state.count());
                let found = state.finalize();
                assert_eq!(expected, found);
            }
        }
    }
}
