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

use crate::guts;
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

const DEGREE: usize = 4;

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
    transposed_leaf_words: [guts::u64x4; 8],
    root_words: guts::u64x8,
    // Note that this buffer is twice as large as what compress4 needs. That guarantees that we
    // have enough input when we compress to know we don't need to finalize any of the leaves.
    buf: [u8; 8 * BLOCKBYTES],
    buflen: u16,
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
        let transposed_leaf_words = implementation.transpose4(
            &leaf_words(0),
            &leaf_words(1),
            &leaf_words(2),
            &leaf_words(3),
        );
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
        let mut buflen = 0;
        if params.key_length > 0 {
            for i in 0..DEGREE {
                let keybytes = &params.key[..params.key_length as usize];
                buf[i * BLOCKBYTES..][..keybytes.len()].copy_from_slice(keybytes);
                buflen = BLOCKBYTES * DEGREE;
            }
        }

        Self {
            transposed_leaf_words,
            root_words,
            buf,
            buflen: buflen as u16,
            count: 0, // count gets updated in self.compress()
            hash_length: params.hash_length,
            implementation,
        }
    }

    fn fill_buf(&mut self, input: &mut &[u8]) {
        let take = cmp::min(self.buf.len() - self.buflen as usize, input.len());
        self.buf[self.buflen as usize..self.buflen as usize + take].copy_from_slice(&input[..take]);
        self.buflen += take as u16;
        *input = &input[take..];
    }

    fn compress(
        input: &[u8; DEGREE * BLOCKBYTES],
        state: &mut [guts::u64x4; 8],
        count: &mut u128,
        implementation: guts::Implementation,
    ) {
        let msg_refs = array_refs!(input, BLOCKBYTES, BLOCKBYTES, BLOCKBYTES, BLOCKBYTES);
        // Note that count is incremented by *one* block, not four.
        *count += BLOCKBYTES as u128;
        let count_low = guts::u64x4([*count as u64; 4]);
        let count_high = guts::u64x4([(*count >> 64) as u64; 4]);
        let lastblock = guts::u64x4([0; 4]);
        let lastnode = guts::u64x4([0; 4]);
        implementation.compress4(
            state,
            msg_refs.0,
            msg_refs.1,
            msg_refs.2,
            msg_refs.3,
            &count_low,
            &count_high,
            &lastblock,
            &lastnode,
        );
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
                Self::compress(
                    array_ref!(self.buf, 0, DEGREE * BLOCKBYTES),
                    &mut self.transposed_leaf_words,
                    &mut self.count,
                    self.implementation,
                );
                self.buflen -= (DEGREE * BLOCKBYTES) as u16;
                // Now, if there's enough input still coming that all four leaves are going to get
                // more, we can do the second compression and clear the buffer. Otherwise, we have
                // to shift the remainder of the buffer to the left (and we know in this case the
                // direct-from-memory loop will get skipped too).
                if input.len() > (DEGREE - 1) * BLOCKBYTES {
                    Self::compress(
                        array_ref!(self.buf, DEGREE * BLOCKBYTES, DEGREE * BLOCKBYTES),
                        &mut self.transposed_leaf_words,
                        &mut self.count,
                        self.implementation,
                    );
                    self.buflen = 0;
                } else {
                    let (left, right) = self.buf.split_at_mut(DEGREE * BLOCKBYTES);
                    left[..self.buflen as usize].copy_from_slice(&right[..self.buflen as usize]);
                }
            }
        }

        // While there are more than 7 input blocks coming, then we know that we can perform a
        // compression and still have more input coming for each leaf. (We also know that the
        // buffer must have been emptied above.)
        while input.len() > ((2 * DEGREE) - 1) * BLOCKBYTES {
            debug_assert_eq!(0, self.buflen);
            let block = array_ref!(input, 0, DEGREE * BLOCKBYTES);
            Self::compress(
                block,
                &mut self.transposed_leaf_words,
                &mut self.count,
                self.implementation,
            );
            input = &input[DEGREE * BLOCKBYTES..];
        }

        // Buffer any remaining input, to be either compressed or finalized in a subsequent call.
        self.fill_buf(&mut input);
        debug_assert_eq!(0, input.len());
        self
    }

    /// Finalize the state and return a `Hash`. This method is idempotent, and calling it multiple
    /// times will give the same result. It's also possible to `update` with more input in between.
    pub fn finalize(&mut self) -> Hash {
        // Zero the buffer tail, since it might contain bytes from previous
        // compressions.
        let buflen = self.buflen as usize;
        for i in buflen..self.buf.len() {
            self.buf[i] = 0;
        }

        // Split the buffer into an array of blocks.
        let blocks = array_refs!(
            &self.buf, BLOCKBYTES, BLOCKBYTES, BLOCKBYTES, BLOCKBYTES, BLOCKBYTES, BLOCKBYTES,
            BLOCKBYTES, BLOCKBYTES
        );
        let blocks = [
            blocks.0, blocks.1, blocks.2, blocks.3, blocks.4, blocks.5, blocks.6, blocks.7,
        ];

        // Clone the leaf words. That keeps this method idempotent.
        let mut leaves_copy = self.transposed_leaf_words;

        // Count how many bytes each leaf still needs to compress. Note that
        // even if the number is zero, each leaf is going to get at least one
        // more compression to finalize it.
        let remaining_bytes_fn = |i: usize| {
            let first_block_start = i * BLOCKBYTES;
            let first_block_end = first_block_start + BLOCKBYTES;
            let second_block_start = (DEGREE + i) * BLOCKBYTES;
            let second_block_end = second_block_start + BLOCKBYTES;
            if buflen < first_block_start {
                0
            } else if buflen < first_block_end {
                buflen - first_block_start
            } else if buflen < second_block_start {
                BLOCKBYTES
            } else if buflen < second_block_end {
                BLOCKBYTES + buflen - second_block_start
            } else {
                2 * BLOCKBYTES
            }
        };
        let mut remaining = [
            remaining_bytes_fn(0),
            remaining_bytes_fn(1),
            remaining_bytes_fn(2),
            remaining_bytes_fn(3),
        ];

        // While all leaves still have compressions remaining, run them in a
        // batch. This might finalize some of the leaves. This loop will either
        // run once (even if some leaves have no input) or twice (if all leaves
        // have more than one block).
        let mut blocks_handled = 0;
        let mut count = self.count;
        loop {
            let mut count_low = guts::u64x4([count as u64; 4]);
            for i in 0..DEGREE {
                let take = cmp::min(BLOCKBYTES, remaining[i]);
                count_low[i] += take as u64;
                remaining[i] -= take;
            }
            let count_high = guts::u64x4([0; 4]);
            let mut lastblock = guts::u64x4([0; 4]);
            for i in 0..DEGREE {
                lastblock[i] = if remaining[i] == 0 { !0 } else { 0 };
            }
            let lastnode = guts::u64x4([0, 0, 0, lastblock[DEGREE - 1]]);
            self.implementation.compress4(
                &mut leaves_copy,
                &blocks[blocks_handled + 0],
                &blocks[blocks_handled + 1],
                &blocks[blocks_handled + 2],
                &blocks[blocks_handled + 3],
                &count_low,
                &count_high,
                &lastblock,
                &lastnode,
            );
            blocks_handled += DEGREE;
            // Note that at this point `count` only applies to leaves that
            // haven't been finalized.
            count += BLOCKBYTES as u128;
            if remaining.iter().any(|&rem| rem == 0) {
                break;
            }
        }

        // We just finished all the batch compressions we could. Some of the
        // leaves might have one more block left to finalize them. Untranspose
        // the state and then finalize those leaves, if any.
        let mut leaves_untransposed = [guts::u64x8([0u64; 8]); 4];
        let &mut [ref mut state0, ref mut state1, ref mut state2, ref mut state3] =
            &mut leaves_untransposed;
        self.implementation
            .untranspose4(&leaves_copy, state0, state1, state2, state3);
        for i in 0..DEGREE {
            if remaining[i] > 0 {
                self.implementation.compress(
                    &mut leaves_untransposed[i],
                    &blocks[blocks_handled + i],
                    count + remaining[i] as u128,
                    !0,
                    if i == DEGREE - 1 { !0 } else { 0 },
                );
            }
        }

        // Compress each of the four untransposed, finalized hashes into the
        // root words as input, using two compressions. Again we copy the words
        // to keep this method idempotent. Note that this uses the full-length
        // leaf hashes, not the shortened versions, even if the hash_length
        // parameter is set to a short value. Note also that, as mentioned
        // above, the root node doesn't hash any key bytes.
        let mut root_words_copy = self.root_words;
        for i in 0..DEGREE / 2 {
            let mut block = [0; BLOCKBYTES];
            LittleEndian::write_u64_into(&leaves_untransposed[2 * i][..], &mut block[0..OUTBYTES]);
            LittleEndian::write_u64_into(
                &leaves_untransposed[2 * i + 1][..],
                &mut block[OUTBYTES..2 * OUTBYTES],
            );
            self.implementation.compress(
                &mut root_words_copy,
                &block,
                ((i + 1) * BLOCKBYTES) as u128,
                if i == DEGREE / 2 - 1 { !0 } else { 0 },
                if i == DEGREE / 2 - 1 { !0 } else { 0 },
            );
        }

        Hash {
            bytes: crate::state_words_to_bytes(&root_words_copy),
            len: self.hash_length,
        }
    }

    /// Return the total number of bytes input so far.
    pub fn count(&self) -> u128 {
        4 * self.count + self.buflen as u128
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
    use byteorder::{ByteOrder, LittleEndian};

    // Paint a byte pattern that won't repeat, so that we don't accidentally miss buffer offset
    // bugs. This is the same as what Bao uses in its tests.
    pub(crate) fn paint_input(buf: &mut [u8]) {
        let mut offset = 0;
        let mut counter: u32 = 1;
        while offset < buf.len() {
            let mut bytes = [0; 4];
            LittleEndian::write_u32(&mut bytes, counter);
            let take = cmp::min(4, buf.len() - offset);
            buf[offset..][..take].copy_from_slice(&bytes[..take]);
            counter += 1;
            offset += take;
        }
    }

    // This is a simple reference implementation without the complicated buffering or parameter
    // support of the real implementation. We need this because the official test vectors don't
    // include any inputs large enough to exercise all the branches in the buffering logic.
    fn blake2bp_reference(input: &[u8]) -> Hash {
        let mut leaves = [
            Blake2bParams::new()
                .fanout(4)
                .max_depth(2)
                .node_offset(0)
                .inner_hash_length(OUTBYTES)
                .to_state(),
            Blake2bParams::new()
                .fanout(4)
                .max_depth(2)
                .node_offset(1)
                .inner_hash_length(OUTBYTES)
                .to_state(),
            Blake2bParams::new()
                .fanout(4)
                .max_depth(2)
                .node_offset(2)
                .inner_hash_length(OUTBYTES)
                .to_state(),
            Blake2bParams::new()
                .fanout(4)
                .max_depth(2)
                .node_offset(3)
                .inner_hash_length(OUTBYTES)
                .last_node(true)
                .to_state(),
        ];
        for (i, chunk) in input.chunks(BLOCKBYTES).enumerate() {
            leaves[i % 4].update(chunk);
        }
        let mut root = Blake2bParams::new()
            .fanout(4)
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
        paint_input(&mut buf);
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
