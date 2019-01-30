use crate::*;
use core::mem;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub const MAX_DEGREE: usize = 4;

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub const MAX_DEGREE: usize = 4;

// Variants other than Portable are unreachable in no_std, unless CPU features
// are explicitly enabled for the build with e.g. RUSTFLAGS="-C target-feature=avx2".
// This might change in the future if is_x86_feature_detected moves into libcore.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Platform {
    Portable,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    SSE41,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    AVX2,
}

#[derive(Clone, Copy, Debug)]
pub struct Implementation(Platform);

impl Implementation {
    pub fn detect() -> Self {
        // Try the different implementations in order of how fast/modern they
        // are. Currently on non-x86, everything just uses portable.
        if let Some(avx2_impl) = Self::avx2_if_supported() {
            avx2_impl
        } else if let Some(sse41_impl) = Self::sse41_if_supported() {
            sse41_impl
        } else {
            Self::portable()
        }
    }

    pub fn portable() -> Self {
        Implementation(Platform::Portable)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[allow(unreachable_code)]
    pub fn sse41_if_supported() -> Option<Self> {
        // Check whether SSE4.1 support is assumed by the build.
        #[cfg(target_feature = "sse4.1")]
        {
            return Some(Implementation(Platform::SSE41));
        }
        // Otherwise dynamically check for support if we can.
        #[cfg(feature = "std")]
        {
            if is_x86_feature_detected!("sse4.1") {
                return Some(Implementation(Platform::SSE41));
            }
        }
        None
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[allow(unreachable_code)]
    pub fn avx2_if_supported() -> Option<Self> {
        // Check whether AVX2 support is assumed by the build.
        #[cfg(target_feature = "avx2")]
        {
            return Some(Implementation(Platform::AVX2));
        }
        // Otherwise dynamically check for support if we can.
        #[cfg(feature = "std")]
        {
            if is_x86_feature_detected!("avx2") {
                return Some(Implementation(Platform::AVX2));
            }
        }
        None
    }

    pub fn degree(&self) -> usize {
        match self.0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Platform::AVX2 => 4,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Platform::SSE41 => 2,
            Platform::Portable => 1,
        }
    }

    pub fn compress(
        &self,
        state_words: &mut u64x8,
        msg: &[u8; BLOCKBYTES],
        count: u128,
        lastblock: u64,
        lastnode: u64,
    ) {
        match self.0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Platform::AVX2 => unsafe {
                avx2::compress(state_words, msg, count, lastblock, lastnode)
            },
            // The SSE4.1 implementation of compress hasn't yet been ported
            // from https://github.com/BLAKE2/BLAKE2/blob/master/sse/blake2b-round.h,
            // so for SSE4.1 falls back to portable.
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Platform::SSE41 => portable::compress(state_words, msg, count, lastblock, lastnode),
            Platform::Portable => portable::compress(state_words, msg, count, lastblock, lastnode),
        }
    }

    pub fn transpose2(&self, words0: &u64x8, words1: &u64x8) -> [u64x2; 8] {
        // Currently there's only the portable implementation of transpose2.
        portable::transpose2(words0, words1)
    }

    pub fn untranspose2(&self, transposed: &[u64x2; 8], out0: &mut u64x8, out1: &mut u64x8) {
        // Currently there's only the portable implementation of untranspose2.
        portable::untranspose2(transposed, out0, out1)
    }

    pub fn compress2(
        &self,
        transposed_state_words: &mut [u64x2; 8],
        msg0: &[u8; BLOCKBYTES],
        msg1: &[u8; BLOCKBYTES],
        count_low: &u64x2,
        count_high: &u64x2,
        lastblock: &u64x2,
        lastnode: &u64x2,
    ) {
        match self.0 {
            // Currently there's no AVX2 implementation of compress2, fall back to SSE4.1.
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Platform::AVX2 | Platform::SSE41 => unsafe {
                sse41::compress2_transposed(
                    transposed_state_words,
                    msg0,
                    msg1,
                    count_low,
                    count_high,
                    lastblock,
                    lastnode,
                );
            },
            Platform::Portable => {
                portable::compress2_transposed(
                    transposed_state_words,
                    msg0,
                    msg1,
                    count_low,
                    count_high,
                    lastblock,
                    lastnode,
                );
            }
        }
    }

    pub fn transpose4(
        &self,
        words0: &u64x8,
        words1: &u64x8,
        words2: &u64x8,
        words3: &u64x8,
    ) -> [u64x4; 8] {
        match self.0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Platform::AVX2 => unsafe { avx2::transpose4(words0, words1, words2, words3) },
            // There is no SSE4.1 implementation of transpose4 yet.
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Platform::SSE41 => portable::transpose4(words0, words1, words2, words3),
            Platform::Portable => portable::transpose4(words0, words1, words2, words3),
        }
    }

    pub fn untranspose4(
        &self,
        transposed: &[u64x4; 8],
        out0: &mut u64x8,
        out1: &mut u64x8,
        out2: &mut u64x8,
        out3: &mut u64x8,
    ) {
        match self.0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Platform::AVX2 => unsafe { avx2::untranspose4(transposed, out0, out1, out2, out3) },
            // There is no SSE4.1 implementation of untranspose4 yet.
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Platform::SSE41 => portable::untranspose4(transposed, out0, out1, out2, out3),
            Platform::Portable => portable::untranspose4(transposed, out0, out1, out2, out3),
        }
    }

    pub fn compress4(
        &self,
        transposed_state_words: &mut [u64x4; 8],
        msg0: &[u8; BLOCKBYTES],
        msg1: &[u8; BLOCKBYTES],
        msg2: &[u8; BLOCKBYTES],
        msg3: &[u8; BLOCKBYTES],
        count_low: &u64x4,
        count_high: &u64x4,
        lastblock: &u64x4,
        lastnode: &u64x4,
    ) {
        match self.0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Platform::AVX2 => unsafe {
                avx2::compress4_transposed(
                    transposed_state_words,
                    msg0,
                    msg1,
                    msg2,
                    msg3,
                    count_low,
                    count_high,
                    lastblock,
                    lastnode,
                );
            },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Platform::SSE41 => unsafe {
                sse41::compress4_transposed(
                    transposed_state_words,
                    msg0,
                    msg1,
                    msg2,
                    msg3,
                    count_low,
                    count_high,
                    lastblock,
                    lastnode,
                );
            },
            Platform::Portable => {
                portable::compress4_transposed(
                    transposed_state_words,
                    msg0,
                    msg1,
                    msg2,
                    msg3,
                    count_low,
                    count_high,
                    lastblock,
                    lastnode,
                );
            }
        }
    }

    pub fn compress1_loop(
        &self,
        state: &mut u64x8,
        input: &[u8],
        count: u128,
        last_block: u64,
        last_node: u64,
        blocks: usize,
        stride: usize,
        buffer_tail: usize,
    ) {
        match self.0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Platform::AVX2 => unsafe {
                avx2::compress1_loop(
                    state,
                    input,
                    count,
                    last_block,
                    last_node,
                    blocks,
                    stride,
                    buffer_tail,
                );
            },
            // Note that there's an SSE version of compress1 in the official C
            // implementation, but I haven't ported it yet.
            _ => {
                portable::compress1_loop(
                    state,
                    input,
                    count,
                    last_block,
                    last_node,
                    blocks,
                    stride,
                    buffer_tail,
                );
            }
        }
    }

    pub fn compress2_loop(
        &self,
        state0: &mut u64x8,
        state1: &mut u64x8,
        input0: &[u8],
        input1: &[u8],
        count_low: &u64x2,
        count_high: &u64x2,
        last_block: &u64x2,
        last_node: &u64x2,
        blocks: usize,
        stride: usize,
        buffer_tail: &u64x2,
    ) {
        match self.0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Platform::AVX2 | Platform::SSE41 => unsafe {
                sse41::compress2_loop(
                    state0,
                    state1,
                    input0,
                    input1,
                    count_low,
                    count_high,
                    last_block,
                    last_node,
                    blocks,
                    stride,
                    buffer_tail,
                );
            },
            Platform::Portable => {
                self.compress1_loop(
                    state0,
                    input0,
                    count_low[0] as u128 + ((count_high[0] as u128) << 64),
                    last_block[0],
                    last_node[0],
                    blocks,
                    stride,
                    buffer_tail[0] as usize,
                );
                self.compress1_loop(
                    state1,
                    input1,
                    count_low[1] as u128 + ((count_high[1] as u128) << 64),
                    last_block[1],
                    last_node[1],
                    blocks,
                    stride,
                    buffer_tail[1] as usize,
                );
            }
        }
    }

    pub fn compress4_loop(
        &self,
        state0: &mut u64x8,
        state1: &mut u64x8,
        state2: &mut u64x8,
        state3: &mut u64x8,
        input0: &[u8],
        input1: &[u8],
        input2: &[u8],
        input3: &[u8],
        count_low: &u64x4,
        count_high: &u64x4,
        last_block: &u64x4,
        last_node: &u64x4,
        blocks: usize,
        stride: usize,
        buffer_tail: &u64x4,
    ) {
        match self.0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Platform::AVX2 => unsafe {
                avx2::compress4_loop(
                    state0,
                    state1,
                    state2,
                    state3,
                    input0,
                    input1,
                    input2,
                    input3,
                    count_low,
                    count_high,
                    last_block,
                    last_node,
                    blocks,
                    stride,
                    buffer_tail,
                );
            },
            _ => {
                // Performance note: Would it be faster to add a compress4_loop
                // interface to the sse41 implementation, which used a single
                // loop to process the inputs together instead of one after the
                // other? For BLAKE2bp it probably would be, because you'll get
                // better cache performance by traversing the input once
                // instead of twice. But for tree hashes probably not, since
                // the inputs are usually adjacent or nearly-adjacent rather
                // than overlapping. Tree hash performance is our priority
                // here, and also doing things this way is simpler.
                self.compress2_loop(
                    state0,
                    state1,
                    input0,
                    input1,
                    &count_low.split()[0],
                    &count_high.split()[0],
                    &last_block.split()[0],
                    &last_node.split()[0],
                    blocks,
                    stride,
                    &buffer_tail.split()[0],
                );
                self.compress2_loop(
                    state2,
                    state3,
                    input2,
                    input3,
                    &count_low.split()[1],
                    &count_high.split()[1],
                    &last_block.split()[1],
                    &last_node.split()[1],
                    blocks,
                    stride,
                    &buffer_tail.split()[1],
                );
            }
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C, align(16))]
pub struct u64x2(pub [u64; 2]);

impl core::ops::Deref for u64x2 {
    type Target = [u64; 2];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl core::ops::DerefMut for u64x2 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C, align(32))]
pub struct u64x4(pub [u64; 4]);

impl u64x4 {
    #[inline(always)]
    pub(crate) fn split(&self) -> &[u64x2; 2] {
        // Safety note: The 32-byte alignment of u64x4 guarantees that each
        // half of it will be 16-byte aligned, and the C repr guarantees that
        // the layout is exactly four packed u64's.
        unsafe { mem::transmute(self) }
    }

    #[inline(always)]
    pub(crate) fn split_mut(&mut self) -> &mut [u64x2; 2] {
        // Safety note: The 32-byte alignment of u64x4 guarantees that each
        // half of it will be 16-byte aligned, and the C repr guarantees that
        // the layout is exactly four packed u64's.
        unsafe { mem::transmute(self) }
    }
}

impl core::ops::Deref for u64x4 {
    type Target = [u64; 4];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl core::ops::DerefMut for u64x4 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C, align(64))]
pub struct u64x8(pub [u64; 8]);

impl u64x8 {
    #[inline(always)]
    pub(crate) fn split(&self) -> &[u64x4; 2] {
        // Safety note: The 64-byte alignment of u64x8 guarantees that each
        // half of it will be 32-byte aligned, and the C repr guarantees that
        // the layout is exactly eight packed u64's.
        unsafe { mem::transmute(self) }
    }

    #[inline(always)]
    pub(crate) fn split_mut(&mut self) -> &mut [u64x4; 2] {
        // Safety note: The 64-byte alignment of u64x8 guarantees that each
        // half of it will be 32-byte aligned, and the C repr guarantees that
        // the layout is exactly eight packed u64's.
        unsafe { mem::transmute(self) }
    }
}

impl core::ops::Deref for u64x8 {
    type Target = [u64; 8];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl core::ops::DerefMut for u64x8 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_detection() {
        assert_eq!(Platform::Portable, Implementation::portable().0);

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[cfg(feature = "std")]
        {
            if is_x86_feature_detected!("avx2") {
                assert_eq!(Platform::AVX2, Implementation::detect().0);
                assert_eq!(
                    Platform::AVX2,
                    Implementation::avx2_if_supported().unwrap().0
                );
                assert_eq!(
                    Platform::SSE41,
                    Implementation::sse41_if_supported().unwrap().0
                );
            } else if is_x86_feature_detected!("sse4.1") {
                assert_eq!(Platform::SSE41, Implementation::detect().0);
                assert!(Implementation::avx2_if_supported().is_none());
                assert_eq!(
                    Platform::SSE41,
                    Implementation::sse41_if_supported().unwrap().0
                );
            } else {
                assert_eq!(Platform::Portable, Implementation::detect().0);
                assert!(Implementation::avx2_if_supported().is_none());
                assert!(Implementation::sse41_if_supported().is_none());
            }
        }
    }

    fn input_state_words(i: u64) -> u64x8 {
        let mut words = u64x8([0; 8]);
        for j in 0..words.len() {
            words[j] = i + j as u64;
        }
        words
    }

    fn input_msg_block(i: u64) -> [u8; 128] {
        let mut block = [0; 128];
        for j in 0..block.len() {
            block[j] = (i + j as u64) as u8;
        }
        block
    }

    fn exercise_1(imp: Implementation, i: u64) -> u64x8 {
        let mut state = input_state_words(i);
        let block = input_msg_block(0x10 + i);
        let count_low = 0x20 + i;
        let count_high = 0x30 + i;
        let count = count_low as u128 + ((count_high as u128) << 64);
        let lastblock = 0x40 + i;
        let lastnode = 0x50 + i;
        imp.compress(&mut state, &block, count, lastblock, lastnode);
        state
    }

    fn exercise_2(imp: Implementation, i: u64) -> [u64x8; 2] {
        let mut state0 = input_state_words(i);
        let mut state1 = input_state_words(i + 1);
        let block0 = input_msg_block(0x10 + i);
        let block1 = input_msg_block(0x10 + i + 1);
        let count_low = u64x2([0x20 + i, 0x20 + i + 1]);
        let count_high = u64x2([0x30 + i, 0x30 + i + 1]);
        let lastblock = u64x2([0x40 + i, 0x40 + i + 1]);
        let lastnode = u64x2([0x50 + i, 0x50 + i + 1]);
        let mut transposed = imp.transpose2(&state0, &state1);
        imp.compress2(
            &mut transposed,
            &block0,
            &block1,
            &count_low,
            &count_high,
            &lastblock,
            &lastnode,
        );
        imp.untranspose2(&transposed, &mut state0, &mut state1);
        [state0, state1]
    }

    fn exercise_4(imp: Implementation, i: u64) -> [u64x8; 4] {
        let mut state0 = input_state_words(i);
        let mut state1 = input_state_words(i + 1);
        let mut state2 = input_state_words(i + 2);
        let mut state3 = input_state_words(i + 3);
        let block0 = input_msg_block(0x10 + i);
        let block1 = input_msg_block(0x10 + i + 1);
        let block2 = input_msg_block(0x10 + i + 2);
        let block3 = input_msg_block(0x10 + i + 3);
        let count_low = u64x4([0x20 + i, 0x20 + i + 1, 0x20 + i + 2, 0x20 + i + 3]);
        let count_high = u64x4([0x30 + i, 0x30 + i + 1, 0x30 + i + 2, 0x30 + i + 3]);
        let lastblock = u64x4([0x40 + i, 0x40 + i + 1, 0x40 + i + 2, 0x40 + i + 3]);
        let lastnode = u64x4([0x50 + i, 0x50 + i + 1, 0x50 + i + 2, 0x50 + i + 3]);
        let mut transposed = imp.transpose4(&state0, &state1, &state2, &state3);
        imp.compress4(
            &mut transposed,
            &block0,
            &block1,
            &block2,
            &block3,
            &count_low,
            &count_high,
            &lastblock,
            &lastnode,
        );
        imp.untranspose4(
            &transposed,
            &mut state0,
            &mut state1,
            &mut state2,
            &mut state3,
        );
        [state0, state1, state2, state3]
    }

    // Make sure the different portable APIs all agree with each other. We
    // don't use known test vectors here; that happens in vector_tests.rs.
    #[test]
    fn test_portable() {
        let portable = Implementation::portable();

        // Gather the one-at-a-time compression results.
        let expected0 = exercise_1(portable, 0);
        let expected1 = exercise_1(portable, 1);
        let expected2 = exercise_1(portable, 2);
        let expected3 = exercise_1(portable, 3);

        // Check that compress2 gives the same answer.
        let two_at_a_time01 = exercise_2(portable, 0);
        let two_at_a_time12 = exercise_2(portable, 2);
        assert_eq!(expected0, two_at_a_time01[0]);
        assert_eq!(expected1, two_at_a_time01[1]);
        assert_eq!(expected2, two_at_a_time12[0]);
        assert_eq!(expected3, two_at_a_time12[1]);

        // Check that compress4 gives the same answer.
        let four_at_a_time = exercise_4(portable, 0);
        assert_eq!(expected0, four_at_a_time[0]);
        assert_eq!(expected1, four_at_a_time[1]);
        assert_eq!(expected2, four_at_a_time[2]);
        assert_eq!(expected3, four_at_a_time[3]);
    }

    // Make sure that SSE41 agrees with portable. We don't use known test
    // vectors here; that happens in vector_tests.rs.
    #[test]
    fn test_sse41() {
        let portable = Implementation::portable();
        let sse41 = if let Some(imp) = Implementation::sse41_if_supported() {
            imp
        } else {
            // No SSE4.1 support. Short circuit the test.
            return;
        };

        assert_eq!(exercise_1(portable, 0), exercise_1(sse41, 0));
        assert_eq!(exercise_2(portable, 0), exercise_2(sse41, 0));
        assert_eq!(exercise_4(portable, 0), exercise_4(sse41, 0));
    }

    // Make sure that AVX2 agrees with portable. We don't use known test
    // vectors here; that happens in vector_tests.rs.
    #[test]
    fn test_avx2() {
        let portable = Implementation::portable();
        let avx2 = if let Some(imp) = Implementation::avx2_if_supported() {
            imp
        } else {
            // No AVX2 support. Short circuit the test.
            return;
        };

        assert_eq!(exercise_1(portable, 0), exercise_1(avx2, 0));
        assert_eq!(exercise_2(portable, 0), exercise_2(avx2, 0));
        assert_eq!(exercise_4(portable, 0), exercise_4(avx2, 0));
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_avx2_compress4_loop() {
        if Implementation::avx2_if_supported().is_none() {
            return;
        }

        let mut inputs = [[0; BLOCKBYTES * 4]; 4];
        for i in 0..4 {
            for j in 0..BLOCKBYTES * 4 {
                inputs[i][j] = (i + j) as u8;
            }
        }

        // First compute expected;
        let mut expected = [
            input_state_words(0),
            input_state_words(1),
            input_state_words(2),
            input_state_words(3),
        ];
        for i in 0..4 {
            for block in 0..4 {
                portable::compress(
                    &mut expected[i],
                    array_ref!(inputs[i], block * BLOCKBYTES, BLOCKBYTES),
                    ((block + 1) * BLOCKBYTES) as u128,
                    if block == 3 { !0 } else { 0 },
                    if block == 3 { !0 } else { 0 },
                );
            }
        }

        // Now do the loop implementation.
        let mut loop_state0 = input_state_words(0);
        let mut loop_state1 = input_state_words(1);
        let mut loop_state2 = input_state_words(2);
        let mut loop_state3 = input_state_words(3);
        unsafe {
            avx2::compress4_loop(
                &mut loop_state0,
                &mut loop_state1,
                &mut loop_state2,
                &mut loop_state3,
                &inputs[0],
                &inputs[1],
                &inputs[2],
                &inputs[3],
                &u64x4([0; 4]),
                &u64x4([0; 4]),
                &u64x4([!0; 4]),
                &u64x4([!0; 4]),
                4,
                1,
                &u64x4([0; 4]),
            );
        }

        assert_eq!(expected[0], loop_state0);
        assert_eq!(expected[1], loop_state1);
        assert_eq!(expected[2], loop_state2);
        assert_eq!(expected[3], loop_state3);
    }

    fn exercise_cases<F>(mut f: F)
    where
        F: FnMut(usize, usize, u128, bool, bool, usize, usize),
    {
        // Chose counts to hit the relevant overflow cases.
        let counts = &[
            0u128,
            (1u128 << 64) - BLOCKBYTES as u128,
            0u128.wrapping_sub(BLOCKBYTES as u128),
        ];
        for invocations in 1..=2 {
            for blocks_per_invoc in 1..=3 {
                for &count in counts {
                    for &last_block in &[true, false] {
                        for &last_node in &[true, false] {
                            for stride in 1..=3 {
                                for &buffer_tail in &[0, 1, BLOCKBYTES - 1] {
                                    eprintln!("\ncase -----");
                                    dbg!(invocations);
                                    dbg!(blocks_per_invoc);
                                    dbg!(count);
                                    dbg!(last_block);
                                    dbg!(last_node);
                                    dbg!(stride);
                                    dbg!(buffer_tail);
                                    f(
                                        invocations,
                                        blocks_per_invoc,
                                        count,
                                        last_block,
                                        last_node,
                                        stride,
                                        buffer_tail,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // For various loop lengths and finalization parameters, make sure that the
    // implementation gives the same answer as the portable implementation does
    // when invoked one block at a time. (So even the portable implementation
    // itself is being tested here, to make sure its loop is correct.) Note
    // that this doesn't include any fixed test vectors; those are taken from
    // the blake2-kat.json file (copied from upstream) and tested elsewhere.
    fn exercise_compress1_loop(implementation: Implementation) {
        let mut input = [0; 100 * BLOCKBYTES];
        paint_test_input(&mut input);
        exercise_cases(
            |invocations, blocks_per_invoc, count, last_block, last_node, stride, buffer_tail| {
                // Use the portable implementation, one block at a time, to
                // compute the final state that we expect.
                let mut reference_state = input_state_words(0);
                for block in 0..invocations * blocks_per_invoc {
                    let input_block = array_ref!(&input, block * stride * BLOCKBYTES, BLOCKBYTES);
                    let is_last_block = block == invocations * blocks_per_invoc - 1;
                    let maybe_tail = if is_last_block { buffer_tail } else { 0 };
                    portable::compress1_loop(
                        &mut reference_state,
                        input_block,
                        count.wrapping_add((block * BLOCKBYTES) as u128),
                        u64_flag(is_last_block && last_block),
                        u64_flag(is_last_block && last_node),
                        1, // blocks, one at a time
                        stride,
                        maybe_tail,
                    );
                }

                // Do the same thing with the implementation
                // under test, and make sure they're the same.
                let mut test_state = input_state_words(0);
                for invocation in 0..invocations {
                    let is_last_invoc = invocation == invocations - 1;
                    let maybe_tail = if is_last_invoc { buffer_tail } else { 0 };
                    implementation.compress1_loop(
                        &mut test_state,
                        &input[invocation * blocks_per_invoc * stride * BLOCKBYTES..],
                        count.wrapping_add((invocation * blocks_per_invoc * BLOCKBYTES) as u128),
                        u64_flag(is_last_invoc && last_block),
                        u64_flag(is_last_invoc && last_node),
                        blocks_per_invoc,
                        stride,
                        maybe_tail,
                    );
                }
                assert_eq!(reference_state, test_state);
            },
        );
    }

    #[test]
    fn test_compress1_loop_portable() {
        exercise_compress1_loop(Implementation::portable());
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_compress1_loop_sse41() {
        // Currently this just falls back to portable, but we test it anyway.
        if let Some(imp) = Implementation::sse41_if_supported() {
            exercise_compress1_loop(imp);
        }
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_compress1_loop_avx2() {
        if let Some(imp) = Implementation::avx2_if_supported() {
            exercise_compress1_loop(imp);
        }
    }

    // Similar to exercise_compress1_loop above.
    fn exercise_compress2_loop(implementation: Implementation) {
        let mut input_buffer = [0; 100 * BLOCKBYTES];
        paint_test_input(&mut input_buffer);
        let inputs = [&input_buffer[0..], &input_buffer[1..]];
        exercise_cases(
            |invocations, blocks_per_invoc, count, last_block, last_node, stride, buffer_tail| {
                // Use the portable compress1_loop implementation to compute a
                // reference state for each input separately.
                let mut reference_states = [input_state_words(0), input_state_words(1)];
                for i in 0..reference_states.len() {
                    portable::compress1_loop(
                        &mut reference_states[i],
                        inputs[i],
                        count,
                        u64_flag(last_block),
                        u64_flag(last_node),
                        invocations * blocks_per_invoc,
                        stride,
                        buffer_tail,
                    );
                }

                // Do the same thing in parallel with the
                // implementation under test under test, and
                // make sure the result is the same.
                let mut test_state0 = input_state_words(0);
                let mut test_state1 = input_state_words(1);
                for invocation in 0..invocations {
                    let is_last_invoc = invocation == invocations - 1;
                    let invoc_count =
                        count.wrapping_add((invocation * blocks_per_invoc * BLOCKBYTES) as u128);
                    let count_low = u64x2([invoc_count as u64; 2]);
                    let count_high = u64x2([(invoc_count >> 64) as u64; 2]);
                    let last_block = u64x2([u64_flag(is_last_invoc && last_block); 2]);
                    let last_node = u64x2([u64_flag(is_last_invoc && last_node); 2]);
                    let maybe_tail = u64x2([if is_last_invoc { buffer_tail as u64 } else { 0 }; 2]);
                    implementation.compress2_loop(
                        &mut test_state0,
                        &mut test_state1,
                        &inputs[0][invocation * blocks_per_invoc * stride * BLOCKBYTES..],
                        &inputs[1][invocation * blocks_per_invoc * stride * BLOCKBYTES..],
                        &count_low,
                        &count_high,
                        &last_block,
                        &last_node,
                        blocks_per_invoc,
                        stride,
                        &maybe_tail,
                    );
                }
                assert_eq!(reference_states[0], test_state0);
                assert_eq!(reference_states[1], test_state1);
            },
        );
    }

    #[test]
    fn test_compress2_loop_portable() {
        exercise_compress2_loop(Implementation::portable());
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_compress2_loop_sse41() {
        if let Some(imp) = Implementation::sse41_if_supported() {
            exercise_compress2_loop(imp);
        }
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_compress2_loop_avx2() {
        // Currently this just falls back to SSE4.1, but we test it anyway.
        if let Some(imp) = Implementation::avx2_if_supported() {
            exercise_compress2_loop(imp);
        }
    }
}
