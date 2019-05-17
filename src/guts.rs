use crate::*;
use arrayref::array_ref;
use core::cmp;
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
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if let Some(avx2_impl) = Self::avx2_if_supported() {
                return avx2_impl;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if let Some(sse41_impl) = Self::sse41_if_supported() {
                return sse41_impl;
            }
        }
        Self::portable()
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

    pub fn compress1_loop(
        &self,
        input: &[u8],
        words: &mut u64x8,
        count: u128,
        last_node: LastNode,
        finalize: Finalize,
        stride: Stride,
    ) {
        match self.0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Platform::AVX2 => unsafe {
                avx2::compress1_loop(input, words, count, last_node, finalize, stride);
            },
            // Note that there's an SSE version of compress1 in the official C
            // implementation, but I haven't ported it yet.
            _ => {
                portable::compress1_loop(input, words, count, last_node, finalize, stride);
            }
        }
    }

    pub fn compress2_loop(&self, jobs: &mut [Job; 2], finalize: Finalize, stride: Stride) {
        match self.0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Platform::AVX2 | Platform::SSE41 => unsafe {
                sse41::compress2_loop(jobs, finalize, stride)
            },
            _ => panic!("unsupported"),
        }
    }

    pub fn compress4_loop(&self, jobs: &mut [Job; 4], finalize: Finalize, stride: Stride) {
        match self.0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Platform::AVX2 => unsafe { avx2::compress4_loop(jobs, finalize, stride) },
            _ => panic!("unsupported"),
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

pub struct Job<'a, 'b> {
    pub input: &'a [u8],
    pub words: &'b mut u64x8,
    pub count: u128,
    pub last_node: LastNode,
}

impl<'a, 'b> core::fmt::Debug for Job<'a, 'b> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // NB: Don't print the words. Leaking them would allow length extension.
        write!(
            f,
            "Job {{ input_len: {}, count: {}, last_node: {} }}",
            self.input.len(),
            self.count,
            self.last_node.yes(),
        )
    }
}

// Finalize could just be a bool, but this is easier to read at callsites.
#[derive(Clone, Copy, Debug)]
pub enum Finalize {
    Yes,
    No,
}

impl Finalize {
    pub fn yes(&self) -> bool {
        match self {
            Finalize::Yes => true,
            Finalize::No => false,
        }
    }
}

// Like Finalize, this is easier to read at callsites.
#[derive(Clone, Copy, Debug)]
pub enum LastNode {
    Yes,
    No,
}

impl LastNode {
    pub fn yes(&self) -> bool {
        match self {
            LastNode::Yes => true,
            LastNode::No => false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Stride {
    Serial,   // BLAKE2b/BLAKE2s
    Parallel, // BLAKE2bp/BLAKE2sp
}

impl Stride {
    #[inline(always)]
    pub fn padded_blockbytes(&self) -> usize {
        match self {
            Stride::Serial => BLOCKBYTES,
            Stride::Parallel => blake2bp::DEGREE * BLOCKBYTES,
        }
    }
}

#[inline(always)]
pub(crate) fn u64_flag(flag: bool) -> u64 {
    if flag {
        !0
    } else {
        0
    }
}

// Pull a array reference at the given offset straight from the input, if
// there's a full block of input available. If there's only a partial block,
// copy it into the provided buffer, and return an array reference that. Along
// with the array, return the number of bytes of real input, and whether the
// input can be finalized (i.e. whether there aren't any more bytes after this
// block). Note that this is written so that the optimizer can elide bounds
// checks, see: https://godbolt.org/z/0hH2bC
#[inline(always)]
pub fn final_block<'a>(
    input: &'a [u8],
    offset: usize,
    buffer: &'a mut [u8; BLOCKBYTES],
    stride: Stride,
) -> (&'a [u8; BLOCKBYTES], usize, bool) {
    let capped_offset = cmp::min(offset, input.len());
    let offset_slice = &input[capped_offset..];
    if offset_slice.len() >= BLOCKBYTES {
        let block = array_ref!(offset_slice, 0, BLOCKBYTES);
        let should_finalize = offset_slice.len() <= stride.padded_blockbytes();
        (block, BLOCKBYTES, should_finalize)
    } else {
        // Copy the final block to the front of the block buffer. The rest of
        // the buffer is assumed to be initialized to zero.
        buffer[..offset_slice.len()].copy_from_slice(offset_slice);
        (buffer, offset_slice.len(), true)
    }
}

#[inline(always)]
pub fn input_debug_asserts(input: &[u8], finalize: Finalize) {
    // If we're not finalizing, the input must not be empty, and it must be an
    // even multiple of the block size.
    if !finalize.yes() {
        debug_assert!(!input.is_empty());
        debug_assert_eq!(0, input.len() % BLOCKBYTES);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use arrayvec::ArrayVec;

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

    fn exercise_cases<F>(mut f: F)
    where
        F: FnMut(Stride, usize, LastNode, Finalize, u128),
    {
        // Chose counts to hit the relevant overflow cases.
        let counts = &[
            0u128,
            (1u128 << 64) - BLOCKBYTES as u128,
            0u128.wrapping_sub(BLOCKBYTES as u128),
        ];
        for &stride in &[Stride::Serial, Stride::Parallel] {
            let lengths = [
                0,
                1,
                BLOCKBYTES - 1,
                BLOCKBYTES,
                BLOCKBYTES + 1,
                2 * BLOCKBYTES - 1,
                2 * BLOCKBYTES,
                2 * BLOCKBYTES + 1,
                stride.padded_blockbytes() - 1,
                stride.padded_blockbytes(),
                stride.padded_blockbytes() + 1,
                2 * stride.padded_blockbytes() - 1,
                2 * stride.padded_blockbytes(),
                2 * stride.padded_blockbytes() + 1,
            ];
            for &length in &lengths {
                for &last_node in &[LastNode::No, LastNode::Yes] {
                    for &finalize in &[Finalize::No, Finalize::Yes] {
                        if !finalize.yes() && (length == 0 || length % BLOCKBYTES != 0) {
                            // Skip these cases, they're invalid.
                            continue;
                        }
                        for &count in counts {
                            // eprintln!("\ncase -----");
                            // dbg!(stride);
                            // dbg!(length);
                            // dbg!(last_node);
                            // dbg!(finalize);
                            // dbg!(count);

                            f(stride, length, last_node, finalize, count);
                        }
                    }
                }
            }
        }
    }

    fn initial_test_words(input_index: usize) -> u64x8 {
        crate::Params::new()
            .node_offset(input_index as u64)
            .to_words()
    }

    // Use the portable implementation, one block at a time, to compute the
    // final state words expected for a given test case.
    fn reference_compression(
        input: &[u8],
        stride: Stride,
        last_node: LastNode,
        finalize: Finalize,
        mut count: u128,
        input_index: usize,
    ) -> u64x8 {
        let mut words = initial_test_words(input_index);
        let mut offset = 0;
        while offset == 0 || offset < input.len() {
            let block_size = cmp::min(BLOCKBYTES, input.len() - offset);
            let maybe_finalize = if offset + stride.padded_blockbytes() < input.len() {
                Finalize::No
            } else {
                finalize
            };
            portable::compress1_loop(
                &input[offset..][..block_size],
                &mut words,
                count,
                last_node,
                maybe_finalize,
                Stride::Serial,
            );
            offset += stride.padded_blockbytes();
            count = count.wrapping_add(BLOCKBYTES as u128);
        }
        words
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

        exercise_cases(|stride, length, last_node, finalize, count| {
            let reference_words =
                reference_compression(&input[..length], stride, last_node, finalize, count, 0);

            let mut test_words = initial_test_words(0);
            implementation.compress1_loop(
                &input[..length],
                &mut test_words,
                count,
                last_node,
                finalize,
                stride,
            );
            assert_eq!(reference_words, test_words);
        });
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

    // I use ArrayVec everywhere in here becuase currently these tests pass
    // under no_std. I might decide that's not worth maintaining at some point,
    // since really all we care about with no_std is that the library builds,
    // but for now it's here. Everything is keyed off of this N constant so
    // that it's easy to copy the code to exercise_compress4_loop.
    fn exercise_compress2_loop(implementation: Implementation) {
        const N: usize = 2;

        let mut input_buffer = [0; 100 * BLOCKBYTES];
        paint_test_input(&mut input_buffer);
        let mut inputs = ArrayVec::<[_; N]>::new();
        for i in 0..N {
            inputs.push(&input_buffer[i..]);
        }

        exercise_cases(|stride, length, last_node, finalize, count| {
            let mut reference_words = ArrayVec::<[_; N]>::new();
            for i in 0..N {
                let words = reference_compression(
                    &inputs[i][..length],
                    stride,
                    last_node,
                    finalize,
                    count.wrapping_add((i * BLOCKBYTES) as u128),
                    i,
                );
                reference_words.push(words);
            }

            let mut test_words = ArrayVec::<[_; N]>::new();
            for i in 0..N {
                test_words.push(initial_test_words(i));
            }
            let mut jobs = ArrayVec::<[_; N]>::new();
            for (i, words) in test_words.iter_mut().enumerate() {
                jobs.push(Job {
                    input: &inputs[i][..length],
                    words,
                    count: count.wrapping_add((i * BLOCKBYTES) as u128),
                    last_node,
                });
            }
            let mut jobs = jobs.into_inner().expect("full");
            implementation.compress2_loop(&mut jobs, finalize, stride);

            for i in 0..N {
                assert_eq!(reference_words[i], test_words[i], "words {} unequal", i);
            }
        });
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

    // Copied from exercise_compress2_loop, with a different value of N and an
    // interior call to compress4_loop.
    fn exercise_compress4_loop(implementation: Implementation) {
        const N: usize = 4;

        let mut input_buffer = [0; 100 * BLOCKBYTES];
        paint_test_input(&mut input_buffer);
        let mut inputs = ArrayVec::<[_; N]>::new();
        for i in 0..N {
            inputs.push(&input_buffer[i..]);
        }

        exercise_cases(|stride, length, last_node, finalize, count| {
            let mut reference_words = ArrayVec::<[_; N]>::new();
            for i in 0..N {
                let words = reference_compression(
                    &inputs[i][..length],
                    stride,
                    last_node,
                    finalize,
                    count.wrapping_add((i * BLOCKBYTES) as u128),
                    i,
                );
                reference_words.push(words);
            }

            let mut test_words = ArrayVec::<[_; N]>::new();
            for i in 0..N {
                test_words.push(initial_test_words(i));
            }
            let mut jobs = ArrayVec::<[_; N]>::new();
            for (i, words) in test_words.iter_mut().enumerate() {
                jobs.push(Job {
                    input: &inputs[i][..length],
                    words,
                    count: count.wrapping_add((i * BLOCKBYTES) as u128),
                    last_node,
                });
            }
            let mut jobs = jobs.into_inner().expect("full");
            implementation.compress4_loop(&mut jobs, finalize, stride);

            for i in 0..N {
                assert_eq!(reference_words[i], test_words[i], "words {} unequal", i);
            }
        });
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_compress4_loop_avx2() {
        if let Some(imp) = Implementation::avx2_if_supported() {
            exercise_compress4_loop(imp);
        }
    }
}
