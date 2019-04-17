//! This module provides interfaces for hashing multiple inputs at once. This
//! can be more efficient than hashing inputs one at a time, because it reduces
//! the overhead of using SIMD. The throughput of these interfaces is
//! comparable to BLAKE2bp, about two times the the throughput of BLAKE2b when
//! AVX2 is available.
//!
//! This implementation keeps working in parallel even when inputs are of
//! different lengths, by managing a working set of jobs whose input isn't yet
//! exhausted. However, if one or two inputs are much longer than the others,
//! and they're encountered only at the end, there might not be any remaining
//! work to parallelize them with. In this case, sorting the inputs
//! longest-first can minimize the time spent falling back to slower serial
//! hashing.

use crate::guts::{self, u64x8, Finalize, Implementation, Job, Stride};
use crate::state_words_to_bytes;
use crate::Hash;
use crate::Params;
use crate::State;
use crate::BLOCKBYTES;
use arrayvec::ArrayVec;

type JobsVec<'a, 'b> = ArrayVec<[Job<'a, 'b>; guts::MAX_DEGREE]>;

fn fill_jobs_vec<'a, 'b>(
    jobs_iter: &mut impl Iterator<Item = Job<'a, 'b>>,
    vec: &mut JobsVec<'a, 'b>,
    target_len: usize,
) {
    while vec.len() < target_len {
        if let Some(job) = jobs_iter.next() {
            vec.push(job);
        } else {
            break;
        }
    }
}

fn evict_finished<'a, 'b>(vec: &mut JobsVec<'a, 'b>, num_jobs: usize) {
    // Iterate backwards so that removal doesn't cause an out-of-bounds panic.
    for i in (0..num_jobs).rev() {
        // Note that is_empty() is only valid because we know all these jobs
        // have been run at least once. Otherwise we could confuse the empty
        // input for a finished job, which would be incorrect.
        if vec[i].input.is_empty() {
            // Note that calling remove() repeatedly has some overhead, because
            // later elements need to be shifted up. However, the JobsVec is
            // small, and this approach guarantees that jobs are encountered in
            // order.
            vec.remove(i);
        }
    }
}

pub fn hash_many_inner<'a, 'b, I>(jobs: I, imp: Implementation, stride: Stride)
where
    I: IntoIterator<Item = Job<'a, 'b>>,
{
    // Fuse is important for correctness, since each of these blocks tries to
    // advance the iterator, even if a previous block emptied it.
    let mut jobs_iter = jobs.into_iter().fuse();
    let mut jobs_vec = JobsVec::new();

    if imp.degree() >= 4 {
        loop {
            fill_jobs_vec(&mut jobs_iter, &mut jobs_vec, 4);
            if jobs_vec.len() < 4 {
                break;
            }
            let jobs_array = array_mut_ref!(jobs_vec, 0, 4);
            imp.compress4_loop(jobs_array, stride);
            evict_finished(&mut jobs_vec, 4);
        }
    }

    if imp.degree() >= 2 {
        loop {
            fill_jobs_vec(&mut jobs_iter, &mut jobs_vec, 2);
            if jobs_vec.len() < 2 {
                break;
            }
            let jobs_array = array_mut_ref!(jobs_vec, 0, 2);
            imp.compress2_loop(jobs_array, stride);
            evict_finished(&mut jobs_vec, 2);
        }
    }

    for job in jobs_vec.into_iter().chain(jobs_iter) {
        imp.compress1_loop(job, stride);
    }
}

pub struct HashManyJob<'a> {
    words: u64x8,
    finalize: Finalize,
    hash_length: u8,
    input: &'a [u8],
}

impl<'a> HashManyJob<'a> {
    pub fn new(params: &Params, input: &'a [u8]) -> Self {
        Self {
            words: params.to_state_words(),
            finalize: if params.last_node {
                guts::Finalize::YesLastNode
            } else {
                guts::Finalize::YesOrdinary
            },
            hash_length: params.hash_length,
            input,
        }
    }

    pub fn to_hash(&self) -> Hash {
        // TODO: assert this isn't called early
        Hash {
            bytes: state_words_to_bytes(&self.words),
            len: self.hash_length,
        }
    }
}

pub fn hash_many<'a, 'b, I>(hash_many_jobs: I)
where
    'b: 'a,
    I: IntoIterator<Item = &'a mut HashManyJob<'b>>,
{
    let imp = Implementation::detect();
    let jobs = hash_many_jobs
        .into_iter()
        .map(|j| Job::new(&mut j.words, 0, j.input, j.finalize));
    hash_many_inner(jobs, imp, Stride::Normal);
}

pub fn update_many<'a, 'b, I>(pairs: I)
where
    'b: 'a,
    I: IntoIterator<Item = (&'a mut State, &'b [u8])>,
{
    let imp = Implementation::detect();
    let jobs = pairs.into_iter().flat_map(|(state, mut input)| {
        // For each pair, if the State has some input in its buffer, try to
        // finish that buffer. If there wasn't enough input to do that --
        // or if the input was empty to begin with -- skip this pair.
        state.compress_buffer_if_possible(&mut input);
        if input.is_empty() {
            return None;
        }
        // Now we know the buffer is empty and there's more input. Make sure we
        // buffer the final block, because update() doesn't finalize.
        let mut last_block_start = input.len() - 1;
        last_block_start -= last_block_start % BLOCKBYTES;
        let (blocks, last_block) = input.split_at(last_block_start);
        state.buf[..last_block.len()].copy_from_slice(last_block);
        state.buflen = last_block.len() as u8;
        // Finally, if the full blocks slice is non-empty, prepare that job for
        // compression, and bump the State count.
        if blocks.is_empty() {
            None
        } else {
            let count = state.count;
            state.count = state.count.wrapping_add(blocks.len() as u128);
            Some(Job::new(&mut state.words, count, blocks, Finalize::NotYet))
        }
    });
    hash_many_inner(jobs, imp, Stride::Normal);
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::guts;
    use crate::paint_test_input;
    use crate::BLOCKBYTES;
    use arrayvec::ArrayVec;

    #[test]
    fn test_hash_many() {
        // Use a length of inputs that will exercise all of the power-of-two loops.
        const LEN: usize = 2 * guts::MAX_DEGREE - 1;

        // Rerun LEN inputs LEN different times, with the empty input starting in a
        // different spot each time.
        let mut input = [0; LEN * BLOCKBYTES];
        paint_test_input(&mut input);
        for start_offset in 0..LEN {
            let mut inputs: [&[u8]; LEN] = [&[]; LEN];
            for i in 0..LEN {
                let chunks = (i + start_offset) % LEN;
                inputs[i] = &input[..chunks * BLOCKBYTES];
            }

            let mut params: ArrayVec<[Params; LEN]> = ArrayVec::new();
            for i in 0..LEN {
                let mut param = Params::new();
                param.node_offset(i as u64);
                param.last_node(i % 2 == 1);
                params.push(param);
            }

            let mut jobs: ArrayVec<[HashManyJob; LEN]> = ArrayVec::new();
            for i in 0..LEN {
                jobs.push(HashManyJob::new(&params[i], inputs[i]));
            }

            hash_many(&mut jobs);

            // Check the outputs.
            for i in 0..LEN {
                let expected = params[i].to_state().update(inputs[i]).finalize();
                assert_eq!(expected, jobs[i].to_hash());
            }
        }
    }

    #[test]
    fn test_update_many() {
        // Use a length of inputs that will exercise all of the power-of-two loops.
        const LEN: usize = 2 * guts::MAX_DEGREE - 1;

        // Rerun LEN inputs LEN different times, with the empty input starting in a
        // different spot each time.
        let mut input = [0; LEN * BLOCKBYTES];
        paint_test_input(&mut input);
        for start_offset in 0..LEN {
            let mut inputs: [&[u8]; LEN] = [&[]; LEN];
            for i in 0..LEN {
                let chunks = (i + start_offset) % LEN;
                inputs[i] = &input[..chunks * BLOCKBYTES];
            }

            let mut params: ArrayVec<[Params; LEN]> = ArrayVec::new();
            for i in 0..LEN {
                let mut param = Params::new();
                param.node_offset(i as u64);
                param.last_node(i % 2 == 1);
                params.push(param);
            }

            let mut states: ArrayVec<[State; LEN]> = ArrayVec::new();
            for i in 0..LEN {
                states.push(params[i].to_state());
            }

            // Run each input twice through, to exercise buffering.
            update_many(states.iter_mut().zip(inputs.iter().cloned()));
            update_many(states.iter_mut().zip(inputs.iter().cloned()));

            // Check the outputs.
            for i in 0..LEN {
                let mut reference_state = params[i].to_state();
                // Again, run the input twice.
                reference_state.update(inputs[i]);
                reference_state.update(inputs[i]);
                assert_eq!(reference_state.finalize(), states[i].finalize());
            }
        }
    }
}
