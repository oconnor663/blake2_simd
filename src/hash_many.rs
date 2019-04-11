use crate::guts::{self, u64x8, Finalize, Implementation, Job, Stride};
use crate::state_words_to_bytes;
use crate::Hash;
use crate::Params;
use arrayvec::ArrayVec;

type JobsVec<'a, 'b> = ArrayVec<[Job<'a, 'b>; guts::MAX_DEGREE]>;

fn fill_jobs_vec<'a, 'b>(
    jobs_iter: &mut impl Iterator<Item = Job<'a, 'b>>,
    vec: &mut JobsVec<'a, 'b>,
) {
    while vec.len() < vec.capacity() {
        if let Some(job) = jobs_iter.next() {
            vec.push(job);
        } else {
            break;
        }
    }
}

fn evict_finished<'a, 'b>(vec: &mut JobsVec<'a, 'b>, num_jobs: usize) {
    // Iterate backwards so that swap_remove doesn't swap in a finished job.
    for i in (vec.len() - num_jobs..vec.len()).rev() {
        // Strictly greater is important here, otherwise we'd get an extra empty
        // block hashed in.
        if vec[i].input.is_empty() {
            // TODO: Make a guarantee about hashing inputs in order?
            vec.swap_remove(i);
        }
    }
}

pub fn hash_many_inner<'a, 'b, I>(jobs: I, imp: Implementation, stride: Stride)
where
    I: IntoIterator<Item = Job<'a, 'b>>,
{
    let mut jobs_iter = jobs.into_iter().fuse();
    let mut jobs_vec = JobsVec::new();
    fill_jobs_vec(&mut jobs_iter, &mut jobs_vec);

    if imp.degree() >= 4 {
        while jobs_vec.len() >= 4 {
            let jobs_array = array_mut_ref!(jobs_vec, jobs_vec.len() - 4, 4);
            imp.compress4_loop(jobs_array, stride);
            evict_finished(&mut jobs_vec, 4);
            fill_jobs_vec(&mut jobs_iter, &mut jobs_vec);
        }
    }

    if imp.degree() >= 2 {
        while jobs_vec.len() >= 2 {
            let jobs_array = array_mut_ref!(jobs_vec, jobs_vec.len() - 2, 2);
            imp.compress2_loop(jobs_array, stride);
            evict_finished(&mut jobs_vec, 2);
            fill_jobs_vec(&mut jobs_iter, &mut jobs_vec);
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
}
