use arrayvec::ArrayVec;
use blake2b_simd::*;
use rand::Rng;
use rand::RngCore;
use rand::SeedableRng;

const MAX_N: usize = 2 * many::MAX_DEGREE;
const MAX_LEN: usize = 3 * BLOCKBYTES;

fn random_params(rng: &mut rand_chacha::ChaChaRng) -> Params {
    let mut params = Params::new();
    // hash_length, key, and last_node are all things that need to be passed
    // from the Params through to the State or whatever. Randomize those.
    // Everything else just winds up in the state words and doesn't really need
    // to be exercised here.
    params.hash_length(rng.gen_range(1, OUTBYTES + 1));
    if rng.gen() {
        let len: usize = rng.gen_range(1, KEYBYTES + 1);
        let key_buf = &[1; KEYBYTES];
        params.key(&key_buf[..len]);
    }
    params.last_node(rng.gen());
    params
}

// Do more tests in release mode, but try to keep execution time under 1 sec.
#[cfg(debug_assertions)]
const NUM_TESTS: usize = 1_000;
#[cfg(not(debug_assertions))]
const NUM_TESTS: usize = 100_000;

fn with_random_inputs(mut f: impl FnMut(&[Params], &[&[u8]])) {
    let mut rng = rand_chacha::ChaChaRng::seed_from_u64(0);

    // Generate randomized input buffers to reuse in each test case.
    let mut input_bufs = [[0; MAX_LEN]; MAX_N];
    for input in input_bufs.iter_mut() {
        rng.fill_bytes(input);
    }

    for _ in 0..NUM_TESTS {
        // Select a random number of random length input slices from the
        // buffers.
        let num_inputs: usize = rng.gen_range(0, MAX_N + 1);
        let mut inputs = ArrayVec::<[&[u8]; MAX_N]>::new();
        for i in 0..num_inputs {
            let input_length = rng.gen_range(0, MAX_LEN + 1);
            inputs.push(&input_bufs[i][..input_length]);
        }

        // For each input slice, create a random Params object.
        let mut params = ArrayVec::<[Params; MAX_N]>::new();
        for _ in 0..num_inputs {
            params.push(random_params(&mut rng));
        }

        f(&params, &inputs);
    }
}

#[test]
fn fuzz_hash_many() {
    with_random_inputs(|params, inputs| {
        // Compute the hash of each input independently.
        let mut expected = ArrayVec::<[Hash; MAX_N]>::new();
        for (param, input) in params.iter().zip(inputs.iter()) {
            expected.push(param.hash(input));
        }

        // Now compute the same hashes in a batch, and check that this gives
        // the same result.
        let mut jobs: ArrayVec<[many::HashManyJob; MAX_N]> = inputs
            .iter()
            .zip(params.iter())
            .map(|(input, param)| many::HashManyJob::new(param, input))
            .collect();
        many::hash_many(&mut jobs);
        for i in 0..jobs.len() {
            assert_eq!(&expected[i], &jobs[i].to_hash(), "job {} mismatch", i);
        }
    });
}

#[test]
fn fuzz_update_many() {
    with_random_inputs(|params, inputs| {
        // Compute the hash of each input independently. Feed each into the
        // state twice, to exercise buffering.
        let mut expected = ArrayVec::<[Hash; MAX_N]>::new();
        for (param, input) in params.iter().zip(inputs.iter()) {
            let mut state = param.to_state();
            state.update(input);
            state.update(input);
            expected.push(state.finalize());
        }

        // Now compute the same hashes in a batch, and check that this gives
        // the same result.
        let mut states = ArrayVec::<[State; MAX_N]>::new();
        for param in params {
            states.push(param.to_state());
        }
        many::update_many(states.iter_mut().zip(inputs.iter()));
        many::update_many(states.iter_mut().zip(inputs.iter()));
        for i in 0..states.len() {
            assert_eq!(2 * inputs[i].len() as u128, states[i].count());
            assert_eq!(&expected[i], &states[i].finalize(), "state {} mismatch", i);
        }
    });
}
