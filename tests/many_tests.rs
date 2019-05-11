use arrayvec::ArrayVec;
use blake2b_simd::*;
use rand::Rng;
use rand::RngCore;
use rand::SeedableRng;

const INPUT_LENGTHS: &[usize] = &[
    0,
    1,
    BLOCKBYTES,
    BLOCKBYTES + 1,
    2 * BLOCKBYTES,
    // 2 * BLOCKBYTES + 1,
    // MAX_LEN,
];

const MAX_N: usize = 2 * many::MAX_DEGREE;
const MAX_LEN: usize = 3 * BLOCKBYTES;

type SizeVec = ArrayVec<[usize; MAX_N]>;

fn with_length_permutations_n(n: usize, buf: &mut SizeVec, f: &mut dyn FnMut(&mut SizeVec)) {
    buf.clear();
    if n == 0 {
        f(buf);
        return;
    }
    with_length_permutations_n(n - 1, buf, &mut |buf| {
        for &len in INPUT_LENGTHS {
            buf.push(len);
            f(buf);
            buf.pop();
        }
    });
}

fn all_length_permutations(max_len: usize, f: &mut dyn FnMut(&mut SizeVec)) {
    let mut buf = SizeVec::new();
    for n in 0..=max_len {
        with_length_permutations_n(n, &mut buf, f);
    }
}

fn random_params(rng: &mut rand_chacha::ChaChaRng) -> Params {
    let mut params = Params::new();
    params.hash_length(rng.gen_range(1, OUTBYTES + 1));
    if rng.gen() {
        let len: usize = rng.gen_range(1, KEYBYTES + 1);
        let key_buf = &[1; KEYBYTES];
        params.key(&key_buf[..len]);
    }
    params.last_node(rng.gen());
    params
}

#[test]
fn test_hash_many() {
    let mut rng = rand_chacha::ChaChaRng::seed_from_u64(0);
    let mut input_bufs = [[0; MAX_LEN]; MAX_N];
    for input in input_bufs.iter_mut() {
        rng.fill_bytes(input);
    }

    all_length_permutations(many::degree(), &mut |lengths| {
        // Create all the input slices.
        let mut inputs = ArrayVec::<[&[u8]; MAX_N]>::new();
        for i in 0..lengths.len() {
            inputs.push(&input_bufs[i][..lengths[i]]);
        }

        // For each input slice, create a random Params object.
        let mut params = ArrayVec::<[Params; MAX_N]>::new();
        for _ in 0..lengths.len() {
            params.push(random_params(&mut rng));
        }

        // Compute the hash of each input independently.
        let mut expected = ArrayVec::<[Hash; MAX_N]>::new();
        for (param, input) in params.iter().zip(inputs.iter()) {
            expected.push(param.hash(input));
        }

        // Now compute the same hashes in a batch. We'll check that this gives
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
