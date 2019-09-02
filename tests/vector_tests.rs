//! The tests in this file run the standard set of test vectors from upstream:
//! https://github.com/BLAKE2/BLAKE2/blob/320c325437539ae91091ce62efec1913cd8093c2/testvectors/blake2-kat.json
//!
//! Currently those cover default hashing and keyed hashing in BLAKE2b and BLAKE2bp. But they don't
//! test the other associated data features, and they don't test any inputs longer than a couple
//! blocks.

use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};

lazy_static! {
    static ref TEST_CASES: Vec<TestCase> =
        serde_json::from_str(include_str!("blake2-kat.json")).unwrap();
}

#[derive(Debug, Serialize, Deserialize)]
struct TestCase {
    hash: String,
    #[serde(rename = "in")]
    in_: String,
    key: String,
    out: String,
}

#[test]
fn blake2b_vectors() {
    let mut test_num = 0u64;
    for case in TEST_CASES.iter() {
        if &case.hash == "blake2b" {
            test_num += 1;
            println!("input {:?}, key {:?}", case.in_, case.key);

            let input_bytes = hex::decode(&case.in_).unwrap();
            let mut params = blake2b_simd::Params::new();
            if !case.key.is_empty() {
                let key_bytes = hex::decode(&case.key).unwrap();
                params.key(&key_bytes);
            }

            // Assert the all-at-once result.
            assert_eq!(case.out, &*params.hash(&input_bytes).to_hex());

            // Assert the State result.
            let mut state = params.to_state();
            state.update(&input_bytes);
            assert_eq!(case.out, &*state.finalize().to_hex());
            assert_eq!(input_bytes.len() as u128, state.count());
        }
    }

    // Make sure we don't accidentally skip all the tests somehow. If the
    // number of test vectors changes in the future, we'll need to update this
    // count.
    assert_eq!(512, test_num);
}

#[test]
fn blake2bp_vectors() {
    let mut test_num = 0u64;
    for case in TEST_CASES.iter() {
        if &case.hash == "blake2bp" {
            test_num += 1;
            println!("input {:?}, key {:?}", case.in_, case.key);

            let input_bytes = hex::decode(&case.in_).unwrap();
            let mut params = blake2b_simd::blake2bp::Params::new();
            if !case.key.is_empty() {
                let key_bytes = hex::decode(&case.key).unwrap();
                params.key(&key_bytes);
            }

            // Assert the all-at-once result.
            assert_eq!(case.out, &*params.hash(&input_bytes).to_hex());

            // Assert the State result.
            let mut state = params.to_state();
            state.update(&input_bytes);
            assert_eq!(case.out, &*state.finalize().to_hex());
            assert_eq!(input_bytes.len() as u128, state.count());
        }
    }

    // Make sure we don't accidentally skip all the tests somehow. If the
    // number of test vectors changes in the future, we'll need to update this
    // count.
    assert_eq!(512, test_num);
}

#[test]
fn blake2s_vectors() {
    let mut test_num = 0u64;
    for case in TEST_CASES.iter() {
        if &case.hash == "blake2s" {
            test_num += 1;
            println!("input {:?}, key {:?}", case.in_, case.key);

            let input_bytes = hex::decode(&case.in_).unwrap();
            let mut params = blake2s_simd::Params::new();
            if !case.key.is_empty() {
                let key_bytes = hex::decode(&case.key).unwrap();
                params.key(&key_bytes);
            }

            // Assert the all-at-once result.
            assert_eq!(case.out, &*params.hash(&input_bytes).to_hex());

            // Assert the State result.
            let mut state = params.to_state();
            state.update(&input_bytes);
            assert_eq!(case.out, &*state.finalize().to_hex());
            assert_eq!(input_bytes.len() as u64, state.count());
        }
    }

    // Make sure we don't accidentally skip all the tests somehow. If the
    // number of test vectors changes in the future, we'll need to update this
    // count.
    assert_eq!(512, test_num);
}

#[test]
fn blake2sp_vectors() {
    let mut test_num = 0u64;
    for case in TEST_CASES.iter() {
        if &case.hash == "blake2sp" {
            test_num += 1;
            println!("input {:?}, key {:?}", case.in_, case.key);

            let input_bytes = hex::decode(&case.in_).unwrap();
            let mut params = blake2s_simd::blake2sp::Params::new();
            if !case.key.is_empty() {
                let key_bytes = hex::decode(&case.key).unwrap();
                params.key(&key_bytes);
            }

            // Assert the all-at-once result.
            assert_eq!(case.out, &*params.hash(&input_bytes).to_hex());

            // Assert the State result.
            let mut state = params.to_state();
            state.update(&input_bytes);
            assert_eq!(case.out, &*state.finalize().to_hex());
            assert_eq!(input_bytes.len() as u64, state.count());
        }
    }

    // Make sure we don't accidentally skip all the tests somehow. If the
    // number of test vectors changes in the future, we'll need to update this
    // count.
    assert_eq!(512, test_num);
}

fn blake2x_test<F: Fn(&[u8], &[u8], u64) -> Vec<u8>, F2: Fn(&[u8], u64, usize) -> Vec<u8>>(h0_hasher: F, b2_hasher: F2, variant_hash_length: usize, variant_name: &str) {
    let mut test_num = 0u64;
    for case in TEST_CASES.iter() {
        if &case.hash == variant_name {
            test_num += 1;

            let input_bytes = hex::decode(&case.in_).unwrap();
            let key = if !case.key.is_empty() {
                hex::decode(&case.key).unwrap()
            } else {
                vec![]
            };

            let output_length = case.out.len()/2;
            let encoded_output_length = (((output_length & ((1 << 8) - 1)) << 32) | ((output_length >> 8) << 40)) as u64;
            let h0 = h0_hasher(&input_bytes, &key, encoded_output_length);

            let num_hashes = (output_length + variant_hash_length - 1)/variant_hash_length;
            let mut buf = vec![];
            for i in 0..num_hashes {
                let hash_length = {
                    if i == (num_hashes - 1) && (output_length % variant_hash_length) != 0 {
                        output_length % variant_hash_length
                    } else {
                        variant_hash_length
                    }
                };

                let b2_out = b2_hasher(&h0, (i as u64) | encoded_output_length, hash_length);
                buf.extend_from_slice(&b2_out);
            }
            assert_eq!(case.out, hex::encode(&buf[..output_length]));
        }
    }

    // Make sure we don't accidentally skip all the tests somehow. If the
    // number of test vectors changes in the future, we'll need to update this
    // count.
    assert_eq!(512, test_num);
}

#[test]
fn blake2xs_vectors() {
    let blake2xs_h0_hasher = |input_bytes: &[u8], key: &[u8], encoded_output_length: u64| -> Vec<u8> {
        let mut params = blake2s_simd::Params::new();
        if key.len() > 0 {
            params.key(key);
        }
        params
            .hash_length(32)
            .node_offset(encoded_output_length);
        let mut state = params.to_state();
        state.update(&input_bytes);
        let h0 = state.finalize().as_ref().to_vec();
        h0
    };
    let blake2xs_b2_hasher = |input_bytes: &[u8], encoded_output_length: u64, hash_length: usize| -> Vec<u8> {
        let mut params = blake2s_simd::Params::new();
        params
            .hash_length(hash_length)
            .max_leaf_length(32)
            .inner_hash_length(32)
            .fanout(0)
            .max_depth(0)
            .node_offset(encoded_output_length);
        let mut state = params.to_state();
        state.update(&input_bytes);
        let b2_out = state.finalize().as_ref().to_vec();
        b2_out
    };

    blake2x_test(blake2xs_h0_hasher, blake2xs_b2_hasher, 32, "blake2xs");
}

#[test]
fn blake2xb_vectors() {
    let blake2xb_h0_hasher = |input_bytes: &[u8], key: &[u8], encoded_output_length: u64| -> Vec<u8> {
        let mut params = blake2b_simd::Params::new();
        if key.len() > 0 {
            params.key(key);
        }
        params
            .hash_length(64)
            .node_offset(encoded_output_length);
        let mut state = params.to_state();
        state.update(&input_bytes);
        let h0 = state.finalize().as_ref().to_vec();
        h0
    };
    let blake2xb_b2_hasher = |input_bytes: &[u8], encoded_output_length: u64, hash_length: usize| -> Vec<u8> {
        let mut params = blake2b_simd::Params::new();
        params
            .hash_length(hash_length)
            .max_leaf_length(64)
            .inner_hash_length(64)
            .fanout(0)
            .max_depth(0)
            .node_offset(encoded_output_length);
        let mut state = params.to_state();
        state.update(&input_bytes);
        let b2_out = state.finalize().as_ref().to_vec();
        b2_out
    };

    blake2x_test(blake2xb_h0_hasher, blake2xb_b2_hasher, 64, "blake2xb");
}
