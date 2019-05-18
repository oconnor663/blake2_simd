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
            assert_eq!(input_bytes.len() as u128, state.count());
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
            assert_eq!(input_bytes.len() as u128, state.count());
        }
    }

    // Make sure we don't accidentally skip all the tests somehow. If the
    // number of test vectors changes in the future, we'll need to update this
    // count.
    assert_eq!(512, test_num);
}
