use super::*;

const EMPTY_HASH: &str = "786a02f742015903c6c6fd852552d272912f4740e15847618a86e217f71f5419d25e1031afee585313896444934eb04b903a685b1448b755d56f701afe9be2ce";
const ABC_HASH: &str = "ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d17d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923";
const THOUSAND_HASH: &str = "1ee4e51ecab5210a518f26150e882627ec839967f19d763e1508b12cfefed14858f6a1c9d1f969bc224dc9440f5a6955277e755b9c513f9ba4421c5e50c8d787";

#[test]
fn test_vectors() {
    let io = &[
        (&b""[..], EMPTY_HASH),
        (&b"abc"[..], ABC_HASH),
        (&[0; 1000], THOUSAND_HASH),
    ];
    for &(input, output) in io {
        let hash = blake2b(input);
        assert_eq!(&hash.hex(), output, "hash mismatch");
    }
}

#[test]
fn test_multiple_finalizes() {
    let mut state = State::new();
    assert_eq!(&state.finalize().hex(), EMPTY_HASH, "hash mismatch");
    assert_eq!(&state.finalize().hex(), EMPTY_HASH, "hash mismatch");
    assert_eq!(&state.finalize().hex(), EMPTY_HASH, "hash mismatch");
    state.update(b"abc");
    assert_eq!(&state.finalize().hex(), ABC_HASH, "hash mismatch");
    assert_eq!(&state.finalize().hex(), ABC_HASH, "hash mismatch");
    assert_eq!(&state.finalize().hex(), ABC_HASH, "hash mismatch");
}

#[test]
fn test_a_thousand_one_by_one() {
    let mut state = State::new();
    for _ in 0..1000 {
        state.update(&[0]);
    }
    let hash = state.finalize();
    assert_eq!(&hash.hex(), THOUSAND_HASH, "hash mismatch");
}

#[test]
fn test_two_times_five_hundred() {
    let mut state = State::new();
    state.update(&[0; 500]);
    state.update(&[0; 500]);
    let hash = state.finalize();
    assert_eq!(&hash.hex(), THOUSAND_HASH, "hash mismatch");
}

#[cfg(feature = "std")]
#[test]
fn test_write() {
    use std::io::prelude::*;

    let mut state = State::new();
    state.write_all(&[0; 1000]).unwrap();
    let hash = state.finalize();
    assert_eq!(&hash.hex(), THOUSAND_HASH, "hash mismatch");
}
