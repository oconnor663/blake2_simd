use super::*;

const EMPTY_HASH: &str = "786a02f742015903c6c6fd852552d272912f4740e15847618a86e217f71f5419\
                          d25e1031afee585313896444934eb04b903a685b1448b755d56f701afe9be2ce";
const ABC_HASH: &str = "ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d1\
                        7d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923";
const ONE_BLOCK_HASH: &str = "865939e120e6805438478841afb739ae4250cf372653078a065cdcfffca4caf7\
                              98e6d462b65d658fc165782640eded70963449ae1500fb0f24981d7727e22c41";
const THOUSAND_HASH: &str = "1ee4e51ecab5210a518f26150e882627ec839967f19d763e1508b12cfefed148\
                             58f6a1c9d1f969bc224dc9440f5a6955277e755b9c513f9ba4421c5e50c8d787";

#[test]
fn test_vectors() {
    let io = &[
        (&b""[..], EMPTY_HASH),
        (&b"abc"[..], ABC_HASH),
        (&[0; BLOCKBYTES], ONE_BLOCK_HASH),
        (&[0; 1000], THOUSAND_HASH),
    ];
    // Test each input all at once.
    for &(input, output) in io {
        let hash = blake2b(input);
        assert_eq!(&hash.to_hex(), output, "hash mismatch");
    }
    // Now in two chunks. This is especially important for the ONE_BLOCK case, because it would be
    // a mistake for update() to call compress, even though the buffer is full.
    for &(input, output) in io {
        let mut state = State::new();
        let split = input.len() / 2;
        state.update(&input[..split]);
        assert_eq!(split as u128, state.count());
        state.update(&input[split..]);
        assert_eq!(input.len() as u128, state.count());
        let hash = state.finalize();
        assert_eq!(&hash.to_hex(), output, "hash mismatch");
    }
    // Now one byte at a time.
    for &(input, output) in io {
        let mut state = State::new();
        let mut count = 0;
        for &b in input {
            state.update(&[b]);
            count += 1;
            assert_eq!(count, state.count());
        }
        let hash = state.finalize();
        assert_eq!(&hash.to_hex(), output, "hash mismatch");
    }
}

#[test]
fn test_multiple_finalizes() {
    let mut state = State::new();
    assert_eq!(&state.finalize().to_hex(), EMPTY_HASH, "hash mismatch");
    assert_eq!(&state.finalize().to_hex(), EMPTY_HASH, "hash mismatch");
    assert_eq!(&state.finalize().to_hex(), EMPTY_HASH, "hash mismatch");
    state.update(b"abc");
    assert_eq!(&state.finalize().to_hex(), ABC_HASH, "hash mismatch");
    assert_eq!(&state.finalize().to_hex(), ABC_HASH, "hash mismatch");
    assert_eq!(&state.finalize().to_hex(), ABC_HASH, "hash mismatch");
}

#[cfg(feature = "std")]
#[test]
fn test_write() {
    use std::io::prelude::*;

    let mut state = State::new();
    state.write_all(&[0; 1000]).unwrap();
    let hash = state.finalize();
    assert_eq!(&hash.to_hex(), THOUSAND_HASH, "hash mismatch");
}

// You can check this case against the equivalent Python:
//
// import hashlib
// hashlib.blake2b(
//     b'foo',
//     digest_size=18,
//     key=b"bar",
//     salt=b"bazbazbazbazbazb",
//     person=b"bing bing bing b",
//     fanout=2,
//     depth=3,
//     leaf_size=0x04050607,
//     node_offset=0x08090a0b0c0d0e0f,
//     node_depth=16,
//     inner_size=17,
//     last_node=True,
// ).hexdigest()
#[test]
fn test_all_parameters() {
    let mut params = Params::default();
    params.hash_length(18);
    params.key(b"bar");
    params.salt(b"bazbazbazbazbazb");
    params.personal(b"bing bing bing b");
    params.fanout(2);
    params.max_depth(3);
    params.max_leaf_length(0x04050607);
    params.node_offset(0x08090a0b0c0d0e0f);
    params.node_depth(16);
    params.inner_hash_length(17);
    let mut state = State::with_params(&params);
    state.set_last_node(true);
    state.update(b"foo");
    let hash = state.finalize();
    assert_eq!("ec0f59cb65f92e7fcca1280ba859a6925ded", &hash.to_hex());
}

#[test]
#[should_panic]
fn test_short_hash_length_panics() {
    Params::new().hash_length(0);
}

#[test]
#[should_panic]
fn test_long_hash_length_panics() {
    Params::new().hash_length(OUTBYTES + 1);
}

#[test]
#[should_panic]
fn test_long_key_panics() {
    Params::new().key(&[0; KEYBYTES + 1]);
}

#[test]
#[should_panic]
fn test_long_salt_panics() {
    Params::new().salt(&[0; SALTBYTES + 1]);
}

#[test]
#[should_panic]
fn test_long_personal_panics() {
    Params::new().personal(&[0; PERSONALBYTES + 1]);
}

#[test]
#[should_panic]
fn test_zero_max_depth_panics() {
    Params::new().max_depth(0);
}

#[test]
#[should_panic]
fn test_long_inner_hash_length_panics() {
    Params::new().inner_hash_length(OUTBYTES + 1);
}

#[cfg(feature = "blake2bp")]
#[test]
fn test_blake2bp() {
    // From https://raw.githubusercontent.com/BLAKE2/BLAKE2/master/testvectors/blake2-kat.json.
    let vectors: &[(&[u8], &str)] = &[
        // Note that memory mapping doesn't work on zero-length input.
        (
            b"\x00",
            "a139280e72757b723e6473d5be59f36e9d50fc5cd7d4585cbc09804895a36c52\
             1242fb2789f85cb9e35491f31d4a6952f9d8e097aef94fa1ca0b12525721f03d",
        ),
        (
            b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\
              \x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\
              \x20\x21\x22\x23\x24\x25\x26\x27\x28\x29\x2a\x2b\x2c\x2d\x2e\x2f\
              \x30\x31\x32\x33\x34\x35\x36\x37\x38\x39\x3a\x3b\x3c\x3d\x3e\x3f\
              \x40\x41\x42\x43\x44\x45\x46\x47\x48\x49\x4a\x4b\x4c\x4d\x4e\x4f\
              \x50\x51\x52\x53\x54\x55\x56\x57\x58\x59\x5a\x5b\x5c\x5d\x5e\x5f\
              \x60\x61\x62\x63\x64\x65\x66\x67\x68\x69\x6a\x6b\x6c\x6d\x6e\x6f\
              \x70\x71\x72\x73\x74\x75\x76\x77\x78\x79\x7a\x7b\x7c\x7d\x7e\x7f\
              \x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x8b\x8c\x8d\x8e\x8f\
              \x90\x91\x92\x93\x94\x95\x96\x97\x98\x99\x9a\x9b\x9c\x9d\x9e\x9f\
              \xa0\xa1\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xab\xac\xad\xae\xaf\
              \xb0\xb1\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xbb\xbc\xbd\xbe\xbf\
              \xc0\xc1\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xcb\xcc\xcd\xce\xcf\
              \xd0\xd1\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xdb\xdc\xdd\xde\xdf\
              \xe0\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xeb\xec\xed\xee\xef\
              \xf0\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xfb\xfc\xfd\xfe",
            "3f35c45d24fcfb4acca651076c08000e279ebbff37a1333ce19fd577202dbd24\
             b58c514e36dd9ba64af4d78eea4e2dd13bc18d798887dd971376bcae0087e17e",
        ),
    ];

    for &(input, expected) in vectors {
        let found = blake2bp(input, 64);
        assert_eq!(&*expected, &*found.to_hex());
    }
}
