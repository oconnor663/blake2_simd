extern crate rayon;

use std::cmp;
use Hash;
use Params;
use BLOCKBYTES;

/// Compute the parallel BLAKE2bp hash of a slice of bytes.
///
/// Although BLAKE2bp is based on BLAKE2b, it's a different function, and it produces a different
/// hash. It's uses four worker threads in parallel, for higher performance on multi-core machines.
///
/// Note that BLAKE2bp sets its own values for all of the tree hashing parameters (fanout, max
/// depth, max leaf length, node offset, node depth, inner hash length, and last node). Only the
/// general parameters (hash length, key, salt, and personalization) from the `params` argument are
/// respected.
pub fn blake2bp(input: &[u8], params: &Params) -> Hash {
    let worker = |index| {
        let mut state = params
            .clone()
            .fanout(4)
            .max_depth(2)
            .max_leaf_length(0)
            .node_offset(index as u64)
            .node_depth(0)
            .inner_hash_length(params.hash_length as usize)
            .last_node(index == 3)
            .to_state();
        let mut start = index * BLOCKBYTES;
        while start < input.len() {
            let blocklen = cmp::min(input.len() - start, BLOCKBYTES);
            state.update(&input[start..][..blocklen]);
            start += 4 * BLOCKBYTES;
        }
        state.finalize()
    };

    // Performance note: This works best when we configure Rayon to use exactly 4 threads. The
    // b2sum binary does this, and our benchmarks do also. But it would be inappropriate for the
    // library code itself to interfere with the global config.
    let ((leaf0, leaf1), (leaf2, leaf3)) = rayon::join(
        || rayon::join(|| worker(0), || worker(1)),
        || rayon::join(|| worker(2), || worker(3)),
    );

    let mut root_state = params
        .clone()
        .fanout(4)
        .max_depth(2)
        .max_leaf_length(0)
        .node_offset(0)
        .node_depth(1)
        .inner_hash_length(params.hash_length as usize)
        .last_node(false)
        .to_state();
    root_state.set_last_node(true);
    root_state.update(leaf0.as_bytes());
    root_state.update(leaf1.as_bytes());
    root_state.update(leaf2.as_bytes());
    root_state.update(leaf3.as_bytes());
    root_state.finalize()
}

#[cfg(test)]
mod test {
    use super::*;

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
            let found = blake2bp(input, &Params::new().hash_length(64));
            assert_eq!(&*expected, &*found.to_hex());
        }
    }
}
