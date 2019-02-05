use crate::guts::{u64x2, u64x4, u64x8, Implementation};
use crate::state_words_to_bytes;
use crate::u64_flag;
use crate::Hash;
use crate::Params;
use crate::BLOCKBYTES;
use std::cmp;

fn hash1(
    implementation: Implementation,
    state: &mut u64x8,
    input: &[u8],
    last_node: u64,
    count: u128,
) {
    // If the final block is uneven (or if the whole input is empty), the last
    // compression will be in a local buffer.
    let partial_block_len = input.len() % BLOCKBYTES;
    let use_local_buffer = input.is_empty() || partial_block_len != 0;
    let blocks = input.len() / BLOCKBYTES;
    debug_assert!(blocks > 0 || use_local_buffer, "at least one compression");
    if blocks > 0 {
        let last_block = u64_flag(!use_local_buffer);
        implementation.compress1_loop(
            state,
            input,
            count,
            last_block,
            last_block & last_node,
            blocks,
            1, // stride
            0, // buffer_tail
        );
    }
    // We need to assemble the last block if the input is uneven, and also in
    // the special case that the input is empty.
    if use_local_buffer {
        let mut buffer = [0; BLOCKBYTES];
        buffer[..partial_block_len].copy_from_slice(&input[input.len() - partial_block_len..]);
        let updated_count = count + (blocks * BLOCKBYTES) as u128;
        let buffer_tail = BLOCKBYTES - partial_block_len;
        implementation.compress1_loop(
            state,
            &buffer,
            updated_count,
            !0,
            last_node,
            1, // blocks
            1, // stride
            buffer_tail,
        );
    }
}

fn hash2(
    implementation: Implementation,
    state0: &mut u64x8,
    state1: &mut u64x8,
    input0: &[u8],
    input1: &[u8],
    last_node: &u64x2,
) {
    // Figure out how many blocks we can compress together. Skip this part
    // entirely if the answer is zero.
    let min_len = cmp::min(input0.len(), input1.len());
    let batch_blocks = min_len / BLOCKBYTES;
    let batch_bytes = batch_blocks * BLOCKBYTES;
    if batch_blocks > 0 {
        let last_block_fn = |input: &[u8]| {
            if input.len() == batch_bytes {
                !0
            } else {
                0
            }
        };
        let last_block = u64x2([last_block_fn(input0), last_block_fn(input1)]);
        let last_node_maybe = u64x2([last_block[0] & last_node[0], last_block[1] & last_node[1]]);

        // Do the main loop compression.
        implementation.compress2_loop(
            state0,
            state1,
            input0,
            input1,
            &u64x2([0; 2]),
            &u64x2([0; 2]),
            &last_block,
            &last_node_maybe,
            batch_blocks,
            1,
            &u64x2([0; 2]),
        );
    }

    // If any of the inputs weren't finished in the 2-way loop above, finish
    // them individually. Note that if any of the inputs is empty, the loop
    // above doesn't do any work, and all of the inputs need to be finalized
    // individually.
    let mut states = [state0, state1];
    let inputs = [input0, input1];
    for i in 0..2 {
        if batch_bytes == 0 || inputs[i].len() != batch_bytes {
            hash1(
                implementation,
                &mut states[i],
                &inputs[i][batch_bytes..],
                last_node[i],
                batch_bytes as u128,
            );
        }
    }
}

fn hash4(
    implementation: Implementation,
    state0: &mut u64x8,
    state1: &mut u64x8,
    state2: &mut u64x8,
    state3: &mut u64x8,
    input0: &[u8],
    input1: &[u8],
    input2: &[u8],
    input3: &[u8],
    last_node: &u64x4,
) {
    // Figure out how many blocks we can compress together. Skip this part
    // entirely if the answer is zero.
    let min_len = cmp::min(
        cmp::min(input0.len(), input1.len()),
        cmp::min(input2.len(), input3.len()),
    );
    let batch_blocks = min_len / BLOCKBYTES;
    let batch_bytes = batch_blocks * BLOCKBYTES;
    if batch_blocks > 0 {
        let last_block_fn = |input: &[u8]| {
            if input.len() == batch_bytes {
                !0
            } else {
                0
            }
        };
        let last_block = u64x4([
            last_block_fn(input0),
            last_block_fn(input1),
            last_block_fn(input2),
            last_block_fn(input3),
        ]);
        let last_node_maybe = u64x4([
            last_block[0] & last_node[0],
            last_block[1] & last_node[1],
            last_block[2] & last_node[2],
            last_block[3] & last_node[3],
        ]);

        // Do the main loop compression.
        implementation.compress4_loop(
            state0,
            state1,
            state2,
            state3,
            input0,
            input1,
            input2,
            input3,
            &u64x4([0; 4]),
            &u64x4([0; 4]),
            &last_block,
            &last_node_maybe,
            batch_blocks,
            1,
            &u64x4([0; 4]),
        );
    }

    // If any of the inputs weren't finished in the 4-way loop above, finish
    // them individually. Note that if any of the inputs is empty, the loop
    // above doesn't do any work, and all of the inputs need to be finalized
    // individually.
    let mut states = [state0, state1, state2, state3];
    let inputs = [input0, input1, input2, input3];
    for i in 0..4 {
        if batch_bytes == 0 || inputs[i].len() != batch_bytes {
            hash1(
                implementation,
                &mut states[i],
                &inputs[i][batch_bytes..],
                last_node[i],
                batch_bytes as u128,
            );
        }
    }
}

pub fn hash_many(inputs: &[&[u8]], outputs: &mut [Hash], params: &[Params]) {
    assert_eq!(inputs.len(), outputs.len());
    assert_eq!(inputs.len(), params.len());
    let implementation = Implementation::detect();
    let mut index = 0;

    while inputs.len() - index >= 4 {
        let params0 = &params[index + 0];
        let params1 = &params[index + 1];
        let params2 = &params[index + 2];
        let params3 = &params[index + 3];
        let mut words0 = params0.to_state_words();
        let mut words1 = params1.to_state_words();
        let mut words2 = params2.to_state_words();
        let mut words3 = params3.to_state_words();
        let last_node = u64x4([
            u64_flag(params0.last_node),
            u64_flag(params1.last_node),
            u64_flag(params2.last_node),
            u64_flag(params3.last_node),
        ]);
        hash4(
            implementation,
            &mut words0,
            &mut words1,
            &mut words2,
            &mut words3,
            inputs[index + 0],
            inputs[index + 1],
            inputs[index + 2],
            inputs[index + 3],
            &last_node,
        );
        outputs[index + 0] = Hash {
            bytes: state_words_to_bytes(&words0),
            len: params0.hash_length,
        };
        outputs[index + 1] = Hash {
            bytes: state_words_to_bytes(&words1),
            len: params1.hash_length,
        };
        outputs[index + 2] = Hash {
            bytes: state_words_to_bytes(&words2),
            len: params2.hash_length,
        };
        outputs[index + 3] = Hash {
            bytes: state_words_to_bytes(&words3),
            len: params3.hash_length,
        };
        index += 4;
    }

    while inputs.len() - index >= 2 {
        let params0 = &params[index + 0];
        let params1 = &params[index + 1];
        let mut words0 = params0.to_state_words();
        let mut words1 = params1.to_state_words();
        let last_node = u64x2([u64_flag(params0.last_node), u64_flag(params1.last_node)]);
        hash2(
            implementation,
            &mut words0,
            &mut words1,
            inputs[index + 0],
            inputs[index + 1],
            &last_node,
        );
        outputs[index + 0] = Hash {
            bytes: state_words_to_bytes(&words0),
            len: params0.hash_length,
        };
        outputs[index + 1] = Hash {
            bytes: state_words_to_bytes(&words1),
            len: params1.hash_length,
        };
        index += 2;
    }

    while inputs.len() - index >= 1 {
        let mut words = params[index].to_state_words();
        hash1(
            implementation,
            &mut words,
            inputs[index],
            u64_flag(params[index].last_node),
            0,
        );
        outputs[index] = Hash {
            bytes: state_words_to_bytes(&words),
            len: params[index].hash_length,
        };
        index += 1;
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::guts;
    use crate::paint_test_input;
    use arrayvec::ArrayVec;

    #[test]
    fn test_hash_many() {
        // Use a length of inputs that will exercise all of the power-of-two loops.
        const LEN: usize = 2 * guts::MAX_DEGREE - 1;

        let mut params: ArrayVec<[Params; LEN]> = ArrayVec::new();
        for i in 0..LEN {
            let mut param = Params::new();
            param.node_offset(i as u64);
            param.last_node(i % 2 == 1);
            params.push(param);
        }

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

            let mut outputs: ArrayVec<[Hash; LEN]> = ArrayVec::new();
            for _ in 0..LEN {
                outputs.push(Hash::empty());
            }

            hash_many(&inputs, &mut outputs, &params);

            // Check the outputs.
            for i in 0..LEN {
                let expected = params[i].to_state().update(inputs[i]).finalize();
                assert_eq!(&expected, &outputs[i]);
            }
        }
    }
}
