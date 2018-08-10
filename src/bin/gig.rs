extern crate blake2b_simd;

use std::time::Instant;

fn main() {
    let before = Instant::now();
    let input = vec![0; 1_000_000_000];
    let alloc_time = Instant::now() - before;
    println!("alloc {:?}", alloc_time);

    let before = Instant::now();
    for &x in input.iter() {
        if x != 0 {
            panic!();
        }
    }
    let read_time = Instant::now() - before;
    println!("read {:?}", read_time);

    let before = Instant::now();
    let hash = blake2b_simd::blake2b(&input);
    let hash_time = Instant::now() - before;
    println!("hash {:?}", hash_time);

    for b in &hash[..] {
        print!("{:x}", b);
    }
    println!();
}
