extern crate blake2b_simd;

use std::time::Instant;

fn main() {
    let start_t = Instant::now();

    let input = vec![0; 1_000_000_000];

    let alloc_t = Instant::now();

    let hash = blake2b_simd::blake2b(&input);

    let hash_t = Instant::now();

    for b in &hash[..] {
        print!("{:x}", b);
    }
    println!();

    println!("alloc {:?}", alloc_t - start_t);
    println!("hash {:?}", hash_t - alloc_t);
}
