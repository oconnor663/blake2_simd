//! This is a toy version of the b2sum program from coreutils, mostly for benchmarking.

extern crate blake2b_simd;

use std::io::prelude::*;

fn main() {
    let stdin = std::io::stdin();
    let mut stdin = stdin.lock();
    let mut buf = [0; 1 << 16];
    let mut state = blake2b_simd::State::new();
    loop {
        let n = stdin.read(&mut buf).unwrap();
        if n == 0 {
            println!("{}", state.finalize().hex());
            return;
        }
        state.update(&buf[..n]);
    }
}
