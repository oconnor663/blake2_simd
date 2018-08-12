extern crate blake2b_simd;
extern crate os_pipe;
#[macro_use]
extern crate structopt;

use blake2b_simd::{Hash, Params, State};
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::process::exit;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(author = "")]
struct Opt {
    #[structopt(parse(from_os_str), default_value = "-")]
    input: Vec<PathBuf>,

    #[structopt(short = "l", long = "length", default_value = "512")]
    /// The size of the output in bits. Must be a multiple of 8.
    length: usize,
}

fn open_input(path: &Path) -> io::Result<File> {
    if path == Path::new("-") {
        os_pipe::dup_stdin().map(From::from)
    } else {
        File::open(path)
    }
}

fn hash_file(mut f: File, hash_length: usize) -> io::Result<Hash> {
    let mut params = Params::default();
    params.hash_length(hash_length);
    let mut state = State::with_params(&params);
    io::copy(&mut f, &mut state)?;
    Ok(state.finalize())
}

fn exit_path_error(path: &Path, e: io::Error) {
    eprintln!("b2sum: {}: {}", path.to_string_lossy(), e);
    exit(1);
}

fn main() {
    let opt = Opt::from_args();

    if opt.length == 0 || opt.length > 512 || opt.length % 8 != 0 {
        eprintln!("Invalid length.");
        exit(1);
    }
    let hash_length = opt.length / 8;

    let mut inputs = Vec::new();
    for path in &opt.input {
        match open_input(path) {
            Ok(file) => inputs.push((path, file)),
            Err(e) => exit_path_error(path, e),
        }
    }

    for (path, file) in inputs {
        match hash_file(file, hash_length) {
            Ok(hash) => println!("{}  {}", hash.hex(), path.to_string_lossy()),
            Err(e) => exit_path_error(path, e),
        }
    }
}
