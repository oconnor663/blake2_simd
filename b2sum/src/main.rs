extern crate blake2b_simd;
extern crate memmap;
extern crate os_pipe;
#[macro_use]
extern crate structopt;

use blake2b_simd::{Hash, Params, State};
use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::path::{Path, PathBuf};
use std::process::exit;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(author = "")]
struct Opt {
    #[structopt(parse(from_os_str), default_value = "-")]
    /// Any number of filepaths, or - for standard input.
    input: Vec<PathBuf>,

    #[structopt(short = "l", long = "length", default_value = "512")]
    /// The size of the output in bits. Must be a multiple of 8.
    length: usize,

    #[structopt(long = "mmap")]
    /// Read input with memory mapping.
    mmap: bool,
}

enum Input {
    Stdin,
    File(File),
    Mmap(memmap::Mmap),
}

fn open_input(path: &Path, mmap: bool) -> io::Result<Input> {
    Ok(if path == Path::new("-") {
        if mmap {
            let stdin_file = os_pipe::dup_stdin()?.into();
            Input::Mmap(unsafe { memmap::Mmap::map(&stdin_file)? })
        } else {
            Input::Stdin
        }
    } else {
        let file = File::open(path)?;
        if mmap {
            Input::Mmap(unsafe { memmap::Mmap::map(&file)? })
        } else {
            Input::File(file)
        }
    })
}

fn hash_one(input: Input, hash_length: usize) -> io::Result<Hash> {
    let mut params = Params::default();
    params.hash_length(hash_length);
    let mut state = State::with_params(&params);
    match input {
        Input::Stdin => {
            let stdin = io::stdin();
            let mut stdin = stdin.lock();
            read_write_all(&mut stdin, &mut state)?;
        }
        Input::File(mut file) => {
            read_write_all(&mut file, &mut state)?;
        }
        Input::Mmap(mmap) => {
            state.update(&mmap);
        }
    }
    Ok(state.finalize())
}

fn read_write_all<R: Read, W: Write>(reader: &mut R, writer: &mut W) -> io::Result<()> {
    // Why 32728 (2^15)? Basically, that's just what coreutils uses. When I benchmark lots of
    // different sizes, a 4 MiB heap buffer actually seems to be the best size, possibly 8% faster
    // than this. Though repeatedly hashing a gigabyte of random data might not reflect real world
    // usage, who knows. At the end of the day, when we really care about speed, we're going to use
    // --mmap and skip buffering entirely. The main goal of this program is to compare the
    // underlying hash implementations (which is to say OpenSSL, which coreutils links against),
    // and to get an honest comparison we might as well use the same buffer size.
    let mut buf = [0; 32768];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            return Ok(());
        }
        writer.write_all(&buf[..n])?;
    }
}

fn main() {
    let opt = Opt::from_args();

    if opt.length == 0 || opt.length > 512 || opt.length % 8 != 0 {
        eprintln!("Invalid length.");
        exit(1);
    }
    let hash_length = opt.length / 8;

    let mut did_error = false;
    for path in &opt.input {
        match open_input(path, opt.mmap) {
            Ok(input) => match hash_one(input, hash_length) {
                Ok(hash) => println!("{}  {}", hash.hex(), path.to_string_lossy()),
                Err(e) => {
                    did_error = true;
                    eprintln!("b2sum: {}: {}", path.to_string_lossy(), e);
                }
            },
            Err(e) => {
                did_error = true;
                eprintln!("b2sum: {}: {}", path.to_string_lossy(), e);
            }
        }
    }
    if did_error {
        exit(1);
    }
}
