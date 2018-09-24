extern crate blake2b_simd;
extern crate memmap;
extern crate os_pipe;
extern crate rayon;
#[macro_use]
extern crate structopt;

use blake2b_simd::{blake2bp, Hash, Params, State};
use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::isize;
use std::path::{Path, PathBuf};
use std::process::exit;
use structopt::StructOpt;

#[cfg(test)]
mod test;
#[cfg(test)]
#[macro_use]
extern crate duct;
#[cfg(test)]
extern crate tempfile;

#[derive(Debug, StructOpt)]
#[structopt(author = "")]
struct Opt {
    #[structopt(parse(from_os_str), default_value = "-")]
    /// Any number of filepaths, or - for standard input.
    input: Vec<PathBuf>,

    #[structopt(short = "l", long = "length", default_value = "512")]
    /// The size of the output in bits. Must be a multiple of 8.
    length_bits: usize,

    #[structopt(long = "mmap")]
    /// Read input with memory mapping.
    mmap: bool,

    #[structopt(long = "blake2bp")]
    /// Use the BLAKE2bp parallel hash function. Implies --mmap.
    blake2bp: bool,
}

fn hash_one(path: &Path, opt: &Opt) -> io::Result<Hash> {
    let hash_length = opt.length_bits / 8;
    let mut state = Params::new().hash_length(hash_length).to_state();
    if path == Path::new("-") {
        if opt.blake2bp {
            let stdin_file = os_pipe::dup_stdin()?.into();
            let map = mmap_file(&stdin_file)?;
            return Ok(blake2bp(&map[..], hash_length));
        } else if opt.mmap {
            let stdin_file = os_pipe::dup_stdin()?.into();
            let map = mmap_file(&stdin_file)?;
            state.update(&map);
        } else {
            let stdin = io::stdin();
            let mut stdin = stdin.lock();
            read_write_all(&mut stdin, &mut state)?;
        }
    } else {
        let mut file = File::open(path)?;
        if opt.blake2bp {
            let map = mmap_file(&file)?;
            return Ok(blake2bp(&map[..], hash_length));
        } else if opt.mmap {
            let map = mmap_file(&file)?;
            state.update(&map);
        } else {
            read_write_all(&mut file, &mut state)?;
        }
    }
    Ok(state.finalize())
}

fn mmap_file(file: &File) -> io::Result<memmap::Mmap> {
    let metadata = file.metadata()?;
    let len = metadata.len();
    // See https://github.com/danburkert/memmap-rs/issues/69.
    if len > isize::MAX as u64 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "mmap length would overflow isize",
        ));
    }
    // Note that this will currently fail if the file is empty. b2sum treats the --mmap flag as
    // mandatory, so we don't try to recover. See https://github.com/danburkert/memmap-rs/issues/72
    // for a discussion about whether this case should succeed in the future.
    unsafe { memmap::MmapOptions::new().len(len as usize).map(file) }
}

fn read_write_all<R: Read>(reader: &mut R, writer: &mut State) -> io::Result<()> {
    // Why not just use std::io::copy? Because it uses an 8192 byte buffer, and using a larger
    // buffer is measurably faster.
    //
    // How did we pick 32768 (2^15) specifically? It's just what coreutils uses. When I benchmark
    // lots of different sizes, a 4 MiB heap buffer actually seems to be the best size, possibly 8%
    // faster than this. Though repeatedly hashing a gigabyte of random data might not reflect real
    // world usage, who knows. At the end of the day, when we really care about speed, we're going
    // to use --mmap and skip buffering entirely. The main goal of this program is to compare the
    // underlying hash implementations (which is to say OpenSSL, which coreutils links against),
    // and to get an honest comparison we might as well use the same buffer size.
    let mut buf = [0; 32768];
    loop {
        match reader.read(&mut buf) {
            Ok(0) => return Ok(()),
            Ok(n) => writer.write_all(&buf[..n])?,
            Err(e) => if e.kind() != io::ErrorKind::Interrupted {
                return Err(e);
            },
        }
    }
}

fn main() {
    let opt = Opt::from_args();

    if opt.length_bits == 0 || opt.length_bits > 512 || opt.length_bits % 8 != 0 {
        eprintln!("Invalid length.");
        exit(1);
    }

    // BLAKE2bp requires exactly 4 threads.
    rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build_global()
        .unwrap();

    let mut did_error = false;
    for path in &opt.input {
        let path_str = path.to_string_lossy();
        match hash_one(path, &opt) {
            Ok(hash) => println!("{}  {}", hash.to_hex(), path_str),
            Err(e) => {
                did_error = true;
                eprintln!("b2sum: {}: {}", path_str, e);
            }
        }
    }
    if did_error {
        exit(1);
    }
}
