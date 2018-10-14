extern crate blake2b_simd;
extern crate hex;
extern crate memmap;
extern crate rayon;
#[macro_use]
extern crate structopt;

use blake2b_simd::{blake2bp, Hash, Params, State};
use std::error::Error;
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

const CANT_MMAP_ERROR: &str = "memory mapping requires a filepath";

#[derive(Debug, StructOpt)]
#[structopt(author = "")]
struct Opt {
    #[structopt(parse(from_os_str), default_value = "-")]
    /// Any number of filepaths, or - for standard input.
    input: Vec<PathBuf>,

    #[structopt(long = "mmap")]
    /// Read input with memory mapping.
    mmap: bool,

    #[structopt(long = "blake2bp")]
    /// Use the BLAKE2bp parallel hash function. Implies --mmap.
    blake2bp: bool,

    #[structopt(long = "portable")]
    /// Always use the portable (non-AVX2) BLAKE2b implementation.
    portable: bool,

    #[structopt(short = "l", long = "length")]
    /// The size of the output in bits. Must be a multiple of 8.
    length_bits: Option<usize>,

    #[structopt(long = "key")]
    /// Set the BLAKE2 key parameter with a hex string.
    key: Option<String>,

    #[structopt(long = "salt")]
    /// Set the BLAKE2 salt parameter with a hex string.
    salt: Option<String>,

    #[structopt(long = "personal")]
    /// Set the BLAKE2 personalization parameter with a hex string.
    personal: Option<String>,

    #[structopt(long = "fanout")]
    /// Set the BLAKE2 fanout parameter, 1 byte.
    fanout: Option<u8>,

    #[structopt(long = "max-depth")]
    /// Set the BLAKE2 max depth parameter, 1 bytes.
    max_depth: Option<u8>,

    #[structopt(long = "max-leaf-length")]
    /// Set the BLAKE2 max leaf length parameter, 4 bytes.
    max_leaf_length: Option<u32>,

    #[structopt(long = "node-offset")]
    /// Set the BLAKE2 node offset parameter, 8 bytes.
    node_offset: Option<u64>,

    #[structopt(long = "node-depth")]
    /// Set the BLAKE2 node depth parameter, 1 byte.
    node_depth: Option<u8>,

    #[structopt(long = "inner-hash-length")]
    /// Set the BLAKE2 inner hash length parameter, 1 byte.
    inner_hash_length: Option<u8>,

    #[structopt(long = "last-node")]
    /// Set the BLAKE2 last node flag.
    last_node: bool,
}

fn hash_one(path: &Path, opt: &Opt, params: &Params) -> io::Result<Hash> {
    let mut state = params.to_state();
    if opt.portable {
        blake2b_simd::benchmarks::force_portable(&mut state);
    }
    if path == Path::new("-") {
        if opt.blake2bp || opt.mmap {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, CANT_MMAP_ERROR));
        } else {
            let stdin = io::stdin();
            let mut stdin = stdin.lock();
            read_write_all(&mut stdin, &mut state)?;
        }
    } else {
        let mut file = File::open(path)?;
        if opt.blake2bp {
            let map = mmap_file(&file)?;
            return Ok(blake2bp(&map[..], params));
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

fn make_params(opt: &Opt) -> Result<Params, Box<Error>> {
    let mut params = Params::new();
    if let Some(length_bits) = opt.length_bits {
        if length_bits == 0 || length_bits > 512 || length_bits % 8 != 0 {
            return Err("Invalid length.".into());
        }
        params.hash_length(length_bits / 8);
    }
    if let Some(ref key) = opt.key {
        params.key(&hex::decode(key)?);
    }
    if let Some(ref salt) = opt.salt {
        params.salt(&hex::decode(salt)?);
    }
    if let Some(ref personal) = opt.personal {
        params.personal(&hex::decode(personal)?);
    }
    if let Some(fanout) = opt.fanout {
        params.fanout(fanout);
    }
    if let Some(max_depth) = opt.max_depth {
        params.max_depth(max_depth);
    }
    if let Some(max_leaf_length) = opt.max_leaf_length {
        params.max_leaf_length(max_leaf_length);
    }
    if let Some(node_offset) = opt.node_offset {
        params.node_offset(node_offset);
    }
    if let Some(node_depth) = opt.node_depth {
        params.node_depth(node_depth);
    }
    if let Some(inner_hash_length) = opt.inner_hash_length {
        params.inner_hash_length(inner_hash_length as usize);
    }
    params.last_node(opt.last_node);
    Ok(params)
}

fn main() {
    let opt = Opt::from_args();

    let params = match make_params(&opt) {
        Ok(params) => params,
        Err(e) => {
            eprintln!("{}", e);
            exit(1);
        }
    };

    // BLAKE2bp requires exactly 4 threads.
    rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build_global()
        .unwrap();

    let mut did_error = false;
    for path in &opt.input {
        let path_str = path.to_string_lossy();
        match hash_one(path, &opt, &params) {
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
