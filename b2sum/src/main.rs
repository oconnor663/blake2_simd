extern crate blake2b_simd;
extern crate hex;
extern crate memmap;
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
    /// The size of the output in bits. Must be a multiple of 8. Max 512.
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
    /// Set the BLAKE2 fanout parameter.
    fanout: Option<u8>,

    #[structopt(long = "max-depth")]
    /// Set the BLAKE2 max depth parameter.
    max_depth: Option<u8>,

    #[structopt(long = "max-leaf-length")]
    /// Set the BLAKE2 max leaf length parameter.
    max_leaf_length: Option<u32>,

    #[structopt(long = "node-offset")]
    /// Set the BLAKE2 node offset parameter.
    node_offset: Option<u64>,

    #[structopt(long = "node-depth")]
    /// Set the BLAKE2 node depth parameter.
    node_depth: Option<u8>,

    #[structopt(long = "inner-hash-length")]
    /// Set the BLAKE2 inner hash length parameter, in bits like --length.
    inner_hash_length_bits: Option<usize>,

    #[structopt(long = "last-node")]
    /// Set the BLAKE2 last node flag.
    last_node: bool,
}

#[derive(Clone, Debug)]
enum EitherState {
    Blake2b(State),
    Blake2bp(blake2bp::State),
}

impl EitherState {
    fn finalize(&mut self) -> Hash {
        match *self {
            EitherState::Blake2b(ref mut state) => state.finalize(),
            EitherState::Blake2bp(ref mut state) => state.finalize(),
        }
    }

    fn force_portable(&mut self) {
        match *self {
            EitherState::Blake2b(ref mut state) => {
                blake2b_simd::benchmarks::force_portable(state);
            }
            EitherState::Blake2bp(ref mut state) => {
                blake2b_simd::benchmarks::force_portable_blake2bp(state);
            }
        }
    }
}

impl Write for EitherState {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match *self {
            EitherState::Blake2b(ref mut state) => {
                state.update(&buf);
            }
            EitherState::Blake2bp(ref mut state) => {
                state.update(&buf);
            }
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn hash_one(path: &Path, opt: &Opt, state: &EitherState) -> io::Result<Hash> {
    let mut state = state.clone();
    if path == Path::new("-") {
        if opt.mmap {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, CANT_MMAP_ERROR));
        } else {
            let stdin = io::stdin();
            let mut stdin = stdin.lock();
            read_write_all(&mut stdin, &mut state)?;
        }
    } else {
        let mut file = File::open(path)?;
        if opt.mmap {
            let map = mmap_file(&file)?;
            state.write_all(&map).unwrap();
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

fn read_write_all<R: Read>(reader: &mut R, writer: &mut EitherState) -> io::Result<()> {
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

fn bits_to_bytes(bits: usize) -> Result<usize, Box<Error>> {
    if bits == 0 || bits > 512 || bits % 8 != 0 {
        Err("Invalid number of bits.".into())
    } else {
        Ok(bits / 8)
    }
}

fn make_state(opt: &Opt) -> Result<EitherState, Box<Error>> {
    let mut params = Params::new();
    let mut blake2bp_params = blake2bp::Params::new();
    if let Some(length_bits) = opt.length_bits {
        let length_bytes = bits_to_bytes(length_bits)?;
        params.hash_length(length_bytes);
        blake2bp_params.hash_length(length_bytes);
    }
    if let Some(ref key) = opt.key {
        let key_bytes = hex::decode(key)?;
        params.key(&key_bytes);
        blake2bp_params.key(&key_bytes);
    }
    if let Some(ref salt) = opt.salt {
        let salt_bytes = hex::decode(salt)?;
        params.salt(&salt_bytes);
        if opt.blake2bp {
            return Err("BLAKE2bp doesn't support --salt.".into());
        }
    }
    if let Some(ref personal) = opt.personal {
        let personal_bytes = hex::decode(personal)?;
        params.personal(&personal_bytes);
        if opt.blake2bp {
            return Err("BLAKE2bp doesn't support --personal.".into());
        }
    }
    if let Some(fanout) = opt.fanout {
        params.fanout(fanout);
        if opt.blake2bp {
            return Err("BLAKE2bp doesn't support --fanout.".into());
        }
    }
    if let Some(max_depth) = opt.max_depth {
        params.max_depth(max_depth);
        if opt.blake2bp {
            return Err("BLAKE2bp doesn't support --max-depth.".into());
        }
    }
    if let Some(max_leaf_length) = opt.max_leaf_length {
        params.max_leaf_length(max_leaf_length);
        if opt.blake2bp {
            return Err("BLAKE2bp doesn't support --max-leaf-length.".into());
        }
    }
    if let Some(node_offset) = opt.node_offset {
        params.node_offset(node_offset);
        if opt.blake2bp {
            return Err("BLAKE2bp doesn't support --node-offset.".into());
        }
    }
    if let Some(node_depth) = opt.node_depth {
        params.node_depth(node_depth);
        if opt.blake2bp {
            return Err("BLAKE2bp doesn't support --node-depth.".into());
        }
    }
    if let Some(inner_hash_length_bits) = opt.inner_hash_length_bits {
        let inner_hash_length_bytes = bits_to_bytes(inner_hash_length_bits)?;
        params.inner_hash_length(inner_hash_length_bytes);
        if opt.blake2bp {
            return Err("BLAKE2bp doesn't support --inner-hash-length.".into());
        }
    }
    if opt.last_node {
        params.last_node(true);
        if opt.blake2bp {
            return Err("BLAKE2bp doesn't support --last-node.".into());
        }
    }
    let mut ret = if opt.blake2bp {
        EitherState::Blake2bp(blake2bp_params.to_state())
    } else {
        EitherState::Blake2b(params.to_state())
    };
    if opt.portable {
        ret.force_portable();
    }
    Ok(ret)
}

fn main() {
    let opt = Opt::from_args();

    let state = match make_state(&opt) {
        Ok(params) => params,
        Err(e) => {
            eprintln!("{}", e);
            exit(1);
        }
    };

    let mut did_error = false;
    for path in &opt.input {
        let path_str = path.to_string_lossy();
        match hash_one(path, &opt, &state) {
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
