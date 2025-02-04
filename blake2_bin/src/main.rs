use anyhow::bail;
use clap::Parser;
use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::path::{Path, PathBuf};
use std::process::exit;

#[derive(Debug, Parser)]
struct Args {
    /// Any number of filepaths, or empty for standard input.
    inputs: Vec<PathBuf>,

    #[arg(long = "mmap")]
    /// Read input with memory mapping.
    mmap: bool,

    #[arg(short = 'b')]
    /// Use the BLAKE2b hash function (default).
    big: bool,

    #[arg(short = 's')]
    /// Use the BLAKE2s hash function.
    small: bool,

    #[arg(short = 'p')]
    /// Use the parallel variant, BLAKE2bp or BLAKE2sp.
    parallel: bool,

    #[arg(long = "length")]
    /// Set the length of the output in bytes.
    length: Option<usize>,

    #[arg(long = "key")]
    /// Set the key parameter with a hex string.
    key: Option<String>,

    #[arg(long = "salt")]
    /// Set the salt parameter with a hex string.
    salt: Option<String>,

    #[arg(long = "personal")]
    /// Set the personalization parameter with a hex string.
    personal: Option<String>,

    #[arg(long = "fanout")]
    /// Set the fanout parameter.
    fanout: Option<u8>,

    #[arg(long = "max-depth")]
    /// Set the max depth parameter.
    max_depth: Option<u8>,

    #[arg(long = "max-leaf-length")]
    /// Set the max leaf length parameter.
    max_leaf_length: Option<u32>,

    #[arg(long = "node-offset")]
    /// Set the node offset parameter.
    node_offset: Option<u64>,

    #[arg(long = "node-depth")]
    /// Set the node depth parameter.
    node_depth: Option<u8>,

    #[arg(long = "inner-hash-length")]
    /// Set the inner hash length parameter.
    inner_hash_length: Option<usize>,

    #[arg(long = "last-node")]
    /// Set the last node flag.
    last_node: bool,
}

enum Params {
    Blake2b(blake2b_simd::Params),
    Blake2bp(blake2b_simd::blake2bp::Params),
    Blake2s(blake2s_simd::Params),
    Blake2sp(blake2s_simd::blake2sp::Params),
}

impl Params {
    fn to_state(&self) -> State {
        match self {
            Params::Blake2b(p) => State::Blake2b(p.to_state()),
            Params::Blake2s(p) => State::Blake2s(p.to_state()),
            Params::Blake2bp(p) => State::Blake2bp(p.to_state()),
            Params::Blake2sp(p) => State::Blake2sp(p.to_state()),
        }
    }
}

#[derive(Clone)]
enum State {
    Blake2b(blake2b_simd::State),
    Blake2bp(blake2b_simd::blake2bp::State),
    Blake2s(blake2s_simd::State),
    Blake2sp(blake2s_simd::blake2sp::State),
}

impl State {
    fn update(&mut self, input: &[u8]) {
        match self {
            State::Blake2b(s) => {
                s.update(input);
            }
            State::Blake2s(s) => {
                s.update(input);
            }
            State::Blake2bp(s) => {
                s.update(input);
            }
            State::Blake2sp(s) => {
                s.update(input);
            }
        }
    }

    fn finalize(&mut self) -> String {
        match self {
            State::Blake2b(s) => s.finalize().to_hex().to_string(),
            State::Blake2s(s) => s.finalize().to_hex().to_string(),
            State::Blake2bp(s) => s.finalize().to_hex().to_string(),
            State::Blake2sp(s) => s.finalize().to_hex().to_string(),
        }
    }
}

fn read_write_all<R: Read>(mut reader: R, state: &mut State) -> io::Result<()> {
    // Why not just use std::io::copy? Because it uses an 8192 byte buffer, and
    // using a larger buffer is measurably faster.
    // https://github.com/rust-lang/rust/commit/8128817119e479b0610685e3fc7a6ff21cde5abc
    // describes how Rust picked its default buffer size.
    //
    // How did we pick 32768 (2^15) specifically? It's just what coreutils
    // uses. When I benchmark lots of different sizes, a 4 MiB heap buffer
    // actually seems to be the best size, possibly 8% faster than this. Though
    // repeatedly hashing a gigabyte of random data might not reflect real
    // world usage, who knows. At the end of the day, when we really care about
    // speed, we're going to use --mmap and skip buffering entirely. The main
    // goal of this program is to compare the underlying hash implementations
    // (which is to say OpenSSL, which coreutils links against), and to get an
    // honest comparison we might as well use the same buffer size.
    let mut buf = [0; 32768];
    loop {
        match reader.read(&mut buf) {
            Ok(0) => return Ok(()),
            Ok(n) => state.update(&buf[..n]),
            Err(e) => {
                if e.kind() != io::ErrorKind::Interrupted {
                    return Err(e);
                }
            }
        }
    }
}

fn make_params(args: &Args) -> anyhow::Result<Params> {
    if args.big && args.small {
        bail!("-b and -s can't be used together");
    }
    let mut params = if args.small {
        if args.parallel {
            Params::Blake2sp(blake2s_simd::blake2sp::Params::new())
        } else {
            Params::Blake2s(blake2s_simd::Params::new())
        }
    } else {
        if args.parallel {
            Params::Blake2bp(blake2b_simd::blake2bp::Params::new())
        } else {
            Params::Blake2b(blake2b_simd::Params::new())
        }
    };
    if let Some(length) = args.length {
        match &mut params {
            Params::Blake2b(p) => {
                p.hash_length(length);
            }
            Params::Blake2s(p) => {
                p.hash_length(length);
            }
            Params::Blake2bp(p) => {
                p.hash_length(length);
            }
            Params::Blake2sp(p) => {
                p.hash_length(length);
            }
        }
    }
    if let Some(ref key) = args.key {
        let key_bytes = hex::decode(key)?;
        match &mut params {
            Params::Blake2b(p) => {
                p.key(&key_bytes);
            }
            Params::Blake2s(p) => {
                p.key(&key_bytes);
            }
            Params::Blake2bp(p) => {
                p.key(&key_bytes);
            }
            Params::Blake2sp(p) => {
                p.key(&key_bytes);
            }
        }
    }
    if let Some(ref salt) = args.salt {
        let salt_bytes = hex::decode(salt)?;
        match &mut params {
            Params::Blake2b(p) => {
                p.salt(&salt_bytes);
            }
            Params::Blake2s(p) => {
                p.salt(&salt_bytes);
            }
            _ => bail!("--salt not supported"),
        }
    }
    if let Some(ref personal) = args.personal {
        let personal_bytes = hex::decode(personal)?;
        match &mut params {
            Params::Blake2b(p) => {
                p.personal(&personal_bytes);
            }
            Params::Blake2s(p) => {
                p.personal(&personal_bytes);
            }
            _ => bail!("--personal not supported"),
        }
    }
    if let Some(fanout) = args.fanout {
        match &mut params {
            Params::Blake2b(p) => {
                p.fanout(fanout);
            }
            Params::Blake2s(p) => {
                p.fanout(fanout);
            }
            _ => bail!("--fanout not supported"),
        }
    }
    if let Some(max_depth) = args.max_depth {
        match &mut params {
            Params::Blake2b(p) => {
                p.max_depth(max_depth);
            }
            Params::Blake2s(p) => {
                p.max_depth(max_depth);
            }
            _ => bail!("--max-depth not supported"),
        }
    }
    if let Some(max_leaf_length) = args.max_leaf_length {
        match &mut params {
            Params::Blake2b(p) => {
                p.max_leaf_length(max_leaf_length);
            }
            Params::Blake2s(p) => {
                p.max_leaf_length(max_leaf_length);
            }
            _ => bail!("--max-leaf-length not supported"),
        }
    }
    if let Some(node_offset) = args.node_offset {
        match &mut params {
            Params::Blake2b(p) => {
                p.node_offset(node_offset);
            }
            Params::Blake2s(p) => {
                p.node_offset(node_offset);
            }
            _ => bail!("--node-offset not supported"),
        }
    }
    if let Some(node_depth) = args.node_depth {
        match &mut params {
            Params::Blake2b(p) => {
                p.node_depth(node_depth);
            }
            Params::Blake2s(p) => {
                p.node_depth(node_depth);
            }
            _ => bail!("--node-depth not supported"),
        }
    }
    if let Some(inner_hash_length) = args.inner_hash_length {
        match &mut params {
            Params::Blake2b(p) => {
                p.inner_hash_length(inner_hash_length);
            }
            Params::Blake2s(p) => {
                p.inner_hash_length(inner_hash_length);
            }
            _ => bail!("--inner-hash-length not supported"),
        }
    }
    if args.last_node {
        match &mut params {
            Params::Blake2b(p) => {
                p.last_node(true);
            }
            Params::Blake2s(p) => {
                p.last_node(true);
            }
            _ => bail!("--last-node not supported"),
        }
    }
    Ok(params)
}

fn hash_file(args: &Args, params: &Params, path: &Path) -> anyhow::Result<String> {
    let mut state = params.to_state();
    let mut file = File::open(path)?;
    if args.mmap {
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        state.update(&mmap);
    } else {
        read_write_all(&mut file, &mut state)?;
    }
    Ok(state.finalize())
}

fn hash_stdin(args: &Args, params: &Params) -> anyhow::Result<String> {
    if args.mmap {
        bail!("--mmap not supported for stdin");
    }
    let mut state = params.to_state();
    read_write_all(std::io::stdin().lock(), &mut state)?;
    Ok(state.finalize())
}

fn main() {
    let args = Args::parse();

    let params = match make_params(&args) {
        Ok(params) => params,
        Err(e) => {
            eprintln!("blake2: {}", e);
            exit(1);
        }
    };

    let mut failed = false;
    if args.inputs.is_empty() {
        match hash_stdin(&args, &params) {
            Ok(hash) => println!("{}", hash),
            Err(e) => {
                eprintln!("blake2: stdin: {}", e);
                failed = true;
            }
        }
    } else {
        for input in &args.inputs {
            match hash_file(&args, &params, input) {
                Ok(hash) => {
                    if args.inputs.len() > 1 {
                        println!("{}  {}", hash, input.to_string_lossy());
                    } else {
                        println!("{}", hash);
                    }
                }
                Err(e) => {
                    eprintln!("blake2: {}: {}", input.to_string_lossy(), e);
                    failed = true;
                }
            }
        }
    }
    if failed {
        exit(1);
    }
}
