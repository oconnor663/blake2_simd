use failure::{bail, Error};
use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::isize;
use std::path::PathBuf;
use std::process::exit;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(author = "")]
struct Opt {
    /// Any number of filepaths, or - for standard input.
    input: Option<PathBuf>,

    #[structopt(long = "mmap")]
    /// Read input with memory mapping.
    mmap: bool,

    #[structopt(long = "blake2b")]
    /// Use the BLAKE2b hash function (default).
    blake2b: bool,

    #[structopt(long = "blake2s")]
    /// Use the BLAKE2s hash function.
    blake2s: bool,

    #[structopt(long = "blake2bp")]
    /// Use the BLAKE2bp hash function.
    blake2bp: bool,

    #[structopt(long = "blake2sp")]
    /// Use the BLAKE2sp hash function.
    blake2sp: bool,

    #[structopt(short = "l", long = "length")]
    /// Set the length of the output in bytes.
    length: Option<usize>,

    #[structopt(long = "key")]
    /// Set the key parameter with a hex string.
    key: Option<String>,

    #[structopt(long = "salt")]
    /// Set the salt parameter with a hex string.
    salt: Option<String>,

    #[structopt(long = "personal")]
    /// Set the personalization parameter with a hex string.
    personal: Option<String>,

    #[structopt(long = "fanout")]
    /// Set the fanout parameter.
    fanout: Option<u8>,

    #[structopt(long = "max-depth")]
    /// Set the max depth parameter.
    max_depth: Option<u8>,

    #[structopt(long = "max-leaf-length")]
    /// Set the max leaf length parameter.
    max_leaf_length: Option<u32>,

    #[structopt(long = "node-offset")]
    /// Set the node offset parameter.
    node_offset: Option<u64>,

    #[structopt(long = "node-depth")]
    /// Set the node depth parameter.
    node_depth: Option<u8>,

    #[structopt(long = "inner-hash-length")]
    /// Set the inner hash length parameter.
    inner_hash_length: Option<usize>,

    #[structopt(long = "last-node")]
    /// Set the last node flag.
    last_node: bool,
}

enum Params {
    Blake2b(blake2b_simd::Params),
    Blake2bp(blake2b_simd::blake2bp::Params),
    Blake2s(blake2s_simd::Params),
    Blake2sp(blake2s_simd::blake2sp::Params),
}

#[derive(Clone)]
enum State {
    Blake2b(blake2b_simd::State),
    Blake2bp(blake2b_simd::blake2bp::State),
    Blake2s(blake2s_simd::State),
    Blake2sp(blake2s_simd::blake2sp::State),
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
    // Note that this will currently fail if the file is empty. We treat the --mmap flag as
    // mandatory and don't try to recover. See https://github.com/danburkert/memmap-rs/issues/72
    // for a discussion about whether this case should succeed in the future.
    unsafe { memmap::MmapOptions::new().len(len as usize).map(file) }
}

fn read_write_all<R: Read>(reader: &mut R, state: &mut State) -> io::Result<()> {
    // Why not just use std::io::copy? Because it uses an 8192 byte buffer, and
    // using a larger buffer is measurably faster.
    // https://github.com/rust-lang/rust/commit/8128817119e479b0610685e3fc7a6ff21cde5abc
    // describes how Rust picked its default buffer size.
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
            Ok(n) => match state {
                State::Blake2b(s) => {
                    s.update(&buf[..n]);
                }
                State::Blake2s(s) => {
                    s.update(&buf[..n]);
                }
                State::Blake2bp(s) => {
                    s.update(&buf[..n]);
                }
                State::Blake2sp(s) => {
                    s.update(&buf[..n]);
                }
            },
            Err(e) => {
                if e.kind() != io::ErrorKind::Interrupted {
                    return Err(e);
                }
            }
        }
    }
}

fn make_state(opt: &Opt) -> Result<State, Error> {
    let type_count: u32 = [opt.blake2b, opt.blake2s, opt.blake2bp, opt.blake2sp]
        .iter()
        .map(|b| *b as u32)
        .sum();
    if type_count > 1 {
        bail!("more than one hash function specified");
    }
    let mut params = if opt.blake2s {
        Params::Blake2s(blake2s_simd::Params::new())
    } else if opt.blake2bp {
        Params::Blake2bp(blake2b_simd::blake2bp::Params::new())
    } else if opt.blake2sp {
        Params::Blake2sp(blake2s_simd::blake2sp::Params::new())
    } else {
        Params::Blake2b(blake2b_simd::Params::new())
    };
    if let Some(length) = opt.length {
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
    if let Some(ref key) = opt.key {
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
    if let Some(ref salt) = opt.salt {
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
    if let Some(ref personal) = opt.personal {
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
    if let Some(fanout) = opt.fanout {
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
    if let Some(max_depth) = opt.max_depth {
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
    if let Some(max_leaf_length) = opt.max_leaf_length {
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
    if let Some(node_offset) = opt.node_offset {
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
    if let Some(node_depth) = opt.node_depth {
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
    if let Some(inner_hash_length) = opt.inner_hash_length {
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
    if opt.last_node {
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
    let state = match &params {
        Params::Blake2b(p) => State::Blake2b(p.to_state()),
        Params::Blake2s(p) => State::Blake2s(p.to_state()),
        Params::Blake2bp(p) => State::Blake2bp(p.to_state()),
        Params::Blake2sp(p) => State::Blake2sp(p.to_state()),
    };
    Ok(state)
}

fn run(opt: &Opt, state: &State) -> Result<String, Error> {
    let mut state = state.clone();
    if let Some(path) = &opt.input {
        let mut file = File::open(path)?;
        if opt.mmap {
            let map = mmap_file(&file)?;
            match &mut state {
                State::Blake2b(s) => {
                    s.update(&map);
                }
                State::Blake2s(s) => {
                    s.update(&map);
                }
                State::Blake2bp(s) => {
                    s.update(&map);
                }
                State::Blake2sp(s) => {
                    s.update(&map);
                }
            }
        } else {
            read_write_all(&mut file, &mut state)?;
        }
    } else {
        if opt.mmap {
            bail!("can't mmap standard input");
        } else {
            let stdin = io::stdin();
            let mut stdin = stdin.lock();
            read_write_all(&mut stdin, &mut state)?;
        }
    }
    Ok(match state {
        State::Blake2b(s) => s.finalize().to_hex().to_string(),
        State::Blake2s(s) => s.finalize().to_hex().to_string(),
        State::Blake2bp(s) => s.finalize().to_hex().to_string(),
        State::Blake2sp(s) => s.finalize().to_hex().to_string(),
    })
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

    match run(&opt, &state) {
        Ok(hash) => println!("{}", hash),
        Err(e) => {
            eprintln!("blake2: {}", e);
            exit(1);
        }
    }
}
