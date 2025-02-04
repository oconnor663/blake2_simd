# blake2_bin

The [`blake2_bin`](https://crates.io/crates/blake2_bin) crate provides
the `blake2` command line utility, based on the
[`blake2b_simd`](https://crates.io/crates/blake2b_simd) and
[`blake2s_simd`](https://crates.io/crates/blake2s_simd) implementations
of BLAKE2. It supports BLAKE2b, BLAKE2s, BLAKE2bp, and BLAKE2sp.

# Installation

```
cargo install blake2_bin
```

# Example

```
# Hash standard input, using BLAKE2b by default.
$ echo hello world | blake2
fec91c70284c72d0d4e3684788a90de9338a5b2f47f01fedbe203cafd68708718ae5672d10eca804a8121904047d40d1d6cf11e7a76419357a9469af41f22d01

# Since BLAKE2b is the default, explicitly adding the -b flag gives the same result.
$ echo hello world | blake2 -b
fec91c70284c72d0d4e3684788a90de9338a5b2f47f01fedbe203cafd68708718ae5672d10eca804a8121904047d40d1d6cf11e7a76419357a9469af41f22d01

# Using the -s and -p flags together gives BLAKE2sp.
$ echo hello world | blake2 -sp
43958a843c00345bae4492cc04ecd1e47453469afeae277e067cad66244625eb

# The full set of command line options.
$ blake2 --help
Usage: blake2 [OPTIONS] [INPUTS]...

Arguments:
  [INPUTS]...  Any number of filepaths, or empty for standard input

Options:
  -b                                           Use the BLAKE2b hash function (default)
  -s                                           Use the BLAKE2s hash function
  -p                                           Use the parallel variant, BLAKE2bp or BLAKE2sp
      --length <LENGTH>                        Set the length of the output in bytes
      --key <KEY>                              Set the key parameter with a hex string
      --salt <SALT>                            Set the salt parameter with a hex string
      --personal <PERSONAL>                    Set the personalization parameter with a hex string
      --fanout <FANOUT>                        Set the fanout parameter
      --max-depth <MAX_DEPTH>                  Set the max depth parameter
      --max-leaf-length <MAX_LEAF_LENGTH>      Set the max leaf length parameter
      --node-offset <NODE_OFFSET>              Set the node offset parameter
      --node-depth <NODE_DEPTH>                Set the node depth parameter
      --inner-hash-length <INNER_HASH_LENGTH>  Set the inner hash length parameter
      --last-node                              Set the last node flag
      --mmap                                   Read input with memory mapping
  -h, --help                                   Print help
  -V, --version                                Print version
```
