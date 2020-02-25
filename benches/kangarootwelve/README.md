This is Rust FFI wrapper around the
[XKCP/K12](https://github.com/XKCP/K12) C implementation, which is
vendored here and statically linked. It's intended for benchmarking
only. This crate assumes that it will only run on the machine that
builds it. If the build machine supports AVX2, it builds the "Haswell"
implementation. If the build machine supports AVX-512, it builds the
"SkylakeX" implementation.
