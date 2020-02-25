This is Rust FFI wrapper around the
[XKCP/K12](https://github.com/XKCP/K12) C implementation, which is
vendored here (version ad51d21, 2020-02-16) and statically linked. It's
intended for testing and benchmarking only.

The build is hardcoded to use the `generic64` target, which includes
runtime feature detection for AVX2 and AVX-512. If you're on a 32-bit
machine or cross-compiling, you'll need to manually edit `build.rs` to
build the `generic32` target.
