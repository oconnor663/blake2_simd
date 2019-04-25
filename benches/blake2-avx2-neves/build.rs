fn main() {
    cc::Build::new()
        .file("./blake2-avx2/blake2b.c")
        .file("./blake2-avx2/blake2bp.c")
        .file("./blake2-avx2/blake2sp.c")
        // GCC and Clang
        .flag_if_supported("-mavx2")
        // MSVC
        .flag_if_supported("/arch:AVX2")
        .compile("blake2-avx2");
}
