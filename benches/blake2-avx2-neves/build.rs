fn main() {
    cc::Build::new()
        .file("./blake2-avx2/blake2b.c")
        .file("./blake2-avx2/blake2bp.c")
        .file("./blake2-avx2/blake2sp.c")
        // Enable AVX2 for GCC and Clang.
        .flag_if_supported("-mavx2")
        // Enable AVX2 for MSVC
        .flag_if_supported("/arch:AVX2")
        // The implementation includes two different input loading strategies.
        // Defining this variable enables the alternative, but in my testing
        // it's slower than the default.
        // .define("PERMUTE_WITH_GATHER", "1")
        .compile("blake2-avx2");
}
