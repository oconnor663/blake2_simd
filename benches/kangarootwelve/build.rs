fn main() {
    std::process::Command::new("make")
        .arg("Haswell/libk12.a")
        .current_dir("K12")
        .status()
        .expect("make error");
    println!("cargo:rustc-link-search=./K12/bin/Haswell");
    println!("cargo:rustc-link-lib=static=k12");
}
