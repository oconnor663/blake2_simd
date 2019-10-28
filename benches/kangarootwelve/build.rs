use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir: PathBuf = env::var("CARGO_MANIFEST_DIR").unwrap().into();
    let k12_dir = manifest_dir.join("K12");
    let build_dir = k12_dir.join("bin/Haswell");
    let build_status = Command::new("make")
        .arg("Haswell/libk12.a")
        .current_dir(&k12_dir)
        .status()
        .unwrap();
    assert!(build_status.success());
    println!("cargo:rustc-link-search={}", build_dir.to_str().unwrap());
    println!("cargo:rustc-link-lib=static=k12");
}
