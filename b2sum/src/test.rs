use std::env::consts::EXE_EXTENSION;
use std::io::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::{Once, ONCE_INIT};
use tempfile::{tempfile, NamedTempFile};

pub fn b2sum_exe() -> PathBuf {
    // `cargo test` doesn't automatically run `cargo build`, so we do that ourselves.
    static CARGO_BUILD_ONCE: Once = ONCE_INIT;
    CARGO_BUILD_ONCE.call_once(|| {
        cmd!("cargo", "build", "--quiet")
            .run()
            .expect("build failed");
    });

    Path::new("target")
        .join("debug")
        .join("b2sum")
        .with_extension(EXE_EXTENSION)
}

#[test]
fn test_stdin() {
    let output = cmd!(b2sum_exe(), "-l128")
        .input("abcdef")
        .read()
        .expect("b2sum failed");
    assert_eq!("2465e7ee63a17b4b307c7792c432aef6  -", output);
}

#[test]
fn test_stdin_mmap() {
    let mut file = tempfile().unwrap();
    file.write_all("abcdef".as_bytes()).unwrap();
    file.flush().unwrap();
    // Without --mmap this test would fail, because we haven't done a seek(0).
    let output = cmd!(b2sum_exe(), "-l128", "--mmap")
        .stdin_handle(file)
        .read()
        .expect("b2sum failed");
    assert_eq!("2465e7ee63a17b4b307c7792c432aef6  -", output);
}

#[test]
fn test_input_file() {
    let mut file = NamedTempFile::new().unwrap();
    file.write_all("abcdef".as_bytes()).unwrap();
    file.flush().unwrap();
    let output = cmd!(b2sum_exe(), "-l128", file.path())
        .read()
        .expect("b2sum failed");
    let expected_output = format!(
        "2465e7ee63a17b4b307c7792c432aef6  {}",
        file.path().to_string_lossy()
    );
    assert_eq!(expected_output, output);
}

#[test]
fn test_input_file_mmap() {
    let mut file = NamedTempFile::new().unwrap();
    file.write_all("abcdef".as_bytes()).unwrap();
    file.flush().unwrap();
    let output = cmd!(b2sum_exe(), "-l128", "--mmap", file.path())
        .read()
        .expect("b2sum failed");
    let expected_output = format!(
        "2465e7ee63a17b4b307c7792c432aef6  {}",
        file.path().to_string_lossy()
    );
    assert_eq!(expected_output, output);
}
