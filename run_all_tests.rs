use std::env;
use std::path::{Path, PathBuf};
use std::process::{self, Command};

fn here() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn run_test_command(project_dir: impl AsRef<Path>, flags: &[&str]) {
    let mut relative_dir: &Path = project_dir
        .as_ref()
        .strip_prefix(here())
        .unwrap_or(project_dir.as_ref());
    if relative_dir.components().count() == 0 {
        relative_dir = Path::new(".");
    }
    println!(
        "cd {} && cargo test {}",
        relative_dir.to_string_lossy(),
        flags.join(" ")
    );
    println!();

    let status = Command::new(env!("CARGO"))
        .arg("test")
        .args(flags)
        .current_dir(project_dir)
        .status()
        .expect("spawn failed");

    if !status.success() {
        process::exit(1);
    }
}

fn main() {
    // Set CARGO_TARGET_DIR for all the test runs (unless the caller already set
    // it), so that they can share build artifacts.
    let target_dir = env::var_os("CARGO_TARGET_DIR")
        .map(Into::<PathBuf>::into)
        .unwrap_or(here().join("target"));
    env::set_var("CARGO_TARGET_DIR", &target_dir);

    // Test all the sub-projects under both std and no_std.
    for &project in &["blake2b", "blake2s", ".", "b2sum"] {
        for &no_std in &[false, true] {
            println!("=== TEST COMMAND ===");
            let mut flags = Vec::new();
            if no_std {
                flags.push("--no-default-features");
            }
            let project_dir = Path::new(here()).join(project);
            run_test_command(&project_dir, &flags);
        }
    }

    // In addition, run the root project under release mode. This lets the fuzz
    // tests use a much larger iteration count.
    run_test_command(".", &["--release"]);
}
