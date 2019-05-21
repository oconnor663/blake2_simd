use std::env;
use std::path::{Path, PathBuf};
use std::process::{self, Command};

fn main() {
    // Set CARGO_TARGET_DIR for all the test runs (unless the caller already set
    // it), so that they can share build artifacts.
    let here = Path::new(env!("CARGO_MANIFEST_DIR"));
    let target_dir = env::var_os("CARGO_TARGET_DIR")
        .map(Into::<PathBuf>::into)
        .unwrap_or(here.join("target"));
    env::set_var("CARGO_TARGET_DIR", &target_dir);

    for &project in &["blake2b", "blake2s", ".", "b2sum"] {
        for &no_std in &[false, true] {
            for &release_mode in &[false, true] {
                println!("=== TEST COMMAND ===");
                let mut args = vec!["test"];
                if no_std {
                    args.push("--no-default-features");
                }
                if release_mode {
                    args.push("--release");
                }
                let project_dir = Path::new(here).join(project);
                let mut relative_dir: &Path =
                    project_dir.strip_prefix(here).unwrap_or(&project_dir);
                if relative_dir.components().count() == 0 {
                    relative_dir = Path::new(".");
                }
                println!(
                    "cd {} && cargo {}",
                    relative_dir.to_string_lossy(),
                    args.join(" ")
                );
                println!();

                let status = Command::new(env!("CARGO"))
                    .args(&args)
                    .current_dir(&project_dir)
                    .status()
                    .expect("spawn failed");

                if !status.success() {
                    process::exit(1);
                }
            }
        }
    }
}
