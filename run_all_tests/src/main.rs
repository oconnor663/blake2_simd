use std::env;
use std::path::{Path, PathBuf};
use std::process::{self, Command};

fn project_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("parent failed")
}

fn run_cmd(project: &str, cmd: &[&str]) {
    let project_dir = Path::new(project_root()).join(project);
    println!("=== COMMAND ===");
    println!("cd {} && {}", project_dir.to_string_lossy(), cmd.join(" "));
    println!();
    let status = Command::new(cmd[0])
        .args(&cmd[1..])
        .current_dir(project_dir)
        .status()
        .expect("spawn failed");

    if !status.success() {
        println!("Command failed :(");
        process::exit(1);
    }
}

fn run_cargo_cmd(project: &str, flags: &[&str]) {
    let mut cmd = vec![env!("CARGO")];
    cmd.extend_from_slice(flags);
    run_cmd(project, &cmd);
}

fn main() {
    // Set CARGO_TARGET_DIR for all the test runs (unless the caller already set
    // it), so that they can share build artifacts.
    let target_dir = env::var_os("CARGO_TARGET_DIR")
        .map(Into::<PathBuf>::into)
        .unwrap_or(project_root().join("target"));
    env::set_var("CARGO_TARGET_DIR", &target_dir);

    // Test all the sub-projects under both std and no_std.
    for &project in &["blake2b", "blake2s", ".", "blake2_bin"] {
        for &no_std in &[false, true] {
            let mut flags = vec!["test"];
            if no_std {
                flags.push("--no-default-features");
            }
            run_cargo_cmd(project, &flags);
        }
    }

    // Run the root project under release mode. This lets the "fuzz" unit tests
    // (not to be confused with the actual "cargo fuzz" tests) use a much
    // larger iteration count.
    run_cargo_cmd(".", &["test", "--release"]);

    // Test the uninline_portable feature of blake2b_simd.
    run_cargo_cmd(
        "blake2b",
        &["test", "--release", "--features=uninline_portable"],
    );

    // Make sure the "cargo fuzz" tests can at least build.
    run_cargo_cmd("blake2b/fuzz", &["check"]);
    run_cargo_cmd("blake2s/fuzz", &["check"]);
}
