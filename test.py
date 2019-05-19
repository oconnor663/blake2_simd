#! /usr/bin/env python3

import os
import pathlib
import subprocess

HERE = pathlib.Path(__file__).parent

TARGETS = [
    HERE,
    HERE / "blake2s",
    HERE / "b2sum",
]


def do(*args, cwd):
    print("=== TEST COMMAND ===")
    print("cd", cwd, "&&", " ".join(args))
    print()
    subprocess.run(args, check=True, cwd=cwd)


def main():
    os.environ["CARGO_TARGET_DIR"] = str((HERE / "target").absolute())

    for target in TARGETS:
        do("cargo", "test", "--all-features", cwd=target)
        do("cargo", "test", "--no-default-features", cwd=target)


if __name__ == "__main__":
    main()
