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


def main():
    os.environ["CARGO_TARGET_DIR"] = str((HERE / "target").absolute())

    for target in TARGETS:
        for features in ["--all-features", "--no-default-features"]:
            for release in [False, True]:
                command = ["cargo", "test"]
                command.append(features)
                if release:
                    command.append("--release")
                print("=== TEST COMMAND ===")
                print("cd", target, "&&", " ".join(command))
                print()
                subprocess.run(command, check=True, cwd=target)


if __name__ == "__main__":
    main()
