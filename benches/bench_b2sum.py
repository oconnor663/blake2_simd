#! /usr/bin/env python3

from pathlib import Path
from subprocess import run, DEVNULL
import os
import sys
import time

runs = 10

b2sum_root = Path(__file__).parent.parent / "b2sum"
b2sum_path = str(b2sum_root / "target/release/b2sum")

targets = [
    ["md5sum"],
    ["sha1sum"],
    ["sha512sum"],
    ["/usr/bin/b2sum"],
    [b2sum_path],
    [b2sum_path, "--mmap"],
]


def print_diff(diff, bits, message):
    gb_per_sec = bits / 1_000_000_000.0 / diff
    print("{:.6f} ({:.6f} GB/s)  {}".format(diff, gb_per_sec, message))


def main():
    if len(sys.argv) < 2:
        print("need an input filename")
        sys.exit(1)
    infile_path = sys.argv[1]
    infile_bits = os.stat(infile_path).st_size
    build_command = ["cargo", "+nightly", "build", "--release"]
    print(" ".join(build_command))
    run(build_command, cwd=b2sum_root)

    averages = {}
    bests = {}
    for target in targets:
        target_name = " ".join(target)
        total = 0
        best = float('inf')
        for i in range(runs):
            infile = open(infile_path)
            start = time.perf_counter()
            run(target, stdin=infile, stdout=DEVNULL)
            diff = time.perf_counter() - start
            if i == 0:
                print_diff(diff, infile_bits, target_name + " (ignored)")
            else:
                print_diff(diff, infile_bits, target_name)
                total += diff
                if diff < best:
                    best = diff
        averages[target_name] = total / (runs-1)
        bests[target_name] = best

    print("--- average ---")
    average_list = list(averages.items())
    average_list.sort(key=lambda pair: pair[1])
    for target_name, average in average_list:
        print_diff(average, infile_bits, target_name)

    print("--- best ---")
    best_list = list(bests.items())
    best_list.sort(key=lambda pair: pair[1])
    for target_name, best in best_list:
        print_diff(best, infile_bits, target_name)


if __name__ == "__main__":
    main()
