#! /usr/bin/env python3

from pathlib import Path
from subprocess import run, DEVNULL
import os
import sys
import statistics
import time

NUM_RUNS = 10

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


def print_diff(diff, gigs, message, stdev=None, stdev_rate=None):
    gb_per_sec = gigs / diff
    diff_stdev = ""
    gbps_stdev = ""
    if stdev is not None:
        diff_stdev = " ± {:.6f}".format(stdev)
    if stdev_rate is not None:
        gbps_stdev = " ± {:.6f}".format(stdev_rate)
    print("{:.6f}{} ({:.6f}{} GB/s)  {}".format(diff, diff_stdev, gb_per_sec,
                                                gbps_stdev, message))


def main():
    if len(sys.argv) < 2:
        print("need an input filename")
        sys.exit(1)
    infile_path = sys.argv[1]
    infile_gigs = os.stat(infile_path).st_size / 1_000_000_000
    build_command = ["cargo", "+nightly", "build", "--release"]
    print(" ".join(build_command))
    run(build_command, cwd=b2sum_root)

    averages = {}
    bests = {}
    stdevs = {}
    stdevs_rate = {}
    for target in targets:
        target_name = " ".join(target)
        runs = []
        rates = []
        best = float('inf')
        for i in range(NUM_RUNS):
            infile = open(infile_path)
            start = time.perf_counter()
            run(target, stdin=infile, stdout=DEVNULL)
            diff = time.perf_counter() - start
            if i == 0:
                print_diff(diff, infile_gigs, target_name + " (ignored)")
            else:
                print_diff(diff, infile_gigs, target_name)
                runs.append(diff)
                rates.append(infile_gigs / diff)
        averages[target_name] = sum(runs) / len(runs)
        bests[target_name] = min(runs)
        stdevs[target_name] = statistics.stdev(runs)
        stdevs_rate[target_name] = statistics.stdev(rates)

    print("--- best ---")
    best_list = list(bests.items())
    best_list.sort(key=lambda pair: pair[1])
    for target_name, best in best_list:
        print_diff(best, infile_gigs, target_name)

    print("--- average ---")
    average_list = list(averages.items())
    average_list.sort(key=lambda pair: pair[1])
    for target_name, average in average_list:
        print_diff(
            average,
            infile_gigs,
            target_name,
            stdev=stdevs[target_name],
            stdev_rate=stdevs_rate[target_name])


if __name__ == "__main__":
    main()
