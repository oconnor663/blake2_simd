#! /usr/bin/bash

set -e -u -o pipefail

# To squeeze the absolute most out of this benchmark:
# - Download a source tarball for libsodium. Or if you're using the GitHub
#   repo, make sure to check out the latest release tag. (origin/master is slower
#   for some reason.)
# - Build libsodium with `./configure --enable-opt && make` to enable
#   optimizations specific to the current machine.
# - Link against what you just built by removing `-lsodium` below and adding
#   {libsodium}/src/libsodium/.libs/libsodium.a as an additional source file.

here="$(dirname "$BASH_SOURCE")"
target="$(mktemp --tmpdir="" bench_libsodium.XXXXX)"

set -v
gcc -Wall --pedantic -O3 -lsodium -o "$target" "$here/bench_libsodium.c"
set +v

echo "$target"
"$target"

rm "$target"
