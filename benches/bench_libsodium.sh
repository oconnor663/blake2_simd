#! /usr/bin/bash

set -e -u -o pipefail

here="$(dirname "$BASH_SOURCE")"
target="$(mktemp --tmpdir="" bench_libsodium.XXXXX)"

echo gcc -Wall --pedantic -O3 -lsodium -o "$target" "$here/bench_libsodium.c"
gcc -Wall --pedantic -O3 -lsodium -o "$target" "$here/bench_libsodium.c"

echo "$target"
"$target"

rm "$target"
