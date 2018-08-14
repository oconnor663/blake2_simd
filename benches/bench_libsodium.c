#include <sodium.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#define BUFSIZE 65536
#define HASHSIZE 32
#define NS_PER_SEC 1000000000
#define INPUT_LEN 1000000000
#define RUNS 10

struct timespec now() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t;
}

uint64_t nanosecond_diff(struct timespec tstart, struct timespec tend) {
  uint64_t start_nanos = tstart.tv_sec * NS_PER_SEC + tstart.tv_nsec;
  uint64_t end_nanos = tend.tv_sec * NS_PER_SEC + tend.tv_nsec;
  return end_nanos - start_nanos;
}

void print_nanos(uint64_t nanos, char *message) {
  double secs = (double)nanos / (double)NS_PER_SEC;
  // bits/ns = GB/sec
  double rate = INPUT_LEN / (double)nanos;
  printf("%.6fs (%.6f GB/s) %s\n", secs, rate, message);
}

// Loop over the input and hash it RUNS number of times.
void run_bench(unsigned char *input) {
  struct timespec tstart;
  unsigned char hash[HASHSIZE];

  uint64_t fastest_nanos = INT_MAX;
  uint64_t total_nanos = 0;
  for (int i = 0; i < RUNS; i++) {
    tstart = now();
    crypto_generichash(hash, HASHSIZE, input, INPUT_LEN, NULL, 0);
    uint64_t nanos = nanosecond_diff(tstart, now());
    if (i == 0) {
      // Ignore the first run. It pays costs like zeroing memory pages.
      print_nanos(nanos, "(ignored)");
    } else {
      print_nanos(nanos, "");
      total_nanos += nanos;
      if (nanos < fastest_nanos) {
        fastest_nanos = nanos;
      }
    }
  }
  printf("-----\n");
  print_nanos(total_nanos / (RUNS - 1), "average");
  print_nanos(fastest_nanos, "fastest");
  printf("-----\n");
}

int main() {
  // Allocate a gig of memory.
  unsigned char *input = calloc(INPUT_LEN, 1);

  // Now's we're going to run the benchmarks twice. In between, we'll call
  // sodium_init(), and libsodium will do runtime CPU feature detection to
  // switch to a faster BLAKE2b. That should mean that our first benchmark run
  // is over the portable "ref" implementation, and our second run is over the
  // AVX2 implementation (assuming this is running on a machine that supports
  // AVX2 to begin with). Note that all this was written in August 2018, so if
  // you're running this years later, it's possible libsodium will add more
  // implementations.

  printf("run #1, the ref implementation\n");
  run_bench(input);
  printf("calling sodium_init()\n");
  if (sodium_init() == -1) {
    return 1;
  }
  printf("run #2, the AVX2 implementation (presumably)\n");
  run_bench(input);
}
