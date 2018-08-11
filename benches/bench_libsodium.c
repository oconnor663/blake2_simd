#include <sodium.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#define BUFSIZE 65536
#define HASHSIZE 32

#define INPUT_LEN 1000000000
#define RUNS 10

struct timespec now() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t;
}

int microsecond_diff(struct timespec tstart, struct timespec tend) {
  int start_micros = tstart.tv_sec * 1000000 + tstart.tv_nsec / 1000;
  int end_micros = tend.tv_sec * 1000000 + tend.tv_nsec / 1000;
  return end_micros - start_micros;
}

void print_micros(int micros, char *name) {
  printf("%d.%06ds %s\n", micros / 1000000, micros % 1000000, name);
}

// Loop over the input and hash it RUNS number of times.
void run_bench(unsigned char *input) {
  struct timespec tstart;
  unsigned char hash[HASHSIZE];

  int fastest_micros = INT_MAX;
  int total_micros = 0;
  for (int i = 0; i < RUNS; i++) {
    tstart = now();
    crypto_generichash(hash, HASHSIZE, input, INPUT_LEN, NULL, 0);
    int micros = microsecond_diff(tstart, now());
    if (i == 0) {
      // Ignore the first run. It pays costs like zeroing memory pages.
      print_micros(micros, "hash (ignored)");
    } else {
      print_micros(micros, "hash");
      total_micros += micros;
      if (micros < fastest_micros) {
        fastest_micros = micros;
      }
    }
  }
  printf("-----\n");
  print_micros(total_micros / (RUNS - 1), "average");
  print_micros(fastest_micros, "fastest");
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
