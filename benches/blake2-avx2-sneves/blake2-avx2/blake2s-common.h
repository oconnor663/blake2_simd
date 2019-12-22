#ifndef BLAKE2_AVX2_BLAKE2S_COMMON_H
#define BLAKE2_AVX2_BLAKE2S_COMMON_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <immintrin.h>

#include "blake2.h"

#define LOAD128(p)  _mm_load_si128( (__m128i *)(p) )
#define STORE128(p,r) _mm_store_si128((__m128i *)(p), r)

#define LOADU128(p)  _mm_loadu_si128( (__m128i *)(p) )
#define STOREU128(p,r) _mm_storeu_si128((__m128i *)(p), r)

static INLINE uint32_t LOADU32(void const * p) {
  uint32_t v;
  memcpy(&v, p, sizeof v);
  return v;
}

#define TOF(reg) _mm_castsi128_ps((reg))
#define TOI(reg) _mm_castps_si128((reg))

#define ADD128(a, b) _mm_add_epi32(a, b)
#define SUB128(a, b) _mm_sub_epi32(a, b)

#define XOR128(a, b) _mm_xor_si128(a, b)
#define AND128(a, b) _mm_and_si128(a, b)
#define  OR128(a, b) _mm_or_si128(a, b)

#define ROT16128(x)  OR128(_mm_srli_epi32((x),  16), _mm_slli_epi32((x), 32 - 16))
#define ROT12128(x)  OR128(_mm_srli_epi32((x),  12), _mm_slli_epi32((x), 32 - 12))
#define ROT8128(x)  OR128(_mm_srli_epi32((x),  8), _mm_slli_epi32((x), 32 - 8))
#define ROT7128(x)  OR128(_mm_srli_epi32((x),  7), _mm_slli_epi32((x), 32 - 7))

#define LOAD(p)  _mm256_load_si256( (__m256i *)(p) )
#define STORE(p,r) _mm256_store_si256((__m256i *)(p), r)

#define LOADU(p)  _mm256_loadu_si256( (__m256i *)(p) )
#define STOREU(p,r) _mm256_storeu_si256((__m256i *)(p), r)

#define ADD(a, b) _mm256_add_epi32(a, b)
#define SUB(a, b) _mm256_sub_epi32(a, b)

#define XOR(a, b) _mm256_xor_si256(a, b)
#define AND(a, b) _mm256_and_si256(a, b)
#define  OR(a, b) _mm256_or_si256(a, b)

#define ROT16(x)  _mm256_or_si256(_mm256_srli_epi32((x),  16), _mm256_slli_epi32((x), 32 - 16))
#define ROT12(x)  _mm256_or_si256(_mm256_srli_epi32((x),  12), _mm256_slli_epi32((x), 32 - 12))
#define ROT8(x)  _mm256_or_si256(_mm256_srli_epi32((x),  8), _mm256_slli_epi32((x), 32 - 8))
#define ROT7(x)  _mm256_or_si256(_mm256_srli_epi32((x),  7), _mm256_slli_epi32((x), 32 - 7))

#endif
