/*
Implementation by Gilles Van Assche and Ronny Van Keer, hereby denoted as "the implementer".

For more information, feedback or questions, please refer to our website:
https://keccak.team/

To the extent possible under law, the implementer has waived all copyright
and related or neighboring rights to the source code in this file.
http://creativecommons.org/publicdomain/zero/1.0/

---

Please refer to the XKCP for more details.
*/

#include <stdint.h>
#include <immintrin.h>
#include "KeccakP-1600-SnP.h"

#ifdef ALIGN
#undef ALIGN
#endif

#if defined(__GNUC__)
#define ALIGN(x) __attribute__ ((aligned(x)))
#elif defined(_MSC_VER)
#define ALIGN(x) __declspec(align(x))
#elif defined(__ARMCC_VERSION)
#define ALIGN(x) __align(x)
#else
#define ALIGN(x)
#endif

#define AVX512alignment 64

#define LOAD4_32(a,b,c,d)           _mm_set_epi32((uint64_t)(a), (uint32_t)(b), (uint32_t)(c), (uint32_t)(d))
#define LOAD8_32(a,b,c,d,e,f,g,h)   _mm256_set_epi32((uint64_t)(a), (uint32_t)(b), (uint32_t)(c), (uint32_t)(d), (uint32_t)(e), (uint32_t)(f), (uint32_t)(g), (uint32_t)(h))
#define LOAD_GATHER2_64(idx,p)      _mm_i32gather_epi64( (const void*)(p), idx, 8)
#define LOAD_GATHER4_64(idx,p)      _mm256_i32gather_epi64( (const void*)(p), idx, 8)
#define LOAD_GATHER8_64(idx,p)      _mm512_i32gather_epi64( idx, (const void*)(p), 8)
#define STORE_SCATTER8_64(p,idx, v) _mm512_i32scatter_epi64( (void*)(p), idx, v, 8)


/* Keccak-p[1600]×2 */

#define XOR(a,b)                    _mm_xor_si128(a,b)
#define XOReq(a, b)                 a = _mm_xor_si128(a, b)
#define XOR3(a,b,c)                 _mm_ternarylogic_epi64(a,b,c,0x96)
#define XOR5(a,b,c,d,e)             XOR3(XOR3(a,b,c),d,e)
#define ROL(a,offset)               _mm_rol_epi64(a,offset)
#define Chi(a,b,c)                  _mm_ternarylogic_epi64(a,b,c,0xD2)
#define CONST_64(a)                 _mm_set1_epi64((__m64)(a))
#define LOAD6464(a, b)              _mm_set_epi64((__m64)(a), (__m64)(b))
#define STORE128u(a, b)             _mm_storeu_si128((__m128i *)&(a), b)
#define UNPACKL( a, b )             _mm_unpacklo_epi64((a), (b))
#define UNPACKH( a, b )             _mm_unpackhi_epi64((a), (b))
#define ZERO()              _mm_setzero_si128()

static ALIGN(AVX512alignment) const uint64_t KeccakP1600RoundConstants[24] = {
    0x0000000000000001ULL,
    0x0000000000008082ULL,
    0x800000000000808aULL,
    0x8000000080008000ULL,
    0x000000000000808bULL,
    0x0000000080000001ULL,
    0x8000000080008081ULL,
    0x8000000000008009ULL,
    0x000000000000008aULL,
    0x0000000000000088ULL,
    0x0000000080008009ULL,
    0x000000008000000aULL,
    0x000000008000808bULL,
    0x800000000000008bULL,
    0x8000000000008089ULL,
    0x8000000000008003ULL,
    0x8000000000008002ULL,
    0x8000000000000080ULL,
    0x000000000000800aULL,
    0x800000008000000aULL,
    0x8000000080008081ULL,
    0x8000000000008080ULL,
    0x0000000080000001ULL,
    0x8000000080008008ULL};

#define KeccakP_DeclareVars(type) \
    type    _Ba, _Be, _Bi, _Bo, _Bu; \
    type    _Da, _De, _Di, _Do, _Du; \
    type    _ba, _be, _bi, _bo, _bu; \
    type    _ga, _ge, _gi, _go, _gu; \
    type    _ka, _ke, _ki, _ko, _ku; \
    type    _ma, _me, _mi, _mo, _mu; \
    type    _sa, _se, _si, _so, _su

#define KeccakP_ThetaRhoPiChi( _L1, _L2, _L3, _L4, _L5, _Bb1, _Bb2, _Bb3, _Bb4, _Bb5, _Rr1, _Rr2, _Rr3, _Rr4, _Rr5 ) \
    _Bb1 = XOR(_L1, _Da); \
    _Bb2 = XOR(_L2, _De); \
    _Bb3 = XOR(_L3, _Di); \
    _Bb4 = XOR(_L4, _Do); \
    _Bb5 = XOR(_L5, _Du); \
    if (_Rr1 != 0) _Bb1 = ROL(_Bb1, _Rr1); \
    _Bb2 = ROL(_Bb2, _Rr2); \
    _Bb3 = ROL(_Bb3, _Rr3); \
    _Bb4 = ROL(_Bb4, _Rr4); \
    _Bb5 = ROL(_Bb5, _Rr5); \
    _L1 = Chi( _Ba, _Be, _Bi); \
    _L2 = Chi( _Be, _Bi, _Bo); \
    _L3 = Chi( _Bi, _Bo, _Bu); \
    _L4 = Chi( _Bo, _Bu, _Ba); \
    _L5 = Chi( _Bu, _Ba, _Be);

#define KeccakP_ThetaRhoPiChiIota0( _L1, _L2, _L3, _L4, _L5, _rc ) \
    _Ba = XOR5( _ba, _ga, _ka, _ma, _sa ); /* Theta effect */ \
    _Be = XOR5( _be, _ge, _ke, _me, _se ); \
    _Bi = XOR5( _bi, _gi, _ki, _mi, _si ); \
    _Bo = XOR5( _bo, _go, _ko, _mo, _so ); \
    _Bu = XOR5( _bu, _gu, _ku, _mu, _su ); \
    _Da = ROL( _Be, 1 ); \
    _De = ROL( _Bi, 1 ); \
    _Di = ROL( _Bo, 1 ); \
    _Do = ROL( _Bu, 1 ); \
    _Du = ROL( _Ba, 1 ); \
    _Da = XOR( _Da, _Bu ); \
    _De = XOR( _De, _Ba ); \
    _Di = XOR( _Di, _Be ); \
    _Do = XOR( _Do, _Bi ); \
    _Du = XOR( _Du, _Bo ); \
    KeccakP_ThetaRhoPiChi( _L1, _L2, _L3, _L4, _L5, _Ba, _Be, _Bi, _Bo, _Bu,  0, 44, 43, 21, 14 ); \
    _L1 = XOR(_L1, _rc) /* Iota */

#define KeccakP_ThetaRhoPiChi1( _L1, _L2, _L3, _L4, _L5 ) \
    KeccakP_ThetaRhoPiChi( _L1, _L2, _L3, _L4, _L5, _Bi, _Bo, _Bu, _Ba, _Be,  3, 45, 61, 28, 20 )

#define KeccakP_ThetaRhoPiChi2( _L1, _L2, _L3, _L4, _L5 ) \
    KeccakP_ThetaRhoPiChi( _L1, _L2, _L3, _L4, _L5, _Bu, _Ba, _Be, _Bi, _Bo, 18,  1,  6, 25,  8 )

#define KeccakP_ThetaRhoPiChi3( _L1, _L2, _L3, _L4, _L5 ) \
    KeccakP_ThetaRhoPiChi( _L1, _L2, _L3, _L4, _L5, _Be, _Bi, _Bo, _Bu, _Ba, 36, 10, 15, 56, 27 )

#define KeccakP_ThetaRhoPiChi4( _L1, _L2, _L3, _L4, _L5 ) \
    KeccakP_ThetaRhoPiChi( _L1, _L2, _L3, _L4, _L5, _Bo, _Bu, _Ba, _Be, _Bi, 41,  2, 62, 55, 39 )

#define KeccakP_4rounds( i ) \
    KeccakP_ThetaRhoPiChiIota0(_ba, _ge, _ki, _mo, _su, CONST_64(KeccakP1600RoundConstants[i]) ); \
    KeccakP_ThetaRhoPiChi1(    _ka, _me, _si, _bo, _gu ); \
    KeccakP_ThetaRhoPiChi2(    _sa, _be, _gi, _ko, _mu ); \
    KeccakP_ThetaRhoPiChi3(    _ga, _ke, _mi, _so, _bu ); \
    KeccakP_ThetaRhoPiChi4(    _ma, _se, _bi, _go, _ku ); \
\
    KeccakP_ThetaRhoPiChiIota0(_ba, _me, _gi, _so, _ku, CONST_64(KeccakP1600RoundConstants[i+1]) ); \
    KeccakP_ThetaRhoPiChi1(    _sa, _ke, _bi, _mo, _gu ); \
    KeccakP_ThetaRhoPiChi2(    _ma, _ge, _si, _ko, _bu ); \
    KeccakP_ThetaRhoPiChi3(    _ka, _be, _mi, _go, _su ); \
    KeccakP_ThetaRhoPiChi4(    _ga, _se, _ki, _bo, _mu ); \
\
    KeccakP_ThetaRhoPiChiIota0(_ba, _ke, _si, _go, _mu, CONST_64(KeccakP1600RoundConstants[i+2]) ); \
    KeccakP_ThetaRhoPiChi1(    _ma, _be, _ki, _so, _gu ); \
    KeccakP_ThetaRhoPiChi2(    _ga, _me, _bi, _ko, _su ); \
    KeccakP_ThetaRhoPiChi3(    _sa, _ge, _mi, _bo, _ku ); \
    KeccakP_ThetaRhoPiChi4(    _ka, _se, _gi, _mo, _bu ); \
\
    KeccakP_ThetaRhoPiChiIota0(_ba, _be, _bi, _bo, _bu, CONST_64(KeccakP1600RoundConstants[i+3]) ); \
    KeccakP_ThetaRhoPiChi1(    _ga, _ge, _gi, _go, _gu ); \
    KeccakP_ThetaRhoPiChi2(    _ka, _ke, _ki, _ko, _ku ); \
    KeccakP_ThetaRhoPiChi3(    _ma, _me, _mi, _mo, _mu ); \
    KeccakP_ThetaRhoPiChi4(    _sa, _se, _si, _so, _su )

#define rounds12 \
    KeccakP_4rounds( 12 ); \
    KeccakP_4rounds( 16 ); \
    KeccakP_4rounds( 20 )

#define initializeState(X) \
    X##ba = ZERO(); \
    X##be = ZERO(); \
    X##bi = ZERO(); \
    X##bo = ZERO(); \
    X##bu = ZERO(); \
    X##ga = ZERO(); \
    X##ge = ZERO(); \
    X##gi = ZERO(); \
    X##go = ZERO(); \
    X##gu = ZERO(); \
    X##ka = ZERO(); \
    X##ke = ZERO(); \
    X##ki = ZERO(); \
    X##ko = ZERO(); \
    X##ku = ZERO(); \
    X##ma = ZERO(); \
    X##me = ZERO(); \
    X##mi = ZERO(); \
    X##mo = ZERO(); \
    X##mu = ZERO(); \
    X##sa = ZERO(); \
    X##se = ZERO(); \
    X##si = ZERO(); \
    X##so = ZERO(); \
    X##su = ZERO(); \

#define XORdata16(X, data0, data1) \
    XOReq(X##ba, LOAD6464((data1)[ 0], (data0)[ 0])); \
    XOReq(X##be, LOAD6464((data1)[ 1], (data0)[ 1])); \
    XOReq(X##bi, LOAD6464((data1)[ 2], (data0)[ 2])); \
    XOReq(X##bo, LOAD6464((data1)[ 3], (data0)[ 3])); \
    XOReq(X##bu, LOAD6464((data1)[ 4], (data0)[ 4])); \
    XOReq(X##ga, LOAD6464((data1)[ 5], (data0)[ 5])); \
    XOReq(X##ge, LOAD6464((data1)[ 6], (data0)[ 6])); \
    XOReq(X##gi, LOAD6464((data1)[ 7], (data0)[ 7])); \
    XOReq(X##go, LOAD6464((data1)[ 8], (data0)[ 8])); \
    XOReq(X##gu, LOAD6464((data1)[ 9], (data0)[ 9])); \
    XOReq(X##ka, LOAD6464((data1)[10], (data0)[10])); \
    XOReq(X##ke, LOAD6464((data1)[11], (data0)[11])); \
    XOReq(X##ki, LOAD6464((data1)[12], (data0)[12])); \
    XOReq(X##ko, LOAD6464((data1)[13], (data0)[13])); \
    XOReq(X##ku, LOAD6464((data1)[14], (data0)[14])); \
    XOReq(X##ma, LOAD6464((data1)[15], (data0)[15])); \

#define XORdata21(X, data0, data1) \
    XORdata16(X, data0, data1) \
    XOReq(X##me, LOAD6464((data1)[16], (data0)[16])); \
    XOReq(X##mi, LOAD6464((data1)[17], (data0)[17])); \
    XOReq(X##mo, LOAD6464((data1)[18], (data0)[18])); \
    XOReq(X##mu, LOAD6464((data1)[19], (data0)[19])); \
    XOReq(X##sa, LOAD6464((data1)[20], (data0)[20])); \

#define chunkSize 8192
#define rateInBytes (21*8)

void KangarooTwelve_AVX512_Process2Leaves(const unsigned char *input, unsigned char *output)
{
    KeccakP_DeclareVars(__m128i);
    unsigned int j;

    initializeState(_);

    for(j = 0; j < (chunkSize - rateInBytes); j += rateInBytes) {
        XORdata21(_, (const uint64_t *)input, (const uint64_t *)(input+chunkSize));
        rounds12
        input += rateInBytes;
    }

    XORdata16(_, (const uint64_t *)input, (const uint64_t *)(input+chunkSize));
    XOReq(_me, CONST_64(0x0BULL));
    XOReq(_sa, CONST_64(0x8000000000000000ULL));
    rounds12

    STORE128u( *(__m128i*)&(output[ 0]), UNPACKL( _ba, _be ) );
    STORE128u( *(__m128i*)&(output[16]), UNPACKL( _bi, _bo ) );
    STORE128u( *(__m128i*)&(output[32]), UNPACKH( _ba, _be ) );
    STORE128u( *(__m128i*)&(output[48]), UNPACKH( _bi, _bo ) );
}

#undef XOR
#undef XOReq
#undef XOR3
#undef XOR5
#undef ROL
#undef Chi
#undef CONST_64
#undef LOAD6464
#undef STORE128u
#undef UNPACKL
#undef UNPACKH
#undef ZERO
#undef XORdata16
#undef XORdata21


/* Keccak-p[1600]×4 */

#define XOR(a,b)                    _mm256_xor_si256(a,b)
#define XOReq(a,b)                  a = _mm256_xor_si256(a,b)
#define XOR3(a,b,c)                 _mm256_ternarylogic_epi64(a,b,c,0x96)
#define XOR5(a,b,c,d,e)             XOR3(XOR3(a,b,c),d,e)
#define XOR512(a,b)                 _mm512_xor_si512(a,b)
#define ROL(a,offset)               _mm256_rol_epi64(a,offset)
#define Chi(a,b,c)                  _mm256_ternarylogic_epi64(a,b,c,0xD2)
#define CONST_64(a)                 _mm256_set1_epi64x(a)
#define ZERO()                      _mm256_setzero_si256()
#define LOAD4_64(a, b, c, d)    _mm256_set_epi64x((uint64_t)(a), (uint64_t)(b), (uint64_t)(c), (uint64_t)(d))

#define XORdata16(X, data0, data1, data2, data3) \
    XOReq(X##ba, LOAD4_64((data3)[ 0], (data2)[ 0], (data1)[ 0], (data0)[ 0])); \
    XOReq(X##be, LOAD4_64((data3)[ 1], (data2)[ 1], (data1)[ 1], (data0)[ 1])); \
    XOReq(X##bi, LOAD4_64((data3)[ 2], (data2)[ 2], (data1)[ 2], (data0)[ 2])); \
    XOReq(X##bo, LOAD4_64((data3)[ 3], (data2)[ 3], (data1)[ 3], (data0)[ 3])); \
    XOReq(X##bu, LOAD4_64((data3)[ 4], (data2)[ 4], (data1)[ 4], (data0)[ 4])); \
    XOReq(X##ga, LOAD4_64((data3)[ 5], (data2)[ 5], (data1)[ 5], (data0)[ 5])); \
    XOReq(X##ge, LOAD4_64((data3)[ 6], (data2)[ 6], (data1)[ 6], (data0)[ 6])); \
    XOReq(X##gi, LOAD4_64((data3)[ 7], (data2)[ 7], (data1)[ 7], (data0)[ 7])); \
    XOReq(X##go, LOAD4_64((data3)[ 8], (data2)[ 8], (data1)[ 8], (data0)[ 8])); \
    XOReq(X##gu, LOAD4_64((data3)[ 9], (data2)[ 9], (data1)[ 9], (data0)[ 9])); \
    XOReq(X##ka, LOAD4_64((data3)[10], (data2)[10], (data1)[10], (data0)[10])); \
    XOReq(X##ke, LOAD4_64((data3)[11], (data2)[11], (data1)[11], (data0)[11])); \
    XOReq(X##ki, LOAD4_64((data3)[12], (data2)[12], (data1)[12], (data0)[12])); \
    XOReq(X##ko, LOAD4_64((data3)[13], (data2)[13], (data1)[13], (data0)[13])); \
    XOReq(X##ku, LOAD4_64((data3)[14], (data2)[14], (data1)[14], (data0)[14])); \
    XOReq(X##ma, LOAD4_64((data3)[15], (data2)[15], (data1)[15], (data0)[15])); \

#define XORdata21(X, data0, data1, data2, data3) \
    XORdata16(X, data0, data1, data2, data3) \
    XOReq(X##me, LOAD4_64((data3)[16], (data2)[16], (data1)[16], (data0)[16])); \
    XOReq(X##mi, LOAD4_64((data3)[17], (data2)[17], (data1)[17], (data0)[17])); \
    XOReq(X##mo, LOAD4_64((data3)[18], (data2)[18], (data1)[18], (data0)[18])); \
    XOReq(X##mu, LOAD4_64((data3)[19], (data2)[19], (data1)[19], (data0)[19])); \
    XOReq(X##sa, LOAD4_64((data3)[20], (data2)[20], (data1)[20], (data0)[20])); \

void KangarooTwelve_AVX512_Process4Leaves(const unsigned char *input, unsigned char *output)
{
    KeccakP_DeclareVars(__m256i);
    unsigned int j;

    initializeState(_);

    for(j = 0; j < (chunkSize - rateInBytes); j += rateInBytes) {
        XORdata21(_, (const uint64_t *)input, (const uint64_t *)(input+chunkSize), (const uint64_t *)(input+2*chunkSize), (const uint64_t *)(input+3*chunkSize));
        rounds12
        input += rateInBytes;
    }

    XORdata16(_, (const uint64_t *)input, (const uint64_t *)(input+chunkSize), (const uint64_t *)(input+2*chunkSize), (const uint64_t *)(input+3*chunkSize));
    XOReq(_me, CONST_64(0x0BULL));
    XOReq(_sa, CONST_64(0x8000000000000000ULL));
    rounds12

#define STORE256u(a, b)         _mm256_storeu_si256((__m256i *)&(a), b)
#define UNPACKL( a, b )         _mm256_unpacklo_epi64((a), (b))
#define UNPACKH( a, b )         _mm256_unpackhi_epi64((a), (b))
#define PERM128( a, b, c )      (__m256i)_mm256_permute2f128_ps((__m256)(a), (__m256)(b), c)
    {
        __m256i lanesL01, lanesL23, lanesH01, lanesH23;

        lanesL01 = UNPACKL( _ba, _be );
        lanesH01 = UNPACKH( _ba, _be );
        lanesL23 = UNPACKL( _bi, _bo );
        lanesH23 = UNPACKH( _bi, _bo );
        STORE256u( output[ 0], PERM128( lanesL01, lanesL23, 0x20 ) );
        STORE256u( output[32], PERM128( lanesH01, lanesH23, 0x20 ) );
        STORE256u( output[64], PERM128( lanesL01, lanesL23, 0x31 ) );
        STORE256u( output[96], PERM128( lanesH01, lanesH23, 0x31 ) );
    }
/* TODO: check if something like this would be better:
    index512 = LOAD8_32(3*laneOffset+1, 2*laneOffset+1, 1*laneOffset+1, 0*laneOffset+1, 3*laneOffset, 2*laneOffset, 1*laneOffset, 0*laneOffset);
    STORE_SCATTER8_64(dataAsLanes+0, index512, stateAsLanes512[0/2]);
    STORE_SCATTER8_64(dataAsLanes+2, index512, stateAsLanes512[2/2]);
*/
}

#undef XOR
#undef XOReq
#undef XOR3
#undef XOR5
#undef XOR512
#undef ROL
#undef Chi
#undef CONST_64
#undef ZERO
#undef LOAD4_64
#undef XORdata16
#undef XORdata21


/* Keccak-p[1600]×8 */

#define XOR(a,b)                    _mm512_xor_si512(a,b)
#define XOReq(a,b)                  a = _mm512_xor_si512(a,b)
#define XOR3(a,b,c)                 _mm512_ternarylogic_epi64(a,b,c,0x96)
#define XOR5(a,b,c,d,e)             XOR3(XOR3(a,b,c),d,e)
#define XOReq512(a, b)              a = XOR(a,b)
#define ROL(a,offset)               _mm512_rol_epi64(a,offset)
#define Chi(a,b,c)                  _mm512_ternarylogic_epi64(a,b,c,0xD2)
#define CONST_64(a)                 _mm512_set1_epi64(a)
#define ZERO()                      _mm512_setzero_si512()

#define XORdata16(X, index, dataAsLanes) \
    XOReq(X##ba, LOAD_GATHER8_64(index, (dataAsLanes) +  0)); \
    XOReq(X##be, LOAD_GATHER8_64(index, (dataAsLanes) +  1)); \
    XOReq(X##bi, LOAD_GATHER8_64(index, (dataAsLanes) +  2)); \
    XOReq(X##bo, LOAD_GATHER8_64(index, (dataAsLanes) +  3)); \
    XOReq(X##bu, LOAD_GATHER8_64(index, (dataAsLanes) +  4)); \
    XOReq(X##ga, LOAD_GATHER8_64(index, (dataAsLanes) +  5)); \
    XOReq(X##ge, LOAD_GATHER8_64(index, (dataAsLanes) +  6)); \
    XOReq(X##gi, LOAD_GATHER8_64(index, (dataAsLanes) +  7)); \
    XOReq(X##go, LOAD_GATHER8_64(index, (dataAsLanes) +  8)); \
    XOReq(X##gu, LOAD_GATHER8_64(index, (dataAsLanes) +  9)); \
    XOReq(X##ka, LOAD_GATHER8_64(index, (dataAsLanes) + 10)); \
    XOReq(X##ke, LOAD_GATHER8_64(index, (dataAsLanes) + 11)); \
    XOReq(X##ki, LOAD_GATHER8_64(index, (dataAsLanes) + 12)); \
    XOReq(X##ko, LOAD_GATHER8_64(index, (dataAsLanes) + 13)); \
    XOReq(X##ku, LOAD_GATHER8_64(index, (dataAsLanes) + 14)); \
    XOReq(X##ma, LOAD_GATHER8_64(index, (dataAsLanes) + 15)); \

#define XORdata21(X, index, dataAsLanes) \
    XORdata16(X, index, dataAsLanes) \
    XOReq(X##me, LOAD_GATHER8_64(index, (dataAsLanes) + 16)); \
    XOReq(X##mi, LOAD_GATHER8_64(index, (dataAsLanes) + 17)); \
    XOReq(X##mo, LOAD_GATHER8_64(index, (dataAsLanes) + 18)); \
    XOReq(X##mu, LOAD_GATHER8_64(index, (dataAsLanes) + 19)); \
    XOReq(X##sa, LOAD_GATHER8_64(index, (dataAsLanes) + 20)); \

void KangarooTwelve_AVX512_Process8Leaves(const unsigned char *input, unsigned char *output)
{
    KeccakP_DeclareVars(__m512i);
    unsigned int j;
    const uint64_t *outputAsLanes = (const uint64_t *)output;
    __m256i index;

    initializeState(_);

    index = LOAD8_32(7*(chunkSize / 8), 6*(chunkSize / 8), 5*(chunkSize / 8), 4*(chunkSize / 8), 3*(chunkSize / 8), 2*(chunkSize / 8), 1*(chunkSize / 8), 0*(chunkSize / 8));
    for(j = 0; j < (chunkSize - rateInBytes); j += rateInBytes) {
        XORdata21(_, index, (const uint64_t *)input);
        rounds12
        input += rateInBytes;
    }

    XORdata16(_, index, (const uint64_t *)input);
    XOReq(_me, CONST_64(0x0BULL));
    XOReq(_sa, CONST_64(0x8000000000000000ULL));
    rounds12

    index = LOAD8_32(7*4, 6*4, 5*4, 4*4, 3*4, 2*4, 1*4, 0*4);
    STORE_SCATTER8_64(outputAsLanes+0, index, _ba);
    STORE_SCATTER8_64(outputAsLanes+1, index, _be);
    STORE_SCATTER8_64(outputAsLanes+2, index, _bi);
    STORE_SCATTER8_64(outputAsLanes+3, index, _bo);
}
