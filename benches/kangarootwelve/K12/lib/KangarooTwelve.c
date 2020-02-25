/*
Implementation by Ronny Van Keer, hereby denoted as "the implementer".

For more information, feedback or questions, please refer to our website:
https://keccak.team/

To the extent possible under law, the implementer has waived all copyright
and related or neighboring rights to the source code in this file.
http://creativecommons.org/publicdomain/zero/1.0/
*/

#include <string.h>
#include <stdint.h>
#include "KangarooTwelve.h"

void KangarooTwelve_SetProcessorCapabilities();
int K12_enableSSSE3 = 0;
int K12_enableAVX2 = 0;
int K12_enableAVX512 = 0;

int KeccakWidth1600_12rounds_SpongeInitialize(KeccakWidth1600_12rounds_SpongeInstance *instance, unsigned int rate, unsigned int capacity)
{
    if (rate+capacity != 1600)
        return 1;
    if ((rate <= 0) || (rate > 1600) || ((rate % 8) != 0))
        return 1;
    KeccakP1600_Initialize(instance->state);
    instance->rate = rate;
    instance->byteIOIndex = 0;
    instance->squeezing = 0;

    return 0;
}

/* ---------------------------------------------------------------- */

int KeccakWidth1600_12rounds_SpongeAbsorb(KeccakWidth1600_12rounds_SpongeInstance *instance, const unsigned char *data, size_t dataByteLen)
{
    size_t i, j;
    unsigned int partialBlock;
    const unsigned char *curData;
    unsigned int rateInBytes = instance->rate/8;

    if (instance->squeezing)
        return 1; /* Too late for additional input */

    i = 0;
    curData = data;
    while(i < dataByteLen) {
        if ((instance->byteIOIndex == 0) && (dataByteLen >= (i + rateInBytes))) {
#ifdef KeccakP1600_12rounds_FastLoop_supported
            /* processing full blocks first */
            if ((rateInBytes % (1600/200)) == 0) {
                /* fast lane: whole lane rate */
                j = KeccakP1600_12rounds_FastLoop_Absorb(instance->state, rateInBytes/(1600/200), curData, dataByteLen - i);
                i += j;
                curData += j;
            }
            else {
#endif
                for(j=dataByteLen-i; j>=rateInBytes; j-=rateInBytes) {
                    KeccakP1600_AddBytes(instance->state, curData, 0, rateInBytes);
                    KeccakP1600_Permute_12rounds(instance->state);
                    curData+=rateInBytes;
                }
                i = dataByteLen - j;
#ifdef KeccakP1600_12rounds_FastLoop_supported
            }
#endif
        }
        else {
            /* normal lane: using the message queue */
            partialBlock = (unsigned int)(dataByteLen - i);
            if (partialBlock+instance->byteIOIndex > rateInBytes)
                partialBlock = rateInBytes-instance->byteIOIndex;
            i += partialBlock;

            KeccakP1600_AddBytes(instance->state, curData, instance->byteIOIndex, partialBlock);
            curData += partialBlock;
            instance->byteIOIndex += partialBlock;
            if (instance->byteIOIndex == rateInBytes) {
                KeccakP1600_Permute_12rounds(instance->state);
                instance->byteIOIndex = 0;
            }
        }
    }
    return 0;
}

/* ---------------------------------------------------------------- */

int KeccakWidth1600_12rounds_SpongeAbsorbLastFewBits(KeccakWidth1600_12rounds_SpongeInstance *instance, unsigned char delimitedData)
{
    unsigned int rateInBytes = instance->rate/8;

    if (delimitedData == 0)
        return 1;
    if (instance->squeezing)
        return 1; /* Too late for additional input */

    /* Last few bits, whose delimiter coincides with first bit of padding */
    KeccakP1600_AddByte(instance->state, delimitedData, instance->byteIOIndex);
    /* If the first bit of padding is at position rate-1, we need a whole new block for the second bit of padding */
    if ((delimitedData >= 0x80) && (instance->byteIOIndex == (rateInBytes-1)))
        KeccakP1600_Permute_12rounds(instance->state);
    /* Second bit of padding */
    KeccakP1600_AddByte(instance->state, 0x80, rateInBytes-1);
    KeccakP1600_Permute_12rounds(instance->state);
    instance->byteIOIndex = 0;
    instance->squeezing = 1;
    return 0;
}

/* ---------------------------------------------------------------- */

int KeccakWidth1600_12rounds_SpongeSqueeze(KeccakWidth1600_12rounds_SpongeInstance *instance, unsigned char *data, size_t dataByteLen)
{
    size_t i, j;
    unsigned int partialBlock;
    unsigned int rateInBytes = instance->rate/8;
    unsigned char *curData;

    if (!instance->squeezing)
        KeccakWidth1600_12rounds_SpongeAbsorbLastFewBits(instance, 0x01);

    i = 0;
    curData = data;
    while(i < dataByteLen) {
        if ((instance->byteIOIndex == rateInBytes) && (dataByteLen >= (i + rateInBytes))) {
            for(j=dataByteLen-i; j>=rateInBytes; j-=rateInBytes) {
                KeccakP1600_Permute_12rounds(instance->state);
                KeccakP1600_ExtractBytes(instance->state, curData, 0, rateInBytes);
                curData+=rateInBytes;
            }
            i = dataByteLen - j;
        }
        else {
            /* normal lane: using the message queue */
            if (instance->byteIOIndex == rateInBytes) {
                KeccakP1600_Permute_12rounds(instance->state);
                instance->byteIOIndex = 0;
            }
            partialBlock = (unsigned int)(dataByteLen - i);
            if (partialBlock+instance->byteIOIndex > rateInBytes)
                partialBlock = rateInBytes-instance->byteIOIndex;
            i += partialBlock;

            KeccakP1600_ExtractBytes(instance->state, curData, instance->byteIOIndex, partialBlock);
            curData += partialBlock;
            instance->byteIOIndex += partialBlock;
        }
    }
    return 0;
}

/* ---------------------------------------------------------------- */

#define chunkSize       8192
#define laneSize        8
#define suffixLeaf      0x0B /* '110': message hop, simple padding, inner node */

#define security        128
#define capacity        (2*security)
#define capacityInBytes (capacity/8)
#define rate            (1600-capacity)

#ifndef KeccakP1600_disableParallelism

int KeccakP1600times2_IsAvailable()
{
    int result = 0;
    result |= K12_enableAVX512;
    result |= K12_enableSSSE3;
    return result;
}

const char * KeccakP1600times2_GetImplementation()
{
    if (K12_enableAVX512)
        return "AVX-512 implementation";
    else
    if (K12_enableSSSE3)
        return "SSSE3 implementation";
    else
        return "";
}

void KangarooTwelve_SSSE3_Process2Leaves(const unsigned char *input, unsigned char *output);
void KangarooTwelve_AVX512_Process2Leaves(const unsigned char *input, unsigned char *output);

void KangarooTwelve_Process2Leaves(const unsigned char *input, unsigned char *output)
{
    if (K12_enableAVX512)
        KangarooTwelve_AVX512_Process2Leaves(input, output);
    else
    if (K12_enableSSSE3)
        KangarooTwelve_SSSE3_Process2Leaves(input, output);
}

int KeccakP1600times4_IsAvailable()
{
    int result = 0;
    result |= K12_enableAVX512;
    result |= K12_enableAVX2;
    return result;
}

const char * KeccakP1600times4_GetImplementation()
{
    if (K12_enableAVX512)
        return "AVX-512 implementation";
    else
    if (K12_enableAVX2)
        return "AVX2 implementation";
    else
        return "";
}

void KangarooTwelve_AVX2_Process4Leaves(const unsigned char *input, unsigned char *output);
void KangarooTwelve_AVX512_Process4Leaves(const unsigned char *input, unsigned char *output);

void KangarooTwelve_Process4Leaves(const unsigned char *input, unsigned char *output)
{
    if (K12_enableAVX512)
        KangarooTwelve_AVX512_Process4Leaves(input, output);
    else
    if (K12_enableAVX2)
        KangarooTwelve_AVX2_Process4Leaves(input, output);
}

int KeccakP1600times8_IsAvailable()
{
    int result = 0;
    result |= K12_enableAVX512;
    return result;
}

const char * KeccakP1600times8_GetImplementation()
{
    if (K12_enableAVX512)
        return "AVX-512 implementation";
    else
        return "";
}

void KangarooTwelve_AVX512_Process8Leaves(const unsigned char *input, unsigned char *output);

void KangarooTwelve_Process8Leaves(const unsigned char *input, unsigned char *output)
{
    if (K12_enableAVX512)
        KangarooTwelve_AVX512_Process8Leaves(input, output);
}

#define ProcessLeaves( Parallellism ) \
    while ( inLen >= Parallellism * chunkSize ) { \
        unsigned char intermediate[Parallellism*capacityInBytes]; \
        \
        KangarooTwelve_Process##Parallellism##Leaves(input, intermediate); \
        input += Parallellism * chunkSize; \
        inLen -= Parallellism * chunkSize; \
        ktInstance->blockNumber += Parallellism; \
        if (KeccakWidth1600_12rounds_SpongeAbsorb(&ktInstance->finalNode, intermediate, Parallellism * capacityInBytes) != 0) return 1; \
    }

#endif

static unsigned int right_encode( unsigned char * encbuf, size_t value )
{
    unsigned int n, i;
    size_t v;

    for ( v = value, n = 0; v && (n < sizeof(size_t)); ++n, v >>= 8 )
        ; /* empty */
    for ( i = 1; i <= n; ++i )
        encbuf[i-1] = (unsigned char)(value >> (8 * (n-i)));
    encbuf[n] = (unsigned char)n;
    return n + 1;
}

int KangarooTwelve_Initialize(KangarooTwelve_Instance *ktInstance, size_t outputLen)
{
    KangarooTwelve_SetProcessorCapabilities();
    ktInstance->fixedOutputLength = outputLen;
    ktInstance->queueAbsorbedLen = 0;
    ktInstance->blockNumber = 0;
    ktInstance->phase = ABSORBING;
    return KeccakWidth1600_12rounds_SpongeInitialize(&ktInstance->finalNode, rate, capacity);
}

int KangarooTwelve_Update(KangarooTwelve_Instance *ktInstance, const unsigned char *input, size_t inLen)
{
    if (ktInstance->phase != ABSORBING)
        return 1;

    if ( ktInstance->blockNumber == 0 ) {
        /* First block, absorb in final node */
        unsigned int len = (inLen < (chunkSize - ktInstance->queueAbsorbedLen)) ? inLen : (chunkSize - ktInstance->queueAbsorbedLen);
        if (KeccakWidth1600_12rounds_SpongeAbsorb(&ktInstance->finalNode, input, len) != 0)
            return 1;
        input += len;
        inLen -= len;
        ktInstance->queueAbsorbedLen += len;
        if ( (ktInstance->queueAbsorbedLen == chunkSize) && (inLen != 0) ) {
            /* First block complete and more input data available, finalize it */
            const unsigned char padding = 0x03; /* '110^6': message hop, simple padding */
            ktInstance->queueAbsorbedLen = 0;
            ktInstance->blockNumber = 1;
            if (KeccakWidth1600_12rounds_SpongeAbsorb(&ktInstance->finalNode, &padding, 1) != 0)
                return 1;
            ktInstance->finalNode.byteIOIndex = (ktInstance->finalNode.byteIOIndex + 7) & ~7; /* Zero padding up to 64 bits */
        }
    }
    else if ( ktInstance->queueAbsorbedLen != 0 ) {
        /* There is data in the queue, absorb further in queue until block complete */
        unsigned int len = (inLen < (chunkSize - ktInstance->queueAbsorbedLen)) ? inLen : (chunkSize - ktInstance->queueAbsorbedLen);
        if (KeccakWidth1600_12rounds_SpongeAbsorb(&ktInstance->queueNode, input, len) != 0)
            return 1;
        input += len;
        inLen -= len;
        ktInstance->queueAbsorbedLen += len;
        if ( ktInstance->queueAbsorbedLen == chunkSize ) {
            unsigned char intermediate[capacityInBytes];
            ktInstance->queueAbsorbedLen = 0;
            ++ktInstance->blockNumber;
            if (KeccakWidth1600_12rounds_SpongeAbsorbLastFewBits(&ktInstance->queueNode, suffixLeaf) != 0)
                return 1;
            if (KeccakWidth1600_12rounds_SpongeSqueeze(&ktInstance->queueNode, intermediate, capacityInBytes) != 0)
                return 1;
            if (KeccakWidth1600_12rounds_SpongeAbsorb(&ktInstance->finalNode, intermediate, capacityInBytes) != 0)
                return 1;
        }
    }

#ifndef KeccakP1600_disableParallelism
    if (KeccakP1600times8_IsAvailable()) {
        ProcessLeaves(8);
    }

    if (KeccakP1600times4_IsAvailable()) {
        ProcessLeaves(4);
    }

    if (KeccakP1600times2_IsAvailable()) {
        ProcessLeaves(2);
    }
#endif

    while ( inLen > 0 ) {
        unsigned int len = (inLen < chunkSize) ? inLen : chunkSize;
        if (KeccakWidth1600_12rounds_SpongeInitialize(&ktInstance->queueNode, rate, capacity) != 0)
            return 1;
        if (KeccakWidth1600_12rounds_SpongeAbsorb(&ktInstance->queueNode, input, len) != 0)
            return 1;
        input += len;
        inLen -= len;
        if ( len == chunkSize ) {
            unsigned char intermediate[capacityInBytes];
            ++ktInstance->blockNumber;
            if (KeccakWidth1600_12rounds_SpongeAbsorbLastFewBits(&ktInstance->queueNode, suffixLeaf) != 0)
                return 1;
            if (KeccakWidth1600_12rounds_SpongeSqueeze(&ktInstance->queueNode, intermediate, capacityInBytes) != 0)
                return 1;
            if (KeccakWidth1600_12rounds_SpongeAbsorb(&ktInstance->finalNode, intermediate, capacityInBytes) != 0)
                return 1;
        }
        else
            ktInstance->queueAbsorbedLen = len;
    }

    return 0;
}

int KangarooTwelve_Final(KangarooTwelve_Instance *ktInstance, unsigned char * output, const unsigned char * customization, size_t customLen)
{
    unsigned char encbuf[sizeof(size_t)+1+2];
    unsigned char padding;

    if (ktInstance->phase != ABSORBING)
        return 1;

    /* Absorb customization | right_encode(customLen) */
    if ((customLen != 0) && (KangarooTwelve_Update(ktInstance, customization, customLen) != 0))
        return 1;
    if (KangarooTwelve_Update(ktInstance, encbuf, right_encode(encbuf, customLen)) != 0)
        return 1;

    if ( ktInstance->blockNumber == 0 ) {
        /* Non complete first block in final node, pad it */
        padding = 0x07; /*  '11': message hop, final node */
    }
    else {
        unsigned int n;

        if ( ktInstance->queueAbsorbedLen != 0 ) {
            /* There is data in the queue node */
            unsigned char intermediate[capacityInBytes];
            ++ktInstance->blockNumber;
            if (KeccakWidth1600_12rounds_SpongeAbsorbLastFewBits(&ktInstance->queueNode, suffixLeaf) != 0)
                return 1;
            if (KeccakWidth1600_12rounds_SpongeSqueeze(&ktInstance->queueNode, intermediate, capacityInBytes) != 0)
                return 1;
            if (KeccakWidth1600_12rounds_SpongeAbsorb(&ktInstance->finalNode, intermediate, capacityInBytes) != 0)
                return 1;
        }
        --ktInstance->blockNumber; /* Absorb right_encode(number of Chaining Values) || 0xFF || 0xFF */
        n = right_encode(encbuf, ktInstance->blockNumber);
        encbuf[n++] = 0xFF;
        encbuf[n++] = 0xFF;
        if (KeccakWidth1600_12rounds_SpongeAbsorb(&ktInstance->finalNode, encbuf, n) != 0)
            return 1;
        padding = 0x06; /* '01': chaining hop, final node */
    }
    if (KeccakWidth1600_12rounds_SpongeAbsorbLastFewBits(&ktInstance->finalNode, padding) != 0)
        return 1;
    if ( ktInstance->fixedOutputLength != 0 ) {
        ktInstance->phase = FINAL;
        return KeccakWidth1600_12rounds_SpongeSqueeze(&ktInstance->finalNode, output, ktInstance->fixedOutputLength);
    }
    ktInstance->phase = SQUEEZING;
    return 0;
}

int KangarooTwelve_Squeeze(KangarooTwelve_Instance *ktInstance, unsigned char * output, size_t outputLen)
{
    if (ktInstance->phase != SQUEEZING)
        return 1;
    return KeccakWidth1600_12rounds_SpongeSqueeze(&ktInstance->finalNode, output, outputLen);
}

int KangarooTwelve( const unsigned char * input, size_t inLen, unsigned char * output, size_t outLen, const unsigned char * customization, size_t customLen )
{
    KangarooTwelve_Instance ktInstance;

    if (outLen == 0)
        return 1;
    if (KangarooTwelve_Initialize(&ktInstance, outLen) != 0)
        return 1;
    if (KangarooTwelve_Update(&ktInstance, input, inLen) != 0)
        return 1;
    return KangarooTwelve_Final(&ktInstance, output, customization, customLen);
}

/* Processor capability detection code by Samuel Neves and Jack O'Connor, see
 * https://github.com/BLAKE3-team/BLAKE3/blob/master/c/blake3_dispatch.c
 */

#if defined(__x86_64__) || defined(_M_X64)
#define IS_X86
#define IS_X86_64
#endif

#if defined(__i386__) || defined(_M_IX86)
#define IS_X86
#define IS_X86_32
#endif

#if defined(IS_X86)
static uint64_t xgetbv() {
#if defined(_MSC_VER)
  return _xgetbv(0);
#else
  uint32_t eax = 0, edx = 0;
  __asm__ __volatile__("xgetbv\n" : "=a"(eax), "=d"(edx) : "c"(0));
  return ((uint64_t)edx << 32) | eax;
#endif
}

static void cpuid(uint32_t out[4], uint32_t id) {
#if defined(_MSC_VER)
  __cpuid((int *)out, id);
#else
#if defined(__i386__) || defined(_M_IX86)
  __asm__ __volatile__("movl %%ebx, %1\n"
                       "cpuid\n"
                       "xchgl %1, %%ebx\n"
                       : "=a"(out[0]), "=r"(out[1]), "=c"(out[2]), "=d"(out[3])
                       : "a"(id));
#else
  __asm__ __volatile__("cpuid\n"
                       : "=a"(out[0]), "=b"(out[1]), "=c"(out[2]), "=d"(out[3])
                       : "a"(id));
#endif
#endif
}

static void cpuidex(uint32_t out[4], uint32_t id, uint32_t sid) {
#if defined(_MSC_VER)
  __cpuidex((int *)out, id, sid);
#else
  __asm__ __volatile__("movl %%ebx, %1\n"
                       "cpuid\n"
                       "xchgl %1, %%ebx\n"
                       : "=a"(out[0]), "=r"(out[1]), "=c"(out[2]), "=d"(out[3])
                       : "a"(id), "c"(sid));
#endif
}

#endif

enum cpu_feature {
  SSE2 = 1 << 0,
  SSSE3 = 1 << 1,
  SSE41 = 1 << 2,
  AVX = 1 << 3,
  AVX2 = 1 << 4,
  AVX512F = 1 << 5,
  AVX512VL = 1 << 6,
  /* ... */
  UNDEFINED = 1 << 30
};

static enum cpu_feature g_cpu_features = UNDEFINED;

static enum cpu_feature
    get_cpu_features() {

  if (g_cpu_features != UNDEFINED) {
    return g_cpu_features;
  } else {
#if defined(IS_X86)
    uint32_t regs[4] = {0};
    uint32_t *eax = &regs[0], *ebx = &regs[1], *ecx = &regs[2], *edx = &regs[3];
    (void)edx;
    enum cpu_feature features = 0;
    cpuid(regs, 0);
    const int max_id = *eax;
    cpuid(regs, 1);
#if defined(__amd64__) || defined(_M_X64)
    features |= SSE2;
#else
    if (*edx & (1UL << 26))
      features |= SSE2;
#endif
    if (*ecx & (1UL << 0))
      features |= SSSE3;
    if (*ecx & (1UL << 19))
      features |= SSE41;

    if (*ecx & (1UL << 27)) { // OSXSAVE
      const uint64_t mask = xgetbv();
      if ((mask & 6) == 6) { // SSE and AVX states
        if (*ecx & (1UL << 28))
          features |= AVX;
        if (max_id >= 7) {
          cpuidex(regs, 7, 0);
          if (*ebx & (1UL << 5))
            features |= AVX2;
          if ((mask & 224) == 224) { // Opmask, ZMM_Hi256, Hi16_Zmm
            if (*ebx & (1UL << 31))
              features |= AVX512VL;
            if (*ebx & (1UL << 16))
              features |= AVX512F;
          }
        }
      }
    }
    g_cpu_features = features;
    return features;
#else
    /* How to detect NEON? */
    return 0;
#endif
  }
}

void KangarooTwelve_SetProcessorCapabilities()
{
    enum cpu_feature features = get_cpu_features();
    K12_enableSSSE3 = (features & SSSE3);
    K12_enableAVX2 = (features & AVX2);
    K12_enableAVX512 = (features & AVX512F) && (features & AVX512VL);
}
