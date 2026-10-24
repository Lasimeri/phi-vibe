/*
 * gemm.c: see gemm.h. Card only (clang -march=knc): every floating-point operation here is a vector instruction.
 *
 * GEMM: the output is cut into panels of 48 columns (three vectors) and row groups of 8-row tiles; a task is one row
 * group of one panel. A tile keeps its 8 x 3 vector accumulators in registers over a block of KC reduction steps:
 * per step it loads 3 vectors of B and broadcasts 8 scalars of A (vbroadcastss; the compiler does not fold the
 * broadcast into the FMA's memory operand), so 24 FMAs per 35 vector instructions. Blocks after the first add into C,
 * and the last block applies alpha, beta, bias, ReLU and the mask. Rows past M and lanes past N are never stored.
 * A transposed (ta) is the same loop with the broadcast addresses' strides swapped.
 *
 * Transpose: each output vector is built from 16 broadcasts merged lane by lane (masked integer adds into a zero
 * vector; this compiler has no gather intrinsic). Exact bit copies.
 */
#include "gemm.h"

#include <immintrin.h>
#include <stdint.h>

#define MR 8
#define KC 256
#define PB0 8      /* B rows ahead into L1 (vprefetch0) */
#define PB1 48     /* B rows ahead into L2 (vprefetch1) */
#define PA 32      /* A: reduction steps ahead (one of the 8 rows per step for A, every step for A transposed) */

static inline __mmask16 lanes(int n) { return _mm512_int2mask(n >= 16 ? 0xffff : n <= 0 ? 0 : (1 << n) - 1); }

static inline __m512 relu16(__m512 v)
{
	/* IEEE bits ordered as signed integers for sign bit 0; any negative pattern is below 0 */
	return _mm512_castsi512_ps(_mm512_max_epi32(_mm512_castps_si512(v), _mm512_setzero_epi32()));
}

/* prefetch hints (VPREFETCH0/1, the KNC forms; the compiler drops __builtin_prefetch for this target). They may
 * reach past a tensor: the arena keeps a guard at its end so they never leave the mapping. */
static inline void pf0(const void *p) { __asm__("vprefetch0 %0" : : "m"(*(const char *)p)); }
static inline void pf1(const void *p) { __asm__("vprefetch1 %0" : : "m"(*(const char *)p)); }

static inline __m512 bcast(const float *p)
{
	return _mm512_extload_ps(p, _MM_UPCONV_PS_NONE, _MM_BROADCAST_1X16, _MM_HINT_NONE);
}

/* rows i .. i+mr-1, columns j .. j+16nv-1, reduction k0 .. k1-1 */
static inline __attribute__((always_inline)) void tile(const gemm_t *g, const int mr, const int nv, int i, int j, int k0, int k1,
                                                       int first, int last)
{
	__m512 acc[MR][3];
	for (int r = 0; r < mr; r++)
		for (int v = 0; v < nv; v++) acc[r][v] = _mm512_setzero_ps();
	const long sr = g->ta ? 1 : g->lda, sk = g->ta ? g->lda : 1;
	const float *ap = g->A + (long)i * sr + (long)k0 * sk;
	const float *bp = g->B + (long)k0 * g->ldb + j;
	for (int k = k0; k < k1; k++, ap += sk, bp += g->ldb) {
		for (int v = 0; v < nv; v++) { pf0(bp + PB0 * g->ldb + 16 * v); pf1(bp + PB1 * g->ldb + 16 * v); }
		if (g->ta) pf0(ap + PA * sk);
		else pf0(ap + (k & 7) % mr * sr + PA);
		__m512 b[3];
		for (int v = 0; v < nv; v++) b[v] = _mm512_load_ps(bp + 16 * v);
		for (int r = 0; r < mr; r++) {
			const __m512 x = bcast(ap + r * sr);
			for (int v = 0; v < nv; v++) acc[r][v] = _mm512_fmadd_ps(x, b[v], acc[r][v]);
		}
	}
	const __m512 zero = _mm512_setzero_ps();
	for (int r = 0; r < mr; r++) {
		float *c = g->C + (long)(i + r) * g->ldc + j;
		for (int v = 0; v < nv; v++) {
			const int col = j + 16 * v;
			__m512 x = acc[r][v];
			if (g->alpha != 1.0f) x = _mm512_mul_ps(x, bcast(&g->alpha));
			if (!first || g->beta) x = _mm512_add_ps(x, _mm512_load_ps(c + 16 * v));
			if (last) {
				if (g->bias) x = _mm512_add_ps(x, _mm512_load_ps(g->bias + col));
				if (g->relu) x = relu16(x);
				if (g->Z) {
					const __m512 z = _mm512_load_ps(g->Z + (long)(i + r) * g->ldz + col);
					const __mmask16 pos = _mm512_cmp_ps_mask(zero, z, _CMP_LT_OS);   /* 0 < z */
					x = _mm512_castsi512_ps(_mm512_mask_add_epi32(_mm512_setzero_epi32(), pos, _mm512_castps_si512(x),
					                                              _mm512_setzero_epi32()));
				}
			}
			_mm512_mask_store_ps(c + 16 * v, lanes(g->N - col), x);
		}
	}
}

#define TILE_NV(mr)                                                                                                   \
	static void tile_##mr(const gemm_t *g, int nv, int i, int j, int k0, int k1, int first, int last)                  \
	{                                                                                                                  \
		if (nv >= 3) tile(g, mr, 3, i, j, k0, k1, first, last);                                                    \
		else if (nv == 2) tile(g, mr, 2, i, j, k0, k1, first, last);                                               \
		else tile(g, mr, 1, i, j, k0, k1, first, last);                                                            \
	}
TILE_NV(8)
TILE_NV(4)
TILE_NV(2)
TILE_NV(1)

typedef struct {
	const gemm_t *g;
	int np;     /* panels */
	int nrg;    /* row groups */
	int rg;     /* tiles per row group */
} gjob;

static void gemm_task(void *v, int t)
{
	const gjob *jb = v;
	const gemm_t *g = jb->g;
	const int p = t / jb->nrg, q = t % jb->nrg;
	const int j = p * 48;
	int nv = (g->N - j + 15) / 16;
	if (nv > 3) nv = 3;
	const int i0 = q * jb->rg * MR;
	const int i1 = i0 + jb->rg * MR < g->M ? i0 + jb->rg * MR : g->M;
	int k0 = 0;
	do {
		const int k1 = k0 + KC < g->K ? k0 + KC : g->K;
		const int first = k0 == 0, last = k1 == g->K;
		for (int i = i0; i < i1;) {
			const int left = i1 - i;
			if (left >= 8) { tile_8(g, nv, i, j, k0, k1, first, last); i += 8; }
			else if (left >= 4) { tile_4(g, nv, i, j, k0, k1, first, last); i += 4; }
			else if (left >= 2) { tile_2(g, nv, i, j, k0, k1, first, last); i += 2; }
			else { tile_1(g, nv, i, j, k0, k1, first, last); i += 1; }
		}
		k0 = k1;
	} while (k0 < g->K);
}

void gemm_run(pool *pl, const gemm_t *g)
{
	if (g->M <= 0 || g->N <= 0) return;
	gjob jb = { g, (g->N + 47) / 48, 0, 4 };
	const int tiles = (g->M + MR - 1) / MR;
	const int threads = pl ? pool_threads(pl) : 1;
	while (jb.rg > 1 && jb.np * ((tiles + jb.rg - 1) / jb.rg) < 2 * threads) jb.rg /= 2;
	jb.nrg = (tiles + jb.rg - 1) / jb.rg;
	const int n = jb.np * jb.nrg;
	if (pl) pool_run(pl, gemm_task, &jb, n);
	else for (int t = 0; t < n; t++) gemm_task(&jb, t);
}

typedef struct { const float *A; long lda; int m, n; float *C; long ldc; } tjob;

/* output row j (= input column j), all of it */
static void trans_task(void *v, int j)
{
	const tjob *t = v;
	float *c = t->C + (long)j * t->ldc;
	const __m512i zi = _mm512_setzero_epi32();
	for (int i = 0; i < t->m; i += 16) {
		const int nl = t->m - i < 16 ? t->m - i : 16;
		__m512i acc = zi;
		const float *a = t->A + (long)i * t->lda + j;
		for (int l = 0; l < nl; l++, a += t->lda)
			acc = _mm512_mask_add_epi32(acc, _mm512_int2mask(1 << l), _mm512_castps_si512(bcast(a)), zi);
		_mm512_mask_store_ps(c + i, lanes(nl), _mm512_castsi512_ps(acc));
	}
}

void trans_run(pool *pl, const float *A, long lda, int m, int n, float *C, long ldc)
{
	tjob t = { A, lda, m, n, C, ldc };
	if (pl) pool_run(pl, trans_task, &t, n);
	else for (int j = 0; j < n; j++) trans_task(&t, j);
}

typedef struct { uint8_t *p; size_t n; } zjob;

#define ZCHUNK (64u << 10)

static void zero_task(void *v, int t)
{
	const zjob *z = v;
	const size_t a = (size_t)t * ZCHUNK, b = a + ZCHUNK < z->n ? a + ZCHUNK : z->n;
	const __m512 zero = _mm512_setzero_ps();
	for (size_t o = a; o < b; o += 64) _mm512_store_ps((float *)(z->p + o), zero);
}

void zero_run(pool *pl, void *p, size_t n)
{
	zjob z = { p, n };
	const int tasks = (int)((n + ZCHUNK - 1) / ZCHUNK);
	if (pl) pool_run(pl, zero_task, &z, tasks);
	else for (int t = 0; t < tasks; t++) zero_task(&z, t);
}
