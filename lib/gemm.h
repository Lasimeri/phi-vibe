/* gemm.h: single-precision GEMM, transpose and zero fill on the 7120A's vector unit (lib/gemm.c, built with
 * clang -march=knc), split across a thread pool. Layout rules are those of lib/vg-proto.h: row-major, every ld a
 * multiple of 16 floats, every base 64-byte aligned; nothing past M rows or N columns is written. */
#ifndef GEMM_H
#define GEMM_H

#include <stddef.h>
#include "pool.h"

typedef struct {
	int M, N, K;
	const float *A; long lda; int ta;   /* ta 0: A[i * lda + k]; ta 1: A[k * lda + i] */
	const float *B; long ldb;           /* B[k * ldb + j] */
	float *C; long ldc;
	const float *bias;                  /* N floats, or NULL */
	const float *Z; long ldz;           /* zero C where Z <= 0, or NULL */
	float alpha;
	int beta;                           /* 1: C += ..., 0: C = ... */
	int relu;
} gemm_t;

void gemm_run(pool *pl, const gemm_t *g);

/* C[j][i] = A[i][j]: A is m x n (lda), C is n x m (ldc) */
void trans_run(pool *pl, const float *A, long lda, int m, int n, float *C, long ldc);

/* n bytes from p (both multiples of 64) set to zero */
void zero_run(pool *pl, void *p, size_t n);

#endif
