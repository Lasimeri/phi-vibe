/*
 * vgtest: gates the card's GEMM daemon against a host double-precision reference, then times the encoder's shapes.
 *
 * usage: vgtest [-r REPEATS] [-s SEED] [-q]   (-q: correctness only)
 *
 * Correctness: odd and even shapes, both A layouts, every epilogue flag, alpha != 1, K = 0. Error measure per element:
 * |card - ref| / (sum_k |alpha a b| + |c0| + |bias| + tiny), the max over all elements; fp32 accumulation over K
 * steps stays near K * 2^-24. The padding around every output is filled with a sentinel before the op and checked
 * after (nothing past M x N may be written). Transpose: bit-exact.
 * Speed: each shape REPEATS times in one list with per-op times; GFLOP/s = 2MNK / op time.
 */
#define _GNU_SOURCE
#include "../host/vg.h"
#include "../lib/vg-proto.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static uint64_t rng = 88172645463325252ull;
static float frand(void)
{
	rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
	return (float)((rng >> 11) * (1.0 / 9007199254740992.0)) * 2.0f - 1.0f;
}

static double now(void)
{
	struct timespec s;
	clock_gettime(CLOCK_MONOTONIC, &s);
	return s.tv_sec + s.tv_nsec / 1e9;
}

#define SENT 1234.5f   /* the padding sentinel */

/* a host matrix with the arena layout (rows8 x ld), contents random, padding SENT */
static float *mat(int rows, int cols, int ld, int fill)
{
	const int rows8 = (rows + 7) / 8 * 8;
	float *m = malloc((size_t)rows8 * ld * sizeof(float));
	for (int r = 0; r < rows8; r++)
		for (int c = 0; c < ld; c++) m[(size_t)r * ld + c] = r < rows && c < cols ? (fill ? frand() : 0.f) : SENT;
	return m;
}

typedef struct { int M, N, K, ta, flags; float alpha; } gcase;

static int run_case(vg *v, const gcase *c, double *worst)
{
	const int M = c->M, N = c->N, K = c->K;
	const int ar = c->ta ? K : M, ac = c->ta ? M : K;
	vt A = vg_tensor(v, ar, ac), B = vg_tensor(v, K, N), C = vg_tensor(v, M, N), bias = vg_tensor(v, 1, N), Z = vg_tensor(v, M, N);
	float *a = mat(ar, ac, A.ld, 1), *b = mat(K, N, B.ld, 1), *c0 = mat(M, N, C.ld, 1), *bi = mat(1, N, bias.ld, 1), *z = mat(M, N, Z.ld, 1);
	const int rows8 = (M + 7) / 8 * 8;
	float *out = malloc((size_t)rows8 * C.ld * sizeof(float));
	/* PUT writes the padding as zero; the sentinel goes into C's padding through a full-width PUT of a wider view */
	vt Cw = C;
	Cw.rows = rows8;
	Cw.cols = C.ld;
	vg_put(v, A, a, A.ld);
	vg_put(v, B, b, B.ld);
	vg_put(v, Cw, c0, C.ld);
	vg_put(v, bias, bi, bias.ld);
	vg_put(v, Z, z, Z.ld);
	const int useb = !!(c->flags & VG_BIAS), usez = !!(c->flags & VG_MASK);
	vg_gemm(v, C, A, B, useb ? &bias : NULL, usez ? &Z : NULL, c->alpha, c->flags & (VG_TA | VG_BETA | VG_RELU));
	vg_get(v, Cw, out, C.ld);
	int err = vg_sync(v);
	if (err) { printf("case %dx%dx%d: card error %d\n", M, N, K, err); return 1; }
	double w = 0;
	long pad_bad = 0;
	for (int i = 0; i < rows8; i++)
		for (int j = 0; j < C.ld; j++) {
			const float got = out[(size_t)i * C.ld + j];
			if (i >= M || j >= N) { pad_bad += got != SENT; continue; }
			double s = 0, mag = 0;
			for (int k = 0; k < K; k++) {
				const double x = c->ta ? a[(size_t)k * A.ld + i] : a[(size_t)i * A.ld + k], y = b[(size_t)k * B.ld + j];
				s += x * y;
				mag += fabs(x * y);
			}
			s *= c->alpha;
			mag *= fabs(c->alpha);
			if (c->flags & VG_BETA) { s += c0[(size_t)i * C.ld + j]; mag += fabs(c0[(size_t)i * C.ld + j]); }
			if (useb) { s += bi[j]; mag += fabs(bi[j]); }
			if (c->flags & VG_RELU && s < 0) s = 0;
			if (usez && !(z[(size_t)i * Z.ld + j] > 0)) s = 0;
			const double e = fabs(got - s) / (mag + 1e-30);
			if (e > w) w = e;
		}
	printf("  %5d x %5d x %5d %s alpha %.2f%s%s%s%s: max rel err %.3g, padding changed %ld\n", M, N, K, c->ta ? "TN" : "NN", c->alpha,
	       c->flags & VG_BETA ? " +C" : "", useb ? " +bias" : "", c->flags & VG_RELU ? " relu" : "", usez ? " mask" : "", w, pad_bad);
	if (w > *worst) *worst = w;
	free(a); free(b); free(c0); free(bi); free(z); free(out);
	vg_release(v, A.off);
	return pad_bad != 0 || w > 1e-5;
}

static int run_trans(vg *v, int m, int n)
{
	vt A = vg_tensor(v, m, n), C = vg_tensor(v, n, m);
	float *a = mat(m, n, A.ld, 1), *out = calloc((size_t)((n + 7) / 8 * 8) * C.ld, sizeof(float));
	vg_put(v, A, a, A.ld);
	vg_trans(v, C, A);
	vg_get(v, C, out, C.ld);
	int err = vg_sync(v);
	long bad = 0;
	for (int j = 0; j < n; j++)
		for (int i = 0; i < m; i++) bad += memcmp(&out[(size_t)j * C.ld + i], &a[(size_t)i * A.ld + j], 4) != 0;
	printf("  transpose %d x %d: %ld of %ld differ%s\n", m, n, bad, (long)m * n, err ? " (card error)" : "");
	free(a); free(out);
	vg_release(v, A.off);
	return bad != 0 || err;
}

static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }

static void bench(vg *v, int M, int N, int K, int ta, int reps, const char *what)
{
	const int ar = ta ? K : M, ac = ta ? M : K;
	vt A = vg_tensor(v, ar, ac), B = vg_tensor(v, K, N), C = vg_tensor(v, M, N);
	if (A.off == UINT64_MAX || B.off == UINT64_MAX || C.off == UINT64_MAX) { vg_release(v, A.off); return; }
	float *a = mat(ar, ac, A.ld, 1), *b = mat(K, N, B.ld, 1);
	vg_put(v, A, a, A.ld);
	vg_put(v, B, b, B.ld);
	vg_sync(v);
	vg_stats s0, s1;
	vg_get_stats(v, &s0);
	vg_times(v, 1);
	const double t0 = now();
	for (int r = 0; r < reps; r++) vg_gemm(v, C, A, B, NULL, NULL, 1.0f, ta ? VG_TA : 0);
	vg_sync(v);
	const double wall = now() - t0;
	vg_times(v, 0);
	usleep(250000);   /* one heartbeat: the VPU total is sampled every 100 ms */
	vg_get_stats(v, &s1);
	double ns[256];
	int n = reps < 256 ? reps : 256;
	for (int i = 0; i < n; i++) vg_op_ns(v, i, &ns[i]);
	qsort(ns, (size_t)n, sizeof *ns, cmpd);
	const double fl = 2.0 * M * N * K;
	printf("  %-14s %5d x %4d x %5d %s: GFLOP/s median %.1f (best %.1f, worst %.1f) over %d; op ms median %.3f; list wall %.3f s; "
	       "VPU instr %.3g (%.2f per FMA lane-op/16)\n", what, M, N, K, ta ? "TN" : "NN", fl / ns[n / 2], fl / ns[0], fl / ns[n - 1], n,
	       ns[n / 2] / 1e6, wall, (double)(s1.vpu - s0.vpu), (double)(s1.vpu - s0.vpu) / (fl / 2 / 16 * reps));
	free(a); free(b);
	vg_release(v, A.off);
}

int main(int argc, char **argv)
{
	int reps = 20, quick = 0, opt;
	while ((opt = getopt(argc, argv, "r:s:q")) != -1) {
		if (opt == 'r') reps = atoi(optarg);
		else if (opt == 's') rng = strtoull(optarg, NULL, 10) | 1;
		else if (opt == 'q') quick = 1;
		else { fprintf(stderr, "usage: vgtest [-r REPEATS] [-s SEED] [-q]\n"); return 2; }
	}
	vg *v = vg_open(NULL, 0, 10.0);
	if (!v) return 1;
	printf("vgtest: arena %llu MiB\n", (unsigned long long)(vg_arena(v) >> 20));
	const gcase cases[] = {
		{ 1, 1, 1, 0, 0, 1.0f }, { 37, 53, 29, 0, 0, 1.0f }, { 37, 53, 29, VG_TA, VG_TA | VG_BETA, 1.0f },
		{ 64, 96, 300, 0, VG_BIAS | VG_RELU, 1.0f }, { 65, 47, 513, VG_TA, VG_TA | VG_BIAS, 0.5f },
		{ 128, 256, 512, 0, VG_MASK, 1.0f }, { 129, 257, 255, 0, VG_BETA | VG_BIAS | VG_RELU | VG_MASK, -0.25f },
		{ 13, 300, 0, 0, VG_BETA | VG_BIAS, 1.0f }, { 300, 17, 1000, VG_TA, VG_TA | VG_RELU, 1.0f },
		{ 2048, 256, 96, 0, VG_BIAS | VG_RELU, 1.0f }, { 512, 256, 2048, VG_TA, VG_TA, 1.0f },
	};
	int fails = 0;
	double worst = 0;
	printf("correctness (host double reference):\n");
	for (size_t i = 0; i < sizeof cases / sizeof *cases; i++) {
		gcase c = cases[i];
		c.ta = !!(c.flags & VG_TA);
		fails += run_case(v, &c, &worst);
	}
	fails += run_trans(v, 37, 53);
	fails += run_trans(v, 512, 128);
	printf("correctness: %d failing, worst rel err %.3g\n", fails, worst);
	if (!quick && !fails) {
		printf("speed (%d repeats per shape, one list each):\n", reps);
		bench(v, 32768, 256, 96, 0, reps, "layer1 fwd");
		bench(v, 16384, 256, 512, 0, reps, "layer2 fwd");
		bench(v, 16384, 512, 256, 0, reps, "layer2 dX");
		bench(v, 512, 256, 16384, 1, reps, "layer2 dW");
		bench(v, 2048, 2048, 2048, 0, reps / 4 > 0 ? reps / 4 : 1, "square");
	}
	vg_stats s;
	vg_get_stats(v, &s);
	printf("card totals: %llu lists, %llu ops, %.3f GFLOP, busy %.3f s (GEMM %.3f, DMA %.3f, %.1f MB), VPU instr %llu\n",
	       (unsigned long long)s.lists, (unsigned long long)s.ops, s.flop / 1e9, s.busy_ns / 1e9, s.gemm_ns / 1e9, s.dma_ns / 1e9,
	       s.dma_bytes / 1e6, (unsigned long long)s.vpu);
	vg_close(v);
	return fails != 0;
}
