/*
 * dstest: gates the D* Lite planner (host/dstar.c) on the real vibe graph against a plain Dijkstra over the same
 * costs, through a long walk where the costs keep changing the way the music daemon will change them.
 *
 * usage: dstest [-s STEPS] [-k NEIGHBOURS] [-r SEED] [-c]   (-c: the CSLS graph vibed uses)
 *
 * Each step: the agent moves (to the plan's first song, or 1 in 5 times to a random song: a pick by hand), the song
 * it left stays blocked for a while, and some costs change: every step 3 % of the node costs; 1 in 3 steps a new
 * target vibe (every goal cost); 1 in 20 steps a new angle weight (every edge). After each plan:
 *   the plan's cost == Dijkstra's cost to the goal (relative difference <= 1e-9), and
 *   the plan's edges summed again from ds_edge() == the plan's cost.
 * Printed: mismatches, vertices expanded by each (median and total), and time per plan.
 */
#define _GNU_SOURCE
#include "../host/dstar.h"
#include "../host/emb.h"
#include "../host/vg.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static uint64_t rng = 0x2545f4914f6cdd1dull;
static uint64_t xr(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static double ur(void) { return (xr() >> 11) * (1.0 / 9007199254740992.0); }

static double now(void)
{
	struct timespec s;
	clock_gettime(CLOCK_MONOTONIC, &s);
	return s.tv_sec + s.tv_nsec / 1e9;
}

static int cmpl(const void *a, const void *b) { long x = *(const long *)a, y = *(const long *)b; return x < y ? -1 : x > y; }

/* a target vibe: the unit mean of 3 random songs */
static void target(const emb *E, float *g)
{
	memset(g, 0, (size_t)E->d * sizeof(float));
	for (int t = 0; t < 3; t++) {
		const float *x = E->mean + (size_t)(xr() % (uint64_t)E->n) * E->d;
		for (int k = 0; k < E->d; k++) g[k] += x[k];
	}
	double s = 0;
	for (int k = 0; k < E->d; k++) s += (double)g[k] * g[k];
	for (int k = 0; k < E->d; k++) g[k] = (float)(g[k] / sqrt(s));
}

static double ang(const float *a, const float *b, int d)
{
	double s = 0;
	for (int k = 0; k < d; k++) s += (double)a[k] * b[k];
	return acos(s > 1 ? 1 : s < -1 ? -1 : s);
}

int main(int argc, char **argv)
{
	int steps = 300, K = 16, opt;
	int csls = 0;
	while ((opt = getopt(argc, argv, "s:k:r:c")) != -1) {
		if (opt == 'c') { csls = 1; continue; }
		if (opt == 's') steps = atoi(optarg);
		else if (opt == 'k') K = atoi(optarg);
		else if (opt == 'r') rng = strtoull(optarg, NULL, 10) | 1;
		else { fprintf(stderr, "usage: dstest [-s STEPS] [-k NEIGHBOURS] [-r SEED] [-c]   (-c: the CSLS graph vibed uses)\n"); return 2; }
	}
	emb E;
	if (emb_load(&E, NULL)) return 1;
	vg *v = vg_open(NULL, 0, 10.0);
	if (!v) return 1;
	int *off, *adj;
	double t0 = now();
	if ((csls ? graph_build_csls : graph_build)(v, E.mean, E.n, E.d, K, &off, &adj)) { fprintf(stderr, "dstest: graph build failed\n"); return 1; }
	printf("dstest: %d songs, k %d: %d directed edges (mean degree %.1f), built in %.2f s\n", E.n, K, off[E.n], (double)off[E.n] / E.n, now() - t0);
	vg_close(v);

	const double alpha_floor = 1.0;
	dstar *D = ds_new(E.n, E.mean, E.d, off, adj, alpha_floor);
	float *g = malloc((size_t)E.d * sizeof(float));
	target(&E, g);
	for (int b = 0; b < E.n; b++) {
		ds_set_node(D, b, ur());
		ds_set_goal(D, b, 2.0 * ang(E.mean + (size_t)b * E.d, g, E.d));
	}
	int *path = malloc(64 * sizeof(int)), *blocked_at = malloc((size_t)E.n * sizeof(int));
	for (int b = 0; b < E.n; b++) blocked_at[b] = -1000;
	long *exd = malloc((size_t)steps * sizeof(long)), *exj = malloc((size_t)steps * sizeof(long));
	double tds = 0, tdj = 0;
	int bad = 0, badpath = 0, s = (int)(xr() % (uint64_t)E.n), jumps = 0, retargets = 0, realphas = 0;
	const int hold = 50;   /* a played song stays blocked for 50 steps */
	ds_move(D, s);
	ds_stats st0, st1;
	for (int t = 0; t < steps; t++) {
		/* the agent's song is blocked as a destination; songs blocked long enough come back */
		ds_set_blocked(D, s, 1);
		blocked_at[s] = t;
		for (int b = 0; b < E.n; b++) if (ds_blocked(D, b) && b != s && t - blocked_at[b] >= hold) ds_set_blocked(D, b, 0);
		for (int c = 0; c < E.n * 3 / 100; c++) ds_set_node(D, (int)(xr() % (uint64_t)E.n), ur());
		if (xr() % 3 == 0) {
			retargets++;
			target(&E, g);
			for (int b = 0; b < E.n; b++) ds_set_goal(D, b, 2.0 * ang(E.mean + (size_t)b * E.d, g, E.d));
		}
		if (xr() % 20 == 0) { realphas++; ds_set_alpha(D, alpha_floor + 2.0 * ur()); }
		ds_get_stats(D, &st0);
		double cost, a0 = now();
		const int n = ds_plan(D, path, 64, &cost);
		tds += now() - a0;
		ds_get_stats(D, &st1);
		exd[t] = st1.expanded - st0.expanded;
		a0 = now();
		const double ref = ds_dijkstra(D, &exj[t]);
		tdj += now() - a0;
		const int same = (isinf(cost) && isinf(ref)) || fabs(cost - ref) <= 1e-9 * fmax(1.0, fabs(ref));
		bad += !same;
		/* the plan's own edges */
		double sum = 0;
		int at = s;
		for (int i = 0; i < n; i++) { sum += ds_edge(D, at, path[i]); at = path[i]; }
		sum += ds_edge(D, at, E.n);
		const int pathok = isinf(cost) ? n == 0 : fabs(sum - cost) <= 1e-9 * fmax(1.0, fabs(cost));
		badpath += !pathok;
		if (!same || !pathok)
			printf("  step %d: D* %.12g, Dijkstra %.12g, path sum %.12g over %d songs%s\n", t, cost, ref, sum, n, same ? "" : "  MISMATCH");
		/* move: along the plan, or a jump */
		if (n == 0 || xr() % 5 == 0) { s = (int)(xr() % (uint64_t)E.n); jumps++; }
		else s = path[0];
		ds_move(D, s);
	}
	qsort(exd, (size_t)steps, sizeof(long), cmpl);
	qsort(exj, (size_t)steps, sizeof(long), cmpl);
	long sd = 0, sj = 0;
	for (int t = 0; t < steps; t++) { sd += exd[t]; sj += exj[t]; }
	printf("dstest: %d steps (%d jumps, %d new targets, %d new angle weights): %d cost mismatches, %d path-sum mismatches\n", steps, jumps,
	       retargets, realphas, bad, badpath);
	printf("  vertices expanded per plan: D* Lite median %ld (min %ld, max %ld, total %ld); Dijkstra median %ld (min %ld, max %ld, total %ld)\n",
	       exd[steps / 2], exd[0], exd[steps - 1], sd, exj[steps / 2], exj[0], exj[steps - 1], sj);
	printf("  time per plan: D* Lite %.3f ms, Dijkstra %.3f ms\n", 1e3 * tds / steps, 1e3 * tdj / steps);
	return bad || badpath;
}
