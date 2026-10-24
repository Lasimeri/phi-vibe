/*
 * prefcheck: gates the preference network (host/pref.c) before it sees a real play.
 *
 * usage: prefcheck [-n DECISIONS] [-r SEED]
 *
 * 1. Gradients: random fraction, goal and choice examples on the real vibes; the backward pass (card GEMMs) against
 *    central differences of the total loss for parameters from every block (the 6 largest gradients and 4 random
 *    ones per block). Pass: relative error <= 1e-2 wherever |gradient| >= 1e-4.
 * 2. Cold start: DECISIONS planned songs with the untrained network through D* Lite on the real graph, each played
 *    in full: no song repeated within the run, every hop an edge of the graph, every cost finite, no NaN.
 */
#define _GNU_SOURCE
#include "../host/dstar.h"
#include "../host/emb.h"
#include "../host/pref.h"
#include "../host/vg.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static uint64_t rng = 0x853c49e6748fea9bull;
static uint64_t xr(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static double ur(void) { return (xr() >> 11) * (1.0 / 9007199254740992.0); }

static double now(void)
{
	struct timespec s;
	clock_gettime(CLOCK_MONOTONIC, &s);
	return s.tv_sec + s.tv_nsec / 1e9;
}

static emb E;

/* a context from a random history of up to 8 plays */
static void rand_context(float *x, int cur)
{
	const float *he[8];
	float hf[8];
	const int nh = (int)(xr() % 9);
	for (int j = 0; j < nh; j++) { he[j] = E.mean + (size_t)(xr() % (uint64_t)E.n) * E.d; hf[j] = (float)ur(); }
	pref_context(x, E.mean + (size_t)cur * E.d, he, hf, nh, 1.79e9 + ur() * 1e7);
}

/* the loss whose gradient pref_grad gives for a parameter: the choice loss trains only the mixing weights
 * (alpha, beta, kappa: the network is held fixed in it), the fraction and goal losses train the network */
static double total(pref *p, pref_example *fr, int nfr, pref_goal_ex *go, int ngo, pref_choice *ch, int nch, int mix)
{
	double a, b, c;
	if (pref_grad(p, fr, nfr, go, ngo, ch, nch, &a, &b, &c)) return NAN;
	return mix ? c : a + b;
}

static int gradcheck(pref *p)
{
	enum { NFR = 16, NGO = 8, NCH = 8 };
	static pref_example fr[NFR];
	static float gx[NGO][PREF_X], cx[NCH][PREF_X];
	pref_goal_ex go[NGO];
	pref_choice ch[NCH];
	for (int i = 0; i < NFR; i++) {
		const int s = (int)(xr() % (uint64_t)E.n);
		rand_context(fr[i].x, s);
		fr[i].song = (int)(xr() % (uint64_t)E.n);
		fr[i].r[0] = (float)ur() * 0.3f;
		fr[i].r[1] = (float)ur() * 0.3f;
		fr[i].target = (float)ur();
	}
	for (int i = 0; i < NGO; i++) {
		const int s = (int)(xr() % (uint64_t)E.n);
		rand_context(gx[i], s);
		go[i] = (pref_goal_ex){ gx[i], E.mean + (size_t)s * E.d, E.mean + (size_t)(xr() % (uint64_t)E.n) * E.d, (float)(0.2 + ur()) };
	}
	for (int i = 0; i < NCH; i++) {
		const int s = (int)(xr() % (uint64_t)E.n);
		rand_context(cx[i], s);
		ch[i] = (pref_choice){ cx[i], E.mean + (size_t)s * E.d, s, (int)(xr() % (uint64_t)E.n), (int)(xr() % (uint64_t)E.n),
		                       { 0, 0 }, { 0, 0 } };
	}
	const long np = pref_nparams();
	double l0 = total(p, fr, NFR, go, NGO, ch, NCH, 0);
	float *g = malloc((size_t)np * 4);
	memcpy(g, p->gr, (size_t)np * 4);
	printf("gradient check: %ld parameters, loss %.6f\n", np, l0);
	struct { const char *name; float *base; long n; } blk[] = {
		{ "A", p->A, (long)(p->Wc - p->A) }, { "Wc", p->Wc, (long)(p->b1 - p->Wc) }, { "b1", p->b1, (long)(p->wo - p->b1) },
		{ "wo", p->wo, (long)(p->wr - p->wo) }, { "wr", p->wr, 2 }, { "bo", p->bo, 1 }, { "V", p->V, (long)(p->U - p->V) },
		{ "U", p->U, (long)(p->mix - p->U) }, { "mix", p->mix, 3 },
	};
	int fails = 0, tested = 0;
	double worst = 0;
	for (size_t b = 0; b < sizeof blk / sizeof *blk; b++) {
		/* the block's 6 largest gradients, then 4 random entries */
		long pick[10];
		int k = 0;
		for (int t = 0; t < 6 && t < blk[b].n; t++) {
			long best = -1;
			for (long i = 0; i < blk[b].n; i++) {
				int used = 0;
				for (int q = 0; q < k; q++) used |= pick[q] == i;
				if (!used && (best < 0 || fabsf(g[blk[b].base - p->w + i]) > fabsf(g[blk[b].base - p->w + best]))) best = i;
			}
			pick[k++] = best;
		}
		for (int t = 0; t < 4 && blk[b].n > 6; t++) pick[k++] = (long)(xr() % (uint64_t)blk[b].n);
		double bw = 0;
		int bt = 0;
		for (int q = 0; q < k; q++) {
			const long i = blk[b].base - p->w + pick[q];
			const float keep = p->w[i];
			const double eps = getenv("PREFCHECK_EPS") ? atof(getenv("PREFCHECK_EPS")) : 1e-3;
			p->w[i] = (float)(keep + eps); pref_upload(p);
			const int mix = blk[b].base == p->mix;
			const double lp = total(p, fr, NFR, go, NGO, ch, NCH, mix);
			p->w[i] = (float)(keep - eps); pref_upload(p);
			const double lm = total(p, fr, NFR, go, NGO, ch, NCH, mix);
			p->w[i] = keep; pref_upload(p);
			const double num = (lp - lm) / (2 * eps), ana = g[i];
			if (fabs(ana) < 1e-4 && fabs(num) < 1e-4) continue;
			const double rel = fabs(num - ana) / fmax(fabs(num), fabs(ana));
			bt++;
			if (rel > bw) bw = rel;
			if (rel > 1e-2) { fails++; printf("  %s[%ld]: backprop %.6e, finite difference %.6e, rel %.3e  FAIL\n", blk[b].name, pick[q], ana, num, rel); }
		}
		tested += bt;
		if (bw > worst) worst = bw;
		printf("  %-3s %6ld params: %d checked, worst rel err %.3e\n", blk[b].name, blk[b].n, bt, bw);
	}
	printf("gradient check: %d checked, %d failing, worst rel err %.3e\n", tested, fails, worst);
	free(g);
	return fails;
}

static int coldstart(pref *p, int decisions)
{
	int *off, *adj;
	if (graph_build(p->card, E.mean, E.n, E.d, 16, &off, &adj)) return 1;
	dstar *D = ds_new(E.n, E.mean, E.d, off, adj, PREF_ALPHA_FLOOR);
	ds_set_alpha(D, pref_alpha(p));
	float *f = malloc((size_t)E.n * 4), *gcos = malloc((size_t)E.n * 4), *r = calloc((size_t)E.n * 2, 4), g[PREF_D], x[PREF_X];
	long *lastplay = malloc((size_t)E.n * sizeof(long));
	for (int b = 0; b < E.n; b++) lastplay[b] = -1;
	int *played = malloc((size_t)(decisions + 1) * sizeof(int));
	int s = (int)(xr() % (uint64_t)E.n), repeats = 0, offgraph = 0, nans = 0, empty = 0;
	played[0] = s;
	lastplay[s] = 0;
	double tscore = 0, tplan = 0;
	const double t0 = 1.79e9;
	ds_move(D, s);
	int prev_block = -1;
	for (int t = 1; t <= decisions; t++) {
		/* recency of every song: plays since it last played, hours since (3.5 minutes a song) */
		for (int b = 0; b < E.n; b++) {
			if (lastplay[b] < 0) { r[2 * b] = 0; r[2 * b + 1] = 0; continue; }
			const long ago = (t - 1) - lastplay[b];
			r[2 * b] = (float)(1.0 / (1 + ago));
			r[2 * b + 1] = (float)exp(-(ago * 3.5 / 60.0) / 12.0);
		}
		const float *he[8];
		float hf[8];
		int nh = 0;
		for (int j = t - 2; j >= 0 && nh < 8; j--) { he[nh] = E.mean + (size_t)played[j] * E.d; hf[nh++] = 1.0f; }
		pref_context(x, E.mean + (size_t)s * E.d, he, hf, nh, t0 + t * 210);
		double a0 = now();
		if (pref_score(p, x, E.mean + (size_t)s * E.d, r, f, g, gcos)) return 1;
		tscore += now() - a0;
		const double kap = pref_kappa(p), bet = pref_beta(p);
		for (int b = 0; b < E.n; b++) {
			if (!isfinite(f[b]) || !isfinite(gcos[b])) nans++;
			ds_set_node(D, b, kap * (1 - f[b]));
			ds_set_goal(D, b, bet * acos(gcos[b] > 1 ? 1 : gcos[b] < -1 ? -1 : gcos[b]));
		}
		if (prev_block >= 0) ds_set_blocked(D, prev_block, 0);
		ds_set_blocked(D, s, 1);   /* the one hard rule: not the song playing now */
		prev_block = s;
		int path[16];
		double cost;
		a0 = now();
		const int n = ds_plan(D, path, 16, &cost);
		tplan += now() - a0;
		if (n == 0 || !isfinite(cost)) { empty++; break; }
		const int next = path[0];
		int edge = 0;
		for (int e = off[s]; e < off[s + 1]; e++) edge |= adj[e] == next;
		offgraph += !edge;
		repeats += lastplay[next] >= 0;
		if (t <= 5 || t == decisions) printf("  decision %2d: plan of %d songs, cost %.3f, next f %.3f\n", t, n, cost, f[next]);
		played[t] = next;
		lastplay[next] = t;
		s = next;
		ds_move(D, s);
	}
	printf("cold start: %d decisions: %d repeats, %d hops off the graph, %d empty plans, %d non-finite outputs; "
	       "per decision: card scoring %.2f ms, plan %.2f ms (alpha %.3f beta %.3f kappa %.3f)\n", decisions, repeats, offgraph,
	       empty, nans, 1e3 * tscore / decisions, 1e3 * tplan / decisions, pref_alpha(p), pref_beta(p), pref_kappa(p));
	return repeats || offgraph || empty || nans;
}

int main(int argc, char **argv)
{
	int decisions = 50, opt;
	while ((opt = getopt(argc, argv, "n:r:")) != -1) {
		if (opt == 'n') decisions = atoi(optarg);
		else if (opt == 'r') rng = strtoull(optarg, NULL, 10) | 1;
		else { fprintf(stderr, "usage: prefcheck [-n DECISIONS] [-r SEED]\n"); return 2; }
	}
	setvbuf(stdout, NULL, _IOLBF, 0);
	if (emb_load(&E, NULL)) return 1;
	vg *v = vg_open(NULL, 0, 10.0);
	if (!v) return 1;
	pref *p = pref_new(0xfeed);
	if (pref_bind(p, v, E.mean, E.n)) { fprintf(stderr, "prefcheck: bind failed\n"); return 1; }
	int bad = gradcheck(p);
	pref_free(p);
	p = pref_new(0xfeed);   /* untrained again for the cold start */
	if (pref_bind(p, v, E.mean, E.n)) return 1;
	bad += coldstart(p, decisions);
	vg_close(v);
	return bad != 0;
}
