/* pref.c: see pref.h. */
#define _GNU_SOURCE
#include "pref.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define D PREF_D
#define X PREF_X
#define H PREF_H
#define G PREF_G
#define MARGIN 0.1

enum {
	O_A = 0,
	O_WC = O_A + D * H,
	O_B1 = O_WC + X * H,
	O_WO = O_B1 + H,
	O_WR = O_WO + H,
	O_BO = O_WR + 2,
	O_V = O_BO + 1,
	O_U = O_V + X * G,
	O_MIX = O_U + G * D,
	NPARAMS = O_MIX + 3,
};

long pref_nparams(void) { return NPARAMS; }

static const float *vibes;   /* the songs' vibes on the host (pref_bind) */

static void views(pref *p)
{
	p->A = p->w + O_A; p->Wc = p->w + O_WC; p->b1 = p->w + O_B1; p->wo = p->w + O_WO; p->wr = p->w + O_WR;
	p->bo = p->w + O_BO; p->V = p->w + O_V; p->U = p->w + O_U; p->mix = p->w + O_MIX;
}

static uint64_t rs;
static double urand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return ((rs >> 11) + 0.5) * (1.0 / 9007199254740992.0); }
static double nrand(void) { return sqrt(-2.0 * log(urand())) * cos(2 * M_PI * urand()); }

static double softplus(double z) { return z > 30 ? z : log1p(exp(z)); }
static double sigm(double z) { return 1.0 / (1.0 + exp(-z)); }

pref *pref_new(uint64_t seed)
{
	pref *p = calloc(1, sizeof *p);
	p->w = calloc(NPARAMS, 4); p->m = calloc(NPARAMS, 4); p->v = calloc(NPARAMS, 4); p->gr = calloc(NPARAMS, 4);
	views(p);
	rs = seed | 1;
	for (int i = 0; i < D * H; i++) p->A[i] = (float)(nrand() * 0.05);
	for (int i = 0; i < X * H; i++) p->Wc[i] = (float)(nrand() * 0.02);
	for (int i = 0; i < H; i++) p->wo[i] = (float)(nrand() * 0.05);
	/* before any play: a song played just now is predicted barely played again (u = -8 r1 - 2 r2) */
	p->wr[0] = -8.0f;
	p->wr[1] = -2.0f;
	for (int i = 0; i < X * G; i++) p->V[i] = (float)(nrand() * 0.02);   /* U starts at 0: the goal starts at the current vibe */
	p->mix[0] = 0.0f;   /* alpha = floor + softplus(0) = 1.19 */
	p->mix[1] = 0.5f;   /* beta = 0.97 */
	p->mix[2] = 2.0f;   /* kappa = 2.13 */
	return p;
}

void pref_free(pref *p)
{
	if (!p) return;
	free(p->w); free(p->m); free(p->v); free(p->gr);
	free(p);
}

int pref_save(const pref *p, const char *path)
{
	char tmp[4096];
	snprintf(tmp, sizeof tmp, "%s.tmp", path);
	FILE *f = fopen(tmp, "wb");
	if (!f) return -1;
	const int32_t n = NPARAMS;
	fwrite("VIBEPRF1", 1, 8, f);
	fwrite(&n, 4, 1, f);
	fwrite(&p->step, 8, 1, f);
	fwrite(p->w, 4, NPARAMS, f); fwrite(p->m, 4, NPARAMS, f); fwrite(p->v, 4, NPARAMS, f);
	int bad = ferror(f);
	bad |= fclose(f);
	return bad || rename(tmp, path) ? -1 : 0;
}

pref *pref_load(const char *path)
{
	FILE *f = fopen(path, "rb");
	if (!f) return NULL;
	char mg[8];
	int32_t n = 0;
	pref *p = pref_new(1);
	int ok = fread(mg, 1, 8, f) == 8 && !memcmp(mg, "VIBEPRF1", 8) && fread(&n, 4, 1, f) == 1 && n == NPARAMS &&
	         fread(&p->step, 8, 1, f) == 1 && fread(p->w, 4, NPARAMS, f) == NPARAMS && fread(p->m, 4, NPARAMS, f) == NPARAMS &&
	         fread(p->v, 4, NPARAMS, f) == NPARAMS;
	fclose(f);
	if (!ok) { pref_free(p); return NULL; }
	return p;
}

double pref_alpha(const pref *p) { return PREF_ALPHA_FLOOR + softplus(p->mix[0]); }
double pref_beta(const pref *p) { return softplus(p->mix[1]); }
double pref_kappa(const pref *p) { return softplus(p->mix[2]); }

int pref_upload(pref *p)
{
	vg *v = p->card;
	vg_put(v, p->tA, p->A, H);
	vg_put(v, p->tWc, p->Wc, H);
	vg_put(v, p->tb1, p->b1, H);
	vg_put(v, p->two, p->wo, 1);
	vg_put(v, p->twoT, p->wo, H);
	vg_put(v, p->tV, p->V, G);
	vg_put(v, p->tU, p->U, D);
	vg_trans(v, p->tUT, p->tU);
	return vg_sync(v);
}

int pref_bind(pref *p, vg *card, const float *E, int n)
{
	p->card = card;
	p->n = n;
	vibes = E;
	p->mark = vg_mark(card);
	p->tE = vg_tensor(card, n, D);
	p->tA = vg_tensor(card, D, H);
	p->tWc = vg_tensor(card, X, H);
	p->tb1 = vg_tensor(card, 1, H);
	p->two = vg_tensor(card, H, 1);
	p->twoT = vg_tensor(card, 1, H);
	p->tV = vg_tensor(card, X, G);
	p->tU = vg_tensor(card, G, D);
	p->tUT = vg_tensor(card, D, G);
	if (p->tUT.off == UINT64_MAX) { vg_release(card, p->mark); return -1; }
	vg_put(card, p->tE, E, D);
	return pref_upload(p);
}

void pref_context(float *x, const float *e_cur, const float *const *hist_e, const float *hist_f, int nhist, double unix_time)
{
	memset(x, 0, X * sizeof(float));
	memcpy(x, e_cur, D * sizeof(float));
	double ws = 0;
	for (int j = 0; j < nhist; j++) {
		const double w = 1.0 / (1 + j);   /* j = 0: the latest play */
		ws += w;
		for (int k = 0; k < D; k++) {
			x[D + k] += (float)(w * hist_f[j] * hist_e[j][k]);
			x[2 * D + k] += (float)(w * (1 - hist_f[j]) * hist_e[j][k]);
		}
	}
	if (ws > 0) for (int k = D; k < 3 * D; k++) x[k] = (float)(x[k] / ws);
	time_t t = (time_t)unix_time;
	struct tm lt;
	localtime_r(&t, &lt);
	const double hour = lt.tm_hour + lt.tm_min / 60.0, day = lt.tm_wday + hour / 24.0;
	x[3 * D + 0] = (float)sin(2 * M_PI * hour / 24);
	x[3 * D + 1] = (float)cos(2 * M_PI * hour / 24);
	x[3 * D + 2] = (float)sin(2 * M_PI * day / 7);
	x[3 * D + 3] = (float)cos(2 * M_PI * day / 7);
	x[3 * D + 4] = (float)(nhist / 8.0);
}

static void unit(float *a, int n)
{
	double s = 0;
	for (int k = 0; k < n; k++) s += (double)a[k] * a[k];
	const double r = 1.0 / sqrt(s + 1e-20);
	for (int k = 0; k < n; k++) a[k] = (float)(a[k] * r);
}

int pref_score(pref *p, const float *x, const float *e_cur, const float *r, float *f, float *g, float *gcos)
{
	vg *v = p->card;
	const uint64_t mk = vg_mark(v);
	const int n = p->n;
	vt tx = vg_tensor(v, 1, X), tc1 = vg_tensor(v, 1, H), th = vg_tensor(v, n, H), tu = vg_tensor(v, n, 1);
	vt tz = vg_tensor(v, 1, G), tg = vg_tensor(v, 1, D), tgT = vg_tensor(v, D, 1), tcos = vg_tensor(v, n, 1);
	float *u = malloc((size_t)n * 4), *gt = malloc(D * 4);
	int rc = -1;
	if (tcos.off == UINT64_MAX) goto out;
	vg_put(v, tx, x, X);
	vg_gemm(v, tc1, tx, p->tWc, &p->tb1, NULL, 1.0f, 0);
	vg_gemm(v, th, p->tE, p->tA, &tc1, NULL, 1.0f, VGF_RELU);
	vg_gemm(v, tu, th, p->two, NULL, NULL, 1.0f, 0);
	vg_gemm(v, tz, tx, p->tV, NULL, NULL, 1.0f, 0);
	vg_gemm(v, tg, tz, p->tU, NULL, NULL, 1.0f, 0);
	vg_get(v, tu, u, 1);
	vg_get(v, tg, gt, D);
	if ((rc = vg_sync(v))) goto out;
	for (int k = 0; k < D; k++) g[k] = e_cur[k] + gt[k];
	unit(g, D);
	for (int b = 0; b < n; b++) f[b] = (float)sigm(u[b] + r[2 * b] * p->wr[0] + r[2 * b + 1] * p->wr[1] + p->bo[0]);
	vg_put(v, tgT, g, 1);
	vg_gemm(v, tcos, p->tE, tgT, NULL, NULL, 1.0f, 0);
	vg_get(v, tcos, gcos, 1);
	rc = vg_sync(v);
out:
	free(u); free(gt);
	vg_release(v, mk);
	return rc;
}

/* fraction head forward for n rows: u (n) and, on the card, the rows' X, E, H for the backward pass */
static int frac_forward(pref *p, const float *xs, const int *songs, int n, float *u, vt *tX, vt *tEb, vt *tH)
{
	vg *v = p->card;
	float *eb = malloc((size_t)n * D * 4);
	for (int i = 0; i < n; i++) memcpy(eb + (size_t)i * D, vibes + (size_t)songs[i] * D, D * 4);
	*tX = vg_tensor(v, n, X); *tEb = vg_tensor(v, n, D); *tH = vg_tensor(v, n, H);
	vt tu = vg_tensor(v, n, 1);
	if (tu.off == UINT64_MAX) { free(eb); return -1; }
	vg_put(v, *tX, xs, X);
	vg_put(v, *tEb, eb, D);
	vg_gemm(v, *tH, *tX, p->tWc, &p->tb1, NULL, 1.0f, 0);
	vg_gemm(v, *tH, *tEb, p->tA, NULL, NULL, 1.0f, VGF_BETA | VGF_RELU);
	vg_gemm(v, tu, *tH, p->two, NULL, NULL, 1.0f, 0);
	vg_get(v, tu, u, 1);
	free(eb);
	return vg_sync(v);
}

/* goal head forward for n rows: g (n x D unit), |g~| (n); Z and X stay on the card */
static int goal_forward(pref *p, const float *xs, const float *const *ecur, int n, float *g, float *gn, vt *tX, vt *tZ)
{
	vg *v = p->card;
	*tX = vg_tensor(v, n, X); *tZ = vg_tensor(v, n, G);
	vt tg = vg_tensor(v, n, D);
	if (tg.off == UINT64_MAX) return -1;
	vg_put(v, *tX, xs, X);
	vg_gemm(v, *tZ, *tX, p->tV, NULL, NULL, 1.0f, 0);
	vg_gemm(v, tg, *tZ, p->tU, NULL, NULL, 1.0f, 0);
	vg_get(v, tg, g, D);
	int rc = vg_sync(v);
	if (rc) return rc;
	for (int i = 0; i < n; i++) {
		float *gi = g + (size_t)i * D;
		double s = 0;
		for (int k = 0; k < D; k++) { gi[k] += ecur[i][k]; s += (double)gi[k] * gi[k]; }
		gn[i] = (float)sqrt(s + 1e-20);
		for (int k = 0; k < D; k++) gi[k] /= gn[i];
	}
	return 0;
}

static double angle_of(double c) { return acos(c > 1 ? 1 : c < -1 ? -1 : c); }
static double dotd(const float *a, const float *b) { double s = 0; for (int k = 0; k < D; k++) s += (double)a[k] * b[k]; return s; }

int pref_grad(pref *p, const pref_example *fr, int nfr, const pref_goal_ex *go, int ngo, const pref_choice *ch, int nch,
              double *loss_frac, double *loss_goal, double *loss_choice)
{
	vg *v = p->card;
	memset(p->gr, 0, NPARAMS * sizeof(float));
	*loss_frac = *loss_goal = *loss_choice = 0;
	int rc = 0;
	const uint64_t mk = vg_mark(v);
	/* ---- fraction head ---- */
	if (nfr > 0) {
		float *xs = malloc((size_t)nfr * X * 4), *u = malloc((size_t)nfr * 4), *du = malloc((size_t)nfr * 4);
		int *songs = malloc((size_t)nfr * sizeof(int));
		for (int i = 0; i < nfr; i++) { memcpy(xs + (size_t)i * X, fr[i].x, X * 4); songs[i] = fr[i].song; }
		vt tX, tEb, tH;
		rc = frac_forward(p, xs, songs, nfr, u, &tX, &tEb, &tH);
		if (!rc) {
			double dwr0 = 0, dwr1 = 0, dbo = 0;
			for (int i = 0; i < nfr; i++) {
				const double z = u[i] + fr[i].r[0] * p->wr[0] + fr[i].r[1] * p->wr[1] + p->bo[0], f = sigm(z), e = f - fr[i].target;
				*loss_frac += e * e / nfr;
				du[i] = (float)(2 * e * f * (1 - f) / nfr);
				dwr0 += du[i] * fr[i].r[0]; dwr1 += du[i] * fr[i].r[1]; dbo += du[i];
			}
			p->gr[O_WR] += (float)dwr0; p->gr[O_WR + 1] += (float)dwr1; p->gr[O_BO] += (float)dbo;
			vt tdu = vg_tensor(v, nfr, 1), tdH = vg_tensor(v, nfr, H), tones = vg_tensor(v, nfr, 1);
			vt tdwo = vg_tensor(v, H, 1), tdA = vg_tensor(v, D, H), tdWc = vg_tensor(v, X, H), tdb1 = vg_tensor(v, 1, H);
			float *ones = malloc((size_t)nfr * 4);
			for (int i = 0; i < nfr; i++) ones[i] = 1;
			vg_put(v, tdu, du, 1);
			vg_put(v, tones, ones, 1);
			vg_gemm(v, tdwo, tH, tdu, NULL, NULL, 1.0f, VGF_TA);
			vg_gemm(v, tdH, tdu, p->twoT, NULL, &tH, 1.0f, 0);   /* through the ReLU: zero where h <= 0 */
			vg_gemm(v, tdA, tEb, tdH, NULL, NULL, 1.0f, VGF_TA);
			vg_gemm(v, tdWc, tX, tdH, NULL, NULL, 1.0f, VGF_TA);
			vg_gemm(v, tdb1, tones, tdH, NULL, NULL, 1.0f, VGF_TA);
			vg_get(v, tdwo, p->gr + O_WO, 1);
			vg_get(v, tdA, p->gr + O_A, H);
			vg_get(v, tdWc, p->gr + O_WC, H);
			vg_get(v, tdb1, p->gr + O_B1, H);
			rc = vg_sync(v);
			free(ones);
		}
		free(xs); free(u); free(du); free(songs);
		vg_release(v, mk);
		if (rc) return rc;
	}
	/* ---- goal head ---- */
	if (ngo > 0) {
		double wsum = 0;
		for (int i = 0; i < ngo; i++) wsum += go[i].w;
		if (wsum > 0) {
			float *xs = malloc((size_t)ngo * X * 4), *g = malloc((size_t)ngo * D * 4), *gn = malloc((size_t)ngo * 4), *dg = malloc((size_t)ngo * D * 4);
			const float **ec = malloc((size_t)ngo * sizeof *ec);
			for (int i = 0; i < ngo; i++) { memcpy(xs + (size_t)i * X, go[i].x, X * 4); ec[i] = go[i].e_cur; }
			vt tX, tZ;
			rc = goal_forward(p, xs, ec, ngo, g, gn, &tX, &tZ);
			if (!rc) {
				for (int i = 0; i < ngo; i++) {
					const float *gi = g + (size_t)i * D, *t = go[i].e_next;
					const double c = dotd(gi, t), w = go[i].w / wsum;
					*loss_goal += w * (1 - c);
					for (int k = 0; k < D; k++) dg[(size_t)i * D + k] = (float)(-w * (t[k] - c * gi[k]) / gn[i]);
				}
				vt tdG = vg_tensor(v, ngo, D), tdU = vg_tensor(v, G, D), tdZ = vg_tensor(v, ngo, G), tdV = vg_tensor(v, X, G);
				vg_put(v, tdG, dg, D);
				vg_gemm(v, tdU, tZ, tdG, NULL, NULL, 1.0f, VGF_TA);
				vg_gemm(v, tdZ, tdG, p->tUT, NULL, NULL, 1.0f, 0);
				vg_gemm(v, tdV, tX, tdZ, NULL, NULL, 1.0f, VGF_TA);
				vg_get(v, tdU, p->gr + O_U, D);
				vg_get(v, tdV, p->gr + O_V, G);
				rc = vg_sync(v);
			}
			free(xs); free(g); free(gn); free(dg); free(ec);
			vg_release(v, mk);
			if (rc) return rc;
		}
	}
	/* ---- the mixing weights, from your own choices ---- */
	if (nch > 0) {
		/* the fraction head on rows 2i (your choice q) and 2i+1 (the queued p), both in choice i's context; the goal
		 * head once per choice */
		float *xs = malloc((size_t)2 * nch * X * 4), *xg = malloc((size_t)nch * X * 4), *u = malloc((size_t)2 * nch * 4);
		float *g = malloc((size_t)nch * D * 4), *gn = malloc((size_t)nch * 4);
		int *songs = malloc((size_t)2 * nch * sizeof(int));
		const float **ec = malloc((size_t)nch * sizeof *ec);
		for (int i = 0; i < nch; i++) {
			memcpy(xs + (size_t)(2 * i) * X, ch[i].x, X * 4);
			memcpy(xs + (size_t)(2 * i + 1) * X, ch[i].x, X * 4);
			memcpy(xg + (size_t)i * X, ch[i].x, X * 4);
			songs[2 * i] = ch[i].q;
			songs[2 * i + 1] = ch[i].p;
			ec[i] = ch[i].e_cur;
		}
		vt tX, tEb, tH, tX2, tZ;
		rc = frac_forward(p, xs, songs, 2 * nch, u, &tX, &tEb, &tH);
		if (!rc) rc = goal_forward(p, xg, ec, nch, g, gn, &tX2, &tZ);
		if (!rc) {
			const double a = pref_alpha(p), b = pref_beta(p), k = pref_kappa(p);
			const double sa = sigm(p->mix[0]), sb = sigm(p->mix[1]), sk = sigm(p->mix[2]);
			double ga = 0, gb = 0, gk = 0;
			for (int i = 0; i < nch; i++) {
				const float *es = vibes + (size_t)ch[i].s * D, *eq = vibes + (size_t)ch[i].q * D, *ep = vibes + (size_t)ch[i].p * D;
				const float *gg = g + (size_t)i * D;
				const double fq = sigm(u[2 * i] + ch[i].rq[0] * p->wr[0] + ch[i].rq[1] * p->wr[1] + p->bo[0]);
				const double fp = sigm(u[2 * i + 1] + ch[i].rp[0] * p->wr[0] + ch[i].rp[1] * p->wr[1] + p->bo[0]);
				const double tq = angle_of(dotd(es, eq)), tp = angle_of(dotd(es, ep));
				const double gq = angle_of(dotd(eq, gg)), gp = angle_of(dotd(ep, gg));
				const double sq = a * tq + k * (1 - fq) + b * gq, sp = a * tp + k * (1 - fp) + b * gp;
				const double l = MARGIN + sq - sp;
				if (l > 0) {
					*loss_choice += l / nch;
					ga += sa * (tq - tp) / nch;
					gk += sk * ((1 - fq) - (1 - fp)) / nch;
					gb += sb * (gq - gp) / nch;
				}
			}
			p->gr[O_MIX] += (float)ga; p->gr[O_MIX + 1] += (float)gb; p->gr[O_MIX + 2] += (float)gk;
		}
		free(xs); free(xg); free(u); free(g); free(gn); free(songs); free(ec);
		vg_release(v, mk);
	}
	return rc;
}

void pref_adam(pref *p, float lr, float decay)
{
	const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
	p->step++;
	const float c1 = 1.0f / (1.0f - powf(b1, (float)p->step)), c2 = 1.0f / (1.0f - powf(b2, (float)p->step));
	for (int i = 0; i < NPARAMS; i++) {
		const float gi = p->gr[i];
		p->m[i] = b1 * p->m[i] + (1 - b1) * gi;
		p->v[i] = b2 * p->v[i] + (1 - b2) * gi * gi;
		if (i < O_MIX && decay > 0) p->w[i] -= lr * decay * p->w[i];
		p->w[i] -= lr * (p->m[i] * c1) / (sqrtf(p->v[i] * c2) + eps);
	}
}
