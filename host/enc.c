/* enc.c: see enc.h. */
#define _GNU_SOURCE
#include "enc.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FRAMES 128
#define BANDS 64

static uint64_t rs;
static double urand(void)
{
	rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
	return ((rs >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}
static double nrand(void) { return sqrt(-2.0 * log(urand())) * cos(2 * M_PI * urand()); }

static void layer_alloc(enc_layer *l, int cin, int cout, int relu)
{
	l->cin = cin; l->cout = cout; l->relu = relu;
	const size_t n = (size_t)cin * cout;
	l->W = calloc(n, 4); l->mW = calloc(n, 4); l->vW = calloc(n, 4); l->gW = calloc(n, 4);
	l->b = calloc((size_t)cout, 4); l->mb = calloc((size_t)cout, 4); l->vb = calloc((size_t)cout, 4); l->gb = calloc((size_t)cout, 4);
}

static void shape(enc *e)
{
	for (int i = 0; i < ENC_MERGE; i++) layer_alloc(&e->L[i], i == 0 ? 2 * BANDS : 2 * ENC_H, ENC_H, 1);
	layer_alloc(&e->L[7], ENC_H, ENC_H, 1);
	layer_alloc(&e->L[8], ENC_H, ENC_Z, 0);
}

enc *enc_new(uint64_t seed)
{
	enc *e = calloc(1, sizeof *e);
	shape(e);
	rs = seed | 1;
	for (int i = 0; i < ENC_LAYERS; i++) {
		enc_layer *l = &e->L[i];
		const double sd = sqrt((l->relu ? 2.0 : 1.0) / l->cin);
		for (size_t k = 0; k < (size_t)l->cin * l->cout; k++) l->W[k] = (float)(nrand() * sd);
	}
	for (int b = 0; b < BANDS; b++) { e->mean[b] = 0; e->inv_std[b] = 1; }
	return e;
}

static const char MAGIC[8] = { 'V', 'I', 'B', 'E', 'E', 'N', 'C', '1' };

int enc_save(const enc *e, const char *path)
{
	char tmp[4096];
	snprintf(tmp, sizeof tmp, "%s.tmp", path);
	FILE *f = fopen(tmp, "wb");
	if (!f) return -1;
	int32_t hdr[2] = { ENC_LAYERS, 1 };
	fwrite(MAGIC, 1, 8, f);
	fwrite(hdr, 4, 2, f);
	fwrite(e->mean, 4, BANDS, f);
	fwrite(e->inv_std, 4, BANDS, f);
	fwrite(&e->step, 8, 1, f);
	for (int i = 0; i < ENC_LAYERS; i++) {
		const enc_layer *l = &e->L[i];
		int32_t d[3] = { l->cin, l->cout, l->relu };
		const size_t n = (size_t)l->cin * l->cout;
		fwrite(d, 4, 3, f);
		fwrite(l->W, 4, n, f); fwrite(l->b, 4, (size_t)l->cout, f);
		fwrite(l->mW, 4, n, f); fwrite(l->vW, 4, n, f);
		fwrite(l->mb, 4, (size_t)l->cout, f); fwrite(l->vb, 4, (size_t)l->cout, f);
	}
	int bad = ferror(f);
	bad |= fclose(f);
	if (bad || rename(tmp, path)) return -1;
	return 0;
}

enc *enc_load(const char *path)
{
	FILE *f = fopen(path, "rb");
	if (!f) return NULL;
	char m[8];
	int32_t hdr[2];
	enc *e = calloc(1, sizeof *e);
	shape(e);
	int ok = fread(m, 1, 8, f) == 8 && !memcmp(m, MAGIC, 8) && fread(hdr, 4, 2, f) == 2 && hdr[0] == ENC_LAYERS &&
	         fread(e->mean, 4, BANDS, f) == BANDS && fread(e->inv_std, 4, BANDS, f) == BANDS && fread(&e->step, 8, 1, f) == 1;
	for (int i = 0; ok && i < ENC_LAYERS; i++) {
		enc_layer *l = &e->L[i];
		int32_t d[3];
		const size_t n = (size_t)l->cin * l->cout;
		ok = fread(d, 4, 3, f) == 3 && d[0] == l->cin && d[1] == l->cout && d[2] == l->relu &&
		     fread(l->W, 4, n, f) == n && fread(l->b, 4, (size_t)l->cout, f) == (size_t)l->cout &&
		     fread(l->mW, 4, n, f) == n && fread(l->vW, 4, n, f) == n &&
		     fread(l->mb, 4, (size_t)l->cout, f) == (size_t)l->cout && fread(l->vb, 4, (size_t)l->cout, f) == (size_t)l->cout;
	}
	fclose(f);
	if (!ok) { enc_free(e); return NULL; }
	return e;
}

void enc_free(enc *e)
{
	if (!e) return;
	for (int i = 0; i < ENC_LAYERS; i++) {
		enc_layer *l = &e->L[i];
		free(l->W); free(l->mW); free(l->vW); free(l->gW); free(l->b); free(l->mb); free(l->vb); free(l->gb);
	}
	free(e->xbuf);
	free(e);
}

/* a reshaped view of a dense tensor (ld == cols) */
static vt view(vt t, int rows, int cols)
{
	vt v = t;
	v.rows = rows;
	v.cols = cols;
	v.ld = cols;
	return v;
}

/* rows of layer i's output for n crops */
static int rows_of(int i, int n) { return i < ENC_MERGE ? n * (FRAMES >> (i + 1)) : n; }

/* layer i's input, as the layer reads it, for n crops */
static vt in_of(enc *e, int i, int n)
{
	if (i == 0) return view(e->X, rows_of(0, n), 2 * BANDS);
	if (i < ENC_MERGE) return view(e->Y[i - 1], rows_of(i, n), e->L[i].cin);
	return view(e->Y[i - 1], n, e->L[i].cin);
}

static vt out_of(vt t, int i, int n) { return view(t, rows_of(i, n), t.cols); }

static void upload_weights(enc *e)
{
	for (int i = 0; i < ENC_LAYERS; i++) {
		enc_layer *l = &e->L[i];
		vg_put(e->v, l->tW, l->W, l->cout);
		vg_trans(e->v, l->tWt, l->tW);
		vg_put(e->v, l->tb, l->b, l->cout);
	}
}

int enc_bind(enc *e, vg *v, int batch)
{
	e->v = v;
	e->batch = batch;
	e->mark = vg_mark(v);
	uint64_t fail = 0;
#define T(x, r, c) do { x = vg_tensor(v, r, c); fail |= x.off == UINT64_MAX; } while (0)
	T(e->X, batch * FRAMES, BANDS);
	for (int i = 0; i < ENC_LAYERS; i++) {
		enc_layer *l = &e->L[i];
		T(e->Y[i], rows_of(i, batch), l->cout);
		T(e->dY[i], rows_of(i, batch), l->cout);
		T(l->tW, l->cin, l->cout);
		T(l->tWt, l->cout, l->cin);
		T(l->tb, 1, l->cout);
		T(l->tgW, l->cin, l->cout);
		T(l->tgb, 1, l->cout);
	}
	T(e->ones, rows_of(0, batch), 1);
	T(e->Zn, batch, ENC_Z);
	T(e->ZnT, ENC_Z, batch);
	T(e->S, batch, batch);
	T(e->D, batch, batch);
	T(e->dZn, batch, ENC_Z);
#undef T
	if (fail) { vg_release(v, e->mark); return -1; }
	float *one = malloc((size_t)rows_of(0, batch) * 4);
	for (int r = 0; r < rows_of(0, batch); r++) one[r] = 1.0f;
	vg_put(v, e->ones, one, 1);
	free(one);
	free(e->xbuf);
	e->xbuf = malloc((size_t)batch * FRAMES * BANDS * 4);
	upload_weights(e);
	return vg_sync(v) ? -1 : 0;
}

void enc_input(enc *e, const uint8_t *const *crops, int n, const float *gain)
{
	for (int c = 0; c < n; c++) {
		const float g = gain ? gain[c] : 0.f;
		float *x = e->xbuf + (size_t)c * FRAMES * BANDS;
		const uint8_t *q = crops[c];
		for (int f = 0; f < FRAMES; f++)
			for (int b = 0; b < BANDS; b++) x[f * BANDS + b] = ((float)q[f * BANDS + b] + g - e->mean[b]) * e->inv_std[b];
	}
}

/* queues the input upload and the forward pass for n crops */
static void forward_ops(enc *e, int n)
{
	vg_put(e->v, view(e->X, n * FRAMES, BANDS), e->xbuf, BANDS);
	for (int i = 0; i < ENC_LAYERS; i++) {
		enc_layer *l = &e->L[i];
		vg_gemm(e->v, out_of(e->Y[i], i, n), in_of(e, i, n), l->tW, &l->tb, NULL, 1.0f, l->relu ? VGF_RELU : 0);
	}
}

int enc_forward(enc *e, int n, float *h, float *z)
{
	forward_ops(e, n);
	if (h) vg_get(e->v, view(e->Y[ENC_MERGE - 1], n, ENC_H), h, ENC_H);
	if (z) vg_get(e->v, view(e->Y[ENC_LAYERS - 1], n, ENC_Z), z, ENC_Z);
	return vg_sync(e->v);
}

static void adam(float *w, float *m, float *v, const float *g, size_t n, float lr, long t)
{
	const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
	const float c1 = 1.0f / (1.0f - powf(b1, (float)t)), c2 = 1.0f / (1.0f - powf(b2, (float)t));
	for (size_t i = 0; i < n; i++) {
		m[i] = b1 * m[i] + (1 - b1) * g[i];
		v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
		w[i] -= lr * (m[i] * c1) / (sqrtf(v[i] * c2) + eps);
	}
}

double enc_train_step(enc *e, float tau, float lr, double *acc)
{
	vg *v = e->v;
	const int n = e->batch;
	float *z = malloc((size_t)n * ENC_Z * 4), *zn = malloc((size_t)n * ENC_Z * 4), *nrm = malloc((size_t)n * 4);
	float *S = malloc((size_t)n * n * 4), *D = malloc((size_t)n * n * 4), *dzn = malloc((size_t)n * ENC_Z * 4);
	double loss = 0;
	int hits = 0;
	/* forward to z; normalise on the host */
	forward_ops(e, n);
	vg_get(v, view(e->Y[ENC_LAYERS - 1], n, ENC_Z), z, ENC_Z);
	if (vg_sync(v)) { loss = NAN; goto out; }
	for (int i = 0; i < n; i++) {
		double s = 0;
		for (int k = 0; k < ENC_Z; k++) s += (double)z[i * ENC_Z + k] * z[i * ENC_Z + k];
		nrm[i] = (float)sqrt(s + 1e-12);
		for (int k = 0; k < ENC_Z; k++) zn[i * ENC_Z + k] = z[i * ENC_Z + k] / nrm[i];
	}
	/* similarities on the card: S = Zn Zn^T */
	vg_put(v, e->Zn, zn, ENC_Z);
	vg_trans(v, e->ZnT, e->Zn);
	vg_gemm(v, e->S, e->ZnT, e->ZnT, NULL, NULL, 1.0f, VGF_TA);
	vg_get(v, e->S, S, n);
	if (vg_sync(v)) { loss = NAN; goto out; }
	/* NT-Xent: row i's partner is i ^ 1; D = dL/dS */
	for (int i = 0; i < n; i++) {
		const int p = i ^ 1;
		double mx = -1e30, sum = 0;
		int best = -1;
		for (int k = 0; k < n; k++) if (k != i) { const double s = S[i * n + k] / tau; if (s > mx) { mx = s; best = k; } }
		for (int k = 0; k < n; k++) if (k != i) sum += exp(S[i * n + k] / tau - mx);
		loss += -(S[i * n + p] / tau - mx - log(sum));
		hits += best == p;
		for (int k = 0; k < n; k++) {
			const double pk = k == i ? 0 : exp(S[i * n + k] / tau - mx) / sum;
			D[i * n + k] = (float)((pk - (k == p)) / (n * tau));
		}
	}
	loss /= n;
	*acc = (double)hits / n;
	/* dZn = (D + D^T) Zn on the card */
	vg_put(v, e->D, D, n);
	vg_gemm(v, e->dZn, e->D, e->Zn, NULL, NULL, 1.0f, 0);
	vg_gemm(v, e->dZn, e->D, e->Zn, NULL, NULL, 1.0f, VGF_TA | VGF_BETA);
	vg_get(v, e->dZn, dzn, ENC_Z);
	if (vg_sync(v)) { loss = NAN; goto out; }
	/* through the normalisation: dz = (dzn - zn (zn . dzn)) / |z| */
	for (int i = 0; i < n; i++) {
		double d = 0;
		for (int k = 0; k < ENC_Z; k++) d += (double)zn[i * ENC_Z + k] * dzn[i * ENC_Z + k];
		for (int k = 0; k < ENC_Z; k++) z[i * ENC_Z + k] = (float)((dzn[i * ENC_Z + k] - zn[i * ENC_Z + k] * d) / nrm[i]);
	}
	const int prof = getenv("VIBE_PROF") != NULL;   /* per-op card times of the backward list, once */
	if (prof) vg_times(v, 1);
	/* backward on the card */
	vg_put(v, view(e->dY[ENC_LAYERS - 1], n, ENC_Z), z, ENC_Z);
	for (int i = ENC_LAYERS - 1; i >= 0; i--) {
		enc_layer *l = &e->L[i];
		const vt in = in_of(e, i, n), dy = out_of(e->dY[i], i, n);
		vg_gemm(v, l->tgW, in, dy, NULL, NULL, 1.0f, VGF_TA);
		vt ones = e->ones;   /* a column of ones (ld 16): the bias gradient is its product with dy */
		ones.rows = rows_of(i, n);
		vg_gemm(v, l->tgb, ones, dy, NULL, NULL, 1.0f, VGF_TA);
		if (i > 0) {
			/* the gradient into layer i's input, masked by the ReLU that made it */
			const vt din = i < ENC_MERGE ? view(e->dY[i - 1], rows_of(i, n), l->cin) : view(e->dY[i - 1], n, l->cin);
			vg_gemm(v, din, dy, l->tWt, NULL, &in, 1.0f, 0);
		}
	}
	for (int i = 0; i < ENC_LAYERS; i++) {
		vg_get(v, e->L[i].tgW, e->L[i].gW, e->L[i].cout);
		vg_get(v, e->L[i].tgb, e->L[i].gb, e->L[i].cout);
	}
	if (vg_sync(v)) { loss = NAN; goto out; }
	if (prof) {
		vg_times(v, 0);
		const char *what[3] = { "dW", "db", "dX" };
		int op = 1;   /* op 0 is the PUT of dz */
		for (int i = ENC_LAYERS - 1; i >= 0; i--)
			for (int w = 0; w < (i > 0 ? 3 : 2); w++, op++) {
				double ns = 0;
				vg_op_ns(v, op, &ns);
				const double fl = w == 2 ? 2.0 * rows_of(i, n) * e->L[i].cout * e->L[i].cin : w == 0 ? 2.0 * rows_of(i, n) * e->L[i].cin * e->L[i].cout : 2.0 * rows_of(i, n) * e->L[i].cout;
				fprintf(stderr, "prof layer %d %s: %.3f ms, %.1f GFLOP/s\n", i, what[w], ns / 1e6, fl / ns);
			}
		unsetenv("VIBE_PROF");
	}
	e->step++;
	for (int i = 0; i < ENC_LAYERS; i++) {
		enc_layer *l = &e->L[i];
		adam(l->W, l->mW, l->vW, l->gW, (size_t)l->cin * l->cout, lr, e->step);
		adam(l->b, l->mb, l->vb, l->gb, (size_t)l->cout, lr, e->step);
	}
	upload_weights(e);   /* queued: runs with the next list */
out:
	free(z); free(zn); free(nrm); free(S); free(D); free(dzn);
	return loss;
}
