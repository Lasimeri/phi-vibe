/*
 * vibe-train: trains the vibe encoder (host/enc.h) on the extracted library, measures it, and writes the song
 * embeddings the next-song planner uses.
 *
 * usage: vibe-train [-e EPOCHS] [-s SONGS_PER_BATCH] [-l LR] [-t TAU] [-v EVAL_EVERY] [-r] [-E] [-x]
 *   -r  resume from the saved model    -E  only embed (and measure) with the saved model    -x  no embedding file
 * Files: model ~/.local/share/phivibe/enc.bin, embeddings ~/.local/share/phivibe/emb.bin (see write_emb).
 *
 * Measures (exact, printed with their chance level):
 *   held-out crop retrieval: songs whose path hash % 10 == 0 are never trained on; up to 4 crops of each, and for
 *     every crop its nearest other crop: the fraction from the same song. Chance = sum (crops_s - 1) / (N - 1) / N.
 *   song neighbourhoods: every song's vector (mean of its crops' unit h), its 10 nearest songs: the fraction in the same
 *     folder, and with the same artist (songs named "Artist - Title"); chance from the folder and artist sizes.
 *   the same two for a hand-made baseline (per-band mean and spread of the dB codes, z-scored) and for the encoder
 *     before training.
 */
#define _GNU_SOURCE
#include "../host/enc.h"
#include "../host/knn.h"
#include "../host/songs.h"
#include "../host/vg.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint64_t xr(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static double ur(void) { return (xr() >> 11) * (1.0 / 9007199254740992.0); }

static double now(void)
{
	struct timespec s;
	clock_gettime(CLOCK_MONOTONIC, &s);
	return s.tv_sec + s.tv_nsec / 1e9;
}

static songs L;
static vg *V;
static char dir[4096], model_path[4200], emb_path[4200];

static int held_out(int i) { return songs_hash(L.s[i].rel) % 10 == 0; }

static void unit(float *x, int d)
{
	double s = 0;
	for (int k = 0; k < d; k++) s += (double)x[k] * x[k];
	const float r = (float)(1.0 / sqrt(s + 1e-20));
	for (int k = 0; k < d; k++) x[k] *= r;
}

/* per-band mean and 1/std of the dB codes over random crops */
static void norm_stats(enc *e)
{
	double s[64] = { 0 }, q[64] = { 0 };
	long n = 0;
	for (int t = 0; t < 4096; t++) {
		const int i = (int)(xr() % (uint64_t)L.n), c = (int)(xr() % (uint64_t)L.s[i].ncrops);
		const uint8_t *m = songs_crop(&L, i, c);
		for (int f = 0; f < 128; f++)
			for (int b = 0; b < 64; b++) { s[b] += m[f * 64 + b]; q[b] += (double)m[f * 64 + b] * m[f * 64 + b]; }
		n += 128;
	}
	for (int b = 0; b < 64; b++) {
		const double mu = s[b] / n, var = q[b] / n - mu * mu;
		e->mean[b] = (float)mu;
		e->inv_std[b] = (float)(1.0 / sqrt(var > 1e-6 ? var : 1e-6));
	}
}

/* h of crops (song, crop) pairs in batches; out: n x 256 unit rows */
static int embed_crops(enc *e, const int *song, const int *crop, int n, float *out)
{
	const uint8_t **ptr = malloc((size_t)e->batch * sizeof *ptr);
	for (int a = 0; a < n; a += e->batch) {
		const int b = n - a < e->batch ? n - a : e->batch;
		for (int j = 0; j < b; j++) ptr[j] = songs_crop(&L, song[a + j], crop[a + j]);
		enc_input(e, ptr, b, NULL);
		if (enc_forward(e, b, out + (size_t)a * ENC_H, NULL)) { free(ptr); return -1; }
	}
	free(ptr);
	for (int j = 0; j < n; j++) unit(out + (size_t)j * ENC_H, ENC_H);
	return 0;
}

/* every song's vector: the unit mean of its crops' unit h; also the first and last crop's */
static int embed_songs(enc *e, float *mean, float *first, float *last)
{
	long total = 0;
	for (int i = 0; i < L.n; i++) total += L.s[i].ncrops;
	int *so = malloc((size_t)total * sizeof(int)), *cr = malloc((size_t)total * sizeof(int));
	long k = 0;
	for (int i = 0; i < L.n; i++)
		for (int c = 0; c < L.s[i].ncrops; c++) { so[k] = i; cr[k] = c; k++; }
	float *h = malloc((size_t)total * ENC_H * sizeof(float));
	const double t0 = now();
	int rc = embed_crops(e, so, cr, (int)total, h);
	if (!rc) {
		k = 0;
		for (int i = 0; i < L.n; i++) {
			float *m = mean + (size_t)i * ENC_H;
			memset(m, 0, ENC_H * sizeof(float));
			for (int c = 0; c < L.s[i].ncrops; c++, k++)
				for (int d = 0; d < ENC_H; d++) m[d] += h[(size_t)k * ENC_H + d];
			unit(m, ENC_H);
			if (first) memcpy(first + (size_t)i * ENC_H, h + (size_t)(k - L.s[i].ncrops) * ENC_H, ENC_H * sizeof(float));
			if (last) memcpy(last + (size_t)i * ENC_H, h + (size_t)(k - 1) * ENC_H, ENC_H * sizeof(float));
		}
		printf("  embedded %ld crops of %d songs in %.2f s\n", total, L.n, now() - t0);
	}
	free(so); free(cr); free(h);
	return rc;
}

/* held-out crop retrieval with the encoder */
static void eval_retrieval(enc *e, const char *tag)
{
	int *so = NULL, *cr = NULL, n = 0, cap = 0;
	for (int i = 0; i < L.n; i++) {
		if (!held_out(i) || L.s[i].ncrops < 2) continue;
		const int m = L.s[i].ncrops < 4 ? L.s[i].ncrops : 4;
		for (int j = 0; j < m; j++) {
			if (n == cap) { cap = cap ? 2 * cap : 4096; so = realloc(so, (size_t)cap * sizeof(int)); cr = realloc(cr, (size_t)cap * sizeof(int)); }
			so[n] = i;
			cr[n] = m == 1 ? 0 : j * (L.s[i].ncrops - 1) / (m - 1);
			n++;
		}
	}
	float *h = malloc((size_t)n * ENC_H * sizeof(float));
	int *nb = malloc((size_t)n * sizeof(int));
	float *sim = malloc((size_t)n * sizeof(float));
	if (embed_crops(e, so, cr, n, h) || knn_card(V, h, n, ENC_H, 1, nb, sim)) { printf("  %s retrieval: card error\n", tag); goto out; }
	int hit = 0, nsongs = 0;
	double chance = 0;
	for (int a = 0; a < n; a++) {
		hit += nb[a] >= 0 && so[nb[a]] == so[a];
		nsongs += a == 0 || so[a] != so[a - 1];
		int same = 0;
		for (int b = 0; b < n; b++) same += so[b] == so[a];
		chance += (double)(same - 1) / (n - 1);
	}
	printf("  %s held-out crop retrieval: %d crops of %d songs, top-1 same song %.4f (chance %.4f)\n", tag, n, nsongs,
	       (double)hit / n, chance / n);
out:
	free(so); free(cr); free(h); free(nb); free(sim);
}

/* song neighbourhoods for any song vectors E (L.n x d, unit rows) */
static void eval_neighbours(const float *E, int d, const char *tag)
{
	const int K = 10;
	int *nb = malloc((size_t)L.n * K * sizeof(int));
	float *sim = malloc((size_t)L.n * K * sizeof(float));
	if (knn_card(V, E, L.n, d, K, nb, sim)) { printf("  %s neighbours: card error\n", tag); free(nb); free(sim); return; }
	int *fsize = calloc((size_t)L.nfolders, sizeof(int)), *asize = calloc((size_t)(L.nartists + 1), sizeof(int));
	for (int i = 0; i < L.n; i++) { fsize[L.s[i].folder]++; if (L.s[i].artist >= 0) asize[L.s[i].artist]++; }
	long fh = 0, ah = 0, an = 0;
	double fch = 0, ach = 0;
	for (int i = 0; i < L.n; i++) {
		for (int j = 0; j < K; j++) {
			const int o = nb[(size_t)i * K + j];
			if (o < 0) continue;
			fh += L.s[o].folder == L.s[i].folder;
			if (L.s[i].artist >= 0) ah += L.s[o].artist == L.s[i].artist;
		}
		fch += (double)(fsize[L.s[i].folder] - 1) / (L.n - 1);
		if (L.s[i].artist >= 0) { an++; ach += (double)(asize[L.s[i].artist] - 1) / (L.n - 1); }
	}
	printf("  %s song neighbours @%d: same folder %.4f (chance %.4f), same artist %.4f (chance %.4f, %ld songs with an artist)\n", tag, K,
	       (double)fh / ((double)L.n * K), fch / L.n, an ? (double)ah / ((double)an * K) : 0, an ? ach / an : 0, an);
	free(nb); free(sim); free(fsize); free(asize);
}

/* the hand-made baseline: per-band mean and spread over a song's crops, z-scored over songs */
static void eval_baseline(void)
{
	const int d = 128;
	float *E = calloc((size_t)L.n * d, sizeof(float));
	for (int i = 0; i < L.n; i++) {
		double s[64] = { 0 }, q[64] = { 0 };
		long n = 0;
		for (int c = 0; c < L.s[i].ncrops; c++) {
			const uint8_t *m = songs_crop(&L, i, c);
			for (int f = 0; f < 128; f++) for (int b = 0; b < 64; b++) { s[b] += m[f * 64 + b]; q[b] += (double)m[f * 64 + b] * m[f * 64 + b]; }
			n += 128;
		}
		for (int b = 0; b < 64; b++) {
			const double mu = s[b] / n;
			E[(size_t)i * d + b] = (float)mu;
			E[(size_t)i * d + 64 + b] = (float)sqrt(fmax(q[b] / n - mu * mu, 0));
		}
	}
	for (int k = 0; k < d; k++) {
		double s = 0, q = 0;
		for (int i = 0; i < L.n; i++) { s += E[(size_t)i * d + k]; q += (double)E[(size_t)i * d + k] * E[(size_t)i * d + k]; }
		const double mu = s / L.n, sd = sqrt(fmax(q / L.n - mu * mu, 1e-12));
		for (int i = 0; i < L.n; i++) E[(size_t)i * d + k] = (float)((E[(size_t)i * d + k] - mu) / sd);
	}
	for (int i = 0; i < L.n; i++) unit(E + (size_t)i * d, d);
	eval_neighbours(E, d, "mel-stats baseline");
	free(E);
}

static void eval_encoder(enc *e, const char *tag)
{
	eval_retrieval(e, tag);
	float *E = malloc((size_t)L.n * ENC_H * sizeof(float));
	if (!embed_songs(e, E, NULL, NULL)) eval_neighbours(E, ENC_H, tag);
	free(E);
}

/*
 * emb.bin: "VIBEEMB1", int32 n, int32 d (256), int64 model step, then n records of { int64 mel record, float mean[d],
 * float first[d], float last[d] }; emb.tsv: one path per line in the same order.
 */
static int write_emb(enc *e)
{
	float *M = malloc((size_t)L.n * ENC_H * 4), *F = malloc((size_t)L.n * ENC_H * 4), *Ls = malloc((size_t)L.n * ENC_H * 4);
	int rc = embed_songs(e, M, F, Ls);
	if (!rc) {
		char tmp[4300];
		snprintf(tmp, sizeof tmp, "%s.tmp", emb_path);
		FILE *f = fopen(tmp, "wb");
		int32_t hd[2] = { L.n, ENC_H };
		fwrite("VIBEEMB1", 1, 8, f);
		fwrite(hd, 4, 2, f);
		fwrite(&e->step, 8, 1, f);
		for (int i = 0; i < L.n; i++) {
			fwrite(&L.s[i].rec, 8, 1, f);
			fwrite(M + (size_t)i * ENC_H, 4, ENC_H, f);
			fwrite(F + (size_t)i * ENC_H, 4, ENC_H, f);
			fwrite(Ls + (size_t)i * ENC_H, 4, ENC_H, f);
		}
		rc = ferror(f) | fclose(f);
		char tsv[4300], tsvtmp[4310];
		snprintf(tsv, sizeof tsv, "%.*s.tsv", (int)(strlen(emb_path) - 4), emb_path);
		snprintf(tsvtmp, sizeof tsvtmp, "%s.tmp", tsv);
		FILE *g = fopen(tsvtmp, "w");
		for (int i = 0; i < L.n; i++) fprintf(g, "%s\n", L.s[i].rel);
		rc |= ferror(g) | fclose(g);
		char bintmp[4300];
		snprintf(bintmp, sizeof bintmp, "%s.tmp", emb_path);
		if (!rc) rc = rename(bintmp, emb_path) | rename(tsvtmp, tsv);
		printf("  wrote %s (%d songs) and %s: %s\n", emb_path, L.n, tsv, rc ? "FAILED" : "ok");
	}
	free(M); free(F); free(Ls);
	return rc;
}

int main(int argc, char **argv)
{
	int epochs = 60, per = 256, every = 10, resume = 0, only_embed = 0, no_emb = 0, opt;
	float lr = 5e-4f, tau = 0.1f;
	while ((opt = getopt(argc, argv, "e:s:l:t:v:rEx")) != -1) {
		if (opt == 'e') epochs = atoi(optarg);
		else if (opt == 's') per = atoi(optarg);
		else if (opt == 'l') lr = strtof(optarg, NULL);
		else if (opt == 't') tau = strtof(optarg, NULL);
		else if (opt == 'v') every = atoi(optarg);
		else if (opt == 'r') resume = 1;
		else if (opt == 'E') only_embed = 1;
		else if (opt == 'x') no_emb = 1;
		else { fprintf(stderr, "usage: vibe-train [-e EPOCHS] [-s SONGS_PER_BATCH] [-l LR] [-t TAU] [-v EVAL_EVERY] [-r] [-E] [-x]\n"); return 2; }
	}
	setvbuf(stdout, NULL, _IOLBF, 0);
	snprintf(dir, sizeof dir, "%s/.local/share/phivibe", getenv("HOME"));
	mkdir(dir, 0755);
	snprintf(model_path, sizeof model_path, "%s/enc.bin", dir);
	snprintf(emb_path, sizeof emb_path, "%s/emb.bin", dir);
	if (songs_load(&L, NULL)) return 1;
	int ntrain = 0, nheld = 0;
	for (int i = 0; i < L.n; i++) { if (held_out(i)) nheld++; else ntrain++; }
	printf("vibe-train: %d songs (%d train, %d held out), %d folders, %d artists\n", L.n, ntrain, nheld, L.nfolders, L.nartists);
	V = vg_open(NULL, 0, 10.0);
	if (!V) return 1;
	enc *e = resume || only_embed ? enc_load(model_path) : NULL;
	if ((resume || only_embed) && !e) { fprintf(stderr, "vibe-train: no model at %s\n", model_path); return 1; }
	if (!e) { e = enc_new(0x5eed1234); norm_stats(e); }
	if (enc_bind(e, V, 2 * per)) { fprintf(stderr, "vibe-train: the card's arena is too small for %d crops\n", 2 * per); return 1; }
	if (only_embed) { eval_encoder(e, "saved model"); return no_emb ? 0 : write_emb(e) != 0; }

	if (!resume) { eval_baseline(); eval_encoder(e, "untrained encoder"); }
	int *train = malloc((size_t)ntrain * sizeof(int));
	for (int i = 0, k = 0; i < L.n; i++) if (!held_out(i)) train[k++] = i;
	const uint8_t **ptr = malloc((size_t)2 * per * sizeof *ptr);
	float *gain = malloc((size_t)2 * per * sizeof(float));
	uint8_t *rolled = malloc((size_t)per * 128 * 64);
	vg_stats s0, s1;
	for (int ep = 1; ep <= epochs; ep++) {
		for (int i = ntrain - 1; i > 0; i--) { int j = (int)(xr() % (uint64_t)(i + 1)), t = train[i]; train[i] = train[j]; train[j] = t; }
		const int steps = ntrain / per;
		double lsum = 0, asum = 0, t0 = now();
		vg_get_stats(V, &s0);
		for (int st = 0; st < steps; st++) {
			for (int j = 0; j < per; j++) {
				const int i = train[st * per + j], nc = L.s[i].ncrops;
				const int c0 = (int)(xr() % (uint64_t)nc);
				ptr[2 * j] = songs_crop(&L, i, c0);
				if (nc > 1) {
					int c1 = (int)(xr() % (uint64_t)(nc - 1));
					if (c1 >= c0) c1++;
					ptr[2 * j + 1] = songs_crop(&L, i, c1);
				} else {
					/* one crop only: the same crop rolled in time by 8..120 frames */
					const int r = 8 + (int)(xr() % 113);
					uint8_t *o = rolled + (size_t)j * 128 * 64;
					for (int f = 0; f < 128; f++) memcpy(o + f * 64, ptr[2 * j] + ((f + r) % 128) * 64, 64);
					ptr[2 * j + 1] = o;
				}
				gain[2 * j] = (float)(ur() * 18 - 9);       /* +-9 codes = +-3.9 dB */
				gain[2 * j + 1] = (float)(ur() * 18 - 9);
			}
			enc_input(e, ptr, 2 * per, gain);
			double acc = 0;
			const double l = enc_train_step(e, tau, lr, &acc);
			if (isnan(l)) { fprintf(stderr, "vibe-train: step failed\n"); return 1; }
			lsum += l;
			asum += acc;
		}
		vg_get_stats(V, &s1);
		const double dt = now() - t0;
		printf("epoch %3d: loss %.4f, pair top-1 %.4f, %d steps %.2f s (%.1f ms/step), card GEMM %.1f GFLOP in %.2f s = %.1f GFLOP/s, DMA %.1f MB in %.2f s\n",
		       ep, lsum / steps, asum / steps, steps, dt, 1e3 * dt / steps, (s1.flop - s0.flop) / 1e9, (s1.gemm_ns - s0.gemm_ns) / 1e9,
		       (double)(s1.flop - s0.flop) / (double)(s1.gemm_ns - s0.gemm_ns), (s1.dma_bytes - s0.dma_bytes) / 1e6, (s1.dma_ns - s0.dma_ns) / 1e9);
		if (enc_save(e, model_path)) fprintf(stderr, "vibe-train: save %s: %s\n", model_path, strerror(errno));
		if (every > 0 && (ep % every == 0 || ep == epochs)) {
			char tag[64];
			snprintf(tag, sizeof tag, "epoch %d", ep);
			eval_encoder(e, tag);
		}
	}
	if (!no_emb) write_emb(e);
	vg_close(V);
	return 0;
}
