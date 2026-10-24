/* emb.c: see emb.h. */
#define _GNU_SOURCE
#include "emb.h"
#include "knn.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int emb_load(emb *E, const char *dir)
{
	memset(E, 0, sizeof *E);
	char path[4096];
	if (dir) snprintf(path, sizeof path, "%s/emb.bin", dir);
	else snprintf(path, sizeof path, "%s/.local/share/phivibe/emb.bin", getenv("HOME"));
	FILE *f = fopen(path, "rb");
	if (!f) { perror(path); return -1; }
	char magic[8];
	int32_t hd[2];
	if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "VIBEEMB1", 8) || fread(hd, 4, 2, f) != 2 || fread(&E->step, 8, 1, f) != 1) {
		fprintf(stderr, "%s: not an embedding file\n", path);
		fclose(f);
		return -1;
	}
	E->n = hd[0];
	E->d = hd[1];
	const size_t nd = (size_t)E->n * E->d;
	E->rec = malloc((size_t)E->n * 8);
	E->mean = malloc(nd * 4); E->first = malloc(nd * 4); E->last = malloc(nd * 4);
	int ok = 1;
	for (int i = 0; ok && i < E->n; i++)
		ok = fread(&E->rec[i], 8, 1, f) == 1 && fread(E->mean + (size_t)i * E->d, 4, (size_t)E->d, f) == (size_t)E->d &&
		     fread(E->first + (size_t)i * E->d, 4, (size_t)E->d, f) == (size_t)E->d &&
		     fread(E->last + (size_t)i * E->d, 4, (size_t)E->d, f) == (size_t)E->d;
	fclose(f);
	if (!ok) { fprintf(stderr, "%s: short\n", path); emb_free(E); return -1; }
	snprintf(path + strlen(path) - 4, 8, ".tsv");
	f = fopen(path, "r");
	if (!f) { perror(path); emb_free(E); return -1; }
	E->rel = calloc((size_t)E->n, sizeof(char *));
	char *line = NULL;
	size_t cap = 0;
	ssize_t len;
	int i = 0;
	while (i < E->n && (len = getline(&line, &cap, f)) > 0) {
		if (line[len - 1] == '\n') line[len - 1] = 0;
		E->rel[i++] = strdup(line);
	}
	free(line);
	fclose(f);
	if (i != E->n) { fprintf(stderr, "%s: %d paths for %d songs\n", path, i, E->n); emb_free(E); return -1; }
	return 0;
}

void emb_free(emb *E)
{
	for (int i = 0; E->rel && i < E->n; i++) free(E->rel[i]);
	free(E->rel); free(E->rec); free(E->mean); free(E->first); free(E->last);
	memset(E, 0, sizeof *E);
}

int emb_find(const emb *E, const char *rel)
{
	int lo = 0, hi = E->n - 1;
	while (lo <= hi) {
		const int mid = (lo + hi) / 2, c = strcmp(E->rel[mid], rel);
		if (!c) return mid;
		if (c < 0) lo = mid + 1; else hi = mid - 1;
	}
	return -1;
}

static int cmp_int(const void *a, const void *b) { return *(const int *)a - *(const int *)b; }

/* the CSR graph from each song's k neighbours (nb, n x k, -1 for none) plus the reverse edges, without duplicates */
static int graph_from_knn(int n, int k, const int *nb, int **off, int **adj)
{
	/* degree: own k plus every song that lists it */
	int *deg = calloc((size_t)n, sizeof(int));
	for (int a = 0; a < n; a++)
		for (int j = 0; j < k; j++) { const int b = nb[(size_t)a * k + j]; if (b >= 0) { deg[a]++; deg[b]++; } }
	int *o = malloc((size_t)(n + 1) * sizeof(int));
	o[0] = 0;
	for (int a = 0; a < n; a++) o[a + 1] = o[a] + deg[a];
	int *e = malloc((size_t)o[n] * sizeof(int)), *fill = calloc((size_t)n, sizeof(int));
	for (int a = 0; a < n; a++)
		for (int j = 0; j < k; j++) {
			const int b = nb[(size_t)a * k + j];
			if (b < 0) continue;
			e[o[a] + fill[a]++] = b;
			e[o[b] + fill[b]++] = a;
		}
	/* sort each list, drop duplicates, compact */
	int *o2 = malloc((size_t)(n + 1) * sizeof(int)), m = 0;
	for (int a = 0; a < n; a++) {
		qsort(e + o[a], (size_t)fill[a], sizeof(int), cmp_int);
		o2[a] = m;
		for (int q = 0; q < fill[a]; q++) {
			const int b = e[o[a] + q];
			if (b == a || (m > o2[a] && e[m - 1] == b)) continue;
			e[m++] = b;
		}
	}
	o2[n] = m;
	free(o); free(fill); free(deg);
	*off = o2;
	*adj = realloc(e, (size_t)m * sizeof(int));
	return 0;
}

int graph_build(vg *v, const float *X, int n, int d, int k, int **off, int **adj)
{
	int *nb = malloc((size_t)n * k * sizeof(int));
	float *sim = malloc((size_t)n * k * sizeof(float));
	int rc = knn_card(v, X, n, d, k, nb, sim);
	if (!rc) rc = graph_from_knn(n, k, nb, off, adj);
	free(nb);
	free(sim);
	return rc;
}

const char *vibe_root(void)
{
	static char buf[4096];
	const char *env = getenv("VIBE_LIBRARY"), *home = getenv("HOME") ? getenv("HOME") : ".";
	if (env && *env) return env;
	snprintf(buf, sizeof buf, "%s/.config/phivibe/library", home);
	FILE *f = fopen(buf, "r");
	if (f) {
		const int ok = fgets(buf, sizeof buf, f) != NULL;
		fclose(f);
		buf[strcspn(buf, "\r\n")] = 0;
		if (ok && *buf) return buf;
	}
	snprintf(buf, sizeof buf, "%s/Music", home);
	return buf;
}

/* CSLS (Lample et al. 2018, cross-domain similarity local scaling): 2 cos(a, b) - r_a - r_b, r = a song's mean cosine
 * to its k nearest songs. Songs near the centre of the vibe space (high r) stop being everyone's neighbour. */
int graph_build_csls(vg *v, const float *X, int n, int d, int k, int **off, int **adj)
{
	int *nb = malloc((size_t)n * k * sizeof(int));
	float *sim = malloc((size_t)n * k * sizeof(float)), *r = malloc((size_t)n * sizeof(float));
	if (knn_card(v, X, n, d, k, nb, sim)) { free(nb); free(sim); free(r); return -1; }
	for (int a = 0; a < n; a++) {
		double s = 0;
		for (int j = 0; j < k; j++) s += sim[(size_t)a * k + j];
		r[a] = (float)(s / k);
	}
	/* ranking a's candidates by 2 cos - r_a - r_b is ranking by cos - r_b / 2 */
	for (int a = 0; a < n; a++) r[a] *= 0.5f;
	const int rc = knn_card_bias(v, X, n, d, k, r, nb, sim);
	free(r);
	if (rc) { free(nb); free(sim); return -1; }
	const int out = graph_from_knn(n, k, nb, off, adj);
	free(nb); free(sim);
	return out;
}
