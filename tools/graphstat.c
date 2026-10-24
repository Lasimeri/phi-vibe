/*
 * graphstat: how even the vibe graph is. With k nearest neighbours plus the reverse edges, a song's degree is k plus
 * the number of songs that list it; "hubs" are songs many others list (songs near the centre of the vibe space).
 *
 * usage: graphstat [-k NEIGHBOURS] [-c]   (-c: build with CSLS-corrected similarity, 2 cos(a,b) - r_a - r_b, r = a
 *                                          song's mean cosine to its k nearest songs)
 * Prints the degree distribution, the top hubs, and each top folder's share of songs against its share of edges.
 */
#define _GNU_SOURCE
#include "../host/emb.h"
#include "../host/knn.h"
#include "../host/vg.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int cmpi(const void *a, const void *b) { return *(const int *)a - *(const int *)b; }

int main(int argc, char **argv)
{
	int K = 16, csls = 0, opt;
	while ((opt = getopt(argc, argv, "k:c")) != -1) {
		if (opt == 'k') K = atoi(optarg);
		else if (opt == 'c') csls = 1;
		else { fprintf(stderr, "usage: graphstat [-k NEIGHBOURS] [-c]\n"); return 2; }
	}
	emb E;
	if (emb_load(&E, NULL)) return 1;
	vg *v = vg_open(NULL, 0, 10.0);
	if (!v) return 1;
	int *off, *adj;
	if (csls) {
		if (graph_build_csls(v, E.mean, E.n, E.d, K, &off, &adj)) return 1;
	} else if (graph_build(v, E.mean, E.n, E.d, K, &off, &adj)) return 1;
	int *deg = malloc((size_t)E.n * sizeof(int)), *srt = malloc((size_t)E.n * sizeof(int));
	for (int i = 0; i < E.n; i++) srt[i] = deg[i] = off[i + 1] - off[i];
	qsort(srt, (size_t)E.n, sizeof(int), cmpi);
	printf("graphstat: %d songs, k %d%s: %d edges; degree min %d, median %d, p90 %d, p99 %d, max %d\n", E.n, K, csls ? " (CSLS)" : "", off[E.n],
	       srt[0], srt[E.n / 2], srt[E.n * 9 / 10], srt[E.n * 99 / 100], srt[E.n - 1]);
	printf("top hubs:\n");
	for (int t = 0; t < 10; t++) {
		int best = -1;
		for (int i = 0; i < E.n; i++) if (deg[i] >= 0 && (best < 0 || deg[i] > deg[best])) best = i;
		printf("  %4d  %s\n", deg[best], E.rel[best]);
		deg[best] = -1;
	}
	for (int i = 0; i < E.n; i++) deg[i] = off[i + 1] - off[i];
	/* top folders: share of songs vs share of edge ends */
	char names[64][256];
	long songs_in[64] = { 0 }, ends_in[64] = { 0 };
	int nn = 0;
	for (int i = 0; i < E.n; i++) {
		char top[256];
		snprintf(top, sizeof top, "%s", E.rel[i]);
		char *sl = strchr(top, '/');
		if (sl) *sl = 0; else snprintf(top, sizeof top, "(root)");
		int f = -1;
		for (int q = 0; q < nn; q++) if (!strcmp(names[q], top)) f = q;
		if (f < 0 && nn < 64) { f = nn++; snprintf(names[f], sizeof names[f], "%s", top); }
		if (f >= 0) { songs_in[f]++; ends_in[f] += deg[i]; }
	}
	printf("top folder: share of songs -> share of edge ends (degree mass)\n");
	for (int q = 0; q < nn; q++)
		printf("  %-24s %5.1f %% -> %5.1f %%\n", names[q], 100.0 * songs_in[q] / E.n, 100.0 * ends_in[q] / off[E.n]);
	vg_close(v);
	return 0;
}
