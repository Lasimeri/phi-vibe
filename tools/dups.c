/*
 * dups: how close each song's nearest other song is (cosine of the vibe vectors), to tell copies of one recording
 * (the same audio in two folders or formats) from different songs.
 *
 * usage: dups [-t THRESHOLD] [-l LIST]   prints the distribution of nearest-song cosines, the pairs at or above the
 *                                        threshold (default 0.995), and LIST of them with their paths
 */
#define _GNU_SOURCE
#include "../host/emb.h"
#include "../host/knn.h"
#include "../host/vg.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv)
{
	double thr = 0.995;
	int list = 20, opt;
	while ((opt = getopt(argc, argv, "t:l:")) != -1) {
		if (opt == 't') thr = atof(optarg);
		else if (opt == 'l') list = atoi(optarg);
		else { fprintf(stderr, "usage: dups [-t THRESHOLD] [-l LIST]\n"); return 2; }
	}
	emb E;
	if (emb_load(&E, NULL)) return 1;
	vg *v = vg_open(NULL, 0, 10.0);
	if (!v) return 1;
	int *nb = malloc((size_t)E.n * sizeof(int));
	float *sim = malloc((size_t)E.n * sizeof(float));
	if (knn_card(v, E.mean, E.n, E.d, 1, nb, sim)) return 1;
	const double edges[] = { 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9999, 1.01 };
	int hist[10] = { 0 };
	for (int i = 0; i < E.n; i++) {
		int b = 0;
		while (b < 9 && sim[i] >= edges[b]) b++;
		hist[b]++;
	}
	printf("dups: %d songs; nearest other song's cosine:\n  below %.4f: %d\n", E.n, edges[0], hist[0]);
	for (int b = 1; b < 10; b++) printf("  %.4f .. %.4f: %d\n", edges[b - 1], edges[b] > 1 ? 1.0 : edges[b], hist[b]);
	int shown = 0, pairs = 0;
	for (int i = 0; i < E.n; i++) {
		if (sim[i] < thr || nb[i] < i) continue;   /* each pair once (when mutual) */
		pairs++;
		if (shown++ < list) printf("  %.6f  %s\n            %s\n", sim[i], E.rel[i], E.rel[nb[i]]);
	}
	printf("pairs at or above %.4f: %d\n", thr, pairs);
	/* the pair from the skip test */
	const int a = emb_find(&E, "(0)MainPL/Collections/Infected_Mushroom/Pegboards - Hero (Infected Mushroom Remix).flac");
	const int b = emb_find(&E, "ytmusic/2BSorted/Pegbaord Nerds - Hero (Infected Mushroom Remix).flac");
	if (a >= 0 && b >= 0) {
		double s = 0;
		for (int k = 0; k < E.d; k++) s += (double)E.mean[(size_t)a * E.d + k] * E.mean[(size_t)b * E.d + k];
		printf("skip-test pair (Pegboards - Hero, two folders): cosine %.6f; nearest of the first: %.6f\n", s, sim[a]);
	}
	vg_close(v);
	return 0;
}
