/* songs.c: see songs.h. */
#define _GNU_SOURCE
#include "songs.h"
#include "../lib/vibe.h"

#include <ctype.h>
#include <fcntl.h>
#include <search.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

uint64_t songs_hash(const char *s)
{
	uint64_t h = 1469598103934665603ull;   /* FNV-1a */
	for (; *s; s++) { h ^= (uint8_t)*s; h *= 1099511628211ull; }
	return h;
}

static int cmp_rel(const void *a, const void *b) { return strcmp(((const song *)a)->rel, ((const song *)b)->rel); }

/* index of name in a growing list of distinct strings (sorted later is not needed: linear map via hsearch) */
typedef struct { char **v; int n, cap; struct hsearch_data h; } strset;

static int intern(strset *S, const char *name)
{
	ENTRY e = { (char *)name, NULL }, *r = NULL;
	if (hsearch_r(e, FIND, &r, &S->h) && r) return (int)(intptr_t)r->data;
	if (S->n == S->cap) { S->cap = S->cap ? 2 * S->cap : 256; S->v = realloc(S->v, (size_t)S->cap * sizeof *S->v); }
	char *k = strdup(name);
	S->v[S->n] = k;
	e.key = k;
	e.data = (void *)(intptr_t)S->n;
	hsearch_r(e, ENTER, &r, &S->h);
	return S->n++;
}

int songs_load(songs *L, const char *dir)
{
	memset(L, 0, sizeof *L);
	char path[4096];
	if (dir) snprintf(path, sizeof path, "%s/index.tsv", dir);
	else snprintf(path, sizeof path, "%s/.cache/phivibe/lib/index.tsv", getenv("HOME"));
	FILE *f = fopen(path, "r");
	if (!f) { perror(path); return -1; }
	char *line = NULL;
	size_t cap = 0;
	ssize_t len;
	int scap = 0;
	while ((len = getline(&line, &cap, f)) > 0) {
		if (line[len - 1] == '\n') line[--len] = 0;
		char *fld[7];
		char *s = line;
		int nf = 0;
		while (nf < 6) {
			char *tab = strchr(s, '\t');
			if (!tab) break;
			*tab = 0;
			fld[nf++] = s;
			s = tab + 1;
		}
		if (nf != 6) continue;
		fld[6] = s;
		if (L->n == scap) { scap = scap ? 2 * scap : 8192; L->s = realloc(L->s, (size_t)scap * sizeof *L->s); }
		song *g = &L->s[L->n++];
		memset(g, 0, sizeof *g);
		g->rec = atoll(fld[0]);
		g->size = atoll(fld[1]);
		g->mtime = atoll(fld[2]);
		g->dur_ms = atoi(fld[3]);
		g->ncrops = atoi(fld[4]);
		if (g->ncrops > 32) g->ncrops = 32;
		char *p = fld[5];
		for (int c = 0; c < g->ncrops && *p; c++) { g->start_ms[c] = (int)strtol(p, &p, 10); if (*p == ',') p++; }
		g->rel = strdup(fld[6]);
	}
	free(line);
	fclose(f);
	/* newest record per path: the index is in write order, so keep the last one */
	for (int i = 0; i < L->n; i++) L->s[i].folder = i;   /* original order, as the tie-break */
	qsort(L->s, (size_t)L->n, sizeof *L->s, cmp_rel);
	int k = 0;
	for (int i = 0; i < L->n; i++) {
		if (k && !strcmp(L->s[k - 1].rel, L->s[i].rel)) {
			if (L->s[i].folder > L->s[k - 1].folder) { free(L->s[k - 1].rel); L->s[k - 1] = L->s[i]; }
			else free(L->s[i].rel);
			continue;
		}
		L->s[k++] = L->s[i];
	}
	L->n = k;
	/* folders and artists */
	strset F = { 0 }, A = { 0 };
	hcreate_r(16384, &F.h);
	hcreate_r(16384, &A.h);
	for (int i = 0; i < L->n; i++) {
		song *g = &L->s[i];
		char d[4096];
		snprintf(d, sizeof d, "%s", g->rel);
		char *slash = strrchr(d, '/');
		const char *base = slash ? slash + 1 : d;
		char name[4096];
		snprintf(name, sizeof name, "%s", base);
		if (slash) *slash = 0; else d[0] = 0;
		g->folder = intern(&F, d);
		char *sep = strstr(name, " - ");
		g->artist = -1;
		if (sep && sep > name) {
			*sep = 0;
			for (char *c = name; *c; c++) *c = (char)tolower((unsigned char)*c);
			g->artist = intern(&A, name);
		}
	}
	L->folders = F.v; L->nfolders = F.n;
	L->artists = A.v; L->nartists = A.n;
	hdestroy_r(&F.h);
	hdestroy_r(&A.h);
	/* mel.bin */
	if (dir) snprintf(path, sizeof path, "%s/mel.bin", dir);
	else snprintf(path, sizeof path, "%s/.cache/phivibe/lib/mel.bin", getenv("HOME"));
	int fd = open(path, O_RDONLY);
	struct stat sb;
	if (fd < 0 || fstat(fd, &sb)) { perror(path); return -1; }
	L->mel_size = (uint64_t)sb.st_size;
	L->mel = mmap(NULL, L->mel_size, PROT_READ, MAP_SHARED, fd, 0);
	close(fd);
	if (L->mel == MAP_FAILED) { perror("mmap mel.bin"); return -1; }
	/* drop entries whose record is past the end (a run still writing) */
	k = 0;
	for (int i = 0; i < L->n; i++) {
		if ((uint64_t)(L->s[i].rec + 1) * MEL_REC <= L->mel_size && L->s[i].ncrops > 0) L->s[k++] = L->s[i];
		else free(L->s[i].rel);
	}
	L->n = k;
	return 0;
}

void songs_free(songs *L)
{
	for (int i = 0; i < L->n; i++) free(L->s[i].rel);
	for (int i = 0; i < L->nfolders; i++) free(L->folders[i]);
	for (int i = 0; i < L->nartists; i++) free(L->artists[i]);
	free(L->s); free(L->folders); free(L->artists);
	if (L->mel && L->mel != MAP_FAILED) munmap((void *)L->mel, L->mel_size);
	memset(L, 0, sizeof *L);
}

const uint8_t *songs_crop(const songs *L, int i, int c)
{
	return L->mel + (uint64_t)L->s[i].rec * MEL_REC + (uint64_t)c * MEL_CROP_BYTES;
}

int songs_find(const songs *L, const char *rel)
{
	int lo = 0, hi = L->n - 1;
	while (lo <= hi) {
		int mid = (lo + hi) / 2, c = strcmp(L->s[mid].rel, rel);
		if (!c) return mid;
		if (c < 0) lo = mid + 1; else hi = mid - 1;
	}
	return -1;
}
