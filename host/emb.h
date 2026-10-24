/* emb.h: the song embeddings vibe-train writes (~/.local/share/phivibe/emb.bin + emb.tsv), and the vibe graph over
 * them (k nearest neighbours by cosine, made symmetric). */
#ifndef EMB_H
#define EMB_H

#include <stdint.h>
#include "vg.h"

typedef struct {
	int n, d;
	int64_t step;          /* the encoder's training step */
	int64_t *rec;          /* mel.bin record of each song */
	float *mean;           /* n x d unit vectors: the song's vibe */
	float *first, *last;   /* n x d: its first and last crop */
	char **rel;            /* paths under the library root, sorted */
} emb;

/* dir NULL: ~/.local/share/phivibe. 0, or -1 with a message. */
int emb_load(emb *E, const char *dir);
void emb_free(emb *E);
int emb_find(const emb *E, const char *rel);   /* binary search, -1 when absent */

/* each song's k nearest songs (products on the card) plus the reverse edges, without duplicates, in CSR form */
int graph_build(vg *v, const float *X, int n, int d, int k, int **off, int **adj);

/* the library root: $VIBE_LIBRARY, else the first line of ~/.config/phivibe/library, else ~/Music */
const char *vibe_root(void);

/* the same with CSLS-corrected similarity (2 cos(a, b) - r_a - r_b, r = mean cosine to the k nearest): fewer hubs */
int graph_build_csls(vg *v, const float *X, int n, int d, int k, int **off, int **adj);

#endif
