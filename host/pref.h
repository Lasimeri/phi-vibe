/*
 * pref.h: the preference network: what you would play now, and where your listening is heading. Trained online
 * from plays; every matrix product on the card (host/vg.h), the loss and Adam on the host.
 *
 * Context x (PREF_X floats): the song now playing (256), the last 8 plays' vibes weighted by the fraction of each that
 * was played (256) and by the fraction that was not (256), the time of day and week (4), how full that history is (1).
 *
 * Fraction head: for a candidate b with vibe e_b and recency r_b (1 / (1 + plays since b was last played),
 * exp(-hours since / 12)):
 *     h = relu(e_b A + x Wc + b1)      (64)
 *     u = h wo + r_b wr + bo,   f = sigmoid(u)   = the fraction of b it predicts you would play now
 * trained on every play with the fraction actually played as the target (squared error).
 *
 * Goal head: g = unit(e_cur + (x V) U), the vibe your next plays are heading to, trained on the next song's vibe,
 * weighted by how much of that song was played (1 - cos).
 *
 * Planner costs: node(b) = kappa (1 - f_b), goal(b) = beta angle(b, g), edge angle weight alpha = alpha_floor + ...;
 * kappa, beta and alpha are softplus of trained parameters (always >= 0), trained on the choices you make yourself:
 * when you play q instead of the queued p, q's one-step cost should be below p's (margin 0.1).
 */
#ifndef VIBE_PREF_H
#define VIBE_PREF_H

#include <stdint.h>
#include "vg.h"

#define PREF_D 256
#define PREF_X 784
#define PREF_H 64
#define PREF_G 16
#define PREF_ALPHA_FLOOR 0.5

typedef struct {
	float *w;          /* the parameters, PREF_NPARAMS floats in the order of the views below */
	float *m, *v;      /* Adam */
	float *gr;         /* gradients */
	long step;
	/* views into w */
	float *A, *Wc, *b1, *wo, *wr, *bo, *V, *U, *mix;   /* mix: alpha, beta, kappa raw */
	/* card */
	vg *card;
	int n;             /* songs whose vibes are on the card */
	vt tE, tA, tWc, tb1, two, twoT, tV, tU, tUT;
	uint64_t mark;
} pref;

typedef struct {
	float x[PREF_X];
	int song;          /* index into the vibes */
	float r[2];        /* recency inputs */
	float target;      /* fraction played, or -1 when none */
} pref_example;

pref *pref_new(uint64_t seed);
int pref_save(const pref *p, const char *path);
pref *pref_load(const char *path);
void pref_free(pref *p);
long pref_nparams(void);

/* puts the song vibes (n x 256) and the weights on the card; 0 or -1 */
int pref_bind(pref *p, vg *card, const float *E, int n);
int pref_upload(pref *p);   /* the weights again, after a change */

double pref_alpha(const pref *p);
double pref_beta(const pref *p);
double pref_kappa(const pref *p);

/* the context vector */
void pref_context(float *x, const float *e_cur, const float *const *hist_e, const float *hist_f, int nhist, double unix_time);

/* every song's predicted fraction (f, n floats) given x and the songs' recency (r, n x 2), the goal vibe g (256) and every
 * song's cosine to it (gcos, n floats) */
int pref_score(pref *p, const float *x, const float *e_cur, const float *r, float *f, float *g, float *gcos);

/* one Adam step. fraction examples (target >= 0), goal examples (x, the current vibe, the next vibe, its weight),
 * choice pairs (state song s, the user's q and the queued p, each with its recency, and the x then). Returns the
 * loss terms; lr is the Adam step. Gradients are left in p->gr. */
typedef struct { const float *x; const float *e_cur, *e_next; float w; } pref_goal_ex;
typedef struct { const float *x; const float *e_cur; int s, q, p; float rq[2], rp[2]; } pref_choice;
int pref_grad(pref *p, const pref_example *fr, int nfr, const pref_goal_ex *go, int ngo, const pref_choice *ch, int nch,
              double *loss_frac, double *loss_goal, double *loss_choice);
void pref_adam(pref *p, float lr, float decay);

#endif
