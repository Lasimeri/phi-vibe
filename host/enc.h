/*
 * enc.h: the vibe encoder. A crop of 128 mel frames x 64 bands goes through 7 merge layers (two neighbouring frames
 * concatenated, then a dense layer with ReLU: 128 frames -> 1, 64 -> 256 -> ... -> 256 channels) to the crop's vibe
 * vector h (256). A projection head (256 -> 256 ReLU -> 128) gives z, used only by the contrastive training: two crops
 * of the same song should land near each other and far from the other songs' crops (NT-Xent).
 *
 * Every matrix product runs on the card (host/vg.h); the host keeps the weights' master copy, the input
 * normalisation, the loss and the optimiser (Adam).
 */
#ifndef VIBE_ENC_H
#define VIBE_ENC_H

#include <stdint.h>
#include "vg.h"

#define ENC_LAYERS 9          /* 7 merge layers + 2 head layers */
#define ENC_MERGE 7
#define ENC_H 256             /* the vibe vector */
#define ENC_Z 128             /* the projection */

typedef struct {
	int cin, cout, relu;
	float *W, *b;             /* cin x cout, cout */
	float *mW, *vW, *mb, *vb; /* Adam moments */
	float *gW, *gb;           /* the last step's gradients */
	vt tW, tWt, tb, tgW, tgb; /* on the card */
} enc_layer;

typedef struct {
	enc_layer L[ENC_LAYERS];
	float mean[64], inv_std[64];   /* input normalisation per band, from the u8 dB codes */
	long step;
	/* card state for a batch of `batch` crops */
	vg *v;
	int batch;
	vt X, Y[ENC_LAYERS], dY[ENC_LAYERS], ones, Zn, ZnT, S, D, dZn;
	uint64_t mark;
	float *xbuf;              /* batch x 128 x 64 floats */
} enc;

/* new random weights (He), or NULL */
enc *enc_new(uint64_t seed);
int enc_save(const enc *e, const char *path);
enc *enc_load(const char *path);
void enc_free(enc *e);

/* binds the model to the card for batches of `batch` crops (a multiple of 8): allocates the activations, uploads
 * the weights. 0, or -1 when the arena is too small. */
int enc_bind(enc *e, vg *v, int batch);

/* u8 crops (n x 128 x 64, n <= batch) into the normalised input; gain shifts the dB codes per crop (NULL: none) */
void enc_input(enc *e, const uint8_t *const *crops, int n, const float *gain);

/* forward to h (n x 256) and z (n x 128, NULL to skip) for the crops in the input */
int enc_forward(enc *e, int n, float *h, float *z);

/* one training step on a batch of pairs: crops 2i and 2i+1 are the same song. Returns the loss; *acc gets the
 * fraction of crops whose nearest other crop is their partner. tau: temperature, lr: Adam step. */
double enc_train_step(enc *e, float tau, float lr, double *acc);

#endif
