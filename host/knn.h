/* knn.h: nearest neighbours by dot product (cosine for unit rows), the products on the card. */
#ifndef KNN_H
#define KNN_H

#include "vg.h"

/* for each of the n rows of E (n x d), the k other rows with the largest dot product, best first, into nb (n x k)
 * and sim (n x k). 0, or the card's error. */
int knn_card(vg *v, const float *E, int n, int d, int k, int *nb, float *sim);

/* the same ranked by dot product minus colbias[j] (a per-candidate offset; NULL for none) */
int knn_card_bias(vg *v, const float *E, int n, int d, int k, const float *colbias, int *nb, float *sim);

#endif
