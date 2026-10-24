/* knn.c: see knn.h. The similarity products run on the card, the top-k selection on the host. */
#include "knn.h"

#include <stdlib.h>
#include <string.h>

#define BLOCK 512

int knn_card(vg *v, const float *E, int n, int d, int k, int *nb, float *sim)
{
	const uint64_t mark = vg_mark(v);
	vt tE = vg_tensor(v, n, d), tET = vg_tensor(v, d, n), S = vg_tensor(v, BLOCK, n);
	if (tE.off == UINT64_MAX || tET.off == UINT64_MAX || S.off == UINT64_MAX) { vg_release(v, mark); return -1; }
	vg_put(v, tE, E, d);
	vg_trans(v, tET, tE);
	float *s = malloc((size_t)BLOCK * n * sizeof(float));
	int rc = 0;
	for (int r0 = 0; r0 < n && !rc; r0 += BLOCK) {
		const int b = n - r0 < BLOCK ? n - r0 : BLOCK;
		vt A = tE, C = S;
		A.off = tE.off + (uint64_t)r0 * tE.ld * 4;
		A.rows = b;
		C.rows = b;
		vg_gemm(v, C, A, tET, NULL, NULL, 1.0f, 0);
		vg_get(v, C, s, n);
		if ((rc = vg_sync(v))) break;
		for (int i = 0; i < b; i++) {
			const float *row = s + (size_t)i * n;
			int *bi = nb + (size_t)(r0 + i) * k;
			float *bs = sim + (size_t)(r0 + i) * k;
			int have = 0;
			for (int j = 0; j < n; j++) {
				if (j == r0 + i) continue;
				const float x = row[j];
				if (have == k && x <= bs[k - 1]) continue;
				int p = have < k ? have++ : k - 1;
				while (p > 0 && bs[p - 1] < x) { bs[p] = bs[p - 1]; bi[p] = bi[p - 1]; p--; }
				bs[p] = x;
				bi[p] = j;
			}
			for (int q = have; q < k; q++) { bi[q] = -1; bs[q] = -2.0f; }
		}
	}
	free(s);
	vg_release(v, mark);
	return rc;
}
