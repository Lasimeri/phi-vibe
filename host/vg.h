/* vg.h: the host side of the card's GEMM daemon (card/vgd.c, lib/vg-proto.h). The host lays out the card's tensor
 * arena (a bump allocator), queues ops, and vg_sync posts the list and waits for it. Tensors are row-major float
 * matrices padded to 8-row multiples and 16-float row strides; the padding stays zero (PUT writes it zero, ops never
 * write past rows x cols). */
#ifndef VG_H
#define VG_H

#include <stddef.h>
#include <stdint.h>

typedef struct vg vg;
typedef struct { uint64_t off; int rows, cols, ld; } vt;

/* path NULL: /dev/shm/phi-hostmem; part_mib 0: the default. Waits up to timeout_s for the card to take the session.
 * NULL with a reason on stderr. */
vg *vg_open(const char *path, unsigned part_mib, double timeout_s);
void vg_close(vg *v);
int vg_alive(vg *v);                     /* 1 while the card's heartbeat moves */

vt vg_tensor(vg *v, int rows, int cols); /* in the arena; off == UINT64_MAX when it is full */
uint64_t vg_mark(vg *v);                 /* the allocator's position, for vg_release */
void vg_release(vg *v, uint64_t mark);   /* frees every tensor allocated after mark */
uint64_t vg_arena(vg *v);                /* arena bytes on the card */

/* queued ops; vg_sync runs them. src/dst have ld floats between rows. */
void vg_put(vg *v, vt t, const float *src, int ld);
void vg_get(vg *v, vt t, float *dst, int ld);
void vg_zero(vg *v, vt t);
void vg_trans(vg *v, vt c, vt a);        /* c = a^T */

#define VGF_TA 1      /* A is stored transposed (K x M) */
#define VGF_BETA 2    /* C += */
#define VGF_RELU 8
/* C = alpha op(A) B (+ C) (+ bias) (relu) (masked where Z <= 0); bias and Z may be NULL */
void vg_gemm(vg *v, vt c, vt a, vt b, const vt *bias, const vt *z, float alpha, int flags);

/* posts the queued ops and waits; 0, or the card's error (VG_E_*), or -1 (timeout, card gone) */
int vg_sync(vg *v);

/* per-op times of the last list (vg_times(v, 1) before the ops are queued), nanoseconds */
void vg_times(vg *v, int on);
int vg_op_ns(vg *v, int i, double *ns);

typedef struct { uint64_t lists, ops, flop, busy_ns, gemm_ns, dma_ns, dma_bytes, vpu, list_ns; } vg_stats;
void vg_get_stats(vg *v, vg_stats *s);

#endif
