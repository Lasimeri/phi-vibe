/* vg.c: see vg.h and lib/vg-proto.h. */
#define _GNU_SOURCE
#include "vg.h"
#include "../lib/vg-proto.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

typedef struct { float *dst; int ld, rows, cols, tld; uint64_t data; } getrec;

struct vg {
	uint8_t *part;
	uint64_t session, posted;
	uint64_t arena, top;          /* arena bytes; the allocator's next free offset */
	vg_op *ops;                   /* the queued list (host copy) */
	int nops, cap;
	uint64_t data;                /* next free byte of the data area (part offset) */
	getrec *gets;
	int ngets, capgets;
	int times;
	uint64_t heart;
	double heart_seen;
};

static double now(void)
{
	struct timespec s;
	clock_gettime(CLOCK_MONOTONIC, &s);
	return s.tv_sec + s.tv_nsec / 1e9;
}

#define LD32(v, o) __atomic_load_n((uint32_t *)((v)->part + (o)), __ATOMIC_ACQUIRE)
#define LD64(v, o) __atomic_load_n((uint64_t *)((v)->part + (o)), __ATOMIC_ACQUIRE)
#define ST32(v, o, x) __atomic_store_n((uint32_t *)((v)->part + (o)), (x), __ATOMIC_RELEASE)
#define ST64(v, o, x) __atomic_store_n((uint64_t *)((v)->part + (o)), (x), __ATOMIC_RELEASE)

int vg_alive(vg *v)
{
	uint64_t h = LD64(v, VG_C_HEART);
	double t = now();
	if (h != v->heart) { v->heart = h; v->heart_seen = t; }
	return t - v->heart_seen < 5.0 && LD64(v, VG_C_SESSION) == v->session && LD32(v, VG_C_STATE) == 1;
}

vg *vg_open(const char *path, unsigned part_mib, double timeout_s)
{
	int fd = open(path ? path : "/dev/shm/phi-hostmem", O_RDWR);
	if (fd < 0) { perror("vg: /dev/shm/phi-hostmem"); return NULL; }
	off_t len = lseek(fd, 0, SEEK_END);
	off_t off = (off_t)(part_mib ? part_mib : VG_PART_MIB) << 20;
	if (len < off + (off_t)VG_PART_SIZE) { fprintf(stderr, "vg: region %lld B has no part at %lld\n", (long long)len, (long long)off); close(fd); return NULL; }
	uint8_t *m = mmap(NULL, VG_PART_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, off);
	close(fd);
	if (m == MAP_FAILED) { perror("vg: mmap"); return NULL; }
	vg *v = calloc(1, sizeof *v);
	v->part = m;
	v->cap = 1024;
	v->ops = calloc((size_t)v->cap, sizeof *v->ops);
	v->data = VG_DATA;
	ST32(v, VG_H_MAGIC, VG_MAGIC);
	ST32(v, VG_H_VERSION, VG_VERSION);
	ST64(v, VG_H_POSTED, 0);
	struct timespec t;
	clock_gettime(CLOCK_REALTIME, &t);
	v->session = (uint64_t)t.tv_sec * 1000000u + (uint64_t)t.tv_nsec / 1000u;
	ST64(v, VG_H_SESSION, v->session);
	v->heart = LD64(v, VG_C_HEART);
	v->heart_seen = now();
	double t0 = now();
	while (LD64(v, VG_C_SESSION) != v->session || LD32(v, VG_C_STATE) != 1) {
		if (now() - t0 > timeout_s) { fprintf(stderr, "vg: the card did not take the session within %.1f s (is vgd running?)\n", timeout_s); vg_close(v); return NULL; }
		usleep(1000);
	}
	v->arena = LD64(v, VG_C_ARENA);
	return v;
}

void vg_close(vg *v)
{
	if (!v) return;
	munmap(v->part, VG_PART_SIZE);
	free(v->ops);
	free(v->gets);
	free(v);
}

uint64_t vg_arena(vg *v) { return v->arena; }
uint64_t vg_mark(vg *v) { return v->top; }
void vg_release(vg *v, uint64_t mark) { if (mark <= v->top) v->top = mark; }

static uint64_t tbytes(vt t) { return (uint64_t)((t.rows + 7) / 8 * 8) * (uint64_t)t.ld * 4; }

vt vg_tensor(vg *v, int rows, int cols)
{
	vt t = { UINT64_MAX, rows, cols, (cols + 15) / 16 * 16 };
	if (t.ld == 0) t.ld = 16;
	const uint64_t n = tbytes(t);
	if (v->top + n > v->arena) { fprintf(stderr, "vg: arena full (%llu + %llu of %llu)\n", (unsigned long long)v->top, (unsigned long long)n, (unsigned long long)v->arena); return t; }
	t.off = v->top;
	v->top += (n + 63) / 64 * 64;
	return t;
}

/* room for one more op and n bytes of staging: posts the queued list first when either is full */
static void room(vg *v, uint64_t n)
{
	if (v->nops >= (int)VG_OPS_MAX || v->data + (n + 63) / 64 * 64 > VG_PART_SIZE) vg_sync(v);
}

static vg_op *add(vg *v)
{
	if (v->nops == v->cap) { v->cap *= 2; v->ops = realloc(v->ops, (size_t)v->cap * sizeof *v->ops); }
	vg_op *o = &v->ops[v->nops++];
	memset(o, 0, sizeof *o);
	return o;
}

/* n bytes of staging (room() made space): the part offset */
static uint64_t stage(vg *v, uint64_t n)
{
	uint64_t at = v->data;
	v->data += (n + 63) / 64 * 64;
	return at;
}

/* the largest whole number of rows (multiple of 8) that fits the data area */
static int chunk_rows(vt t)
{
	int r = (int)(VG_DATA_SIZE / ((uint64_t)t.ld * 4)) / 8 * 8;
	return r < 8 ? 8 : r;
}

void vg_put(vg *v, vt t, const float *src, int ld)
{
	const int rows8 = (t.rows + 7) / 8 * 8, cr = chunk_rows(t);
	for (int r0 = 0; r0 < rows8; r0 += cr) {
		const int nr = rows8 - r0 < cr ? rows8 - r0 : cr;
		const uint64_t n = (uint64_t)nr * t.ld * 4;
		room(v, n);
		const uint64_t at = stage(v, n);
		float *d = (float *)(v->part + at);
		for (int r = 0; r < nr; r++) {
			float *row = d + (size_t)r * t.ld;
			const int gr = r0 + r;
			if (gr < t.rows) {
				memcpy(row, src + (size_t)gr * ld, (size_t)t.cols * 4);
				memset(row + t.cols, 0, (size_t)(t.ld - t.cols) * 4);
			} else memset(row, 0, (size_t)t.ld * 4);
		}
		vg_op *o = add(v);
		o->code = VG_PUT;
		o->a = at;
		o->c = t.off + (uint64_t)r0 * t.ld * 4;
		o->m = (uint32_t)n;
	}
}

void vg_get(vg *v, vt t, float *dst, int ld)
{
	const int cr = chunk_rows(t);
	for (int r0 = 0; r0 < t.rows; r0 += cr) {
		const int nr = t.rows - r0 < cr ? t.rows - r0 : cr;
		const uint64_t n = ((uint64_t)nr * t.ld * 4 + 63) / 64 * 64;
		room(v, n);
		const uint64_t at = stage(v, n);
		vg_op *o = add(v);
		o->code = VG_GET;
		o->a = t.off + (uint64_t)r0 * t.ld * 4;
		o->c = at;
		o->m = (uint32_t)n;
		if (v->ngets == v->capgets) { v->capgets = v->capgets ? 2 * v->capgets : 64; v->gets = realloc(v->gets, (size_t)v->capgets * sizeof *v->gets); }
		v->gets[v->ngets++] = (getrec){ dst + (size_t)r0 * ld, ld, nr, t.cols, t.ld, at };
	}
}

void vg_zero(vg *v, vt t)
{
	room(v, 0);
	vg_op *o = add(v);
	o->code = VG_ZERO;
	o->c = t.off;
	o->m = (uint32_t)tbytes(t);
}

void vg_trans(vg *v, vt c, vt a)
{
	room(v, 0);
	vg_op *o = add(v);
	o->code = VG_TRANS;
	o->a = a.off;
	o->lda = (uint32_t)a.ld;
	o->m = (uint32_t)a.rows;
	o->n = (uint32_t)a.cols;
	o->c = c.off;
	o->ldc = (uint32_t)c.ld;
}

void vg_gemm(vg *v, vt c, vt a, vt b, const vt *bias, const vt *z, float alpha, int flags)
{
	room(v, 0);
	vg_op *o = add(v);
	o->code = VG_GEMM;
	o->flags = (uint32_t)(flags & (VG_TA | VG_BETA | VG_RELU));
	o->m = (uint32_t)c.rows;
	o->n = (uint32_t)c.cols;
	o->k = (uint32_t)(flags & VG_TA ? a.rows : a.cols);
	o->alpha = alpha;
	o->a = a.off; o->lda = (uint32_t)a.ld;
	o->b = b.off; o->ldb = (uint32_t)b.ld;
	o->c = c.off; o->ldc = (uint32_t)c.ld;
	if (bias) { o->flags |= VG_BIAS; o->bias = bias->off; }
	if (z) { o->flags |= VG_MASK; o->z = z->off; o->ldz = (uint32_t)z->ld; }
}

void vg_times(vg *v, int on) { v->times = on; }

int vg_op_ns(vg *v, int i, double *ns)
{
	if (i < 0 || (uint64_t)i * 4 >= VG_OPS - VG_TIMES) return -1;
	*ns = LD32(v, VG_TIMES + 4 * i) * 64.0;
	return 0;
}

int vg_sync(vg *v)
{
	if (!v->nops) return 0;
	memcpy(v->part + VG_OPS, v->ops, (size_t)v->nops * sizeof *v->ops);
	ST32(v, VG_H_NOPS, (uint32_t)v->nops);
	ST32(v, VG_H_TIMES, (uint32_t)v->times);
	const uint64_t n = ++v->posted;
	__atomic_thread_fence(__ATOMIC_SEQ_CST);
	ST64(v, VG_H_POSTED, n);
	v->nops = 0;
	v->data = VG_DATA;
	int spins = 0;
	while (LD64(v, VG_C_DONE) < n) {
		if (++spins > 2000) {
			if (!vg_alive(v)) { fprintf(stderr, "vg: the card stopped answering (list %llu)\n", (unsigned long long)n); v->ngets = 0; return -1; }
			usleep(50);
		}
	}
	const int err = (int)LD32(v, VG_C_ERR);
	if (err) fprintf(stderr, "vg: list %llu: op %u refused, error %d\n", (unsigned long long)n, LD32(v, VG_C_ERROP), err);
	for (int i = 0; i < v->ngets && !err; i++) {
		const getrec *g = &v->gets[i];
		for (int r = 0; r < g->rows; r++)
			memcpy(g->dst + (size_t)r * g->ld, v->part + g->data + (size_t)r * g->tld * 4, (size_t)g->cols * 4);
	}
	v->ngets = 0;
	return err;
}

void vg_get_stats(vg *v, vg_stats *s)
{
	s->lists = LD64(v, VG_C_LISTS);
	s->ops = LD64(v, VG_C_OPS);
	s->flop = LD64(v, VG_C_FLOP);
	s->busy_ns = LD64(v, VG_C_BUSY_NS);
	s->gemm_ns = LD64(v, VG_C_GEMM_NS);
	s->dma_ns = LD64(v, VG_C_DMA_NS);
	s->dma_bytes = LD64(v, VG_C_DMA_BYTES);
	s->vpu = LD64(v, VG_C_VPU);
	s->list_ns = LD64(v, VG_C_LIST_NS);
}
