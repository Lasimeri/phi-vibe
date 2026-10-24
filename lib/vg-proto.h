/*
 * vg-proto.h: the vibe GEMM daemon's part of the shared host-memory region (docs/protocol.md). The host maps it from
 * /dev/shm/phi-hostmem, the card from /dev/phihost (uncached, for the control fields) and through its DMA engine (the
 * op lists and tensor data). Little-endian; offsets are bytes from the part's start. Host-written and card-written
 * fields sit on separate 64-byte lines.
 *
 * The host owns the card's tensor arena: it places every tensor at an arena byte offset it chose, and sends op lists
 * (PUT data into the arena, GEMM, TRANSPOSE, GET data back into the region). The card runs a list front to back, one
 * op at a time across its thread pool, then raises DONE.
 */
#ifndef VG_PROTO_H
#define VG_PROTO_H

#include <stdint.h>

#define VG_MAGIC 0x47424956u        /* "VIBG" */
#define VG_VERSION 1
#define VG_PART_MIB 1792            /* default part offset in the 2 GiB region: its last 256 MiB */
#define VG_PART_SIZE (256u << 20)

/* written by the host */
#define VG_H_MAGIC 0                /* u32 */
#define VG_H_VERSION 4              /* u32 */
#define VG_H_SESSION 8              /* u64 raised by each host open */
#define VG_H_POSTED 64              /* u64 op lists posted this session; list n is the one at VG_OPS */
#define VG_H_NOPS 72                /* u32 ops in the posted list */
#define VG_H_TIMES 76               /* u32 1: write each op's time into VG_TIMES */

/* written by the card */
#define VG_C_SESSION 128            /* u64 the session the card has taken */
#define VG_C_STATE 136              /* u32 0 starting, 1 ready, 2 error */
#define VG_C_THREADS 140            /* u32 */
#define VG_C_ARENA 144              /* u64 arena bytes */
#define VG_C_DONE 192               /* u64 lists finished this session */
#define VG_C_ERR 200                /* i32 0, or the first failing op's error (VG_E_*) in the last list */
#define VG_C_ERROP 204              /* u32 that op's index */
#define VG_C_LIST_NS 208            /* u64 the last list's time on the card, DMA included */
#define VG_C_HEART 256              /* u64 counted up about every 100 ms */
/* totals since the daemon started */
#define VG_C_LISTS 320              /* u64 */
#define VG_C_OPS 328                /* u64 */
#define VG_C_FLOP 336               /* u64 floating-point operations of GEMMs (2 per multiply-add) */
#define VG_C_BUSY_NS 344            /* u64 time running lists */
#define VG_C_GEMM_NS 352            /* u64 of which GEMM */
#define VG_C_DMA_NS 360             /* u64 of which DMA (op list pulls, PUT, GET) */
#define VG_C_DMA_BYTES 368          /* u64 */
#define VG_C_VPU 376                /* u64 VPU instructions executed by the pool's threads (KNC event 0x2016, user) */

#define VG_TIMES (16u << 10)        /* u32 per op of the last list: nanoseconds / 64 (when VG_H_TIMES) */
#define VG_OPS (64u << 10)          /* the op list */
#define VG_OPS_MAX ((VG_DATA - VG_OPS) / sizeof(vg_op))
#define VG_DATA (1u << 20)          /* staging for PUT and GET: the rest of the part */
#define VG_DATA_SIZE (VG_PART_SIZE - VG_DATA)

enum {
	VG_PUT = 1,       /* arena[c ..] = data[a ..], m bytes (multiples of 64, both offsets too) */
	VG_GET = 2,       /* data[c ..] = arena[a ..], m bytes */
	VG_GEMM = 3,      /* C = alpha * op(A) B (+ C if VG_BETA) (+ bias) then act; see below */
	VG_TRANS = 4,     /* C[j][i] = A[i][j]: A is m x n with lda, C n x m with ldc */
	VG_ZERO = 5,      /* arena[c ..] = 0, m bytes */
};

/* GEMM: M x N output, K reduction. A at a: M x K with lda (A[i][k]), or with VG_TA stored K x M (A[k][i]). B at b:
 * K x N with ldb. C at c: M x N with ldc. Every ld is in floats, a multiple of 16; a, b, c, bias, z are 64-byte
 * aligned. Rows and columns past M and N are neither read into C nor written. */
#define VG_TA 1           /* A transposed (stored K x M) */
#define VG_BETA 2         /* add to C instead of overwriting it */
#define VG_BIAS 4         /* add bias[j] (N floats at bias) */
#define VG_RELU 8         /* max(x, 0) on the result */
#define VG_MASK 16        /* zero where Z <= 0 (Z at z, M x N with ldz): the ReLU's backward pass */

/* VG_E_*: the card's error for a refused op */
#define VG_E_OP 1         /* unknown op code */
#define VG_E_RANGE 2      /* an offset or size outside the arena or the data area */
#define VG_E_ALIGN 3      /* an offset, size or ld that breaks the rules above */
#define VG_E_DMA 4        /* a DMA copy timed out */

typedef struct {
	uint32_t code, flags;
	uint32_t m, n, k;
	float alpha, gamma;
	uint32_t lda, ldb, ldc, ldz;
	uint32_t pad0;
	uint64_t a, b, c, bias, z;
	uint8_t pad1[40];
} vg_op;                  /* 128 bytes */

#endif
