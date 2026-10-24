/*
 * vgd: the vibe GEMM daemon on the card. Runs op lists from its part of the shared host-memory region
 * (lib/vg-proto.h): data comes in and goes out through a card-owned DMA channel, GEMMs and transposes run on the
 * vector unit (lib/gemm.c) over a tensor arena of huge pages that the host lays out.
 *
 * usage: vgd [-o PART_MIB] [-c DMA_CHANNEL] [-t THREADS] [-p THREADS_PER_CORE] [-a ARENA_MIB]
 * Defaults: 1792, 4, 118, 2 (cores 0 to 58, two hardware threads each), 1024.
 *
 * With a host-memory region under 4 GiB the card's init makes the whole region swap (/dev/phiblk1); vgd turns that
 * swap off first, since the region is then shared memory, not swap.
 *
 * The card-side scalar work is the control around the vector code: the op loop, DMA descriptor writes and polling,
 * task hand-out, loop counters and addresses.
 */
#define _GNU_SOURCE
#include "cdma.h"
#include "../lib/gemm.h"
#include "../lib/vg-proto.h"

#include <errno.h>
#include <fcntl.h>
#include <linux/perf_event.h>
#include <sched.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/swap.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define HUGE (2u << 20)
#define CHUNK (512u << 10)    /* DMA descriptor size: at most 16383 lines, and 2 MiB pages split evenly */
#define GUARD (4u << 20)      /* mapped past the arena's end: the GEMM's prefetches may run ahead of a tensor */

static volatile uint8_t *part;   /* the part's first VG_OPS bytes, uncached through /dev/phihost */

static void say(const char *fmt, ...)
{
	char ts[32];
	time_t t = time(NULL);
	strftime(ts, sizeof ts, "%F %T", localtime(&t));
	va_list ap;
	va_start(ap, fmt);
	fprintf(stderr, "%s vgd: ", ts);
	vfprintf(stderr, fmt, ap);
	fputc('\n', stderr);
	va_end(ap);
}

static double now(void)
{
	struct timespec s;
	clock_gettime(CLOCK_MONOTONIC, &s);
	return s.tv_sec + s.tv_nsec / 1e9;
}

static inline void fence(void) { __asm__ volatile("lock addq $0, (%%rsp)" ::: "memory"); }
static inline uint32_t rd32(int off) { return *(volatile uint32_t *)(part + off); }
static inline uint64_t rd64(int off) { return *(volatile uint64_t *)(part + off); }
static inline void wr32(int off, uint32_t v) { *(volatile uint32_t *)(part + off) = v; }
static inline void wr64(int off, uint64_t v) { *(volatile uint64_t *)(part + off) = v; }

static int swap_off(void)
{
	FILE *f = fopen("/proc/swaps", "r");
	char line[256];
	int on = 0;
	while (f && fgets(line, sizeof line, f)) on |= !strncmp(line, "/dev/phiblk1", 12);
	if (f) fclose(f);
	if (!on) return 0;
	if (swapoff("/dev/phiblk1")) { say("swapoff /dev/phiblk1: %s", strerror(errno)); return -1; }
	say("swap on /dev/phiblk1 turned off: the region is shared memory now");
	return 0;
}

static int hugepages(long need)
{
	FILE *f = fopen("/proc/meminfo", "r");
	char line[256];
	long total = 0, freep = 0;
	while (f && fgets(line, sizeof line, f)) {
		sscanf(line, "HugePages_Total: %ld", &total);
		sscanf(line, "HugePages_Free: %ld", &freep);
	}
	if (f) fclose(f);
	if (freep >= need) return 0;
	f = fopen("/proc/sys/vm/nr_hugepages", "w");
	if (!f) { say("nr_hugepages: %s", strerror(errno)); return -1; }
	fprintf(f, "%ld\n", total + (need - freep));
	fclose(f);
	return 0;
}

/* ---- the arena and its DMA ---- */

static cdma D;
static uint64_t part_addr;      /* the part's card address (in the region) */
static uint8_t *arena;
static uint64_t *arena_phys;    /* per 2 MiB page */
static size_t arena_size;

/* region <-> arena copy of n bytes; to_arena 1: region part offset ro -> arena offset ao. 0, or VG_E_DMA */
static int dma_copy(int to_arena, uint64_t ro, uint64_t ao, uint64_t n)
{
	uint64_t seq = 0;
	for (uint64_t o = 0; o < n;) {
		const uint64_t a = ao + o;
		const uint64_t in_page = HUGE - a % HUGE;
		uint64_t len = n - o < CHUNK ? n - o : CHUNK;
		if (len > in_page) len = in_page;
		const uint64_t ap = arena_phys[a / HUGE] + a % HUGE, rp = part_addr + ro + o;
		seq = to_arena ? cdma_submit(&D, rp, ap, (uint32_t)len, 0, 0) : cdma_submit(&D, ap, rp, (uint32_t)len, 0, 0);
		if (!seq) return VG_E_DMA;
		o += len;
	}
	return seq && cdma_wait(&D, seq, 5.0) ? VG_E_DMA : 0;
}

/* ---- VPU instruction counters, one per pool thread ---- */

static int nthreads;
static int *pfd;
static volatile int arrived;

static void counter_task(void *v, int t)
{
	(void)v;
	struct perf_event_attr pa;
	memset(&pa, 0, sizeof pa);
	pa.size = sizeof pa;
	pa.type = PERF_TYPE_RAW;
	pa.config = 0x2016;              /* KNC VPU_INSTRUCTIONS_EXECUTED */
	pa.exclude_kernel = 1;
	pa.exclude_hv = 1;
	pfd[t] = (int)syscall(__NR_perf_event_open, &pa, 0, -1, -1, 0);
	/* every thread holds here until all have opened one, so each takes exactly one task */
	__atomic_fetch_add(&arrived, 1, __ATOMIC_ACQ_REL);
	while (__atomic_load_n(&arrived, __ATOMIC_ACQUIRE) < nthreads) {}
}

static uint64_t vpu_total(void)
{
	uint64_t sum = 0, v;
	for (int i = 0; i < nthreads; i++)
		if (pfd[i] >= 0 && read(pfd[i], &v, sizeof v) == sizeof v) sum += v;
	return sum;
}

/* ---- ops ---- */

static int in_arena(uint64_t off, uint64_t n) { return off <= arena_size && n <= arena_size - off; }

/* bytes a rows x cols matrix with ld spans */
static uint64_t span(uint64_t rows, uint64_t cols, uint64_t ld) { return rows ? ((rows - 1) * ld + cols) * 4 : 0; }

static int run_op(pool *pl, const vg_op *o, uint64_t *flop, double *gemm_s, double *dma_s, uint64_t *dma_bytes)
{
	double t0 = now();
	int e = 0;
	switch (o->code) {
	case VG_PUT:
	case VG_GET: {
		const uint64_t ro = o->code == VG_PUT ? o->a : o->c, ao = o->code == VG_PUT ? o->c : o->a, n = o->m;
		if ((ro | ao | n) % 64) return VG_E_ALIGN;
		if (ro < VG_DATA || ro > VG_PART_SIZE || n > VG_PART_SIZE - ro || !in_arena(ao, n)) return VG_E_RANGE;
		e = dma_copy(o->code == VG_PUT, ro, ao, n);
		*dma_s += now() - t0;
		*dma_bytes += n;
		return e;
	}
	case VG_ZERO:
		if ((o->c | o->m) % 64) return VG_E_ALIGN;
		if (!in_arena(o->c, o->m)) return VG_E_RANGE;
		zero_run(pl, arena + o->c, o->m);
		return 0;
	case VG_TRANS:
		if ((o->lda | o->ldc) % 16 || (o->a | o->c) % 64) return VG_E_ALIGN;
		if (o->lda < o->n || o->ldc < o->m) return VG_E_ALIGN;
		if (!in_arena(o->a, span(o->m, o->n, o->lda)) || !in_arena(o->c, span(o->n, o->m, o->ldc))) return VG_E_RANGE;
		trans_run(pl, (const float *)(arena + o->a), o->lda, (int)o->m, (int)o->n, (float *)(arena + o->c), o->ldc);
		return 0;
	case VG_GEMM: {
		const int ta = !!(o->flags & VG_TA);
		const uint64_t M = o->m, N = o->n, K = o->k;
		if ((o->lda | o->ldb | o->ldc) % 16 || (o->a | o->b | o->c) % 64) return VG_E_ALIGN;
		if (o->ldb < N || o->ldc < N || o->lda < (ta ? M : K)) return VG_E_ALIGN;
		/* B, C and Z are read a whole vector at a time: up to N rounded to 16 in the last row */
		const uint64_t N16 = (N + 15) / 16 * 16;
		if (!in_arena(o->a, ta ? span(K, M, o->lda) : span(M, K, o->lda)) || !in_arena(o->b, span(K, N16, o->ldb)) ||
		    !in_arena(o->c, span(M, N16, o->ldc)))
			return VG_E_RANGE;
		gemm_t g = { (int)M, (int)N, (int)K, (const float *)(arena + o->a), o->lda, ta, (const float *)(arena + o->b), o->ldb,
		             (float *)(arena + o->c), o->ldc, NULL, NULL, 0, o->alpha, !!(o->flags & VG_BETA), !!(o->flags & VG_RELU) };
		if (o->flags & VG_BIAS) {
			if (o->bias % 64) return VG_E_ALIGN;
			if (!in_arena(o->bias, (N + 15) / 16 * 64)) return VG_E_RANGE;
			g.bias = (const float *)(arena + o->bias);
		}
		if (o->flags & VG_MASK) {
			if (o->z % 64 || o->ldz % 16 || o->ldz < N) return VG_E_ALIGN;
			if (!in_arena(o->z, span(M, N16, o->ldz))) return VG_E_RANGE;
			g.Z = (const float *)(arena + o->z);
			g.ldz = o->ldz;
		}
		gemm_run(pl, &g);
		*flop += 2 * M * N * K;
		*gemm_s += now() - t0;
		return 0;
	}
	default:
		return VG_E_OP;
	}
}

int main(int argc, char **argv)
{
	uint64_t part_mib = VG_PART_MIB, arena_mib = 1024;
	int threads = 118, tpc = 2, chan = 4, opt;
	while ((opt = getopt(argc, argv, "o:c:t:p:a:")) != -1) {
		if (opt == 'o') part_mib = strtoull(optarg, NULL, 10);
		else if (opt == 'c') chan = atoi(optarg);
		else if (opt == 't') threads = atoi(optarg);
		else if (opt == 'p') tpc = atoi(optarg);
		else if (opt == 'a') arena_mib = strtoull(optarg, NULL, 10);
		else { fprintf(stderr, "usage: vgd [-o PART_MIB] [-c DMA_CHANNEL] [-t THREADS] [-p THREADS_PER_CORE] [-a ARENA_MIB]\n"); return 2; }
	}
	if (threads < 1 || tpc < 1 || tpc > 4 || threads > 59 * tpc) { say("threads %d at %d per core: cores 0 to 58 only", threads, tpc); return 2; }
	const uint64_t off = part_mib << 20;
	arena_size = (size_t)arena_mib << 20;
	const long pages = (long)((arena_size + GUARD) / HUGE);
	if (swap_off() || hugepages(pages + 2)) return 1;
	if (cdma_open(&D, chan)) return 1;
	if (off + VG_PART_SIZE > D.region_size) {
		say("part at %llu MiB does not fit the %llu MiB region", (unsigned long long)part_mib, (unsigned long long)(D.region_size >> 20));
		return 1;
	}
	part_addr = D.region_addr + off;
	int fd = open("/dev/phihost", O_RDWR | O_SYNC);
	part = fd < 0 ? MAP_FAILED : mmap(NULL, VG_OPS, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)off);
	if (part == MAP_FAILED) { say("/dev/phihost: %s", strerror(errno)); return 1; }
	uint64_t ops_phys;
	vg_op *ops = huge_alloc(HUGE, &ops_phys, 1);
	arena_phys = calloc((size_t)pages, sizeof *arena_phys);
	arena = huge_alloc(arena_size + GUARD, arena_phys, (int)pages);
	if (!ops || !arena) { say("no huge pages for the arena (%ld) or the op landing page", pages); return 1; }

	/* worker k (the caller is 0) on core k / tpc, hardware thread k % tpc: CPU 1 + 4 * core + thread */
	int *cpus = calloc((size_t)threads, sizeof *cpus);
	for (int k = 1; k < threads; k++) cpus[k - 1] = 1 + 4 * (k / tpc) + k % tpc;
	cpu_set_t set;
	CPU_ZERO(&set);
	CPU_SET(1, &set);
	sched_setaffinity(0, sizeof set, &set);
	pool *pl = pool_open(threads, cpus);
	pool_set_spin(pl, 0.2);   /* lists come back to back during training: keep the workers awake between them */
	nthreads = threads;
	pfd = calloc((size_t)threads, sizeof *pfd);
	pool_run(pl, counter_task, NULL, threads);
	int counted = 0;
	for (int i = 0; i < threads; i++) counted += pfd[i] >= 0;

	wr32(VG_C_STATE, 0);
	wr32(VG_C_THREADS, (uint32_t)threads);
	wr64(VG_C_ARENA, arena_size);
	uint64_t lists = 0, nops_total = 0, flop = 0, dma_bytes = 0, heart = 0;
	double busy = 0, gemm_s = 0, dma_s = 0;
	wr64(VG_C_LISTS, 0);
	wr64(VG_C_OPS, 0);
	wr64(VG_C_FLOP, 0);
	wr64(VG_C_BUSY_NS, 0);
	wr64(VG_C_GEMM_NS, 0);
	wr64(VG_C_DMA_NS, 0);
	wr64(VG_C_DMA_BYTES, 0);
	wr64(VG_C_VPU, 0);
	say("serving: part at %llu MiB (card address 0x%llx), DMA channel %d, %d threads (%d per core), arena %llu MiB, "
	    "VPU counters on %d threads", (unsigned long long)part_mib, (unsigned long long)part_addr, chan, threads, tpc,
	    (unsigned long long)arena_mib, counted);

	uint64_t session = rd64(VG_C_SESSION), done = 0;
	double next_heart = 0;
	struct timespec nap = { 0, 20000 };
	for (;;) {
		double t = now();
		if (t >= next_heart) {
			wr64(VG_C_HEART, ++heart);
			wr64(VG_C_VPU, vpu_total());
			next_heart = t + 0.1;
		}
		const uint64_t s = rd64(VG_H_SESSION);
		if (s != session && rd32(VG_H_MAGIC) == VG_MAGIC && rd32(VG_H_VERSION) == VG_VERSION) {
			done = rd64(VG_H_POSTED);
			wr64(VG_C_DONE, done);
			wr32(VG_C_ERR, 0);
			fence();
			wr64(VG_C_SESSION, s);
			wr32(VG_C_STATE, 1);
			session = s;
			say("session %llu", (unsigned long long)s);
		}
		if (s != session || rd64(VG_H_POSTED) <= done) { nanosleep(&nap, NULL); continue; }

		/* list done + 1: pull it, run it, answer */
		const uint64_t n = done + 1;
		const uint32_t nops = rd32(VG_H_NOPS), times = rd32(VG_H_TIMES);
		double l0 = now();
		int err = 0;
		uint32_t errop = 0;
		if (nops > VG_OPS_MAX) { err = VG_E_RANGE; }
		else if (nops) {
			const uint32_t bytes = nops * (uint32_t)sizeof(vg_op);
			uint64_t seq = 0;
			for (uint32_t o = 0; o < bytes; o += CHUNK) {
				const uint32_t len = bytes - o < CHUNK ? bytes - o : CHUNK;
				seq = cdma_submit(&D, part_addr + VG_OPS + o, ops_phys + o, len, 0, 0);
				if (!seq) break;
			}
			if (!seq || cdma_wait(&D, seq, 5.0)) err = VG_E_DMA;
			dma_s += now() - l0;
		}
		for (uint32_t i = 0; !err && i < nops; i++) {
			const double o0 = now();
			if (o0 >= next_heart) { wr64(VG_C_HEART, ++heart); next_heart = o0 + 0.1; }   /* alive during long lists */
			err = run_op(pl, &ops[i], &flop, &gemm_s, &dma_s, &dma_bytes);
			if (err) errop = i;
			if (times) wr32(VG_TIMES + 4 * (int)i, (uint32_t)((now() - o0) * 1e9 / 64));
		}
		const double l1 = now();
		busy += l1 - l0;
		lists++;
		nops_total += nops;
		wr32(VG_C_ERR, (uint32_t)err);
		wr32(VG_C_ERROP, errop);
		wr64(VG_C_LIST_NS, (uint64_t)((l1 - l0) * 1e9));
		wr64(VG_C_LISTS, lists);
		wr64(VG_C_OPS, nops_total);
		wr64(VG_C_FLOP, flop);
		wr64(VG_C_BUSY_NS, (uint64_t)(busy * 1e9));
		wr64(VG_C_GEMM_NS, (uint64_t)(gemm_s * 1e9));
		wr64(VG_C_DMA_NS, (uint64_t)(dma_s * 1e9));
		wr64(VG_C_DMA_BYTES, dma_bytes);
		if (err) say("list %llu: op %u refused (error %d, code %u)", (unsigned long long)n, errop, err, ops[errop].code);
		fence();
		wr64(VG_C_DONE, n);
		done = n;
	}
}
