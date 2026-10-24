/* Copied from Intel-Phi-Worldgen card/lib/cdma.c at 420329d (same author, MIT). */
/* cdma.c: a card-owned channel of the SBOX DMA engine, driven from user space on the card. See cdma.md. */
#include "cdma.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define SBOX_DMA_PAGE 0x08007DA000ULL /* SBOX + 0xA000 (SSDG 328207-002 2.1.8.2.1; the stack's defs.inc) */
#define REGION_HEADER 0x10000000ULL   /* phictl's ring region header: "PHIR", the window's address and size */
#define CHAN_STRIDE 0x40
#define R_DCAR 0x00
#define R_DHPR 0x04
#define R_DTPR 0x08
#define R_DRAR_LO 0x14
#define R_DRAR_HI 0x18
#define R_DCHERRMSK 0x30
#define R_DCR 0x280
#define DCAR_MASKS 0x03000000u /* both interrupt masks: the card kernel has no DMA handler */
#define RING 256               /* descriptors of 16 bytes: one 4 KiB page */
#define HUGE (2u << 20)
#define MASK40 ((1ULL << 40) - 1)

static inline void fence(void)
{
    /* the card has no mfence; a locked read-modify-write is the full fence */
    __asm__ volatile("lock addq $0, (%%rsp)" ::: "memory");
}

static inline uint32_t rr(cdma *d, int off) { return d->regs[(d->chan * CHAN_STRIDE + off) / 4]; }
static inline void wr(cdma *d, int off, uint32_t v) { d->regs[(d->chan * CHAN_STRIDE + off) / 4] = v; }

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static uint64_t phys_of(const void *p)
{
    int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0) return 0;
    uint64_t e = 0;
    off_t at = (off_t)((uintptr_t)p / 4096) * 8;
    ssize_t r = pread(fd, &e, 8, at);
    close(fd);
    if (r != 8 || !(e >> 63)) return 0;
    return (e & ((1ULL << 55) - 1)) * 4096 + ((uintptr_t)p & 4095);
}

void *huge_alloc(size_t n, uint64_t *phys, int pages)
{
    size_t len = (n + HUGE - 1) / HUGE * HUGE;
    if ((int)(len / HUGE) > pages) return NULL;
    uint8_t *p = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE, -1, 0);
    if (p == MAP_FAILED) return NULL;
    mlock(p, len);
    memset(p, 0, len);
    for (size_t i = 0; i < len / HUGE; i++) {
        phys[i] = phys_of(p + i * HUGE);
        /* the page must be one physical 2 MiB frame: its last 4 KiB must follow its first */
        if (!phys[i] || phys_of(p + i * HUGE + HUGE - 4096) != phys[i] + HUGE - 4096) {
            munmap(p, len);
            return NULL;
        }
    }
    return p;
}

int cdma_open(cdma *d, int chan)
{
    memset(d, 0, sizeof *d);
    d->chan = chan;
    d->lockfd = -1;
    pthread_mutex_init(&d->m, NULL);
    char lock[64];
    snprintf(lock, sizeof lock, "/tmp/phi-cdma%d.lock", chan);
    d->lockfd = open(lock, O_CREAT | O_RDWR, 0600);
    if (d->lockfd < 0 || flock(d->lockfd, LOCK_EX | LOCK_NB)) {
        fprintf(stderr, "cdma: channel %d is locked by another process (%s)\n", chan, lock);
        return -1;
    }
    int mem = open("/dev/mem", O_RDWR | O_SYNC);
    if (mem < 0) { perror("cdma: /dev/mem"); return -1; }
    d->regs = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, mem, (off_t)SBOX_DMA_PAGE);
    const volatile uint8_t *hdr = mmap(NULL, 4096, PROT_READ, MAP_SHARED, mem, (off_t)REGION_HEADER);
    close(mem);
    if (d->regs == MAP_FAILED || hdr == MAP_FAILED) { perror("cdma: mmap /dev/mem"); return -1; }
    if (memcmp((const void *)hdr, "PHIR", 4) != 0) {
        fprintf(stderr, "cdma: no PHIR region header at 0x%llx\n", (unsigned long long)REGION_HEADER);
        return -1;
    }
    memcpy(&d->region_addr, (const void *)(hdr + 32), 8);
    memcpy(&d->region_size, (const void *)(hdr + 40), 8);
    munmap((void *)hdr, 4096);
    if (!d->region_addr || !d->region_size || (d->region_addr & 4095)) {
        fprintf(stderr, "cdma: implausible region 0x%llx size %llu\n", (unsigned long long)d->region_addr,
                (unsigned long long)d->region_size);
        return -1;
    }

    d->page = huge_alloc(HUGE, &d->page_phys, 1);
    if (!d->page) { fprintf(stderr, "cdma: no huge page for the ring\n"); return -1; }
    d->status = (volatile uint64_t *)(d->page + 4096);

    /* take the channel: owner card (bit 2n = 0), disabled (bit 2n+1 = 0), every other bit unchanged */
    uint32_t dcr = d->regs[R_DCR / 4];
    uint32_t mine = 3u << (2 * chan);
    if ((dcr & (2u << (2 * chan))) && !(dcr & (1u << (2 * chan)))) {
        /* enabled and card owned: a previous owner died with the lock released; let its copies drain */
        double t0 = now();
        while (rr(d, R_DTPR) != rr(d, R_DHPR) && now() - t0 < 2.0) {}
        if (rr(d, R_DTPR) != rr(d, R_DHPR)) { fprintf(stderr, "cdma: channel %d busy\n", chan); return -1; }
    }
    d->regs[R_DCR / 4] = dcr & ~mine;
    uint32_t back = d->regs[R_DCR / 4];
    if ((back & ~mine) != (dcr & ~mine) || (back & mine)) {
        fprintf(stderr, "cdma: DCR 0x%08x -> 0x%08x: other channels changed\n", dcr, back);
        return -1;
    }
    wr(d, R_DCAR, DCAR_MASKS);
    wr(d, R_DCHERRMSK, 0);
    wr(d, R_DRAR_LO, (uint32_t)d->page_phys);
    wr(d, R_DRAR_HI, (RING << 4) | (uint32_t)((d->page_phys >> 32) & 0xF));
    /* the head on the first line at or after the tail; the slots between are NOPs (the page is zero) */
    uint32_t tail = rr(d, R_DTPR) % RING;
    d->head = (tail + 3) / 4 * 4 % RING;
    fence();
    wr(d, R_DHPR, d->head);
    d->regs[R_DCR / 4] = (d->regs[R_DCR / 4] & ~mine) | (2u << (2 * chan));
    if (!(d->regs[R_DCR / 4] & (2u << (2 * chan)))) { fprintf(stderr, "cdma: enable did not stick\n"); return -1; }
    double t0 = now();
    while (rr(d, R_DTPR) % RING != d->head && now() - t0 < 2.0) {}
    return 0;
}

uint64_t cdma_submit(cdma *d, uint64_t src, uint64_t dst, uint32_t bytes, uint64_t host_status, uint64_t host_value)
{
    if (!bytes || bytes % 64 || bytes / 64 > 16383 || (src | dst) % 64) return 0;
    pthread_mutex_lock(&d->m);
    while (d->seq - *d->status >= RING / 4 - 1) {} /* at most 63 lines in flight */
    uint64_t seq = ++d->seq;
    volatile uint64_t *q = (volatile uint64_t *)(d->page + d->head * 16);
    q[0] = (src & MASK40) | ((uint64_t)(bytes / 64) << 46); /* memcpy */
    q[1] = (dst & MASK40) | (1ULL << 60);
    q[2] = host_status ? host_value : 0; /* status into host memory, or a NOP */
    q[3] = host_status ? (host_status & MASK40) | (2ULL << 60) : 0;
    q[4] = seq; /* status into the ring page: this process's completion */
    q[5] = ((d->page_phys + 4096) & MASK40) | (2ULL << 60);
    q[6] = 0; /* NOP */
    q[7] = 0;
    d->head = (d->head + 4) % RING;
    fence();
    wr(d, R_DHPR, d->head);
    pthread_mutex_unlock(&d->m);
    return seq;
}

int cdma_wait(cdma *d, uint64_t seq, double timeout_s)
{
    if (*d->status >= seq) return 0;
    double t0 = now();
    struct timespec nap = { 0, 5000 };
    while (*d->status < seq) {
        if (now() - t0 > timeout_s) return -1;
        nanosleep(&nap, NULL);
    }
    return 0;
}

void cdma_close(cdma *d)
{
    if (!d->regs || d->regs == MAP_FAILED) return;
    d->regs[R_DCR / 4] &= ~(2u << (2 * d->chan));
    if (d->lockfd >= 0) close(d->lockfd);
}
