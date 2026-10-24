/* Copied from Intel-Phi-Worldgen card/lib/cdma.h at 420329d (same author, MIT). */
/* cdma.h: a card-owned channel of the SBOX DMA engine, driven from user space on the card. See cdma.md. */
#ifndef CDMA_H
#define CDMA_H

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    int chan;
    volatile uint32_t *regs;           /* SBOX + 0xA000: the eight channels' registers and DCR */
    uint8_t *page;                     /* one 2 MiB huge page: the ring at 0, the status word at 4096 */
    uint64_t page_phys;
    volatile uint64_t *status;         /* the sequence number of the last completed line */
    uint64_t region_addr, region_size; /* the shared host-memory region: its card address and size */
    uint32_t head;                     /* the next descriptor slot, a multiple of 4 */
    uint64_t seq;                      /* the last sequence number submitted */
    int lockfd;
    pthread_mutex_t m;
} cdma;

/* Takes channel chan (0 to 7) for this process; 0, or -1 with a message on stderr. */
int cdma_open(cdma *d, int chan);

/*
 * One line of four descriptors: copy bytes (a multiple of 64, at most 16383 lines) from src to dst, both card
 * addresses (GDDR physical, or region_addr + an offset in the shared host-memory region); then, when host_status
 * is nonzero, write host_value there (8 bytes, in host memory, after the copy's data); then the line's sequence
 * number to the status word. Returns the sequence number, or 0.
 */
uint64_t cdma_submit(cdma *d, uint64_t src, uint64_t dst, uint32_t bytes, uint64_t host_status, uint64_t host_value);

/* Waits until line seq has completed; 0, or -1 after timeout_s seconds. */
int cdma_wait(cdma *d, uint64_t seq, double timeout_s);

void cdma_close(cdma *d);

/* n bytes (rounded up to 2 MiB pages) of huge pages, populated and locked, each page's physical address in phys[]
 * (at most pages entries); NULL when the kernel has too few free. */
void *huge_alloc(size_t n, uint64_t *phys, int pages);

#endif
