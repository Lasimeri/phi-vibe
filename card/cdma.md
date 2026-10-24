<!-- Copied from Intel-Phi-Worldgen card/lib/cdma.md at 420329d (same author, MIT). -->
# cdma.h, cdma.c

In this project `card/vgd.c` takes channel 4 (see `docs/protocol.md`); the notes below are the original ones.

One channel of the card's SBOX DMA engine, taken and driven from user space
on the card. pwgd uses channel 6.

The other channels:
- phictl owns 0 (disk) and 1 (host memory) from the host;
- the AVX-512 wrapper's worker takes 7 the same way as this file does (its `card/vpu/cdma.md` documents the method).

The register layout and descriptor formats are the stack's:
- **Registers:** `host/asm/phictl/defs.inc` and `dma.md` in the Intel Phi 7120A tree, from Intel SSDG 328207-002 2.1.8.2.1 and MPSS `mic_dma_md.c`. Channel n's registers sit at SBOX + 0xA000 + 0x40·n: DCAR, DHPR (head), DTPR (tail), DRAR_LO, DRAR_HI (the ring size in bits 20:4, address bits 35:32), DCHERRMSK. DCR is at 0xA280: bit 2n is the owner (1 host, 0 card), bit 2n+1 enables the channel.
- **Descriptors:** 16 bytes each:
  - memcpy: 40-bit source, length in 64-byte lines in bits 59:46, 40-bit destination, type 1 in bits 63:60;
  - status: a 64-bit value written to a 40-bit address, type 2;
  - zero is a NOP.

**`cdma_open(d, chan)`:**

1. Takes `/tmp/phi-cdma<chan>.lock` exclusively.
2. Maps the SBOX DMA page (`0x08007DA000`) and phictl's region header (`0x10000000`: `PHIR`, the shared host-memory region's card address at +32, its size at +40) through `/dev/mem`.
3. Gets one 2 MiB huge page for the ring (256 descriptors) and the status word (at 4096), its physical frame read from `/proc/self/pagemap`.
4. Writes the channel's DCR bits as card owned and disabled, checking that no other channel's bits moved.
5. Masks both interrupts (the card kernel has no DMA handler).
6. Writes the ring's address and size, puts the head on the first line at or after the tail, and enables the channel.

A channel found enabled and card-owned was a dead owner's: its copies get
2 s to drain first.

**`cdma_submit(src, dst, bytes, host_status, host_value)`:**

- writes one 64-byte line of four descriptors:
  - the copy;
  - a status into host memory (or a NOP), so a host reader sees the value only after the copy's data, both being posted writes on the same path;
  - a status into the ring page with the line's sequence number;
  - a NOP;
- then a fence (`lock addq`: the card has no `mfence`) and the new head into DHPR.

Lengths are multiples of 64 and at most 16,383 lines: 1 MiB is one line
too many. At most 63 lines are in flight.

**`cdma_wait`** naps 5 µs between looks at the status word.

**Measured** on the 7120A, 2026-10-22 (`card/pwgd/cdmatest.c`):

| Check | Result |
| --- | --- |
| 64 KiB pulled from the shared region | 0 of 8,192 words differ |
| 64 KiB pushed into it, with a host status word | 0 differ; the host read the same words and the status from `/dev/shm/phi-hostmem` |
| 64 MiB, as 128 copies of 512 KiB | 3,622.8 MB/s into the card, 3,637.4 MB/s out |
