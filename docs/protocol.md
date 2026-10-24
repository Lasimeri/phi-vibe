# The card's GEMM protocol

`lib/vg-proto.h` is the authority; this page is the walk-through.

## Where the bytes live

The card stack's `phictl --host-mem 2G` gives the card a window onto 2 GiB of host memory:
- the host maps it from `/dev/shm/phi-hostmem`;
- the card maps it uncached from `/dev/phihost`, and reaches it at card address `0x8100000000` (read from the PHIR header at card physical `0x10000000`) through its DMA engine.

This project uses the last 256 MiB of that window (the *part*, at 1792 MiB) and DMA channel 4, a card-owned channel driven from user space (`card/cdma.c`).

| Part offset | Size | Written by | Holds |
|---|---|---|---|
| 0 | 64 B | host | magic `VIBG`, version, session |
| 64 | 64 B | host | lists posted, ops in the posted list, per-op timing on/off |
| 128 | 128 B | card | session taken, state, threads, arena size, lists done, error and failing op, last list time |
| 256 | 64 B | card | heartbeat (about every 100 ms, also between ops) |
| 320 | 64 B | card | totals since start: lists, ops, GEMM FLOP, busy ns, GEMM ns, DMA ns, DMA bytes, VPU instructions |
| 16 KiB | 48 KiB | card | per-op time of the last list (ns / 64), when the host asked for it |
| 64 KiB | 960 KiB | host | the op list: 128-byte `vg_op` records |
| 1 MiB | 255 MiB | both | staging for PUT (host to card) and GET (card to host) |

Host-written and card-written fields never share a 64-byte line.

## A session

1. The host writes magic, version, `posted = 0` and a new session number (the time in microseconds).
2. The card sees the new session, sets its done count to the host's posted count, acknowledges the session, and sets state 1.
3. Before every op the host checks that the card's heartbeat moved within the last 5 s.

## A list

1. The host fills the op list and any PUT data in the staging area, writes the op count, then raises `posted` (release order).
2. The card pulls the op list into a huge page with one DMA copy and runs the ops in order. Each op is spread over the card's thread pool: 118 threads, two on each of cores 0 to 58.
3. The card writes the error (0 if none), the totals and the time, then raises `done`.
4. The host copies the GET results out of staging.

Only one list is in flight at a time. The host's allocator places every tensor in the card's arena, 1 GiB of 2 MiB huge pages, plus a 4 MiB guard so prefetches running past a tensor stay inside the mapping. DMA copies are split at every 2 MiB page boundary and into 512 KiB descriptors.

## Ops

| Op | Effect |
|---|---|
| `PUT` | arena[c..] = staging[a..], m bytes (multiples of 64) |
| `GET` | staging[c..] = arena[a..], m bytes |
| `GEMM` | C = alpha·op(A)·B (+C) (+bias) (ReLU) (zero where Z ≤ 0) |
| `TRANS` | C = Aᵀ |
| `ZERO` | arena[c..] = 0 |

GEMM rules:
- Matrices are row-major, every ld is a multiple of 16 floats, and every base is 64-byte aligned.
- With `VG_TA`, A is stored K × M, so the same kernel computes Aᵀ·B, which the weight gradients use.
- Only rows below M and columns below N are written. The host keeps tensor padding at zero, so reshaping a dense `R × C` tensor into `R/2 × 2C` is free; the encoder's merge layers rely on that.
- The card refuses any op that breaks a rule (`VG_E_*`) and skips the rest of the list.

## What runs where

On the card:
- every floating-point operation in a GEMM, a transpose or a zero fill, as VPU instructions (`lib/gemm.c`);
- the scalar control around them: the op loop, DMA descriptors and polling, task hand-out, loop counters and addresses.

On the host CPU: everything else (decoding, FFT and mel, the loss, Adam, normalisation, neighbour selection, the planner, VLC).
