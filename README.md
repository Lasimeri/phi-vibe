# phi-vibe

A music-taste network that learns the "vibe" of every song in a local library from the audio itself, and will pick
the song VLC plays next. Its matrix products run on the vector unit of an Intel Xeon Phi 7120A coprocessor card;
everything else runs on the host CPU.

This repository holds the parts that are built and measured on the real hardware:

| Part | State |
|---|---|
| `vibe-mel`: the library as log-mel crops | done, gated, ran over 6,027 songs |
| `vgd`: GEMM daemon on the card's vector unit, host client `host/vg.c` | done, gated against a double-precision reference |
| `vibe-train`: the vibe encoder, trained on the card | done, measured against chance and a hand-made baseline |
| Next-song planner (D* Lite over the vibe graph) | in progress, not in this repository yet |
| Preference network that learns from plays | in progress, not in this repository yet |
| VLC daemon (watches plays over MPRIS, queues the pick) | in progress, not in this repository yet |

## How it works

```
╔═ host CPU (Linux desktop) ═════════════════╗      ╔═ Xeon Phi 7120A ═══════════════════╗
║ ┌────────────────────────────────────────┐ ║      ║ ┌────────────────────────────────┐ ║
║ │ vibe-mel: libav decode, FFT, 64 mel    │ ║      ║ │ vgd: op lists, DMA channel 4   │ ║
║ │ bands, <= 32 crops of 6 s per song     │ ║      ║ │ 118 threads (cores 0-58 x 2)   │ ║
║ └───────────────────┬────────────────────┘ ║      ║ │ arena: 1 GiB of huge pages     │ ║
║                     ▼ mel.bin, index.tsv   ║      ║ └───────────────┬────────────────┘ ║
║ ┌────────────────────────────────────────┐ ║ PUT  ║ ┌───────────────▼────────────────┐ ║
║ │ vibe-train: batches, NT-Xent loss,     │─╫─────▶║ │ lib/gemm.c on the VPU: GEMM,   │ ║
║ │ Adam, normalisation (host/enc.c)       │◀╫──────║ │ transpose, zero (8x48 tiles)   │ ║
║ └───────────────────┬────────────────────┘ ║ GET  ║ └────────────────────────────────┘ ║
║                     ▼ enc.bin, emb.bin     ║      ╚════════════════════════════════════╝
║ ┌────────────────────────────────────────┐ ║
║ │ next: planner + VLC daemon (planned)   │ ║   part: last 256 MiB of the 2 GiB
║ └────────────────────────────────────────┘ ║   /dev/shm/phi-hostmem window
╚════════════════════════════════════════════╝
```

**Features.** `vibe-mel` decodes every audio file with FFmpeg's libraries:
- 22,050 Hz mono, Hann 2048, hop 1024;
- 64 mel bands (30 Hz to 11,025 Hz), in dB relative to a full-scale sine, stored as 8-bit codes over −100 to +10 dB;
- up to 32 crops of 128 frames (5.99 s) per song, one centred in each equal segment of the song, so crops never overlap;
- long gaps are skipped by seeking, so an hour-long mix reads about 6 s per crop instead of the whole file.

It runs at idle I/O priority, because the library usually sits on the disk the music is playing from. Runs are resumable.

**The vibe encoder** (`host/enc.c`) turns one crop into a 256-number vibe vector:
- seven merge layers each concatenate two neighbouring frames and apply a dense layer with ReLU, so 128 frames shrink to 1 and 64 bands grow to 256 channels;
- a projection head (256 → 256 → 128) is used only for training.

No labels are involved. It learns contrastively (NT-Xent, temperature 0.1): two crops of the same song should land near each other and far from every other song's crops in the batch. A song's vibe is the mean of its crops' unit vectors.

**On the card.** Every matrix product, forward and backward, runs on the 7120A, as does every transpose and the similarity matrix. The host keeps the master weights, the loss, Adam and the input normalisation. The card side is a daemon (`card/vgd.c`):
- it takes op lists through the last 256 MiB of the card stack's shared host-memory window;
- it moves data with a card-owned DMA channel;
- it runs `lib/gemm.c`, an 8 × 48 register-tiled SGEMM written with KNC intrinsics (24 accumulators in registers, two-level software prefetch), on 118 hardware threads.

The protocol is in [`docs/protocol.md`](docs/protocol.md).

## Measured

All on one 7120A (61 cores, 1.238 GHz, 16 GB GDDR5) in a Ryzen 7 5800X desktop.

**Mel extraction**
- Gate: a 440 Hz sine of amplitude 1/8 lands in band 9 at −25.4 dB (hand-computed −25.3), and 3000 Hz lands in band 37 as predicted.
- Library: 6,028 files (one folder of talk excluded), 641.4 h of audio, 1,183.7 s wall time with 3 threads.
- Errors: 1 file with no audio stream.
- Output: `mel.bin` of 1,579,941,888 B.

**GEMM correctness** (`vgtest -q`)
- 11 shapes cover odd sizes, both A layouts, accumulate, bias, ReLU, mask, alpha 0.5 and −0.25, and K = 0.
- Worst relative error is 2.25 × 10⁻⁷ against a host double-precision reference.
- The sentinel padding around every output stays untouched, and transposes are bit-exact.

**GEMM speed** (`vgtest -r 20`, median GFLOP/s; peak with 59 cores × 2 threads is 2,337, calculated)

| Shape (M × N × K) | No prefetch | With prefetch |
|---|---|---|
| 32768 × 256 × 96 | 366.1 | 346.9 |
| 16384 × 256 × 512 | 248.0 | 436.5 |
| 16384 × 512 × 256 | 194.4 | 350.1 |
| 512 × 256 × 16384, A transposed | 83.5 | 259.5 |
| 2048 × 2048 × 2048 | 87.0 | 217.2 |

The card counts 1.48 to 1.55 VPU instructions per 16 multiply-adds (event 0x2016). The inner loop is 35 vector instructions per 24 FMAs: 3 loads, 8 broadcasts and 24 FMAs.

**Training** (6,027 songs: 5,430 train, 597 held out by path hash)
- 126 ms per step of 512 crops: about 30.2 GFLOP on the card at 302 GFLOP/s, plus 27 MB of DMA.
- 2.66 s per epoch.

| | Held-out crop retrieval, top-1 same song | 10 nearest songs: same folder | 10 nearest songs: same artist |
|---|---|---|---|
| chance | 0.0013 | 0.0744 | 0.0008 |
| hand-made baseline (band mean and spread) | | 0.2657 | 0.0286 |
| encoder, untrained | 0.0302 | 0.1631 | 0.0312 |
| encoder, epoch 25 | 0.2451 | 0.3232 | 0.0460 |
| encoder, epoch 75 | 0.3168 | 0.3544 | 0.0502 |
| encoder, epoch 150 (shipped) | 0.3508 | 0.3656 | 0.0542 |

Chance levels are exact, computed from the folder and artist sizes. Retrieval is measured on songs never trained on.

## Next-song planner (in progress)

The plan follows Garry's Mod's NextBot NPCs, which chase a moving target across a navigation mesh, with A* re-planned as the world changes. The modern incremental form is D* Lite (Koenig and Likhachev, AAAI 2002). Here:
- **The mesh:** the nearest-neighbour graph of the songs' vibe vectors.
- **The agent:** stands on the song now playing.
- **Edge costs:** the angle between two songs' vibes plus a learned cost of hearing that song now.
- **The target:** a virtual goal node whose entry cost from each song is the song's angle to the vibe a network predicts you are heading toward.

Plays (finished, skipped, picked by hand) train the networks by gradient descent: no count tables and no play probabilities. How much of a song was played, and how recently each song played, are network inputs rather than fixed rules. D* Lite's own heuristic is only the search's distance estimate: it speeds the search and does not change the chosen path, which the planned gate checks against a full Dijkstra.

## Requirements

- An Intel Xeon Phi x100 card booted with the Intel Phi 7120A stack (`phictl` with `--host-mem 2G`, which provides `/dev/shm/phi-hostmem` on the host and `/dev/phihost` on the card), with root ssh to the card.
- The Intel-Phi-LLVM clang (`-march=knc`) for `lib/gemm.c` and `lib/pool.c`, and the stack's `knc-cc` and `phi-isa-audit`. `build.sh` finds them under `~/.cache/intel-phi-llvm-build` and `~/Intel Phi 7120A`; override with `PHI_LLVM_BIN`, `PHI_SYSROOT_7120A` and `PHI_TREE`.
- Host: gcc, FFmpeg 9 libraries (libavformat, libavcodec, libswresample, libavutil), FFTW3 (single precision).

## Build and run

```sh
./build.sh                                   # build/vibe-mel, vgtest, vibe-train, vgd-card
echo /path/to/music > ~/.config/phivibe/library
echo 'Podcasts' > ~/.config/phivibe/exclude  # optional: top folders to leave out
systemd-run --user -p IOSchedulingClass=idle -p Nice=19 build/vibe-mel -j 3

# on the card (vgd turns the card's swap on the shared window off, then serves; it does not survive a card reboot)
ssh card 'mkdir -p /tmp/vibe && cat > /tmp/vibe/vgd && chmod +x /tmp/vibe/vgd' < build/vgd-card
ssh card 'cd /tmp/vibe && setsid nohup ./vgd > vgd.log 2>&1 < /dev/null &'

build/vgtest -q                              # correctness gate
build/vgtest -r 20                           # speed
build/vibe-train -e 150 -v 25                # train, measure every 25 epochs, write the embeddings
```

Outputs:
- `~/.cache/phivibe/lib/`: `mel.bin`, `index.tsv`, `errors.tsv`;
- `~/.local/share/phivibe/`: `enc.bin` (the model and its Adam state), `emb.bin` and `emb.tsv` (per song: the mean, first-crop and last-crop vibe).

## Notes on the card

- **Intrinsics.** This compiler's `kncintrin.h` has no gather, masked move or `set_epi32`. The transpose is built from broadcasts merged with masked integer adds.
- **Broadcasts.** The broadcast into the FMA's memory operand is not folded: each one is a separate `vbroadcastss`.
- **Prefetch.** `__builtin_prefetch` emits nothing for this target, so prefetches are inline `vprefetch0` and `vprefetch1`.
- **Idle workers.** Idle pool workers spin for a set time using `DELAY`, KNC's replacement for `PAUSE`. Letting them sleep after about 1.6 ms made every short op wait milliseconds for wake-ups: 5.6 ms instead of 0.13 ms for the head's weight gradient. Fixing it halved the training step, from 275 ms to 129 ms.
- **ISA audit.** `phi-isa-audit` reports 5 CMOVs in `gemm_task`. These are a decoder desync inside three-byte-VEX `vprefetch0` instructions: `llvm-objdump --mcpu=knc` decodes the same bytes as `vprefetch0` and finds no CMOV in the binary, and the binary runs on the card.

## Licence

MIT, see [`LICENSE.md`](LICENSE.md). The DMA helper and the thread pool are copied from the author's other MIT
projects; each file names its source.
