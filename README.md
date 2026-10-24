# phi-vibe

A music-taste network that learns the "vibe" of every song in a local library from the audio itself, watches what
you play in VLC, learns from every play, and lines up the next song. Its matrix products run on the vector unit of
an Intel Xeon Phi 7120A coprocessor card; everything else runs on the host CPU.

| Part | State |
|---|---|
| `vibe-mel`: the library as log-mel crops | done, gated, ran over 6,027 songs |
| `vgd`: GEMM daemon on the card's vector unit, host client `host/vg.c` | done, gated against a double-precision reference |
| `vibe-train`: the vibe encoder, trained on the card | done, measured against chance and a hand-made baseline |
| Next-song planner: D* Lite over the vibe graph (`host/dstar.c`) | done, gated against Dijkstra |
| Preference network that learns from plays (`host/pref.c`) | done, gradient-checked, cold-start gated |
| `vibed`: watches VLC over MPRIS, learns, queues the pick, re-plans after a skip | running; picks played, skips re-planned (see Measured) |

## How it works

```
╔═ host CPU (Linux desktop) ═══════════════════════════╗      ╔═ Xeon Phi 7120A ════════════════╗
║ ┌──────────────┐  MPRIS: state, plays, AddTrack      ║      ║ ┌─────────────────────────────┐ ║
║ │ VLC          │◀──────────────────────────────┐     ║      ║ │ vgd: op lists, DMA ch. 4    │ ║
║ └──────────────┘                               │     ║      ║ │ 118 threads, cores 0-58     │ ║
║ ┌──────────────────────────────────────────────┴───┐ ║      ║ │ arena 1 GiB of huge pages   │ ║
║ │ vibed: play log (fraction played, how it ended)  │ ║ PUT  ║ └──────────────┬──────────────┘ ║
║ │ preference net: Adam, losses (host/pref.c)       │─╫─────▶║ ┌──────────────▼──────────────┐ ║
║ │ D* Lite over the CSLS vibe graph (host/dstar.c)  │◀╫──────║ │ lib/gemm.c on the VPU:      │ ║
║ └──────────────────────────────────────────────────┘ ║ GET  ║ │ GEMM, transpose, zero fill  │ ║
║ ┌──────────────────────────────────────────────────┐ ║      ║ └─────────────────────────────┘ ║
║ │ vibe-mel: mel crops; vibe-train: encoder (card)  │ ║      ╚═════════════════════════════════╝
║ └──────────────────────────────────────────────────┘ ║
╚══════════════════════════════════════════════════════╝ part: last 256 MiB of the 2 GiB window
```

**Features.** `vibe-mel` decodes every audio file with FFmpeg's libraries:
- 22,050 Hz mono, Hann 2048, hop 1024;
- 64 mel bands (30 Hz to 11,025 Hz), in dB relative to a full-scale sine, stored as 8-bit codes over −100 to +10 dB;
- up to 32 crops of 128 frames (5.99 s) per song, one centred in each equal segment of the song;
- long gaps are skipped by seeking, so an hour-long mix reads about 6 s per crop.

It runs at idle I/O priority, because the library usually sits on the disk the music is playing from.

**The vibe encoder** (`host/enc.c`) turns one crop into a 256-number vibe vector:
- seven merge layers each concatenate two neighbouring frames and apply a dense layer with ReLU;
- a projection head is used only for training.

It learns without labels (NT-Xent): two crops of the same song should land near each other and far from every other song's crops. A song's vibe is the mean of its crops' unit vectors.

**The preference network** (`host/pref.c`, 83,334 parameters) learns from your plays. Every play is logged with what was measured: the fraction played, and how it ended:
- `natural`: VLC reached the end;
- `next`: VLC's lined-up track started early;
- `pick`: you started another track;
- `stop`: playback stopped.

It has two heads and three mixing weights:
- **Fraction head.** Predicts the fraction of each song you would play now. It sees the song's vibe, a context and how recently the song played. The context is the current song, your last eight plays weighted by how much of each you played, and the time of day and week.
- **Goal head.** Predicts the vibe your listening is heading toward.
- **Mixing weights** (α, β, κ). They weigh vibe smoothness, predicted fraction and distance to the goal. They are trained only on your own choices: when you start a song instead of the queued one, yours should cost less.

**When you skip** (Next before the end), the pick VLC starts was planned before the skip. `vibed` handles it like a navigating agent that finds a new obstacle:
- the network learns from the skip at once;
- the plan is repaired from the last song you let play (the latest play that did not end in a skip), with every song skipped since blocked;
- if the repaired plan starts with a different song, VLC switches to it straight away, inserted right after the stale pick. The stale pick, heard for about a tenth of a second, stays in the playlist but is not logged as a play. `vibed` never removes the track VLC is playing;
- a track you lined up yourself is never replaced.

Copies of one recording count as one song: blocking, recency and skips all treat them together. Copies are vibe vectors at cosine ≥ 0.9999; 385 songs in the library have at least one.

There are no play counts, no transition tables and no hand-set thresholds. How much of a song was played, and how recently each song played, are inputs and targets, not rules. The only fixed rule: never the song playing now. Before the first play the recency weights start strongly negative, so a song you just heard is avoided while the network knows nothing else.

**The planner** follows Garry's Mod's NextBot NPCs, which chase a moving target across a navigation mesh with A* and re-plan as the world changes. Here it uses D* Lite (Koenig and Likhachev, AAAI 2002), the incremental re-planning form:
- **The mesh:** the vibe graph, each song linked to its 16 nearest songs plus the reverse links.
- **Neighbour ranking:** CSLS (Lample et al. 2018) instead of plain cosine, so hour-long mixes (whose vibe averages many songs) stop being every song's neighbour.
- **The agent:** stands on the song now playing.
- **An edge:** costs α × the angle between the two vibes, plus κ × (1 − the predicted fraction of the song it leads to).
- **The target:** a virtual goal reached from every song, at β × the song's angle to the predicted goal vibe.

D* Lite's own heuristic (α_floor × angle) only speeds the search: the chosen plan is the cheapest one, which the gate checks against Dijkstra.

**On the card.** Every matrix product, forward and backward, runs on the 7120A:
- the encoder's training and inference;
- every song's score, its cosine to the goal, and the graph's similarity matrix;
- the preference network's training.

The card side is a daemon (`card/vgd.c`). It takes op lists through the last 256 MiB of the card stack's shared host-memory window, moves data with a card-owned DMA channel, and runs `lib/gemm.c`: an 8 × 48 register-tiled SGEMM with KNC intrinsics and two-level software prefetch, on 118 hardware threads. The protocol is in [`docs/protocol.md`](docs/protocol.md).

## Measured

All on one 7120A (61 cores, 1.238 GHz, 16 GB GDDR5) in a Ryzen 7 5800X desktop. Raw logs are kept outside the repository.

**Mel extraction**
- Gate: a 440 Hz sine of amplitude 1/8 lands in band 9 at −25.4 dB (hand-computed −25.3), and 3000 Hz in band 37.
- Library: 6,028 files, 641.4 h of audio, 1,183.7 s with 3 threads, 1 file with no audio stream.

**GEMM**
- Correctness: 11 shapes, worst relative error 2.25 × 10⁻⁷ against double precision. Padding is untouched and transposes are bit-exact.
- Speed: median GFLOP/s, 20 runs each; peak with 59 cores × 2 threads is 2,337, calculated.

| Shape (M × N × K) | No prefetch | With prefetch |
|---|---|---|
| 32768 × 256 × 96 | 366.1 | 346.9 |
| 16384 × 256 × 512 | 248.0 | 436.5 |
| 16384 × 512 × 256 | 194.4 | 350.1 |
| 512 × 256 × 16384, A transposed | 83.5 | 259.5 |
| 2048 × 2048 × 2048 | 87.0 | 217.2 |

**Encoder** (6,027 songs: 5,430 train, 597 held out by path hash)
- 126 ms per step: 30.2 GFLOP on the card at 302 GFLOP/s.
- 150 epochs in about 7 minutes.

| | Held-out crop retrieval, top-1 same song | 10 nearest songs: same folder | 10 nearest songs: same artist |
|---|---|---|---|
| chance | 0.0013 | 0.0744 | 0.0008 |
| hand-made baseline (band mean and spread) | | 0.2657 | 0.0286 |
| encoder, untrained | 0.0302 | 0.1631 | 0.0312 |
| encoder, epoch 25 | 0.2451 | 0.3232 | 0.0460 |
| encoder, epoch 75 | 0.3168 | 0.3544 | 0.0502 |
| encoder, epoch 150 (shipped) | 0.3508 | 0.3656 | 0.0542 |

**Vibe graph** (`graphstat`)

| | Edges | Degree median | p99 | Max | Top hubs |
|---|---|---|---|---|---|
| plain cosine | 136,648 | 20 | 49 | 75 | 8 of 10 are lofi or chillhop mixes |
| CSLS (used) | 115,744 | 18 | 30 | 38 | no mixes |

**Planner** (`dstest -c`): 300 steps on the real graph, with costs changing as the daemon changes them, including 102 new targets, 57 jumps and 17 new α values.
- 0 cost mismatches and 0 path-sum mismatches against Dijkstra.
- The honest cost: because the goal connects to every song, each new target touches every vertex of D* Lite's backward search. D* Lite expands a median of 819 vertices per plan, against 51 for a fresh Dijkstra, and takes 1.685 ms against 0.016 ms. One plan per song makes that irrelevant.

**Preference network** (`prefcheck`)
- Backward pass against central differences, three seeds:
  - every parameter passes at ε = 10⁻³ or 10⁻⁴;
  - the misses at 10⁻³ sit on ReLU kinks, and the misses at 10⁻⁴ are fp32 round-off;
  - one seed passes everything at 10⁻³, worst 6.9 × 10⁻³.
- Cold start: 50 untrained decisions through D* Lite with 0 repeats, 0 hops off the graph, 0 empty plans and 0 NaN. Scoring all 6,027 songs on the card takes 2.08 ms.

**Skips** were tested on a private D-Bus session, with a second VLC on dummy audio and video outputs, so nothing touched the real player or the speakers. Three runs of four skips after the fixes below (`tools/skiptest.sh`; a first run found them):
- **Detection:** every skip was classified `next`. VLC reports `Stopped` for an instant while it switches tracks, so a stop only counts once it has lasted 1 s.
- **Learning and switch:** the network took 16 Adam steps on the card in 87 to 92 ms. VLC played the re-planned song 0.115 to 0.233 s after Next (9 switches in the runs after the fixes).
- **Agreement:** when the repaired plan agreed with the queued pick, the pick was kept.
- **Your tracks:** a track lined up by hand played and was not replaced.
- **Nothing playing is removed:** after every switch, every track that played since Next was still in the playlist (0 removed in 4 skips). An earlier build deleted the stale pick, the song VLC had just started on your Next, which looked like the playing song vanishing.
- **Copies:** a skipped song's copy in another folder was chosen 0 times, after the fix. Before it, a skip of "Pegboards - Hero" brought back its copy from `ytmusic/`.

**Copies** (`dups`): nearest-song cosine ≥ 0.9999 for 385 songs (most exactly 1.000000), against 17 between 0.9995 and 0.9999.

**History**: 72 plays imported from the vlc-discord journal. 13 titles matched no library file. 50 plays were at 90 % or more and 15 under 10 %.

**End to end** (2026-10-24, live):
- The track playing when `vibed` started ran to its natural end (448 of 448 s).
- The network took 16 Adam steps on the card over all 73 logged plays.
- VLC then played the song `vibed` had queued after it, logged with source `vibe`.
- `vibed` queued the next one.

## Requirements

- **Card.** An Intel Xeon Phi x100 card booted with the Intel Phi 7120A stack (`phictl` with `--host-mem 2G`, which provides `/dev/shm/phi-hostmem` on the host and `/dev/phihost` on the card), with root ssh to the card on port 2222.
- **Card toolchain.** The Intel-Phi-LLVM clang (`-march=knc`) for `lib/gemm.c` and `lib/pool.c`, and the stack's `knc-cc` and `phi-isa-audit`. `build.sh` finds them under `~/.cache/intel-phi-llvm-build` and `~/Intel Phi 7120A`; override with `PHI_LLVM_BIN`, `PHI_SYSROOT_7120A` and `PHI_TREE`.
- **Host.** gcc, FFmpeg 9 libraries (libavformat, libavcodec, libswresample, libavutil), FFTW3 (single precision), libsystemd (sd-bus), VLC 3 with its D-Bus control interface.

## Build and run

```sh
./build.sh
echo /path/to/music > ~/.config/phivibe/library
echo 'Podcasts' > ~/.config/phivibe/exclude    # optional: top folders to leave out
systemd-run --user -p IOSchedulingClass=idle -p Nice=19 build/vibe-mel -j 3

# the card daemon (the keeper below does this by itself)
mkdir -p ~/.local/share/phivibe ~/.local/lib/phivibe
cp build/vgd-card ~/.local/share/phivibe/vgd-card
cp scripts/vgd-keeper.sh ~/.local/lib/phivibe/

build/vgtest -q                # GEMM gate
build/vibe-train -e 150 -v 25  # train, measure, write ~/.local/share/phivibe/emb.bin
build/dstest -c                # planner gate
build/prefcheck                # gradient check and cold start

cp build/vibed ~/.local/bin/vibed.new && mv ~/.local/bin/vibed.new ~/.local/bin/vibed
cp units/*.service ~/.config/systemd/user/ && systemctl --user daemon-reload
build/vibed -i -n              # once: import the vlc-discord journal, watch only (VLC untouched); stop with ^C
systemctl --user enable --now phivibe-vgd.service phivibe.service
cat ~/.local/share/phivibe/status.txt
```

**`vibed` and VLC**
- It never pauses, seeks or changes the volume. It adds one track right after the current one. The only time it starts a track itself is the switch to the re-planned song right after you skip one of its picks.
- It adds nothing while you have tracks queued after the current one, or while VLC shuffles. If you add tracks while its pick is waiting, it takes the pick back: at once when VLC sends `TrackAdded` in time, otherwise at the next track change, when the play order can be read exactly.
- Its files live in `~/.local/share/phivibe`: `events.tsv` (every play), `pref.bin` (the network), `status.txt` (now, the plan, the weights) and `pending` (the waiting pick).

## Limits

- **Card boot.** The card has to be booted (`phi up`). The keeper unit restarts `vgd` after a card reboot, but nothing boots the card by itself. Until the card is up, `vibed` logs plays and retries every 30 s.
- **New songs.** Songs added to the library are not picked until `vibe-mel` and `vibe-train -E` run again.
- **History.** 72 plays is little history: the network adapts as you listen.
- **Monitoring.** `phitop` has no line for `vgd` yet. Its threads show in phitop's core grid while it works.

## Notes on the card

- **Intrinsics.** This compiler's `kncintrin.h` has no gather, masked move or `set_epi32`. The transpose is built from broadcasts merged with masked integer adds.
- **Broadcasts.** The broadcast into the FMA's memory operand is not folded: each one is a separate `vbroadcastss`.
- **Prefetch.** `__builtin_prefetch` emits nothing for this target, so prefetches are inline `vprefetch0` and `vprefetch1`.
- **Idle workers.** Idle pool workers spin for a set time using `DELAY`, KNC's replacement for `PAUSE`. Letting them sleep after about 1.6 ms made every short op wait milliseconds for wake-ups. Fixing it halved the training step, from 275 ms to 129 ms.
- **ISA audit.** `phi-isa-audit` reports 5 CMOVs in `gemm_task`. These are a decoder desync inside three-byte-VEX `vprefetch0` instructions: `llvm-objdump --mcpu=knc` decodes the same bytes as `vprefetch0` and finds no CMOV in the binary, and the binary runs on the card.

## Notes on VLC 3

- **Tracks property.** `TrackList.Tracks` is built from VLC's play-order array, which is rebuilt only when playback moves to another item. It is exact right after a track change and stale after an `AddTrack`.
- **TrackAdded signal.** `TrackAdded` arrives under the interface name `MediaPlayer2.TrackList`, without the `org.mpris.` prefix. When it does not arrive in time, `vibed` finds its new item by probing track ids upward, which works because VLC only ever counts them up.
- **Track ids.** Ids restart when VLC restarts. `vibed` therefore checks a track's URL before it removes anything.

## Licence

MIT, see [`LICENSE.md`](LICENSE.md). The DMA helper and the thread pool are copied from the author's other MIT
projects; each file names its source.
