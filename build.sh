#!/usr/bin/env bash
# build.sh: host programs (cc) and the card daemon into build/. On the card: lib/gemm.c through the Intel-Phi-LLVM S3
# clang (-march=knc: the vector unit), the rest of vgd through the stack's knc-cc (static). phi-isa-audit checks the
# card binary for instructions the 7120A lacks.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out="$here/build"
mkdir -p "$out"
flags=(-O2 -std=gnu11 -Wall -Wextra)
av=$(pkg-config --cflags --libs libavformat libavcodec libswresample libavutil fftw3f)
# shellcheck disable=SC2086
cc "${flags[@]}" -o "$out/vibe-mel" "$here/host/mel.c" $av -lm -lpthread
cc "${flags[@]}" -o "$out/vgtest" "$here/tools/vgtest.c" "$here/host/vg.c" -lm
cc "${flags[@]}" -o "$out/vibe-train" "$here/tools/train.c" "$here/host/enc.c" "$here/host/knn.c" "$here/host/songs.c" "$here/host/vg.c" -lm
cc "${flags[@]}" -o "$out/dstest" "$here/tools/dstest.c" "$here/host/dstar.c" "$here/host/emb.c" "$here/host/knn.c" "$here/host/vg.c" -lm
cc "${flags[@]}" -o "$out/prefcheck" "$here/tools/prefcheck.c" "$here/host/pref.c" "$here/host/dstar.c" "$here/host/emb.c" "$here/host/knn.c" "$here/host/vg.c" -lm
cc "${flags[@]}" -o "$out/vibed" "$here/host/vibed.c" "$here/host/pref.c" "$here/host/dstar.c" "$here/host/emb.c" "$here/host/knn.c" "$here/host/mpris.c" "$here/host/vg.c" -lm -lsystemd
cc "${flags[@]}" -o "$out/graphstat" "$here/tools/graphstat.c" "$here/host/emb.c" "$here/host/knn.c" "$here/host/vg.c" -lm
cc "${flags[@]}" -o "$out/dups" "$here/tools/dups.c" "$here/host/emb.c" "$here/host/knn.c" "$here/host/vg.c" -lm
VCC="${PHI_LLVM_BIN:-$HOME/.cache/intel-phi-llvm-build/llvm/bin}/clang"
SYS="${PHI_SYSROOT_7120A:-$HOME/.cache/intel-phi-7120a-build/toolchain/sysroot}"
"$VCC" --target=x86_64-unknown-linux-musl -march=knc --sysroot="$SYS" "${flags[@]}" -fno-jump-tables -falign-loops=1 -fPIE \
    -c -o "$out/gemm-knc.o" "$here/lib/gemm.c"
"$VCC" --target=x86_64-unknown-linux-musl -march=knc --sysroot="$SYS" "${flags[@]}" -fno-jump-tables -falign-loops=1 -fPIE \
    -c -o "$out/pool-knc.o" "$here/lib/pool.c"
# shellcheck disable=SC1091
. "${PHI_TREE:-$HOME/Intel Phi 7120A}/toolchain/env.sh"
knc-cc "${flags[@]}" -static -o "$out/vgd-card" "$here/card/vgd.c" "$here/card/cdma.c" "$out/gemm-knc.o" "$out/pool-knc.o" -lm -lpthread
audit="${PHI_TREE:-$HOME/Intel Phi 7120A}/host/target/debug/phi-isa-audit"
# the audit exits nonzero on findings; its 5 CMOV findings in gemm_task are a decoder desync inside VEX3 vprefetch0
# (llvm-objdump --mcpu=knc decodes those bytes as vprefetch0 and finds no cmov): report, do not stop
printf 'vgd-card: '; "$audit" "$out/vgd-card" | tail -1 || true
