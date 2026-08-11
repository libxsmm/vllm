#!/bin/bash
# Fetch the LIBXSMM checkout that provides the int2 GEMM support.
#
# The int2 microkernel (LIBXSMM_GEMM_FLAG_INTERPRETE_A_AS_INT2_VNNI4_INTLV, and
# the AVX2/SRF generator behind it) is not on libxsmm's main branch. The commit
# below is the one tpp-pytorch-extension's `2bit_gemm` branch pins as its
# libxsmm submodule; no branch points at it, but it can be fetched by SHA.
#
# The build does this automatically if the tree is missing, so this script is
# only needed to pre-populate it or to refresh a stale checkout.
set -eo pipefail

COMMIT=84bff4a46ef27f445259664ac93abf5c33a02148
REPO=https://github.com/libxsmm/libxsmm.git
DST=${1:-/data/nfs_home/egeorgan/cpu_ternary_vllm/third_party/libxsmm}

if [[ -f "$DST/src/libxsmm_main.c" ]]; then
    have=$(git -C "$DST" rev-parse HEAD 2>/dev/null || echo unknown)
    if [[ "$have" == "$COMMIT" ]]; then
        echo "LIBXSMM already at $COMMIT: $DST"
        exit 0
    fi
    echo "LIBXSMM at $DST is $have, expected $COMMIT" >&2
fi

mkdir -p "$DST"
cd "$DST"
git init -q .
git remote add origin "$REPO" 2>/dev/null || git remote set-url origin "$REPO"
git fetch -q --depth 1 origin "$COMMIT"
git checkout -q FETCH_HEAD

test -f src/libxsmm_main.c
grep -rq INTERPRETE_A_AS_INT2_VNNI4_INTLV include/ \
    || { echo "fetched tree lacks the int2 GEMM flag" >&2; exit 1; }

echo "LIBXSMM $(git rev-parse HEAD) -> $DST"
