"""Probe whether the node's unaccounted memory is reclaimable under pressure.

Allocates a bounded balloon, touches it (forcing the kernel to reclaim whatever
it can), then frees it. If the ~88 GiB of unaccounted memory were any kind of
reclaimable cache, `used` would drop afterwards.

Bounded on purpose: this node has already been wedged once by memory pressure.
"""

import ctypes
import os
import sys
import time


def meminfo():
    d = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        d[k] = int(v.split()[0]) // 1024  # MiB
    return d


def show(tag):
    m = meminfo()
    print(f"{tag:24s} MemFree={m['MemFree']:6d} MiB  MemAvailable={m['MemAvailable']:6d} MiB  "
          f"Cached={m['Cached']:6d}  Slab={m['Slab']:5d}  SwapFree={m['SwapFree']:6d}")
    return m


def main():
    gib = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    before = show("before balloon")

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    MADV_DONTNEED = 4

    chunk = 512 * 1024 * 1024
    n = (gib * 1024**3) // chunk
    blocks = []
    print(f"\nallocating {gib} GiB in {n} x 512 MiB chunks and touching every page...")
    try:
        for i in range(n):
            b = bytearray(chunk)
            # touch every 4 KiB page so the kernel must actually back it
            mv = memoryview(b)
            for off in range(0, chunk, 4096):
                mv[off] = 1
            blocks.append(b)
            if (i + 1) % 4 == 0:
                m = meminfo()
                print(f"  +{(i + 1) * chunk // 2**20:6d} MiB touched, "
                      f"MemFree={m['MemFree']:6d}  SwapFree={m['SwapFree']:6d}")
                if m["SwapFree"] < 4096:
                    print("  !! swap nearly exhausted, stopping early")
                    break
    except MemoryError:
        print("  MemoryError - stopping")

    show("\nat peak")

    # hand the pages back to the kernel explicitly, then drop the refs
    for b in blocks:
        addr = ctypes.addressof((ctypes.c_char * len(b)).from_buffer(b))
        libc.madvise(ctypes.c_void_p(addr), ctypes.c_size_t(len(b)), MADV_DONTNEED)
    blocks.clear()
    libc.malloc_trim(0)
    time.sleep(3)

    after = show("after free + trim")
    print(f"\nnet change in MemFree: {after['MemFree'] - before['MemFree']:+d} MiB")
    print("If the unaccounted memory were reclaimable cache, MemFree would now be"
          " far higher than before.")


if __name__ == "__main__":
    main()
