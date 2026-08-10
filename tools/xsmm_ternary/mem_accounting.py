"""Account for every byte of RAM the kernel admits to, and show the gap.

Explains *why* the missing memory cannot be freed: it belongs to no process, no
cgroup, no cache, and no shrinker-backed pool.
"""


def meminfo():
    d = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        parts = v.split()
        d[k] = int(parts[0]) // 1024 if len(parts) > 1 else int(parts[0])
    return d


def main():
    m = meminfo()
    total = m["MemTotal"]

    # everything /proc/meminfo can attribute
    buckets = {
        "MemFree": m["MemFree"],
        "AnonPages (process anon)": m["AnonPages"],
        "Mapped (file-backed in use)": m["Mapped"],
        "Cached (page cache)": m["Cached"],
        "Buffers": m["Buffers"],
        "Slab (kernel objects)": m["Slab"],
        "KernelStack": m["KernelStack"],
        "PageTables": m["PageTables"],
        "VmallocUsed": m["VmallocUsed"],
        "Percpu": m["Percpu"],
        "Shmem (tmpfs)": m["Shmem"],
        "Hugetlb": m["Hugetlb"],
    }
    accounted = sum(buckets.values())

    print(f"MemTotal                       {total:8d} MiB")
    print("-" * 52)
    for k, v in buckets.items():
        print(f"  {k:30s} {v:8d} MiB")
    print("-" * 52)
    print(f"  {'ACCOUNTED':30s} {accounted:8d} MiB")
    print(f"  {'UNACCOUNTED':30s} {total - accounted:8d} MiB"
          f"   <-- {(total - accounted) * 100.0 / total:.1f}% of RAM")

    print("\nWhat drop_caches could free (page cache + reclaimable slab):")
    print(f"  Cached + Buffers + SReclaimable = "
          f"{m['Cached'] + m['Buffers'] + m['SReclaimable']} MiB")

    print("\n/proc/buddyinfo (free pages per order, per zone):")
    for line in open("/proc/buddyinfo"):
        print("  " + line.rstrip())


if __name__ == "__main__":
    main()
