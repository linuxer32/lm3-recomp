#!/usr/bin/env python3
"""Name the functions in a RECOMP_PROFILE sample file.

Usage: profile-report.py <executable> <profile> [top]

Prints the hottest functions overall, the busiest threads, and each busy
thread's hottest functions. Generated guest functions (fn_<address>) keep
their guest address; everything else is grouped by Rust path. Samples in
libraries are named by the library's exported symbols ([libc.so.6]
memmove), or by the library alone when it has none there.
"""

import bisect
import os
import collections
import re
import subprocess
import sys


def symbols(executable):
    output = subprocess.run(
        ["nm", "-C", "--defined-only", "-n", executable],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    addresses, names = [], []
    for line in output.splitlines():
        parts = line.split(" ", 2)
        if len(parts) < 3 or parts[1] not in "tTwW":
            continue
        addresses.append(int(parts[0], 16))
        names.append(parts[2])
    return addresses, names


def library_symbols(path, start):
    """A library's exported functions, by address where it is mapped: its
    executable mapping starts at `start` (the profile's `map` line)."""
    try:
        headers = subprocess.run(
            ["readelf", "-lW", path], check=True, capture_output=True, text=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return [], []
    # The executable LOAD segment's address in the file's own layout.
    text = next(
        (
            int(fields[2], 16)
            for fields in (line.split() for line in headers.splitlines())
            if len(fields) >= 8 and fields[0] == "LOAD" and "E" in fields[6:-1]
        ),
        None,
    )
    if text is None:
        return [], []
    bias = start - (text & ~0xFFF)
    # The static symtab first: an unstripped library (the Android
    # `librecomp_game.so`) names every function there, while its exports
    # alone would name the runtime's few exported entry points for most of
    # it. Stripped system libraries fall back to their dynamic symbols.
    addresses, names = [], []
    for command in (
        ["nm", "-C", "--defined-only", "-n", path],
        ["nm", "-D", "--defined-only", "-n", path],
    ):
        try:
            output = subprocess.run(
                command, check=True, capture_output=True, text=True
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            continue
        for line in output.splitlines():
            parts = line.split(" ", 2)
            if len(parts) < 3 or parts[1] not in "tTiwW":
                continue
            addresses.append(int(parts[0], 16) + bias)
            names.append(parts[2].split("@")[0])
        if addresses:
            break
    return addresses, names


def short(name):
    name = re.sub(r"::h[0-9a-f]{16}$", "", name)
    match = re.search(r"(fn_[0-9a-f]{16})(__p\d+)?", name)
    if match:
        return match.group(1) + (match.group(2) or "")
    return name


def main():
    executable, profile = sys.argv[1], sys.argv[2]
    top = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    addresses, names = symbols(executable)
    samples = []
    libraries = []
    resolved = {}
    with open(profile) as file:
        for line in file:
            if line.startswith("base "):
                continue
            if line.startswith("map "):
                _, start, end, path = line.split(" ", 3)
                path = path.strip()
                # A device's path (Android): the executable given here is
                # the local copy of that library.
                if not os.path.exists(path) and os.path.basename(path) == os.path.basename(
                    executable
                ):
                    path = executable
                libraries.append((int(start, 16), int(end, 16), path))
                continue
            pc, tid = line.split()[-2:] if line.startswith("x ") else line.split()
            if line.startswith("x ") or pc == "-":
                address = int(pc, 16) if pc != "-" else -1
                mapping = next(
                    ((start, path) for start, end, path in libraries if start <= address < end),
                    None,
                )
                if mapping is None:
                    samples.append(("[?]", tid))
                    continue
                if mapping not in resolved:
                    resolved[mapping] = library_symbols(mapping[1], mapping[0])
                library = mapping[1].rsplit("/", 1)[-1]
                lib_addresses, lib_names = resolved[mapping]
                index = bisect.bisect_right(lib_addresses, address) - 1
                name = f" {lib_names[index]}" if index >= 0 else ""
                samples.append((f"[{library}]{name}", tid))
                continue
            index = bisect.bisect_right(addresses, int(pc, 16)) - 1
            samples.append((short(names[index]) if index >= 0 else "?", tid))
    total = len(samples)
    if not total:
        print("no samples")
        return
    print(f"{total} samples")
    print("\nhottest functions:")
    for name, count in collections.Counter(name for name, _ in samples).most_common(top):
        print(f"  {100 * count / total:5.1f}%  {name}")
    threads = collections.Counter(tid for _, tid in samples)
    print("\nthreads:")
    for tid, count in threads.most_common(8):
        print(f"  {100 * count / total:5.1f}%  tid {tid}")
        hot = collections.Counter(name for name, sample_tid in samples if sample_tid == tid)
        for name, hits in hot.most_common(8):
            print(f"           {100 * hits / count:5.1f}%  {name}")


if __name__ == "__main__":
    main()
