#!/usr/bin/env python3
"""Print a title.toml `[runtime.memory_layout]` block for a homebrew intake.

Module segments come from the intake module.json (named
`<module>-text|rodata|data|bss`, which the runtime maps to Horizon memory
states for svcQueryMemory). Heap, main thread stack and the thread-local
region are added at the given addresses.

Usage: nso-memory-layout.py <intake/module.json> [--heap BASE SIZE]
       [--stack TOP SIZE] [--tls BASE]
"""
import argparse
import json


def region(name, base, size, perms):
    r, w, x = (perms[0] == "r", perms[1] == "w", perms[2] == "x")
    flags = ", ".join(f"{k} = {str(v).lower()}" for k, v in (("read", r), ("write", w), ("execute", x)))
    return f'  {{ name = "{name}", base = {base:#x}, size = {size:#x}, permissions = {{ {flags} }} }},'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("module_json")
    parser.add_argument("--heap", nargs=2, default=["0x100000000", "0x100000000"])
    parser.add_argument("--stack", nargs=2, default=["0x80000000", "0x100000"])
    parser.add_argument("--tls", default="0x70000000")
    args = parser.parse_args()
    to_int = lambda value: int(value, 0)

    lines = ["[runtime.memory_layout]", "regions = ["]
    modules = json.load(open(args.module_json))["modules"]
    page = lambda value: (value + 0xFFF) & ~0xFFF
    for module in sorted(modules, key=lambda module: module["base"]):
        base = module["base"]
        bss = module.get("bss") or {}
        segments = sorted(module["segments"], key=lambda segment: segment["memory_offset"])
        for segment in segments:
            start = segment["memory_offset"]
            end = start + segment["memory_size"]
            # Horizon maps segments in whole pages, so a module is contiguous
            # (the SDK walks text -> rodata -> data with svcQueryMemory). .data
            # ends where .bss starts; everything else ends on a page.
            if not (segment["name"] == "data" and bss.get("size")):
                end = page(end)
            lines.append(region(f'{module["name"]}-{segment["name"]}', base + start, end - start,
                                segment["permissions"]))
        if bss.get("size"):
            start = bss["memory_offset"]
            lines.append(region(f'{module["name"]}-bss', base + start, page(start + bss["size"]) - start,
                                "rw-"))
    heap_base, heap_size = map(to_int, args.heap)
    stack_top, stack_size = map(to_int, args.stack)
    lines.append(region("heap", heap_base, heap_size, "rw-"))
    # One 0x200-byte slot per guest thread; the main thread takes the first.
    lines.append(region("tls", to_int(args.tls), 0x40000, "rw-"))
    lines.append(region("stack", stack_top - stack_size, stack_size, "rw-"))
    lines.append("]")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
