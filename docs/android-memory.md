# Android guest memory and thread stacks

Status: `port/andmem`. This documents the runtime's memory behavior on
Android ARM64 and the limits that remain. It is not a device-validation
result: the target check is a cross-compile, and no Android title has run
with this branch yet.

## Guest pages versus host pages

Horizon's page size stays fixed at 4 KiB (`fastmem::PAGE`): the generated
memory layout, the checked access path, `svcQueryMemory`, and the guest's
expectations all use it.

The host page size is detected once with `sysconf(_SC_PAGESIZE)` instead of
assuming 4 KiB. Android 15 devices commonly use 16 KiB pages, and some
Linux hosts use 64 KiB. The detected size controls:

- the size and address alignment of every `mmap`/`MAP_FIXED` commit;
- which host pages are fresh, shared, or released;
- the address and length passed to `mmap` when pages become inaccessible.

`RECOMP_HOST_PAGE_SIZE=<power of two>` overrides detection for tests. The
value must be at least 4 KiB, at most 1 MiB, and a multiple of the real host
page size; otherwise it is ignored with a message. It is a test hook, not a
performance setting.

Because a host page can hold several guest pages, a mapped host page makes
its neighbouring guest pages accessible too. Guest offsets are still 1:1, so
every load and store lands on the same bytes it would on hardware:

- the extra bytes initially read as zero;
- a region that later claims one of those guest pages zeroes the page (or
  only its own bytes when another region shares the guest page);
- releasing a region zeroes its released bytes when a neighbour keeps the
  host page mapped, so old data is not observable.

The direct path still cannot report an unmapped 4 KiB guest page inside a
mapped 16 KiB host page as a fault; the checked path keeps the exact 4 KiB
behavior. This is the deliberate trade-off for using fastmem on 16 KiB
hosts. `write_watch` is disabled when the host page is not 4 KiB because
userfaultfd protection is host-page granular.

## The 39-bit reservation and its fallback

On startup the runtime tries one reservation:

```text
mmap(NULL, 512 GiB, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE)
```

The 39-bit guest address space is 512 GiB. Many Android devices have a
39-bit user virtual address space, so a 512 GiB reservation cannot fit next
to the executable, libraries, heap, and thread stacks. When `mmap` fails:

1. the runtime prints `could not reserve the 39-bit guest address space
   (<errno>); memory accesses stay checked`;
2. no guest base is published (`recomp_abi::memory_base() == 0`);
3. every region is allocated as a checked heap buffer (`Pages::Heap`);
4. all loads, stores, copies, permissions, and unmapped accesses go through
   the checked runtime path, with the same guest addresses and 4 KiB
   semantics.

The runtime never creates a smaller rebased mapping. A windowed 1:1 mapping
would need the ABI's generated memory view to carry a smaller limit; without
that, an access just past the window could hit unrelated host memory. A
region that is outside the reservation (or whose host pages cannot be
committed) disables the direct base the same way, so generated code switches
to the checked path rather than touching an address that is not mapped.

`RECOMP_FASTMEM=0` forces the checked path. This is also how the
`memory_fallback` integration test runs the fallback deterministically on a
desktop host. `RECOMP_WATCH_STORE` disables the direct path because it has to
see every store.

Devices with a 48-bit or larger user address space can still get the full
reservation. Whether a specific device does is visible in the startup log
and can be checked from the guest side by eye on the first run; no Android
device has been run on this branch yet.

## Thread stacks

Host stacks for guest threads were 256 MiB each, which is a large virtual
reservation on a 39-bit device with many guest threads. The defaults are now:

| Host | Default guest-thread stack |
|---|---|
| Linux/desktop (unchanged) | 256 MiB |
| Android (`target_os = "android"`) | 16 MiB |

`RECOMP_THREAD_STACK_MB=<1..1024>` overrides the default on any host. `0`,
a non-number, or an out-of-range value uses the platform default; values
above 1024 are clamped to 1024. The setting applies to threads started by
`svcStartThread`; the runtime's monitor threads keep their small defaults.
Very deep lifted call chains on Android can still exhaust 16 MiB; raise the
setting for such a title and watch for the stack-overflow handler.

## Related host features

- The sampling profiler (`RECOMP_PROFILE`) is Linux-x86-64-only and is not
  built for Android.
- Write watching uses Linux `userfaultfd`/`PAGEMAP_SCAN`; it is not built for
  Android and is additionally disabled on any host whose page size is not
  4 KiB. The renderer's hashing fallback covers vertex ranges.
- Windows has its own reservation and exception layer on `port/winrt`; this
  branch does not change it.

## Verification and remaining risks

Cross-checks run for this change:

```text
cargo test -p recomp-runtime
RECOMP_HOST_PAGE_SIZE=16384 cargo test -p recomp-runtime -- --skip write_watch
cargo clippy -p recomp-runtime --all-targets -- -D warnings
cargo check --target aarch64-linux-android -p recomp-runtime
```

Remaining device risks:

- On a 39-bit Android device fastmem is expected to be unavailable, so guest
  memory access is slower than the desktop direct path. It is correct, but
  performance must be measured on such a device.
- A 16 KiB host page relaxes unmapped-fault granularity within that page on
  the direct path. Released bytes are zeroed, but a title that relies on a
  fault for a 4 KiB hole inside an otherwise mapped host page will only see
  the checked path's exact behavior when `RECOMP_FASTMEM=0`.
- `MAP_NORESERVE` still needs virtual address space; kernels configured with
  strict overcommit (`vm.overcommit_memory=2`) or app-specific address-space
  limits can refuse the reservation even on a 48-bit device. The checked
  fallback handles that, at the same performance cost.
- Android's low-memory killer and per-app memory limits are outside the
  runtime's control. Checked heap regions compile to normal process memory,
  and many guest threads still cost virtual address space for their stacks.
- The 16 MiB Android stack default is a judgment call; a title with unusually
  deep lifted recursion may need `RECOMP_THREAD_STACK_MB` raised.
- No game run, emulator run, or Android device run was performed for this
  branch. The `aarch64-linux-android` result is a type-check only; SIMD,
  signal delivery, and 16 KiB-page behavior still need an on-device smoke
  test.
