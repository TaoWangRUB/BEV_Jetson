# cuNLS CUDA-10.2 / TX2 port patch

`0001-cuda102-tx2-port.patch` adapts **cuNLS** (`Release_07_13_2026`) to build on the
Jetson TX2 — nvcc 10.2 / gcc-8 / C++14 / sm_62. cuNLS is the CUDA nonlinear
least-squares backend cuVSLAM v17 needs for `OdometryMode::Multisensor`
(`-DUSE_CUNLS=ON`); upstream targets CUDA 12 and sm_75+.

cuNLS is not a submodule. `scripts/build_cuvslam_tx2gpu.sh` downloads the release
tarball into `build/`, extracts it, applies this patch, and hands the result to
cuVSLAM's FetchContent via `-DFETCHCONTENT_SOURCE_DIR_CUNLS=...`, so neither the
upstream download nor its `PATCH_COMMAND` ever runs.

## The blocker, and why it is survivable

cuNLS links **cuDSS**, NVIDIA's sparse direct solver. cuDSS ships as a **prebuilt
binary only** — there is no source — and NVIDIA publishes it exclusively for
CUDA 12 and 13 (`libcudss-linux-{x86_64,sbsa,aarch64}-*_cuda1{2,3}-archive.tar.xz`).
There is no CUDA 10.2 build and no Tegra build. It cannot be ported.

It also turns out not to matter: **cuVSLAM never selects cuDSS.**
`libs/pnp/multisensor_pose_estimator.cpp` explicitly asks for
`SparseLinearSolverType::DenseQR` (pure cuSOLVER — `Sgeqrf`/`Sormqr`/`Strsm`, all
present in CUDA 10.2), and every other cuVSLAM path takes cuNLS's own default of
`BlockSparsePCG` (a plain CUDA kernel). So the patch removes the cuDSS backend
outright rather than trying to satisfy it.

## What it changes (22 files)

1. **cuDSS excised** — dropped from `CUNLS_LIBS`, `bundle_static_dependencies`, and
   both object libraries; `cudss_helper.cpp` and `cudss_sparse_linear_solver.cpp` are
   removed from the build (the files remain on disk but are never compiled), and the
   factory's `cuDSS` case now throws. `<cudss.h>` was only ever included by those two
   `.cpp` files, so every header stays cuDSS-free and the public API is unchanged.
2. **Arch / standard / compiler deferred to the parent** — upstream hard-codes
   `CMAKE_CUDA_ARCHITECTURES 75 80 86 89` and `CMAKE_CUDA_COMPILER`, which overrides
   cuVSLAM's sm_62 choice. Now guarded with `if(NOT DEFINED ...)`. Adds
   `CMAKE_CUDA_STANDARD 14` — nvcc 10.2 has no C++17 device mode at all.
3. **`<cuda/std/array>` / `<cuda/std/limits>` replaced.** CUDA 10.2 ships libcu++ 1.0,
   which provides only `atomic, cassert, cfloat, climits, cstddef, cstdint,
   type_traits, version` — neither header exists. `types.h` gains a `DeviceArray`
   aggregate with the same layout and the subset of the `std::array` interface cuNLS
   uses; `numeric_limits<float>::min()` / `<int>::max()` become `FLT_MIN` / `INT_MAX`.
4. **C++17 → C++14 in everything a `.cu` pulls in:** `if constexpr` (every branch is
   well-formed for every instantiation, so a plain `if` is equivalent and still
   folded), nested `namespace cunls::profiler`, `std::string_view` → `const std::string&`
   in `log.h`, `std::tuple_size_v` → `::value`, a fold expression → an initialiser-list
   expansion, and `std::is_base_of_v` → `::value`.
5. **`DomainRange` gains a move constructor.** It is returned by value from
   `Domain::CreateDomainRange`; C++17 elides that construction outright, but C++14
   still requires an accessible move constructor even when eliding, so the deleted one
   is a hard error. The destructor is already guarded on `handle_`, so a
   transfer-and-null move preserves the RAII semantics exactly.

## Regenerating (after a cuNLS version bump)

```bash
CUNLS_VERSION=<tag> ./scripts/port/regen_cunls_patch.sh
```

Then re-run the pre-flight below before spending time on the board.

## Pre-flight (on the dev host, no TX2 needed)

The host's nvcc accepts `-arch=sm_62 -std=c++14`, which catches device-side C++14
breakage without a board round-trip. This is how fixes 4 and 5 were found — a plain
grep for C++17 constructs missed both.

**The host nvcc only warns about C++17.** At `-std=c++14` it accepts structured
bindings with `warning #3356-D: structured bindings are a C++17 feature` and exits 0,
while nvcc 10.2's older frontend rejects them outright. So exit status alone is not a
C++17 oracle — the pre-flight must fail on that warning too:

```bash
D=build/cuNLS-Release_07_13_2026
for f in $(find $D/cunls -name '*.cu'); do
  if nvcc -std=c++14 -arch=sm_62 -Wno-deprecated-gpu-targets -I$D -c "$f" -o /dev/null 2>/tmp/e; then
    grep -qE 'warning #3356|C\+\+17 feature' /tmp/e && echo "C++17 IN $f"
  else echo "FAIL $f"; fi
done
```

Grepping for C++17 constructs is a weak substitute — three were missed that way. Note
in particular that structured bindings need `auto\s*(const)?\s*[&*]*\s*\[`: the
common spelling here is `const auto &[a, b]`, which a plain `auto\s*\[` never matches.

All 54 `.cu` (warning-clean) and all 12 compiled `.cpp` pass. (`cudss_helper.cpp` and
`cudss_sparse_linear_solver.cpp` still fail on the missing `<cudss.h>` — expected;
they are excluded from the build.)

Note this proves C++14/sm_62 *syntax* only. It uses the host's CUDA 12 toolkit, so it
cannot vouch for CUDA-10.2 API availability; that is what the board build is for.
