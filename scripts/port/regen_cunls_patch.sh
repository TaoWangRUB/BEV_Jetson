#!/usr/bin/env bash
# Regenerate patch/cunls/0001-cuda102-tx2-port.patch from a pristine cuNLS tarball.
# cuNLS is a FetchContent dependency of cuVSLAM (USE_CUNLS), needed for
# OdometryMode::Multisensor. Upstream targets CUDA 12 / sm_75+; this replays every
# edit needed for nvcc 10.2 / gcc-8 / C++14 / sm_62. See patch/cunls/README.md.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VER="${CUNLS_VERSION:-Release_07_13_2026}"
OUT="${REPO_ROOT}/patch/cunls/0001-cuda102-tx2-port.patch"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
TARBALL="${REPO_ROOT}/build/cunls-${VER}.tar.gz"

mkdir -p "${REPO_ROOT}/build"
if [[ ! -f "$TARBALL" ]]; then
    echo "downloading cuNLS ${VER} ..."
    curl -sSL -o "$TARBALL" \
      "https://github.com/nvidia-isaac/cuNLS/archive/refs/tags/${VER}.tar.gz"
fi
tar xzf "$TARBALL" -C "$WORK"
SRC="$WORK/cuNLS-${VER}"
cp -r "$SRC" "$WORK/orig"

python3 - "$SRC" <<'PY'
import pathlib, re, sys
root = pathlib.Path(sys.argv[1])
def edit(rel, fn):
    p = root / rel; s = p.read_text(); out = fn(s)
    assert out != s, f"no-op edit on {rel}"
    p.write_text(out)

# ---- 1. top-level CMakeLists: arch/standard/compiler must defer to the parent,
#         and cuDSS is removed entirely (see README: cuVSLAM never selects it).
def top(s):
    s = s.replace(
        "set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)\nset(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)",
        "# TX2 port: never override what the parent project already chose. cuVSLAM sets\n"
        "# sm_62 and its own nvcc; hard-coding sm_75+ here breaks the Jetson TX2 build.\n"
        "if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)\n"
        "  set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)\n"
        "endif()\n"
        "if(NOT DEFINED CMAKE_CUDA_COMPILER)\n"
        "  set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)\n"
        "endif()")
    # nvcc 10.2 has no -std=c++17 for device code at all.
    s = s.replace(
        "set(CMAKE_CXX_STANDARD 17)\nset(CMAKE_CXX_STANDARD_REQUIRED ON)",
        "set(CMAKE_CXX_STANDARD 17)\nset(CMAKE_CXX_STANDARD_REQUIRED ON)\n"
        "# nvcc 10.2 cannot compile C++17 device code; the .cu sources and every header\n"
        "# they pull in are downgraded to C++14 by this patch.\n"
        "if(NOT DEFINED CMAKE_CUDA_STANDARD)\n"
        "  set(CMAKE_CUDA_STANDARD 14)\n"
        "  set(CMAKE_CUDA_STANDARD_REQUIRED ON)\n"
        "endif()")
    s = s.replace("include(cmake/AddCUDSS.cmake)\n", "")
    s = re.sub(r"# cuDSS is fetched as a prebuilt archive.*?add_cudss\(VERSION \"\$\{CUDSS_VERSION\}\"\)\n",
               "# cuDSS REMOVED for the TX2 port: it ships only as a prebuilt binary, and NVIDIA\n"
               "# publishes it for CUDA 12/13 only -- no CUDA 10.2 and no Tegra build exists, and\n"
               "# there is no source to compile. cuVSLAM never selects it (it asks for DenseQR;\n"
               "# cuNLS's own default is BlockSparsePCG), so the backend is dropped outright.\n",
               s, flags=re.S)
    s = s.replace(
        "set(CUNLS_LIBS spdlog::spdlog CUDA::cusparse CUDA::cublas CUDA::cusolver cudss)",
        "set(CUNLS_LIBS spdlog::spdlog CUDA::cusparse CUDA::cublas CUDA::cusolver)")
    s = s.replace("bundle_static_dependencies(cunls spdlog::spdlog cudss)",
                  "bundle_static_dependencies(cunls spdlog::spdlog)")
    return s
edit("CMakeLists.txt", top)

# ---- 2/3. drop the two cuDSS translation units and their link edges
edit("cunls/common/CMakeLists.txt", lambda s: s
     .replace("  cudss_helper.cpp\n", "")
     .replace("target_link_libraries(cunls_common PRIVATE spdlog::spdlog cudss)",
              "target_link_libraries(cunls_common PRIVATE spdlog::spdlog)"))
edit("cunls/linear_solver/CMakeLists.txt", lambda s: s
     .replace("  cudss_sparse_linear_solver.cpp\n", "")
     .replace("\ntarget_link_libraries(cunls_linear_solver PRIVATE cudss)\n", "\n"))

# ---- 4. factory: the cuDSS branch can no longer be built
edit("cunls/linear_solver/sparse_linear_solver.cpp", lambda s: s
     .replace('#include "cunls/common/cudss_helper.h"\n', "")
     .replace("  case SparseLinearSolverType::cuDSS:\n"
              "    return std::make_unique<cuDSSLinearSolver>(config.cudss_solver_options);\n",
              "  case SparseLinearSolverType::cuDSS:\n"
              "    // Removed in the CUDA-10.2 / TX2 port: cuDSS has no build for this toolkit.\n"
              "    throw std::invalid_argument(\n"
              "        \"cuDSS solver unavailable in this build (CUDA-10.2 TX2 port); \"\n"
              "        \"use DenseQR or BlockSparsePCG\");\n"))

# ---- 5. types.h: CUDA 10.2's libcu++ 1.0 has no <cuda/std/array>
edit("cunls/common/types.h", lambda s: s
     .replace("#include <cuda/std/array>\n", "")
     .replace("namespace cunls {",
              "namespace cunls {\n\n"
              "// CUDA 10.2 ships libcu++ 1.0, which has no <cuda/std/array> (it provides only\n"
              "// atomic, cassert, cfloat, climits, cstddef, cstdint, type_traits, version).\n"
              "// DeviceArray is a minimal stand-in with the same aggregate layout and the subset\n"
              "// of the std::array interface cuNLS uses, usable from host and device.\n"
              "template <typename T, int N> struct DeviceArray {\n"
              "  T _elems[N];\n"
              "  __host__ __device__ T &operator[](int i) { return _elems[i]; }\n"
              "  __host__ __device__ constexpr const T &operator[](int i) const { return _elems[i]; }\n"
              "  __host__ __device__ constexpr int size() const { return N; }\n"
              "  __host__ __device__ T *data() { return _elems; }\n"
              "  __host__ __device__ constexpr const T *data() const { return _elems; }\n"
              "  __host__ __device__ T *begin() { return _elems; }\n"
              "  __host__ __device__ constexpr const T *begin() const { return _elems; }\n"
              "  __host__ __device__ T *end() { return _elems + N; }\n"
              "  __host__ __device__ constexpr const T *end() const { return _elems + N; }\n"
              "};", 1)
     .replace("template <int Dim> using Vector = cuda::std::array<float, Dim>;",
              "template <int Dim> using Vector = DeviceArray<float, Dim>;")
     .replace("template <int Dim> using Matrix = cuda::std::array<float, Dim * Dim>;",
              "template <int Dim> using Matrix = DeviceArray<float, Dim * Dim>;"))

for f in ["cunls/factor/point_to_point_factor_batch.h",
          "cunls/factor/point_to_plane_factor_batch.h",
          "cunls/factor/symmetric_point_to_plane_factor_batch.h",
          "cunls/factor/prior_vector_factor_batch.h"]:
    edit(f, lambda s: s.replace("#include <cuda/std/array>\n", ""))

# ---- 6. <cuda/std/limits> is likewise absent; FLT_MIN / INT_MAX are equivalent
for f in ["cunls/robustifier/arctan_loss_function_batch.cu",
          "cunls/robustifier/huber_loss_function_batch.cu",
          "cunls/robustifier/tolerant_loss_function_batch.cu",
          "cunls/robustifier/soft_lone_loss_function_batch.cu",
          "cunls/robustifier/cauchy_loss_function_batch.cu"]:
    edit(f, lambda s: s
         .replace("#include <cuda/std/limits>", "#include <cfloat>")
         .replace("cuda::std::numeric_limits<float>::min()", "FLT_MIN"))
edit("cunls/state/state_batch_ops.cu", lambda s: s
     .replace("#include <cuda/std/limits>", "#include <climits>")
     .replace("cuda::std::numeric_limits<int>::max()", "INT_MAX"))

# ---- 7. C++17 device syntax -> C++14
#  `if constexpr` : every branch is well-formed for every instantiation here
#  (the reduction kernel always takes a, b and w), so a plain `if` on a
#  compile-time-constant condition is equivalent and still folded by nvcc.
edit("cunls/common/helper.h", lambda s: s.replace("if constexpr (is_throw)", "if (is_throw)"))
edit("cunls/minimizer/device_reduction.cu", lambda s: s
     .replace("if constexpr (Mode == 0)", "if (Mode == 0)")
     .replace("} else if constexpr (Mode == 1)", "} else if (Mode == 1)"))
#  nested namespace definitions
for f in ["cunls/common/profiler.h", "cunls/common/profiler.cpp"]:
    edit(f, lambda s: s
         .replace("namespace cunls::profiler {", "namespace cunls { namespace profiler {")
         .replace("} // namespace cunls::profiler", "}} // namespace cunls::profiler"))
#  std::string_view is C++17; log.h is included by .cu translation units
for f in ["cunls/common/log.h", "cunls/common/log.cpp"]:
    edit(f, lambda s: s
         .replace("#include <string_view>\n", "")
         .replace("std::string_view", "const std::string &"))
#  ---- cuSPARSE: CUDA 10.2 ships cuSPARSE 10.3, which predates part of the
#  generic API cuNLS uses. Three separate gaps, handled three ways.

#  (a) cusparseSpMV_preprocess is a CUDA-12 optimisation hint. cusparseSpMV is
#      correct without it (the PCG solver's own comment notes the preprocess is
#      per-structure, not per-solve), so compile it out on older toolkits.
_PRE_PCG = """  THROW_ON_CUSPARSE_ERROR(cusparseSpMV_preprocess(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                                  matA, vecX, &beta, vecY, CUDA_R_32F,
                                                  CUSPARSE_SPMV_ALG_DEFAULT, spmv_buffer_.data()));"""
edit("cunls/linear_solver/block_sparse_pcg_solver.cu", lambda s: s.replace(
    _PRE_PCG,
    "#if defined(CUSPARSE_VERSION) && CUSPARSE_VERSION >= 12000\n"
    "  // Optimisation hint only; absent before CUDA 12 and not required for correctness.\n"
    + _PRE_PCG + "\n#endif"))
_PRE_SM = """  THROW_ON_CUSPARSE_ERROR(cusparseSpMV_preprocess(
      cusparse_handle, operation, &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
      CUSPARSE_SPMV_ALG_DEFAULT, buffer_ptr));"""
edit("cunls/minimizer/sparse_matrix.cu", lambda s: s.replace(
    _PRE_SM,
    "#if defined(CUSPARSE_VERSION) && CUSPARSE_VERSION >= 12000\n"
    "  // Optimisation hint only; absent before CUDA 12 and not required for correctness.\n"
    + _PRE_SM + "\n#endif"))

#  (b) cusparseCsrSetPointers (CUDA 11+) has no 10.2 equivalent, so rebuild the
#      descriptor instead -- a CSR descriptor is just dims plus the three device
#      pointers. Both callers re-fetch it via GetDescription() afterwards, and the
#      SpMV buffer size depends only on dims/nnz, which UpdatePointers never changes.
#      The dimensions are not queryable on 10.2 (no cusparseSpMatGetSize), so they
#      are cached at construction -- and must therefore survive the move-assignment,
#      which is how block_sparse_pcg_solver installs the descriptor.
edit("cunls/common/cusparse_helper.h", lambda s: s.replace(
    "  void *description_ = nullptr; ///< The cuSPARSE matrix descriptor",
    "  void *description_ = nullptr; ///< The cuSPARSE matrix descriptor\n"
    "  int rows_ = 0;                ///< Cached row count (CUDA 10.2 cannot query it back)\n"
    "  int cols_ = 0;                ///< Cached column count"))
edit("cunls/common/cusparse_helper.cpp", lambda s: s
     .replace("""    int num_rows, int num_cols, int num_nonzeros,
    const CSRSparseMatrix &matrix) {
  auto rows_ptr""",
              """    int num_rows, int num_cols, int num_nonzeros,
    const CSRSparseMatrix &matrix)
    : rows_(num_rows), cols_(num_cols) {
  auto rows_ptr""")
     .replace("""cuSPARSEMatrixDescription::cuSPARSEMatrixDescription(int num_rows,
                                                     int num_cols) {""",
              """cuSPARSEMatrixDescription::cuSPARSEMatrixDescription(int num_rows,
                                                     int num_cols)
    : rows_(num_rows), cols_(num_cols) {""")
     .replace("""  description_ = std::exchange(other.description_, nullptr);
  return *this;""",
              """  description_ = std::exchange(other.description_, nullptr);
  rows_ = std::exchange(other.rows_, 0);
  cols_ = std::exchange(other.cols_, 0);
  return *this;""")
     .replace("""  THROW_ON_CUSPARSE_ERROR(
      cusparseCsrSetPointers(static_cast<cusparseSpMatDescr_t>(description_),
                             rows_ptr, cols_ptr, values_ptr));""",
              """  // CUDA 10.2's cuSPARSE has no cusparseCsrSetPointers; rebuild the descriptor.
  if (description_) {
    WARN_ON_CUSPARSE_ERROR(
        cusparseDestroySpMat(static_cast<cusparseSpMatDescr_t>(description_)));
    description_ = nullptr;
  }
  cusparseSpMatDescr_t descr = nullptr;
  THROW_ON_CUSPARSE_ERROR(cusparseCreateCsr(
      &descr, rows_, cols_, static_cast<int>(matrix.NumNonZeros()), rows_ptr,
      cols_ptr, values_ptr, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  description_ = static_cast<void *>(descr);"""))

#  (c) the cuSPARSE A^T*A multiplier needs the SpGEMM-reuse family (CUDA 11.3+)
#      and cusparseSpMatGetSize. Like cuDSS it is an unselected backend -- the
#      default is SparseMatrixMultiplierType::Fast, a custom kernel -- so drop it.
edit("cunls/minimizer/CMakeLists.txt",
     lambda s: s.replace("  cusparse_matrix_multiplier.cpp\n", ""))
edit("cunls/minimizer/sparse_matrix_multiplier.cpp", lambda s: s
     .replace('#include "cunls/minimizer/cusparse_matrix_multiplier.h"\n', "")
     .replace("""  case SparseMatrixMultiplierType::cuSPARSE:
    return std::make_unique<cuSPARSESparseMatrixMultiplier>();""",
              """  case SparseMatrixMultiplierType::cuSPARSE:
    // Removed in the CUDA-10.2 / TX2 port: needs the cuSPARSE SpGEMM-reuse API
    // (CUDA 11.3+). Use SparseMatrixMultiplierType::Fast, which is the default.
    throw std::invalid_argument(
        "cuSPARSE square multiplier unavailable in this build "
        "(CUDA-10.2 TX2 port); use Fast");"""))

#  cublasSgemvStridedBatched does not exist in CUDA 10.2's cuBLAS (only the gemm
#  StridedBatched family does). A batched gemv is exactly a batched gemm with n=1:
#  y(m x 1) = alpha * A(m x k) * x(k x 1) + beta * y. incx/incy are 1 here and the
#  vectors are contiguous, so x and y map to column matrices with ld = residual_size,
#  which satisfies ldb >= k and ldc >= m. Strides carry over unchanged.
edit("cunls/factor/information_factor_batch.cpp", lambda s: s
     .replace("""  const size_t stride = residual_size * residual_size;
  constexpr size_t inc = 1;

  THROW_ON_CUBLAS_ERROR(cublasSgemvStridedBatched(
      static_cast<cublasHandle_t>(cublas_handle), CUBLAS_OP_N, residual_size,
      residual_size, &alpha, sqrt_information, residual_size, stride, residuals,
      inc, residual_size, &beta, residuals, inc, residual_size, num_factors));""",
              """  const size_t stride = residual_size * residual_size;

  // CUDA 10.2 has no cublasSgemvStridedBatched; expressed as a batched gemm with
  // n = 1, which is the same operation (see patch/cunls/README.md).
  const int n = static_cast<int>(residual_size);
  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      static_cast<cublasHandle_t>(cublas_handle), CUBLAS_OP_N, CUBLAS_OP_N, n,
      /*n=*/1, n, &alpha, sqrt_information, n,
      static_cast<long long>(stride), residuals, n,
      static_cast<long long>(residual_size), &beta, residuals, n,
      static_cast<long long>(residual_size),
      static_cast<int>(num_factors)));"""))

#  C++17 variable template -> the C++14 trait spelling
edit("cunls/robustifier/scaled_loss_function_batch.h", lambda s: s
     .replace("std::is_base_of_v<LossFunctionBatch, T>",
              "std::is_base_of<LossFunctionBatch, T>::value"))

#  DomainRange is returned by value from Domain::CreateDomainRange. C++17 elides
#  that construction outright (guaranteed copy elision), but C++14 still requires
#  an accessible move constructor even when it elides the call -- so the deleted
#  one is a hard error under nvcc 10.2. Give it a real move: the destructor is
#  already guarded on handle_, so transferring and nulling the source preserves
#  the RAII semantics exactly (and is a no-op when profiling is compiled out).
edit("cunls/common/profiler.h", lambda s: s
     .replace("  DomainRange(DomainRange &&) = delete;\n",
              "  DomainRange(DomainRange &&other) noexcept\n"
              "      : handle_(other.handle_), name_(std::move(other.name_)) {\n"
              "    other.handle_ = nullptr;\n"
              "  }\n")
     .replace("#include <string>", "#include <string>\n#include <utility>", 1))

#  the pre-C++20 formatting fallback in log.h is itself C++17: a tuple_size_v
#  variable template and a fold expression. Both have direct C++14 spellings.
edit("cunls/common/log.h", lambda s: s
     .replace("Index < std::tuple_size_v<Tuple>",
              "Index < std::tuple_size<Tuple>::value")
     .replace("  (format_arg_at_index<Indices>(oss, args, target_index), ...);",
              "  // C++14 has no fold expressions; expand through an initialiser list, whose\n"
              "  // elements are sequenced left-to-right just like the fold was.\n"
              "  int expand[] = {0, (format_arg_at_index<Indices>(oss, args, target_index), 0)...};\n"
              "  (void)expand;"))
PY

# capture as a unified diff
( cd "$WORK" && diff -ruN "orig" "cuNLS-${VER}" \
    | sed -e "s|^--- orig/|--- a/|" -e "s|^+++ cuNLS-${VER}/|+++ b/|" ) > "$OUT" || true
echo "Regenerated $OUT ($(grep -c '^--- a/' "$OUT") files, $(wc -l < "$OUT") lines)"
