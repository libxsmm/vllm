# LIBXSMM is used for the JIT-generated ternary (int2) x int8 GEMM microkernels.
#
# The int2 support (LIBXSMM_GEMM_FLAG_INTERPRETE_A_AS_INT2_VNNI4_INTLV and the
# AVX2/SRF microkernel) only exists in the pinned checkout below; it is not on
# libxsmm's public main branch, so we build from a local source tree.

set(VLLM_LIBXSMM_SRC "$ENV{LIBXSMM_ROOT}" CACHE PATH
    "Path to a LIBXSMM source tree providing the int2 GEMM support.")

if (NOT VLLM_LIBXSMM_SRC)
    set(VLLM_LIBXSMM_SRC "${CMAKE_CURRENT_LIST_DIR}/../../../third_party/libxsmm")
endif()

get_filename_component(VLLM_LIBXSMM_SRC "${VLLM_LIBXSMM_SRC}" ABSOLUTE)

if (NOT EXISTS "${VLLM_LIBXSMM_SRC}/src/libxsmm_main.c")
    message(FATAL_ERROR
        "LIBXSMM source not found at '${VLLM_LIBXSMM_SRC}'. "
        "Set LIBXSMM_ROOT (env) or -DVLLM_LIBXSMM_SRC=<path> to a checkout that "
        "contains the int2 GEMM support.")
endif()

message(STATUS "Using LIBXSMM from: ${VLLM_LIBXSMM_SRC}")

set(XSMM_STATIC ON CACHE BOOL "" FORCE)
add_subdirectory("${VLLM_LIBXSMM_SRC}" "${CMAKE_BINARY_DIR}/libxsmm-build" EXCLUDE_FROM_ALL)

# linked into python shared modules
set_property(TARGET xsmm PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_options(xsmm PRIVATE -w)

set(VLLM_LIBXSMM_INCLUDE_DIR "${VLLM_LIBXSMM_SRC}/include")
