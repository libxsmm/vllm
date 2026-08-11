# LIBXSMM is used for the JIT-generated ternary (int2) x int8 GEMM microkernels.
#
# The int2 support (LIBXSMM_GEMM_FLAG_INTERPRETE_A_AS_INT2_VNNI4_INTLV and the
# AVX2/SRF microkernel) is not on libxsmm's main branch, so a specific commit is
# pinned. It is the same one that tpp-pytorch-extension's `2bit_gemm` branch
# pins as its libxsmm submodule, and it is fetchable by SHA from the public
# repository even though no branch points at it.

set(VLLM_LIBXSMM_COMMIT "84bff4a46ef27f445259664ac93abf5c33a02148")
set(VLLM_LIBXSMM_REPO "https://github.com/libxsmm/libxsmm.git")

set(VLLM_LIBXSMM_SRC "$ENV{LIBXSMM_ROOT}" CACHE PATH
    "Path to a LIBXSMM source tree providing the int2 GEMM support.")

if (NOT VLLM_LIBXSMM_SRC)
    set(VLLM_LIBXSMM_SRC "${CMAKE_CURRENT_LIST_DIR}/../../../third_party/libxsmm")
endif()

get_filename_component(VLLM_LIBXSMM_SRC "${VLLM_LIBXSMM_SRC}" ABSOLUTE)

if (NOT EXISTS "${VLLM_LIBXSMM_SRC}/src/libxsmm_main.c")
    find_package(Git REQUIRED)
    message(STATUS
        "Fetching LIBXSMM ${VLLM_LIBXSMM_COMMIT} into ${VLLM_LIBXSMM_SRC}")
    file(MAKE_DIRECTORY "${VLLM_LIBXSMM_SRC}")
    # fetch by SHA rather than clone: no branch contains this commit
    execute_process(
        COMMAND ${GIT_EXECUTABLE} init -q .
        WORKING_DIRECTORY "${VLLM_LIBXSMM_SRC}" RESULT_VARIABLE _rc)
    if (_rc EQUAL 0)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} remote add origin "${VLLM_LIBXSMM_REPO}"
            WORKING_DIRECTORY "${VLLM_LIBXSMM_SRC}" ERROR_QUIET)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} fetch -q --depth 1 origin
                    "${VLLM_LIBXSMM_COMMIT}"
            WORKING_DIRECTORY "${VLLM_LIBXSMM_SRC}" RESULT_VARIABLE _rc)
    endif()
    if (_rc EQUAL 0)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} checkout -q FETCH_HEAD
            WORKING_DIRECTORY "${VLLM_LIBXSMM_SRC}" RESULT_VARIABLE _rc)
    endif()
    if (NOT _rc EQUAL 0 OR NOT EXISTS "${VLLM_LIBXSMM_SRC}/src/libxsmm_main.c")
        message(FATAL_ERROR
            "Could not fetch LIBXSMM ${VLLM_LIBXSMM_COMMIT} from "
            "${VLLM_LIBXSMM_REPO} into '${VLLM_LIBXSMM_SRC}'. Clone it manually "
            "(git init && git fetch --depth 1 origin <commit> && git checkout "
            "FETCH_HEAD) and point LIBXSMM_ROOT or -DVLLM_LIBXSMM_SRC at it.")
    endif()
else()
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
        WORKING_DIRECTORY "${VLLM_LIBXSMM_SRC}"
        OUTPUT_VARIABLE _have OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if (_have AND NOT _have STREQUAL "${VLLM_LIBXSMM_COMMIT}")
        message(WARNING
            "LIBXSMM at '${VLLM_LIBXSMM_SRC}' is ${_have}, expected "
            "${VLLM_LIBXSMM_COMMIT}. The int2 kernels may be missing.")
    endif()
endif()

message(STATUS "Using LIBXSMM from: ${VLLM_LIBXSMM_SRC}")

set(XSMM_STATIC ON CACHE BOOL "" FORCE)
add_subdirectory("${VLLM_LIBXSMM_SRC}" "${CMAKE_BINARY_DIR}/libxsmm-build" EXCLUDE_FROM_ALL)

# linked into python shared modules
set_property(TARGET xsmm PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_options(xsmm PRIVATE -w)

set(VLLM_LIBXSMM_INCLUDE_DIR "${VLLM_LIBXSMM_SRC}/include")
