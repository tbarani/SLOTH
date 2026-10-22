# ---------------------------------------------------------------------------
# Coupled SLOTH / alouette tests (the S0 multiscale hook).
#
# These tests need two things that live OUTSIDE this repository: the alouette
# coupling header (`alouette/coupling/alouette_micro.h`) and the built
# `libalouette_core.so`. They are therefore OPT-IN and default OFF, so a plain
# SLOTH checkout configures and builds exactly as before: no machine-local
# absolute path, and no second MPI runtime in the native tree.
#
# Enable with -DSLOTH_BUILD_ALOUETTE_TESTS=ON and point the cache variables
# below at the installation. The third variable is the MPI installation the
# coupled executable must bind to: the alouette coupling library is linked
# against the MPI of the environment it was built in, and the executable must
# use the SAME tree (one MPI runtime per process, or the two libraries
# disagree about communicators).
# ---------------------------------------------------------------------------
option(SLOTH_BUILD_ALOUETTE_TESTS "Build the SLOTH/alouette coupling tests" OFF)

set(ALOUETTE_COUPLING_INCLUDE_DIR
    ""
    CACHE PATH "Directory containing alouette/coupling/alouette_micro.h")
set(ALOUETTE_COUPLING_LIBRARY
    ""
    CACHE FILEPATH "The alouette coupling shared library (libalouette_core.so)")
set(ALOUETTE_MPI_LIBRARY_DIR
    ""
    CACHE PATH "Directory of the libmpi.so the coupled executable must use")

if(SLOTH_BUILD_ALOUETTE_TESTS
   AND EXISTS "${ALOUETTE_COUPLING_INCLUDE_DIR}/alouette/coupling/alouette_micro.h"
   AND EXISTS "${ALOUETTE_COUPLING_LIBRARY}")
  # An empty MPI directory would let `-lmpi` resolve to the system runtime and
  # silently break the one-MPI-runtime contract, so it is refused rather than
  # linked.
  if(NOT ALOUETTE_MPI_LIBRARY_DIR)
    message(FATAL_ERROR
            "SLOTH_BUILD_ALOUETTE_TESTS=ON requires ALOUETTE_MPI_LIBRARY_DIR: the coupled "
            "executable must bind to the same MPI runtime as libalouette_core.so.")
  endif()
  message(STATUS "SLOTH/Alouette coupling tests: ENABLED")
  add_subdirectory(2D)
else()
  message(STATUS "SLOTH/Alouette coupling tests: disabled "
                 "(set SLOTH_BUILD_ALOUETTE_TESTS=ON and point ALOUETTE_COUPLING_INCLUDE_DIR, "
                 "ALOUETTE_COUPLING_LIBRARY and ALOUETTE_MPI_LIBRARY_DIR at the installation)")
endif()
