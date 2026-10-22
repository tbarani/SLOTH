# ---------------------------------------------------------------------------
# Coupled SLOTH / alouette tests (the S0 multiscale hook).
#
# These tests need two things that live OUTSIDE this repository: the alouette
# coupling header (`alouette/coupling/alouette_micro.h`) and the built
# `libalouette_core.so`. The family therefore disables itself when either is
# absent, so a plain SLOTH checkout configures and builds exactly as before.
# Point the cache variables below at the installation to enable it.
#
# The third cache variable is the MPI installation the coupled executable must
# bind to: the alouette coupling library is linked against the MPI of the
# environment it was built in, and the executable must use the SAME tree (one
# MPI runtime per process, or the two libraries disagree about communicators).
# ---------------------------------------------------------------------------
set(ALOUETTE_COUPLING_INCLUDE_DIR
    "/home/tommy/projects/alouette/include"
    CACHE PATH "Directory containing alouette/coupling/alouette_micro.h")
set(ALOUETTE_COUPLING_LIBRARY
    "/home/tommy/projects/alouette/build-sloth/src/libalouette_core.so"
    CACHE FILEPATH "The alouette coupling shared library (libalouette_core.so)")
set(ALOUETTE_MPI_LIBRARY_DIR
    "/home/tommy/projects/alouette/.spack-coupling/.spack-env/view/lib"
    CACHE PATH "Directory of the libmpi.so the coupled executable must use")

if(EXISTS "${ALOUETTE_COUPLING_INCLUDE_DIR}/alouette/coupling/alouette_micro.h"
   AND EXISTS "${ALOUETTE_COUPLING_LIBRARY}")
  message(STATUS "SLOTH/Alouette coupling tests: ENABLED")
  add_subdirectory(2D)
else()
  message(STATUS "SLOTH/Alouette coupling tests: disabled "
                 "(set ALOUETTE_COUPLING_INCLUDE_DIR and ALOUETTE_COUPLING_LIBRARY to enable)")
endif()
