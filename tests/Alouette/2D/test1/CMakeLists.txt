# ---------------------------------------------------------------------------
# The S0 coupled test: HeatTransfer/2D/test1 + the R2 multiscale provider.
#
# NOT `create_test()` on purpose. `create_test()` gives the link line to CMake,
# and the link line IS part of what this test establishes: the executable must
# bind to ONE MPI runtime (the coupling library's), which means an explicit
# `-L<mpi dir> -Wl,--no-as-needed -lmpi` and an rpath with that directory
# first. It is also registered as a manual target rather than a ctest entry:
# it depends on an external library tree, and the PRRTE cross-check (running
# it under both MPI installations' `mpirun`) is done by hand, with the launcher
# recorded.
# ---------------------------------------------------------------------------

# The OpenMPI closure of the coupling view (libmpi's own DT_NEEDED entries).
# libmpi carries an RPATH for most of them, but not for hwloc, and RUNPATH does
# not apply transitively to a dependency's dependencies: the executable's rpath
# is what resolves libhwloc for libmpi, so these directories are load-bearing.
set(ALOUETTE_MPI_CLOSURE_LIBRARY_DIRS
    "/home/tommy/projects/sloth_trials/spack/opt/spack/linux-skylake/openmpi-5.0.10-5yylz57e4m2v3is77jl2z6ixvuo53jxu/lib"
    "/home/tommy/projects/sloth_trials/spack/opt/spack/linux-skylake/libevent-2.1.12-kw5fcnw3as7bpvwxggfbtit6ln2mfrts/lib"
    "/home/tommy/projects/sloth_trials/spack/opt/spack/linux-skylake/hwloc-2.9.3-ddcd7hl3r2zngt3izyd7r67ou3h2hvzg/lib"
    CACHE STRING "Extra rpath directories for the coupling view's MPI closure")

add_executable(Alouette2Dtest1 main.cpp)

target_link_libraries(Alouette2Dtest1 PRIVATE Sloth ${ALOUETTE_COUPLING_LIBRARY})

target_include_directories(Alouette2Dtest1 PRIVATE
    ${CMAKE_SOURCE_DIR}/tests
    ${ALOUETTE_COUPLING_INCLUDE_DIR}
)

set_compile_options(Alouette2Dtest1)
add_dependencies(tests Alouette2Dtest1)

configure_file(${CMAKE_SOURCE_DIR}/Mesh-examples/camembert2D.msh
               ${CMAKE_CURRENT_BINARY_DIR}/camembert2D.msh COPYONLY)

# The rpath, in priority order: the coupling view's libmpi first, then the
# closure directories, the coupling library itself and SLOTH's own shared
# dependencies (derived from the imported locations, so the configure's MFEM
# and HYPRE choices are respected). `SKIP_BUILD_RPATH` stops CMake from
# prepending its own order in front of this one.
set(_alouette_rpath_dirs "${ALOUETTE_MPI_LIBRARY_DIR}")
if(MPI_mpi_LIBRARY)
  get_filename_component(_alouette_mpi_dir "${MPI_mpi_LIBRARY}" DIRECTORY)
  list(APPEND _alouette_rpath_dirs "${_alouette_mpi_dir}")
endif()
get_filename_component(_alouette_coupling_dir "${ALOUETTE_COUPLING_LIBRARY}" DIRECTORY)
list(APPEND _alouette_rpath_dirs "${_alouette_coupling_dir}")
foreach(_alouette_dep IN ITEMS "${HYPRE_LIBRARY}" "${METIS_LIBRARY}")
  if(_alouette_dep)
    get_filename_component(_alouette_dep_dir "${_alouette_dep}" DIRECTORY)
    list(APPEND _alouette_rpath_dirs "${_alouette_dep_dir}")
  endif()
endforeach()
# SuiteSparse is an IMPORTED TARGET here (`SuiteSparse::suitesparse`), not a
# path, so its directory comes from the include tree's sibling `lib`. A
# non-path in the list would otherwise become an EMPTY rpath entry, and an
# empty entry means the current directory to the loader.
if(SuiteSparse_INCLUDE_DIRS)
  get_filename_component(_alouette_suitesparse_root "${SuiteSparse_INCLUDE_DIRS}" DIRECTORY)
  list(APPEND _alouette_rpath_dirs "${_alouette_suitesparse_root}/lib")
endif()
list(APPEND _alouette_rpath_dirs ${ALOUETTE_MPI_CLOSURE_LIBRARY_DIRS})
list(REMOVE_DUPLICATES _alouette_rpath_dirs)

set(_alouette_link_options
    "-L${ALOUETTE_MPI_LIBRARY_DIR}"
    "-Wl,--no-as-needed"
    "-lmpi"
    "-Wl,--as-needed"
)
foreach(_alouette_rpath IN LISTS _alouette_rpath_dirs)
  list(APPEND _alouette_link_options "-Wl,-rpath,${_alouette_rpath}")
endforeach()
target_link_options(Alouette2Dtest1 PRIVATE ${_alouette_link_options})
set_target_properties(Alouette2Dtest1 PROPERTIES SKIP_BUILD_RPATH TRUE)
