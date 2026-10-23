# ---------------------------------------------------------------------------
# The S1 offline chain: the layered macro app + the external alouette batch.
#
# This target links `Sloth` ONLY. That is the point of the offline variant: the
# two programs never share a runtime, so there is no coupling library, no
# include directory and no MPI one-runtime link dance here. The batch is
# invoked by PATH from the chain script, as a separate process.
#
# The chain itself is one ctest entry, `t.layered_macro`, driving
# dump -> batch -> table for three cases (series, parallel, transient) plus a
# constant-lambda* run to compare against. The script asserts the summary
# numbers against hard-coded analytic constants, independently of the app's own
# anchor checks: the app proves its own numbers, the script proves the FILES
# carry them.
# ---------------------------------------------------------------------------

set(CURRENT_EXE Alouette2Dtest2)
add_executable(${CURRENT_EXE} main.cpp)
target_link_libraries(${CURRENT_EXE} PRIVATE Sloth)
target_include_directories(${CURRENT_EXE} PRIVATE ${CMAKE_SOURCE_DIR}/tests)
set_compile_options(${CURRENT_EXE})
add_dependencies(tests ${CURRENT_EXE})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run_layered_macro.sh
               ${CMAKE_CURRENT_BINARY_DIR}/run_layered_macro.sh COPYONLY)

# The app is serial-only (its dump's material points are local element
# indices), so the chain runs without mpirun. The timeout is generous: three
# macro runs per case plus the external batch, and the steady cases' implicit
# steps are stiff (dt = 1e5 s) but few.
add_test(
    NAME t.layered_macro
    COMMAND bash ${CMAKE_CURRENT_BINARY_DIR}/run_layered_macro.sh
            $<TARGET_FILE:${CURRENT_EXE}> ${ALOUETTE_OFFLINE_BATCH}
            ${CMAKE_CURRENT_BINARY_DIR}/work_layered_macro
)
set_tests_properties(t.layered_macro PROPERTIES LABELS "2D ALOUETTE" TIMEOUT 600)
