cmake_minimum_required(VERSION 4.4)
execute_process(COMMAND python3 ${BUILD_SCRIPT_DIR}/GenerateCoefficient.py -r  -f  ${CMAKE_CURRENT_SOURCE_DIR}/coefficients.json
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

set(CURRENT_EXE FissionGas2Dtest1)
add_executable(${CURRENT_EXE} main.cpp)
target_link_libraries(${CURRENT_EXE} Sloth)
set_compile_options(${CURRENT_EXE})
add_dependencies(tests ${CURRENT_EXE})

# Fully implicit time scheme, on a single bubble and a time step small enough for the solution to
# be converged in time: the reference below was checked against a run at half the time step, which
# reproduces it to 2.e-6 on every quantity, and against the same problem solved with the coupling
# iterated to convergence, which reproduces it to 4.e-5. The final time is kept short so that the
# whole test runs in a few seconds.
# The specialized time files are appended to, not truncated, so the results of a previous run would
# otherwise be compared against the reference.
add_test(
    NAME FissionGas2Dtest1
    COMMAND bash -c "rm -rf Saves && mpirun -np 2 $<TARGET_FILE:${CURRENT_EXE}> -fi -nb 1 -t 100 -dtmin 5 -dtmax 5"
)
set_tests_properties(FissionGas2Dtest1 PROPERTIES LABELS "2D FISSIONGAS")

# Order parameters: interfacial energy and amount of bubble phase, i.e. the Allen-Cahn side of the
# model and the switching functions.
create_col_comparison("CompareFissionGas2Dtest1" "op_time_specialized.csv"
    "Saves/OrderParameters/time_specialized.csv" -1 relativeEpsilon 1e-8 FALSE
    "FissionGas2Dtest1" "2D FISSIONGAS")

# Gas chemical potential: the transport equation, the production term confined to the matrix by the
# switching function (MassSource integrator), and the partitioning of the species between the
# phases (PhaseChangeSource integrator). The last column is the gas inventory of the domain.
create_col_comparison("CompareFissionGas2Dtest1Gas" "mug_time_specialized.csv"
    "Saves/MuGas/time_specialized.csv" -1 relativeEpsilon 1e-8 FALSE
    "FissionGas2Dtest1" "2D FISSIONGAS")
