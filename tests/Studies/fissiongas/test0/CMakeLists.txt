execute_process(COMMAND python3 ${BUILD_SCRIPT_DIR}/GenerateCoefficient.py -r  -f  ${CMAKE_CURRENT_SOURCE_DIR}/coefficients.json
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

set(CURRENT_EXE FissionGas2Dtest0)
add_executable(${CURRENT_EXE} main.cpp)
target_link_libraries(${CURRENT_EXE} Sloth)
set_compile_options(${CURRENT_EXE})
add_dependencies(tests ${CURRENT_EXE})
add_test(
    NAME FissionGas2Dtest0
    COMMAND mpirun -np 2 ${CURRENT_EXE}
)
