cmake_minimum_required(VERSION 2.6)

set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

MESSAGE( STATUS "BUILD DUMP_KERNEL_ASSEMBLY:         " ${DUMP_KERNEL_ASSEMBLY})

#clang++ -target amdgcn-amd-amdhsa-opencl  -x cl hello.cl -c -S
  

project(01_cl_hello_world)   



    file(GLOB SOURCE_FILES "*.cpp" "*.h")
    find_package(OpenCL REQUIRED) 
    generate_opencl_diassembly(${PROJECT_NAME} hello.cl)

    add_executable(${PROJECT_NAME} hello.cpp )
    add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy
                   ${CMAKE_CURRENT_SOURCE_DIR}/hello.cl $<TARGET_FILE_DIR:${PROJECT_NAME}>)
    target_link_libraries(${PROJECT_NAME} OpenCL)
