cmake_minimum_required(VERSION 2.6)

set( CMAKE_EXPORT_COMPILE_COMMANDS ON )
    
project(01_cl_hello_world)   
    file(GLOB SOURCE_FILES "*.cpp" "*.h")
    find_package(OpenCL REQUIRED) 

    add_executable(${PROJECT_NAME} hello.cpp )
    add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy
                   ${CMAKE_CURRENT_SOURCE_DIR}/hello.cl $<TARGET_FILE_DIR:${PROJECT_NAME}>)
    target_link_libraries(${PROJECT_NAME} OpenCL)
