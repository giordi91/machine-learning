cmake_minimum_required(VERSION 3.6.0)

#preventing in source build
if ( ${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR} )
        message( FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt." )
    endif()

find_program(
	CLANG_TIDY_EXE
	NAMES "clang-tidy"
	DOC "Path to clang-tidy executable"
)
if(NOT CLANG_TIDY_EXE)
	message(STATUS "clang-tidy not found.")
else()
	message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
	set(DO_CLANG_TIDY "${CLANG_TIDY_EXE}" "-checks=*,-clang-analyzer-alpha.*")
endif()

#compile database
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

#including utils function, this will cascade to all subsequent included cmakes
include (cmakeutils/code_gen.txt)

#options
option(DUMP_KERNEL_ASSEMBLY "Wheter or not to generate readable assembly code from kernels" ON)
option(BUILD_OPENCL "Wheter or not to build the opencl side of code" ON)
option(USE_OPENCL_2_0 "Wheter or not to use opencl 2.0" OFF)

MESSAGE( STATUS "DUMP_KERNEL_ASSEMBLY:         " ${DUMP_KERNEL_ASSEMBLY})
MESSAGE( STATUS "BUILD_OPENCL:                 " ${BUILD_OPENCL})
MESSAGE( STATUS "USE_OPENCL_2_0:               " ${USE_OPENCL_2_0})
#adding opencl examples
if(${BUILD_OPENCL} STREQUAL "ON")
    add_subdirectory(examples/opencl/01_hello_world)
endif()
