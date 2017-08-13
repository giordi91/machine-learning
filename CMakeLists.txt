cmake_minimum_required(VERSION 3.6.0)

#preventing in source build
if ( ${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR} )
        message( FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt." )
    endif()

#compile database
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

#including utils function, this will cascade to all subsequent included cmakes
include (cmakeutils/code_gen.txt)

#setupping clang tidy
init_clang_tidy()

#options
option(DUMP_KERNEL_ASSEMBLY "Wheter or not to generate readable assembly code from kernels" ON)
option(BUILD_OPENCL "Wheter or not to build the opencl side of code" ON)
option(BUILD_CPU  "Wheter or not to build the cpu side of code" ON)
option(BUILD_CUDA "Wheter or not to build the cuda side of code" OFF)
option(USE_OPENCL_2_0 "Wheter or not to use opencl 2.0" ON)
option(BUILD_TESTS "Wheter or not to build the tests" ON)
option(RUN_TESTS "Wheter or not run the tests" ON)
option(BUILD_EXAMPLES "Wheter or not to build the examples" ON)

#check the compiler
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # using Clang
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  # using GCC
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  # using Intel C++
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  # using Visual Studio C++
  #disabling build of kernel assembly for the time being
  SET(DUMP_KERNEL_ASSEMBLY OFF CACHE BOOL "Wheter or not to generate readable assembly code from kernels" FORCE)
endif()


#just an overal log of the passed options
MESSAGE( STATUS "Building with the following options")
MESSAGE( STATUS "DUMP_KERNEL_ASSEMBLY:         " ${DUMP_KERNEL_ASSEMBLY})
MESSAGE( STATUS "BUILD_CPU:                    " ${BUILD_CPU})
MESSAGE( STATUS "BUILD_CUDA:                   " ${BUILD_CUDA})
MESSAGE( STATUS "BUILD_OPENCL:                 " ${BUILD_OPENCL})
MESSAGE( STATUS "USE_OPENCL_2_0:               " ${USE_OPENCL_2_0})
MESSAGE( STATUS "BUILD_EXAMPLES:               " ${BUILD_EXAMPLES})
MESSAGE( STATUS "BUILD TESTS:                  " ${BUILD_TESTS})
MESSAGE( STATUS "RUN TESTS:                    " ${RUN_TESTS})

#defining some useful variables
set(MAIN_LIB_NAME "mg_ml")
#adding cpu core
add_subdirectory(src)

#adding opencl examples
if(${BUILD_OPENCL} STREQUAL "ON" AND ${BUILD_EXAMPLES} STREQUAL "ON")
    add_subdirectory(examples/opencl/01_hello_world)
endif()

#adding test
if(${BUILD_TESTS} STREQUAL "ON")
    add_subdirectory(tests)
endif()

#cpu examples
if(${BUILD_CPU} STREQUAL "ON" AND ${BUILD_EXAMPLES} STREQUAL "ON")
add_subdirectory(examples/cpu/01_linear_regression)
add_subdirectory(examples/cpu/02_close_handler)
add_subdirectory(examples/cpu/03_load_cifar_10)
endif()
