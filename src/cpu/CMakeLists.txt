cmake_minimum_required(VERSION 3.6)



#clang++ -target amdgcn-amd-amdhsa-opencl  -x cl hello.cl -c -S
  
project(mg_ml_cpu)   

    file(GLOB SOURCE_FILES "*.cpp" "*.h")
    include_directories(${CMAKE_SOURCE_DIR}/include ${OpenCL_INCLUDE_DIRS})
    
	#defining standard compiling flags
	if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  /std:c++14 /W4 /arch:AVX2 /fp:fast")
	else()
	     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -pedantic -Wextra -m64 -mavx2 -mfma -ffast-math")
    endif()

	#adding the executable
    add_library(${PROJECT_NAME} STATIC ${SOURCE_FILES} )

	#enabling clang tidy
    enable_clang_tidy_for_project()
