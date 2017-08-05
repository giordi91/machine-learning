cmake_minimum_required(VERSION 3.6)



#clang++ -target amdgcn-amd-amdhsa-opencl  -x cl hello.cl -c -S
  
project(01_cl_hello_world)   

    file(GLOB SOURCE_FILES "*.cpp" "*.h")
    find_package(OpenCL REQUIRED) 
    generate_opencl_diassembly(${PROJECT_NAME} hello.cl)

    include_directories(${CMAKE_SOURCE_DIR}/include)
    if(${USE_OPENCL_2_0} STREQUAL "ON")
    add_definitions(-DUSE_OPENCL_2_0)
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -pedantic -Wextra -m64")

    add_executable(${PROJECT_NAME} hello.cpp )
    set (CMAKE_CXX_STANDARD 11)

	copy_kernel(hello.cl)
    target_link_libraries(${PROJECT_NAME} OpenCL)

    if(CLANG_TIDY_EXE)
	  set_target_properties(
			  ${PROJECT_NAME}  PROPERTIES
				  CXX_CLANG_TIDY "${DO_CLANG_TIDY}"
					)
	endif()
