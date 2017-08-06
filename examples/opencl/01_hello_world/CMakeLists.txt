cmake_minimum_required(VERSION 3.6)



#clang++ -target amdgcn-amd-amdhsa-opencl  -x cl hello.cl -c -S
  
project(01_cl_hello_world)   

    file(GLOB SOURCE_FILES "*.cpp" "*.h")
    find_package(OpenCL REQUIRED) 
    generate_opencl_diassembly(${PROJECT_NAME} hello.cl)

    include_directories(${CMAKE_SOURCE_DIR}/include ${OpenCL_INCLUDE_DIRS})
	#link_directories(${OpenCL_LIBRARIES})
    if(${USE_OPENCL_2_0} STREQUAL "ON")
    add_definitions(-DUSE_OPENCL_2_0)
    endif()

    
	#defining standard compiling flags
	if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  /std:c++14 /W4 /arch:AVX2 /fp:fast")
	else()
	     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -pedantic -Wextra -m64 -mavx2 -mfma -ffast-math")
    endif()

	#adding the executable
    add_executable(${PROJECT_NAME} hello.cpp )
    
	#here we are gonna set the working directory for debugging the exe, this is because most of the time
	#if you load the kernel from file, you want to run the executable from its own path and not a parent one
	set_visual_studio_working_directory_if_needed()

	#copying the kernel in same folder
	copy_kernel(hello.cl)
    target_link_libraries(${PROJECT_NAME} ${OpenCL_LIBRARIES} )

	#enabling clang tidy
    enable_clang_tidy_for_project()
