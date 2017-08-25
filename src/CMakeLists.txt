cmake_minimum_required(VERSION 3.6)



#clang++ -target amdgcn-amd-amdhsa-opencl  -x cl hello.cl -c -S
  
project(${MAIN_LIB_NAME})   

    #looking for the cpu files
    if(${BUILD_CPU} STREQUAL "ON")
         file(GLOB_RECURSE CPU_SOURCE_FILES "cpu/*.cpp" "cpu/*.h")
         
	     message (STATUS "found CPU files")
         foreach(dir ${CPU_SOURCE_FILES})
               message(STATUS "dir='${dir}'")
         endforeach()
    endif()
    #looking for the shared files
    file(GLOB_RECURSE COMMON_SOURCE_FILES "common/*.cpp")
    
	message (STATUS "found common files")
    foreach(dir ${COMMON_SOURCE_FILES})
         message(STATUS "dir='${dir}'")
    endforeach()

    include_directories(${CMAKE_SOURCE_DIR}/include )
	#defining standard compiling flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_CXX_FLAGS}")

	#adding the executable
    add_library(${PROJECT_NAME} STATIC ${CPU_SOURCE_FILES} ${COMMON_SOURCE_FILES})

	#enabling clang tidy
    enable_clang_tidy_for_project()
