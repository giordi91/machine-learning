cmake_minimum_required(VERSION 3.6)



#clang++ -target amdgcn-amd-amdhsa-opencl  -x cl hello.cl -c -S
  
project(04_logistic_regression)   

    file(GLOB SOURCE_FILES "*.cpp" "*.h")
    include_directories(${CMAKE_SOURCE_DIR}/include )
	
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_CXX_FLAGS}")

	#adding the executable
    add_executable(${PROJECT_NAME}  ${SOURCE_FILES})
    target_link_libraries(${PROJECT_NAME} ${MAIN_LIB_NAME} )
    
	#enabling clang tidy
    enable_clang_tidy_for_project()
