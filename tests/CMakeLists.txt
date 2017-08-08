cmake_minimum_required(VERSION 3.6)

project(tests)   

    find_package(GTest REQUIRED) 
    file(GLOB SOURCE_FILES "src/*.cpp" "src/*.h")
    include_directories(${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
	

	#defining standard compiling flags
	if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  /std:c++14 /W4 /arch:AVX2 /fp:fast")
	else()
	     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -pedantic -Wextra -m64 -mavx2 -mfma -ffast-math")
    endif()

	#adding the executable
    add_executable(${PROJECT_NAME} ${SOURCE_FILES} )
    target_link_libraries(${PROJECT_NAME} mg_ml_cpu ${GTEST_BOTH_LIBRARIES} pthread)
