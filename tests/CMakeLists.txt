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


    #here we look for all the test data file, and we are gonna copy that in the output 
    #directory with the executable so there is no running around in finind those file
    #and the code is simpler and more reliable
    file(GLOB files "test_data/*.txt")
    foreach(file ${files})
        copy_file(${file})
    endforeach()
    

    #check if to run the test or not
    if(${RUN_TESTS} STREQUAL "ON")
        #now the problem is that if i am in build, because I did cmake ../, the test executable
        #wont be in the build directory but in a sub folder tests, going into the sub folder
        #run getting out is a pain, so I am adding a custom command to trigger the test after 
        #build and setting the working directo as the current output dir, and expect
        #the test file to be picked as tests/myTestFile.txt
        add_custom_command(TARGET tests 
            COMMAND ./tests/tests
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR} )
    endif()