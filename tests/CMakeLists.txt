cmake_minimum_required(VERSION 3.6)

project(tests)   

    find_package(GTest REQUIRED) 
    file(GLOB SOURCE_FILES "src/*.cpp" "src/*.h")
    include_directories(${CMAKE_SOURCE_DIR}/include ${GTEST_INCLUDE_DIRS})
	

	#defining standard compiling flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_CXX_FLAGS}")

    if(NOT WIN_32)
     set(PTHREAD_LIB "-lpthread")
    endif()
    
	#adding the executable
    add_executable(${PROJECT_NAME} ${SOURCE_FILES} )
    target_link_libraries(${PROJECT_NAME} ${MAIN_LIB_NAME} ${GTEST_BOTH_LIBRARIES} 
                        ${PTHREAD_LIB})


    
    if(WIN32) 
        file(COPY ${CMAKE_SOURCE_DIR}/tests/test_data/datasets DESTINATION ${CMAKE_BINARY_DIR}/${PROJECT_NAME})
        file(GLOB files "test_data/*.txt")
        foreach(file ${files})
            file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/${PROJECT_NAME} )
        endforeach()
    else()
        #here we look for all the test data file, and we are gonna copy that in the output 
        #directory with the executable so there is no running around in finind those file
        #and the code is simpler and more reliable
        file(GLOB files "test_data/*.txt")
        foreach(file ${files})
            copy_file(${file})
        endforeach()
    endif()

    #check if to run the test or not
    if(${RUN_TESTS} STREQUAL "ON")
        if (NOT WIN32)


            #now the problem is that if i am in build, because I did cmake ../, the test executable
            #wont be in the build directory but in a sub folder tests, going into the sub folder
            #run getting out is a pain, so I am adding a custom command to trigger the test after 
            #build and setting the working directo as the current output dir, and expect
            #the test file to be picked as tests/myTestFile.txt

            add_custom_command(
                DEPENDS tests
                COMMAND ./tests/tests
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR} 
                OUTPUT run_my_tests)

            # Custom target will always cause its dependencies to be evaluated and is
            # run by default
            add_custom_target(dummy_target ALL
                DEPENDS
                  run_my_tests 
                  create_dataset_soft_link
                )
            MESSAGE(STATUS "Creating soft link for dataset: \n  ${CMAKE_SOURCE_DIR}/tests/test_data/datasets ->  ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/datasets")
            #command to generate a soft link to the dataset folder
                add_custom_command(
                    DEPENDS tests
                    COMMAND ln -f -s ${CMAKE_SOURCE_DIR}  ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/datasets 
                    OUTPUT create_dataset_soft_link 
                    )
        endif()
        
    endif()

