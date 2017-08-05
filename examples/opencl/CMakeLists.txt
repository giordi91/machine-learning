cmake_minimum_required(VERSION 2.6)



#clang++ -target amdgcn-amd-amdhsa-opencl  -x cl hello.cl -c -S
  
project(01_cl_hello_world)   

    file(GLOB SOURCE_FILES "*.cpp" "*.h")
    find_package(OpenCL REQUIRED) 
    generate_opencl_diassembly(${PROJECT_NAME} hello.cl)

    include_directories(${CMAKE_SOURCE_DIR}/include)
    if(${USE_OPENCL_2_0} STREQUAL "ON")
    add_definitions(-DUSE_OPENCL_2_0)
    endif()


	

    #set(CMAKE_CXX_STANDARD 11) # C++11...
    #set(CMAKE_CXX_STANDARD_REQUIRED ON) #...is required...
    #set(CMAKE_CXX_EXTENSIONS OFF) #...without compiler extensions like gnu++11
    add_executable(${PROJECT_NAME} hello.cpp ${CMAKE_CXX_FLAGS})

    add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy
                   ${CMAKE_CURRENT_SOURCE_DIR}/hello.cl $<TARGET_FILE_DIR:${PROJECT_NAME}>)
    target_link_libraries(${PROJECT_NAME} OpenCL)
