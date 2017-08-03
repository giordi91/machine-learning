cmake_minimum_required(VERSION 2.6.3)
#including utils function, this will cascade to all subsequent included cmakes
include (cmakeutils/code_gen.txt)

#options
option(DUMP_KERNEL_ASSEMBLY "Wheter or not to generate readable assembly code from kernels" ON)

#adding opencl examples
add_subdirectory(examples/opencl/01_hello_world)
