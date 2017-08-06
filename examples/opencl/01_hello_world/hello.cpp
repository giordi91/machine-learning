#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>

#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x100000)

#include <mg_ml/gpu/opencl/cl_wrap.h>

int main() {
  cl_device_id device_id = nullptr;
  cl_context context = nullptr;
  cl_command_queue command_queue = nullptr;
  cl_mem memobj = nullptr;
  cl_program program = nullptr;
  cl_kernel kernel = nullptr;
  cl_platform_id platform_id = nullptr;
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret;

  char string[MEM_SIZE];

  FILE *fp = nullptr;
  char fileName[] = "./hello.cl";
  //char *source_str;
  size_t source_size;

  // Load the source code containing the kernel
  fp = fopen(static_cast<char*>(fileName), "r");
  if (fp == nullptr) {
    std::cerr<< "Failed to load kernel."<<std::endl;
    exit(1);
  }
  auto source_str = std::make_unique<char[]>(MAX_SOURCE_SIZE);
  source_size = fread(source_str.get(), 1, MAX_SOURCE_SIZE, fp);
  fclose(fp);

  // Get Platform and Device Info
  ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
  ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id,
                       &ret_num_devices);

  // Create OpenCL context
  context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &ret);

  // Create Command Queue
  const size_t ZERO= 0;
  command_queue = clCreateCommandQueueWrap(context, device_id, &ZERO, &ret);

  // Create Memory Buffer
  memobj = clCreateBuffer(context, CL_MEM_READ_WRITE, MEM_SIZE * sizeof(char),
                          nullptr, &ret);

  // Create Kernel Program from the source
  const char* raw_source_ptr = source_str.get();
  program = clCreateProgramWithSource(context, 1, &raw_source_ptr,
                                      static_cast<const size_t *>(&source_size), &ret);

  // Build Kernel Program
  ret = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);

  // Create OpenCL Kernel
  kernel = clCreateKernel(program, "hello", &ret);

  // Set OpenCL Kernel Parameters
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), static_cast<void *>(&memobj));

  // Execute OpenCL Kernel
  ret = clEnqueueTaskWrap(command_queue, kernel, 0, nullptr, nullptr);

  // Copy results from the memory buffer
  ret =
      clEnqueueReadBuffer(command_queue, memobj, CL_TRUE, 0,
                          MEM_SIZE * sizeof(char), static_cast<char*>(string), 0, nullptr, nullptr);

  // Display Result
  std::cout<<static_cast<char*>(string)<<std::endl;;

  // Finalization
  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseKernel(kernel);
  ret = clReleaseProgram(program);
  ret = clReleaseMemObject(memobj);
  ret = clReleaseCommandQueue(command_queue);
  ret = clReleaseContext(context);

  return 0;
}
