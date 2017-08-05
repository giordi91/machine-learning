#pragma once
#ifndef USE_OPENCL_2_0
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

const size_t ONE = 1;

#if USE_OPENCL_2_0
#define clCreateCommandQueueWrap(...)                                          \
  clCreateCommandQueueWithProperties(__VA_ARGS__);
#else
#define clCreateCommandQueueWrap(...)                                          \
  clCreateCommandQueue(__VA_ARGS__);
#endif

//this define is used to wrap the enqueTask that is deprecated in opencl 2.0 but 
//hopefully should never be used
#if USE_OPENCL_2_0
#define clEnqueueTaskWrap(command_queue, kernel, eventCount, eventWait,        \
                          outEvent)                                            \
  clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &ONE, &ONE, eventCount,     \
                         eventWait, outEvent);
#else
#define clEnqueueTaskWrap(command_queue, kernel, eventCount, eventWait,        \
                          outEvent)                                            \
  clEnqueueTask(command_queue, kernel, eventCount, eventWait, outEvent);
#endif
