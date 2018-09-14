#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifndef NDEBUG
#define CHECK_STATUS(status) \
    if (status != cudaSuccess) \
        fprintf(stderr, "File: %s\nLine:%d Function:%s>>>%s\n", __FILE__, __LINE__, __FUNCTION__,\
        cudaGetErrorString(status))
#else
#define CHECK_STATUS(status) status
#endif
//////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
    int deviceCount;
    CHECK_STATUS(cudaGetDeviceCount(&deviceCount));
    int device;
    for (device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        CHECK_STATUS(cudaGetDeviceProperties(&deviceProp, device));
        printf("Device %d has compute capability %d.%d.\n",
               device, deviceProp.major, deviceProp.minor);
    }
    return 0;
}
