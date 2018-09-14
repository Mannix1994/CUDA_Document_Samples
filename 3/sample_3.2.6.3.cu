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

__global__ void MyKernel(){

}

int main(int argc, char **argv) {

    CHECK_STATUS(cudaSetDevice(0));     // 选择设备0
    cudaStream_t s0;
    cudaStreamCreate(&s0);              // 创建与设备0关联的流s0
    MyKernel<<<100, 64, 0, s0>>>();     // 在设备0的s0上运行MyKernel
    CHECK_STATUS(cudaGetLastError());

    CHECK_STATUS(cudaSetDevice(1));     // 选择设备1
    cudaStream_t s1;
    cudaStreamCreate(&s1);              // 创建与设备1关联的流s1
    MyKernel<<<100, 64, 0, s1>>>();     // 在设备1的s1上运行MyKernel
    CHECK_STATUS(cudaGetLastError());

    // 这个调用会失败
    MyKernel<<<100, 64, 0, s0>>>();     // 在设备1上，在s0上运行MyKernel

    // 1.内存复制会成功
    // 2.如果流和事件关联的设备不一样，cudaEventRecord()会失败
    // 3.如果输入的两时间关联的设备不同，cudaEventElapsedTime()会失败
    // 4.cudaEventSynchronize()和cudaEventQuery()会成功
    // 5.cudaStreamWaitEvent()会成功

    CHECK_STATUS(cudaStreamDestroy(s0));
    CHECK_STATUS(cudaStreamDestroy(s1));
    return 0;
}
