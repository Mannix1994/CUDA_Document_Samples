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

__global__ void MyKernel(float* data){

}

int main(int argc, char **argv) {
    int can;
    CHECK_STATUS(cudaDeviceCanAccessPeer(&can,0,1));
    printf("是否支持peer-to-peer access：%d",can);

    size_t size = 1024 * sizeof(float);
    CHECK_STATUS(cudaSetDevice(0));     // 选择设备0
    float* p0;
    CHECK_STATUS(cudaMalloc(&p0, size));// 在设备0上分配设备内存

    CHECK_STATUS(cudaSetDevice(1));     // 选择设备1
    float* p1;
    CHECK_STATUS(cudaMalloc(&p1, size));// 在设备1上分配设备内存

    CHECK_STATUS(cudaSetDevice(0));     // 选择设备0
    MyKernel<<<1000, 128>>>(p0);        // 在设备0上运行MyKernel
    CHECK_STATUS(cudaGetLastError());

    CHECK_STATUS(cudaSetDevice(1));     // 选择设备1
    // cudaMemcpyPeer
    // 1.之前的所有任务执行完之后才会被调用
    // 2.完成之后才会执行后面的命令
    CHECK_STATUS(cudaMemcpyPeer(p1, 1, p0, 0, size));   // 把p0复制到p1
    MyKernel<<<1000, 128>>>(p1);        // 在设备1上运行MyKernel
    CHECK_STATUS(cudaGetLastError());

    CHECK_STATUS(cudaFree(p0));
    CHECK_STATUS(cudaFree(p1));
    return 0;
}
