#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_STATUS(status) \
    if (status != cudaSuccess) \
        fprintf(stderr, "File: %s\nLine:%d Function:%s>>>%s\n", __FILE__, __LINE__, __FUNCTION__,\
        cudaGetErrorString(status))
//////////////////////////////////////////////////////////////////////////////////////////////////

// 设备全局常量内存
__constant__ float constData[256];
// 设备全局内存
__device__ float devData;
// 设备全局内存指针
__device__ float* devPointer;


// Device code
__global__ void MyKernel()
{
    printf("%lu\n", sizeof(devData));
}

int main(int argc, char **argv) {
    CHECK_STATUS(cudaSetDevice(0));

    float data[256];
    // 复制数据到设备全局常量内存
    cudaMemcpyToSymbol(constData, data, sizeof(data));
    // 从设备全局常量内存把数据复制主机内存
    cudaMemcpyFromSymbol(data, constData, sizeof(data));

    // 复制数据到设备全局内存
    float value = 3.14f;
    cudaMemcpyToSymbol(devData, &value, sizeof(float));

    float* ptr;
    cudaMalloc(&ptr, 256 * sizeof(float));
    // 把ptr这个指针复制到设备全局内存的指针devPointer。
    // 注意，这里复制的是指针。
    cudaMemcpyToSymbol(devPointer, &ptr, sizeof(ptr));

    // 调用内核
    MyKernel<<<1, 1>>>();

    // 检查错误
    CHECK_STATUS(cudaGetLastError());

    // 释放内存，这里要释放ptr，而不是devPointer
    CHECK_STATUS(cudaFree(ptr));
    return 0;
}
