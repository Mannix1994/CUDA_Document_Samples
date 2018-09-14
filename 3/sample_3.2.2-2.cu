#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_STATUS(status) \
    if (status != cudaSuccess) \
        fprintf(stderr, "File: %s\nLine:%d Function:%s>>>%s\n", __FILE__, __LINE__, __FUNCTION__,\
        cudaGetErrorString(status))

// Device code
__global__ void MyKernel(float* devPtr,
                         size_t pitch, int width, int height)
{
    for (int r = 0; r < height; ++r) {
        float* row = (float*)((char*)devPtr + r * pitch);//第r行
        for (int c = 0; c < width; ++c) {
            float element = row[c];
        }
    }
}

int main(int argc, char **argv) {
    CHECK_STATUS(cudaSetDevice(0));
    size_t width = 64, height = 64;
    float* devPtr;
    size_t pitch;
    // 分配二维数组
    CHECK_STATUS(cudaMallocPitch(&devPtr, &pitch, width * sizeof(float), height));

    // 调用内核
    MyKernel<<<100, 512>>>(devPtr, pitch, width, height);

    // 检查错误
    CHECK_STATUS(cudaGetLastError());

    // 释放内存
    CHECK_STATUS(cudaFree(devPtr));
    return 0;
}
