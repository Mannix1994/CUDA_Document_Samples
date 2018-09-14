#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_STATUS(status) \
    if (status != cudaSuccess) \
        fprintf(stderr, "File: %s\nLine:%d Function:%s>>>%s\n", __FILE__, __LINE__, __FUNCTION__,\
        cudaGetErrorString(status))

// Device code
__global__ void MyKernel(cudaPitchedPtr devPitchedPtr,
                         int width, int height, int depth)
{
    char* devPtr = (char*)devPitchedPtr.ptr;    //获取数据指针
    size_t pitch = devPitchedPtr.pitch;         //获取一行所占的字节数
    size_t slicePitch = pitch * height;         //获取一层的大小，单位为字节
    for (int z = 0; z < depth; ++z) {
        char* slice = devPtr + z * slicePitch;          // 得到第z层的起始地址
        for (int y = 0; y < height; ++y) {
            float* row = (float*)(slice + y * pitch);   //得到第z层中，第y行的起始地址
            for (int x = 0; x < width; ++x) {
                float element = row[x];                 //得到第y行的第x个元素
            }
        }
    }
}
int main(int argc, char **argv) {
    CHECK_STATUS(cudaSetDevice(0));

    size_t width = 64, height = 64, depth = 64;

    // 定义三维数组大小
    cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);

    // 三维数组的一个数据结构
    cudaPitchedPtr devPitchedPtr;

    // 分配三维数组
    CHECK_STATUS(cudaMalloc3D(&devPitchedPtr,extent));

    // 调用内核
    MyKernel<<<100, 512>>>(devPitchedPtr, width, height, depth);

    // 检查错误
    CHECK_STATUS(cudaGetLastError());

    // 释放内存
    CHECK_STATUS(cudaFree(devPitchedPtr.ptr));
    return 0;
}
