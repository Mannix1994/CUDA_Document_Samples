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

__global__ void MyKernel(float* output, float* input, size_t size){
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i>size/ sizeof(float))
        return;
    output[i] = input[i] - 5;
}

int main(int argc, char **argv) {
    CHECK_STATUS(cudaSetDevice(0));

    // 创建两个流
    cudaStream_t stream[2];
    for (int i = 0; i < 2; ++i)
        CHECK_STATUS(cudaStreamCreate(&stream[i]));

    // 在主机内存上分配页锁存(page-locked memory)
    const int N = 512;
    size_t size = N * sizeof(float);
    float *hostPtr;
    CHECK_STATUS(cudaMallocHost(&hostPtr, 2 * size));
    // 初始化
    for(size_t i=0;i<N*2;i++)
        hostPtr[i] = i;

    // 分配设备内存
    float *inputDevPtr,*outputDevPtr;
    CHECK_STATUS(cudaMalloc(&inputDevPtr,2*size));
    CHECK_STATUS(cudaMalloc(&outputDevPtr,2*size));

    // 创建事件
    cudaEvent_t start, stop;
    CHECK_STATUS(cudaEventCreate(&start));
    CHECK_STATUS(cudaEventCreate(&stop));

    // 录制事件
    CHECK_STATUS(cudaEventRecord(start, 0));
    for (int i = 0; i < 2; ++i) {
        // 把数据从页锁存复制到设备内存
        CHECK_STATUS(cudaMemcpyAsync(inputDevPtr + i * N, hostPtr + i * N,
                                     size, cudaMemcpyHostToDevice, stream[i]));
        // 调用kernel
        MyKernel<<<100, 512, 0, stream[i]>>>(outputDevPtr + i * N, inputDevPtr + i * N, size);
        // 检查错误
        CHECK_STATUS(cudaGetLastError());
        // 把数据从设备内存拷贝会主机内存
        CHECK_STATUS(cudaMemcpyAsync(hostPtr + i * N, outputDevPtr + i * N,
                                     size, cudaMemcpyDeviceToHost, stream[i]));
    }
    // 录制事件
    CHECK_STATUS(cudaEventRecord(stop, 0));

    // 调用这个函数之后，stop之前所有的cuda调用完成之后才会返回
    CHECK_STATUS(cudaEventSynchronize(stop));

    // 计算运行CUDA的时间
    float elapsedTime;
    CHECK_STATUS(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("elapsedTime: %fms\n",elapsedTime);

    // 打印数据
    for(size_t i=0;i<10;i++)
    {
        printf("%.2f\t",hostPtr[i]);
    }

    // 销毁流
    for (int i = 0; i < 2; ++i)
        CHECK_STATUS(cudaStreamDestroy(stream[i]));

    // 释放设备内存
    CHECK_STATUS(cudaFreeHost(hostPtr));
    CHECK_STATUS(cudaFree(inputDevPtr));
    CHECK_STATUS(cudaFree(outputDevPtr));

    // 销毁事件
    CHECK_STATUS(cudaEventDestroy(start));
    CHECK_STATUS(cudaEventDestroy(stop));
    return 0;
}
