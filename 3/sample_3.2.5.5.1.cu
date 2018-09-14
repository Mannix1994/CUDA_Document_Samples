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
    cudaDeviceProp prop;
    CHECK_STATUS(cudaGetDeviceProperties(&prop,0));
    printf("Kernel并发执行:%d, 异步引擎数量:%d\n",prop.concurrentKernels,prop.asyncEngineCount);

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

    // 同步，有三种方式，只列出两种，选一种就行
    // 1.等待所有流中的的所有任务完成
    cudaDeviceSynchronize();
    // 2.分别同步不同的流，可以只等待其中的某个流中的任务完成
    for (int i = 0; i < 2; ++i)
        CHECK_STATUS(cudaStreamSynchronize(stream[i]));

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
    return 0;
}
