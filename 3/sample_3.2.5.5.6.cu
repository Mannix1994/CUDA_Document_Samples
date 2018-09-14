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

// stream回调
void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void *data){
    printf("Inside MyCallback %lu\n", (size_t)data);
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
    float *hostPtr[2];
    for (int i = 0; i < 2; ++i)
        CHECK_STATUS(cudaMallocHost(&(hostPtr[i]), size));
    // 初始化
    for (int i = 0; i < 2; ++i)
        for(int j=0;j<N;j++)
            hostPtr[i][j] = j;

    // 分配设备内存
    float *inputDevPtr[2],*outputDevPtr[2];
    for (int i = 0; i < 2; ++i){
        CHECK_STATUS(cudaMalloc(&(inputDevPtr[i]), size));
        CHECK_STATUS(cudaMalloc(&(outputDevPtr[i]), size));
    }

    for (int i = 0; i < 2; ++i) {
        // 把数据从页锁存复制到设备内存
        CHECK_STATUS(cudaMemcpyAsync(inputDevPtr[i], hostPtr[i],
                                     size, cudaMemcpyHostToDevice, stream[i]));
        // 调用kernel
        MyKernel<<<100, 512, 0, stream[i]>>>(outputDevPtr[i], inputDevPtr[i], size);
        // 检查错误
        CHECK_STATUS(cudaGetLastError());
        // 把数据从设备内存拷贝会主机内存
        CHECK_STATUS(cudaMemcpyAsync(hostPtr[i], outputDevPtr[i],
                                     size, cudaMemcpyDeviceToHost, stream[i]));

        // 添加回调。MyCallback会在之前添加到流的任务完成以后被调用
        // 不能在回调里面调用CUDA API，避免造成死锁
        CHECK_STATUS(cudaStreamAddCallback(stream[i], MyCallback, (void*)i, 0));
    }

    // 等待所有流中的的所有任务完成
    cudaDeviceSynchronize();

    // 打印数据
    for(size_t i=0;i<10;i++)
    {
        printf("%.2f\t",hostPtr[0][i]);
    }

    // 销毁流
    for (int i = 0; i < 2; ++i)
        CHECK_STATUS(cudaStreamDestroy(stream[i]));

    // 释放设备内存
    for (int i = 0; i < 2; ++i)
    {
        CHECK_STATUS(cudaFreeHost(hostPtr[i]));
        CHECK_STATUS(cudaFree(inputDevPtr[i]));
        CHECK_STATUS(cudaFree(outputDevPtr[i]));
    }
    return 0;
}
