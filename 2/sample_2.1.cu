#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>

static const int N = 10;

#define CHECK_STATUS(status) \
    if (status != cudaSuccess) \
        fprintf(stderr, "File: %s\nLine:%d Function:%s>>>%s\n", __FILE__, __LINE__, __FUNCTION__,\
        cudaGetErrorString(status))

//
__global__ void VecAdd(float *A, float *B, float *C) {
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main(int argc, char **argv) {
    CHECK_STATUS(cudaSetDevice(0));
    float a[N];
    float b[N];
    for(int i = 0;i<N;i++){
        a[i] = i;
        b[i] = i;
    }
    float c[N];

    float *d_a,*d_b,*d_c;

    //分配显存
    CHECK_STATUS(cudaMalloc(&d_a, N*sizeof(float)));
    CHECK_STATUS(cudaMalloc(&d_b, N*sizeof(float)));
    CHECK_STATUS(cudaMalloc(&d_c, N*sizeof(float)));

    // 把数据从内存复制到显存
    CHECK_STATUS(cudaMemcpy(d_a,a,N* sizeof(float),cudaMemcpyHostToDevice));
    CHECK_STATUS(cudaMemcpy(d_b,b,N* sizeof(float),cudaMemcpyHostToDevice));

    // 调用kernel
    VecAdd<<<1,N>>>(d_a,d_b,d_c);

    // 检查错误
    CHECK_STATUS(cudaGetLastError());

    // 从显存把数据复制到内存
    CHECK_STATUS(cudaMemcpy(c,d_c,N* sizeof(float),cudaMemcpyDeviceToHost));

    // 打印
    for(int i=0;i<N;i++)
        printf("%f ",c[i]);
    printf("\n");
    
    //释放显存
    CHECK_STATUS(cudaFree(d_a));
    CHECK_STATUS(cudaFree(d_b));
    CHECK_STATUS(cudaFree(d_c));
    return 0;
}
