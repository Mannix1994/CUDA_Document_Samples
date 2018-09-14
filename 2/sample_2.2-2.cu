#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>

static const int M = 16;//行
static const int N = 32;//列

#define CHECK_STATUS(status) \
    if (status != cudaSuccess) \
        fprintf(stderr, "File: %s\nLine:%d Function:%s>>>%s\n", __FILE__, __LINE__, __FUNCTION__,\
        cudaGetErrorString(status))

//二维数组相加
__global__ void MatAdd(float *A, float *B, float *C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N) {
        int index = i * N + j;
        C[index] = A[index] + B[index];
    }
}

int main(int argc, char **argv) {
    CHECK_STATUS(cudaSetDevice(0));
    const int SIZE = M * N;
    float a[SIZE];
    float b[SIZE];
    for(int i = 0;i<SIZE;i++){
        a[i] = i;
        b[i] = i;
    }
    float c[SIZE];

    float *d_a,*d_b,*d_c;

    //分配显存
    CHECK_STATUS(cudaMalloc(&d_a, SIZE*sizeof(float)));
    CHECK_STATUS(cudaMalloc(&d_b, SIZE*sizeof(float)));
    CHECK_STATUS(cudaMalloc(&d_c, SIZE*sizeof(float)));

    // 把数据从内存复制到显存
    CHECK_STATUS(cudaMemcpy(d_a,a,SIZE* sizeof(float),cudaMemcpyHostToDevice));
    CHECK_STATUS(cudaMemcpy(d_b,b,SIZE* sizeof(float),cudaMemcpyHostToDevice));

    // 调用kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(M / threadsPerBlock.x, N / threadsPerBlock.y);
    MatAdd<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c);

    // 检查错误
    CHECK_STATUS(cudaGetLastError());

    // 从显存把数据复制到内存
    CHECK_STATUS(cudaMemcpy(c,d_c,SIZE* sizeof(float),cudaMemcpyDeviceToHost));

    // 打印
    for(int i=0;i<M;i++)
    {
        for(int j=0;j<N;j++)
            printf("%f\t",c[i*N + j]);
        printf("\n");
    }

    //释放显存
    CHECK_STATUS(cudaFree(d_a));
    CHECK_STATUS(cudaFree(d_b));
    CHECK_STATUS(cudaFree(d_c));
    return 0;
}
