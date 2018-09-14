#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>

#ifndef NDEBUG
#define CHECK_STATUS(status) \
    if (status != cudaSuccess) \
        fprintf(stderr, "File: %s\nLine:%d Function:%s>>>%s\n", __FILE__, __LINE__, __FUNCTION__,\
        cudaGetErrorString(status))
#else
#define CHECK_STATUS(status) status
#endif
//////////////////////////////////////////////////////////////////////////////////////////////////

// 矩阵数据结构定义，行优先存储。
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;

// block大小
#define BLOCK_SIZE 16

// 获取矩阵的一个元素
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

// 设置矩阵的一个元素
__device__ void SetElement(Matrix A, int row, int col, float value)
{
    A.elements[row * A.stride + col] = value;
}

// 在A中，获取一个大小为BLOCK_SIZExBLOCK_SIZE的子矩阵
__device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}

// 声明矩阵相乘的核函数
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// C=A*B,矩阵乘法
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // 把A和B复制到设备内存
    Matrix d_A;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    CHECK_STATUS(cudaMalloc(&d_A.elements, size));
    CHECK_STATUS(cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice));

    Matrix d_B;
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    CHECK_STATUS(cudaMalloc(&d_B.elements, size));
    CHECK_STATUS(cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice));

    // 在设备上分配C的内存
    Matrix d_C;
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    CHECK_STATUS(cudaMalloc(&d_C.elements, size));

    // 调用kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    CHECK_STATUS(cudaGetLastError());

    // 从设备内存中把C复制到主机内存
    CHECK_STATUS(cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost));

    // 释放设备内存
    CHECK_STATUS(cudaFree(d_A.elements));
    CHECK_STATUS(cudaFree(d_B.elements));
    CHECK_STATUS(cudaFree(d_C.elements));
}

// 矩阵乘法，每个thread完成一行乘以一列
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // 获取本Block的id
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // 每个block计算C的一个子矩阵
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    // 每个线程计算Csub中的一个元素
    float Cvalue = 0;

    // 线程在block内坐标
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    // 遍历计算Csub所需要的A和B的子矩阵
    // 让子矩阵与子矩阵相乘，并累加Cvalue
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {

        // Get sub-matrix Asub of A
        // 获取A的子矩阵Asub
        Matrix Asub = GetSubMatrix(A, blockRow, m);

        // Get sub-matrix Bsub of B
        // 获取B的子矩阵Bsub
        Matrix Bsub = GetSubMatrix(B, m, blockCol);

        // Shared memory used to store Asub and Bsub respectively
        // block内，用来存Asub和Bsub的共享内存
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        // 把数据从设备全局内存复制到共享内存
        // 每个线程复制子矩阵的一个元素
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        // 同步，保证子矩阵的所有元素都已经复制到共享内存之内
        __syncthreads();

        // Multiply Asub and Bsub together
        // 计算As的第row行和Bs的第col列
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        // 同步，保证block的所有线程都完成上述计算
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    // 保存Cvalue到Csub,也就是保存到设备内存
    // 每个线程保存一个元素
    SetElement(Csub, row, col, Cvalue);
}

// 分配矩阵内存并赋值
void newMatrix(Matrix *matrix, uint height, uint width, float value){
    matrix->height = height;
    matrix->width = width;
    matrix->stride = width;
    matrix->elements = new float[width*height];
    for(size_t i=0;i<height*width;i++)
        matrix->elements[i] = value;
}

// 释放矩阵内存
void freeMatrix(Matrix *matrix){
    delete[] matrix->elements;
    matrix->elements = nullptr;
}

// 打印矩阵内存
void printMatrix(Matrix matrix){
    for(size_t i=0;i<matrix.height;i++){
        printf("row:%lu\t",i);
        for(size_t j=0;j<matrix.width;j++){
            printf("%.2f\t",matrix.elements[i*matrix.width+j]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    CHECK_STATUS(cudaSetDevice(0));

    Matrix a,b,c;
    newMatrix(&a,64,16,1);
    newMatrix(&b,16,16,1);
    newMatrix(&c,64,16,1);

    MatMul(a,b,c);

    printMatrix(c);

    freeMatrix(&a);
    freeMatrix(&b);
    freeMatrix(&c);
    return 0;
}
