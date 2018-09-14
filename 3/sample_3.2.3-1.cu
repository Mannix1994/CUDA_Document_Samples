#include <stdio.h>
#include <stdlib.h>

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

// 矩阵数据结构定义，行优先存储。
// M(row,col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

// block大小
#define BLOCK_SIZE 16

// 声明矩阵相乘的核函数
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// C=A*B,矩阵乘法
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // 把矩阵A复制到设备内存
    Matrix d_A;
    d_A.width = A.width;
    d_A.height = A.height;
    size_t size_A = A.width * A.height * sizeof(float);
    CHECK_STATUS(cudaMalloc(&d_A.elements, size_A));
    CHECK_STATUS(cudaMemcpy(d_A.elements, A.elements, size_A, cudaMemcpyHostToDevice));

    // 把矩阵B复制到设备内存
    Matrix d_B;
    d_B.width = B.width;
    d_B.height = B.height;
    size_t size_B = B.width * B.height * sizeof(float);
    CHECK_STATUS(cudaMalloc(&d_B.elements, size_B));
    CHECK_STATUS(cudaMemcpy(d_B.elements, B.elements, size_B, cudaMemcpyHostToDevice));

    // 在设备上分配矩阵C的内存
    Matrix d_C;
    d_C.width = C.width;
    d_C.height = C.height;
    size_t size_C = C.width * C.height * sizeof(float);

    CHECK_STATUS(cudaMalloc(&d_C.elements, size_C));

    // 调用Kernel，完成矩阵乘法
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);//矩阵的大小要分别是BLOCK_SIZE的整数倍
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    CHECK_STATUS(cudaGetLastError());

    // 从设备内存中把数据复制主机内存
    CHECK_STATUS(cudaMemcpy(C.elements, d_C.elements, size_C, cudaMemcpyDeviceToHost));

    // 释放设备内存
    CHECK_STATUS(cudaFree(d_A.elements));
    CHECK_STATUS(cudaFree(d_B.elements));
    CHECK_STATUS(cudaFree(d_C.elements));
}

// 矩阵乘法，每个thread完成一行乘以一列
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e) //累加
        Cvalue += A.elements[row * A.width + e]
                  * B.elements[e * B.width + col];
    C.elements[row * C.width + col] = Cvalue;
}

// 分配矩阵内存并赋值
void newMatrix(Matrix *matrix, uint height, uint width, float value){
    matrix->height = height;
    matrix->width = width;
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
        for(size_t j=0;j<matrix.width;j++){
            printf("%.2f\t",matrix.elements[i*matrix.width+j]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    CHECK_STATUS(cudaSetDevice(0));

    Matrix a,b,c;
    newMatrix(&a,32,16,1);
    newMatrix(&b,16,16,1);
    newMatrix(&c,32,16,1);

    MatMul(a,b,c);

    printMatrix(c);

    freeMatrix(&a);
    freeMatrix(&b);
    freeMatrix(&c);
    return 0;
}
