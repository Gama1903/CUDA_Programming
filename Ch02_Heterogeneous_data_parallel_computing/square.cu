#include <iostream>
#include <cuda_runtime.h>

// 1d square kernel: 向量平方
__global__ void square_kernel(float *input, float *output, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        output[idx] = input[idx] * input[idx];
    }
}

// 2d square kernel: 矩阵平方
__global__ void square_matrix_kernel(float *input, float *output, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols)
    {
        output[row * cols + col] = input[row * cols + col] * input[row * cols + col];
    }
}

int main()
{
    int rows = 1024;
    int cols = 1024;
    size_t size = rows * cols * sizeof(float);

    // Unified memory management: 统一内存管理
    float *input, *output;
    cudaMallocManaged(&input, size);
    cudaMallocManaged(&output, size);

    for (int i = 0; i < rows * cols; ++i)
    {
        input[i] = static_cast<float>(i);
    }

    int deviceId;
    cudaGetDevice(&deviceId); // 获取设备ID

    // data pre-fetch: 数据预取
    cudaMemPrefetchAsync(input, size, deviceId);

    dim3 blockDim(16, 16);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);

    square_matrix_kernel<<<gridDim, blockDim>>>(input, output, rows, cols);

    cudaMemPrefetchAsync(output, size, cudaCpuDeviceId);

    // device synchronize: 设备同步
    cudaDeviceSynchronize();

    // 检查是否有错误发生
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // free memory pointer: 释放内存指针
    cudaFree(input);
    cudaFree(output);

    return 0;
}