#include <iostream>
#include <cuda_runtime.h>

__global__ void vecSquareKernel(float *input, float *output, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        output[idx] = input[idx] * input[idx];
    }
}

// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
        {
            exit(code);
        }
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b)
{
    return (a + b - 1) / b;
}

// host memory manage
__global__ void vecSquare_1(float *input, float *output, int n)
{
    float *a_d, *b_d;
    size_t size = n * sizeof(float);

    cudaMalloc((void **)&a_d, size);
    cudaMalloc((void **)&b_d, size);

    cudaMemcpy(a_d, input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, output, size, cudaMemcpyHostToDevice);

    const unsigned int threadNum = 256;
    unsigned int blockNum = cdiv(n, threadNum);

    vecSquareKernel<<<blockNum, threadNum>>>(a_d, b_d, n);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    cudaMemcpy(output, b_d, size, cudaMemcpyDeviceToHost);

    cudaFree(a_d);
    cudaFree(b_d);
}

// unified memory manage and data prefetch
__global__ void vecSquare_2(float *input, float *output, int n)
{
    int deviceId;
    cudaGetDevice(&deviceId);
    size_t size = n * sizeof(float);

    cudaMallocManaged(&input, size);
    cudaMallocManaged(&output, size);

    cudaMemPrefetchAsync(input, size, deviceId);

    const unsigned int threadNum = 256;
    unsigned int blockNum = cdiv(n, threadNum);

    vecSquareKernel<<<blockNum, threadNum>>>(input, output, n);
    cudaMemPrefetchAsync(output, size, cudaCpuDeviceId);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    cudaFree(input);
    cudaFree(output);
}