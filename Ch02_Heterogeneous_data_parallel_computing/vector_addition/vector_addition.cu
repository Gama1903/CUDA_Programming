// each thread computes one element of the result vector
__global__ void vecAddKernel_1(float *a, float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}

// each thread computes two adjacent elements of the result vector
__global__ void vecAddKernel_2(float *a, float *b, float *c, int n)
{
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    int j = i + 1;
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
    if (j < n)
    {
        c[j] = a[j] + b[j];
    }
}

// each thread computes two elements separated by blockDim.x elements of the result vector
__global__ void vecAddKernel_3(float *a, float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    int j = i + blockDim.x;
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
    if (j < n)
    {
        c[j] = a[j] + b[j];
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

void vecAdd(float *a, float *b, float *c, int n)
{
    float *a_d, *b_d, *c_d;
    int size = n * sizeof(float);

    cudaMalloc((void **)&a_d, size);
    cudaMalloc((void **)&b_d, size);
    cudaMalloc((void **)&c_d, size);

    cudaMemcpy(a_d, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, size, cudaMemcpyHostToDevice);

    const unsigned int threadNum = 256;
    const unsigned int blockNum = cdiv(n, threadNum);

    vecAddKernel_1<<<blockNum, threadNum>>>(a_d, b_d, c_d, n);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    cudaMemcpy(c, c_d, size, cudaMemcpyDeviceToHost);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}
