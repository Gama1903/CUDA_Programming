// Kernel function to add two vectors
__global__ void vectorAddTwoElements_1(const float *a, const float *b, float *c, int n)
{
    // Calculate the global thread index using option (C)
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

    // Ensure we do not go out of bounds
    if (i < n - 1)
    { // Check if both i and i+1 are within bounds
        // Each thread adds two elements
        c[i] = a[i] + b[i];
        c[i + 1] = a[i + 1] + b[i + 1];
    }
}

// Kernel function to add two vectors
__global__ void vectorAddTwoElements_2(const float *a, const float *b, float *c, int n)
{
    // Calculate the global thread index using option (C)
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

    // Ensure we do not go out of bounds
    if (i < n)
    { // Check if both i and i+1 are within bounds
        // Each thread adds two elements
        c[i] = a[i] + b[i];
    }
    int j = i + 1;
    if (j < n)
    {
        c[j] = a[j] + b[j];
    }
}