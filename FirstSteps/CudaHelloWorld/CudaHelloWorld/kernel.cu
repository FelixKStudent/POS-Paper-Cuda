#include <iostream>
#include <vector>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__
void vecaddkernel(float* a, float* b, float* c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

void vecadd(const std::vector<float>& h_a, const std::vector<float>& h_b, std::vector<float>& h_c)
{
    int n = h_a.size();
    int size = n * sizeof(float);
    float* d_a, * d_b, * d_c;

    cudaMalloc((void**)&d_a, size);
    cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_b, size);
    cudaMemcpy(d_b, h_b.data(), size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_c, size);

    vecaddkernel<<<ceil(n / 256.0), 256 >>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c.data(), d_c, size, cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

void fillvecs(std::vector<float>& h_a, std::vector<float>& h_b, int n)
{
    for (int i = 0; i < n; i++) {
        h_a.push_back(static_cast<float>(i));  // example: fill h_a with values 0, 1, 2, ..., n-1
        h_b.push_back(static_cast<float>(2 * i));  // example: fill h_b with values 0, 2, 4, ..., 2n-2
    }
}

int main()
{
    const int n = 100000000;

    std::vector<float> h_a, h_b, h_c;
    h_a.reserve(n);
    h_b.reserve(n);
    h_c.resize(n);

    fillvecs(h_a, h_b, n);

    auto start = std::chrono::high_resolution_clock::now();
    vecadd(h_a, h_b, h_c);
    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "time taken by function: " << duration.count() << " microseconds" << std::endl;

    return 0;
}
