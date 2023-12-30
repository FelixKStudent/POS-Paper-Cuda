
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cuda_runtime_api.h>

const int Width = 4;
const int TILE_WIDTH = 2;

// tag::kernel[]
__global__ void MatrixMulKernel(float* InputA, float* InputB, float* Result, int Width) {
	// Calculate the row index of Result and InputA
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	// Calculate the column index of Result and InputB
	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	if ((Row < Width) && (Col < Width)) {
		float ResultValue = 0;
        // Loop through the row/column and compute the resulting value
		for (int k = 0; k < Width; ++k) {
			ResultValue += InputA[Row * Width + k] * InputB[k * Width + Col];
		}
        Result[Row * Width + Col] = ResultValue;
	}
}
// end::kernel[]


// tag::tiledKernel[]
// tag::shared[]
__global__ void TiledMatrixMulKernel(float* d_InputA, float* d_InputB, float* d_Result, int Width) {
    __shared__ float SharedA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float SharedB[TILE_WIDTH][TILE_WIDTH];
// end::shared[]

// tag::define[]
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify the row and column of the d_Result element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    float ResultValue = 0;
// end::define[]

// tag::loopLoad[]
    // Loop over the d_InputA and d_InputB tiles required to compute d_Result element
    for (int ph = 0; ph < Width / TILE_WIDTH; ++ph) {

        // Collaborative loading of d_InputA and d_InputB tiles into shared memory
        SharedA[ty][tx] = d_InputA[Row * Width + ph * TILE_WIDTH + tx];
        SharedB[ty][tx] = d_InputB[(ph * TILE_WIDTH + ty) * Width + Col];
        __syncthreads();
// end::loopLoad[]

// tag::compute[]
        // Same computation as previously
        for (int k = 0; k < TILE_WIDTH; ++k) {
            ResultValue += SharedA[ty][k] * SharedB[k][tx];
        }
        __syncthreads();
    }
    d_Result[Row * Width + Col] = ResultValue;
}
// end::compute[]
// end::tiledKernel[]

int main() {
    const int matrixSize = Width * Width;

    float h_A[matrixSize];
    float h_B[matrixSize];
    float h_R[matrixSize];

    // Initialize matrices with some values
    for (int i = 0; i < matrixSize; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i + 1);
    }

    float* d_InputA, * d_InputB, * d_Result;
    cudaMalloc((void**)&d_InputA, matrixSize * sizeof(float));
    cudaMalloc((void**)&d_InputB, matrixSize * sizeof(float));
    cudaMalloc((void**)&d_Result, matrixSize * sizeof(float));
    cudaMemcpy(d_InputA, h_A, matrixSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_InputB, h_B, matrixSize * sizeof(float), cudaMemcpyHostToDevice);

    // tag::call[]
    // Define grid and block dimensions & Launch kernel
    dim3 dimGrid(2, 2, 1);
    dim3 dimBlock(2, 2, 1);
    TiledMatrixMulKernel <<<dimGrid, dimBlock >>> (d_InputA, d_InputB, d_Result, Width);
    // end::call[]

    // Copy result from device to host & Free device memory
    cudaMemcpy(h_R, d_Result, matrixSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_InputA);
    cudaFree(d_InputB);
    cudaFree(d_Result);

    // Print the result matrix
    printf("Result Matrix:\n");
    for (int i = 0; i < Width; ++i) {
        for (int j = 0; j < Width; ++j) {
            printf("%f ", h_R[i * Width + j]);
        }
        printf("\n");
    }

    return 0;
}