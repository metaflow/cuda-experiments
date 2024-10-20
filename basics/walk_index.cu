#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void walk() {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  printf("b (x, y, z): %2d, %2d, %2d t (x, y, z): %2d, %2d, %2d => %2d\n",
         blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z, idx);
}

void checkCuda(cudaError_t err) {
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

int main() {
  int threadsPerBlock = 32;
  int numBlocks = 2;

  printf("Launching kernel with %d blocks of %d threads each\n", numBlocks,
         threadsPerBlock);

  walk<<<numBlocks, threadsPerBlock>>>();
  checkCuda(cudaGetLastError());
  checkCuda(cudaDeviceSynchronize());
  return 0;
}
