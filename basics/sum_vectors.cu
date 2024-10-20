#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>
#include <stdlib.h>

inline void checkCuda(cudaError_t err) {
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

void initVector(float *vec, int size) {
  for (int i = 0; i < size; i++) {
    vec[i] = ((float)rand()) / RAND_MAX;
  }
}

// TODO: try with multiple values per thread - either close together or strided.
// Strategy 1: thread gets sequential values.
// __global__ void sumVectorsClose(float *a, float *b, float *c, int size,
//                            int valuesPerThread, int threadsPerBlock) {
//   int idx = (blockIdx.x * blockDim.x + threadIdx.x) * valuesPerThread;

//   for (int i = 0; i < valuesPerThread && idx + i < size; i++) {
//     c[idx + i] = a[idx + i] + b[idx + i];
//   }
// }

// Strategy 2: thread values strided by block size.
// __global__ void sumVectors(float *a, float *b, float *c, int size,
//                            int valuesPerThread, int threadsPerBlock) {
//   int x = blockIdx.x * blockDim.x * valuesPerThread + threadIdx.x;
//   for (int i = 0; x < size && i < valuesPerThread; i++) {
//     c[x] = a[x] + b[x];
//     x += blockDim.x;
//   }
// }

// Strategy 3: thread values strided by block size times threads per block.
__global__ void sumVectors(float *a, float *b, float *c, int size,
                           int valuesPerThread) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = 0; x < size && i < valuesPerThread; i++) {
    c[x] = a[x] + b[x];
    x += blockDim.x * gridDim.x;
  }
}

void sumVectorsOnHost(float *a, float *b, float *c, int size) {
  for (int i = 0; i < size; i++) {
    c[i] = a[i] + b[i];
  }
}

double get_time_sec() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char *argv[]) {
  srand(time(NULL));
  float *h_a, *h_b, *h_c, *h_c_ref;
  float *d_a, *d_b, *d_c;

  int n = 1 << 26;
  if (argc > 1) {
    n = atoi(argv[1]);
  }
  int threadsPerBlock = 32;
  if (argc > 2) {
    threadsPerBlock = atoi(argv[2]);
  }
  int valuesPerThread = 1;
  if (argc > 3) {
    valuesPerThread = atoi(argv[3]);
  }

  size_t size = n * sizeof(float);

  h_a = (float *)malloc(size);
  h_b = (float *)malloc(size);
  h_c = (float *)malloc(size);
  h_c_ref = (float *)malloc(size);

  nvtxRangePush("Init");
  initVector(h_a, n);
  initVector(h_b, n);
  checkCuda(cudaMalloc(&d_a, size));
  checkCuda(cudaMalloc(&d_b, size));
  checkCuda(cudaMalloc(&d_c, size));

  int extra_iterations = 0;
  double t1, t2;
  if (extra_iterations > 0) {
    t1 = get_time_sec();
    for (int i = 0; i < extra_iterations; i++) {
      checkCuda(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
      checkCuda(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));
      checkCuda(cudaDeviceSynchronize());
    }
    t2 = get_time_sec();
    printf("Copy: %fs\n", (t2 - t1) / extra_iterations);
  }
  checkCuda(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));
  checkCuda(cudaDeviceSynchronize());
  nvtxRangePop();

  int blocksPerGrid = (n + threadsPerBlock * valuesPerThread - 1) /
                      (threadsPerBlock * valuesPerThread);
  printf("N: %d, values per thread: %d, threads / block: %d, blocks: %d\n", n,
         valuesPerThread, threadsPerBlock, blocksPerGrid);
  // Warm up runs.
  for (int i = 0; i < 10; i++) {
  }

  if (extra_iterations > 0) {
    for (int i = 0; i < 10; i++) {
      sumVectorsOnHost(h_a, h_b, h_c_ref, n);
    }
    t1 = get_time_sec();
    for (int i = 0; i < extra_iterations; i++) {
      sumVectorsOnHost(h_a, h_b, h_c_ref, n);
    }
    t2 = get_time_sec();
    printf(" CPU: %fs\n", (t2 - t1) / extra_iterations);
  }

  nvtxRangePush("Check correctness");
  sumVectors<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n,
                                                 valuesPerThread);
  checkCuda(cudaGetLastError());
  checkCuda(cudaDeviceSynchronize());
  checkCuda(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));
  sumVectorsOnHost(h_a, h_b, h_c_ref, n);
  for (int i = 0; i < n; i++) {
    if (h_c[i] != h_c_ref[i]) {
      printf("Error: h_c[%d] = %f, h_c_ref[%d] = %f\n", i, h_c[i], i,
             h_c_ref[i]);
      exit(EXIT_FAILURE);
    }
  }
  printf("OK\n");
  nvtxRangePop();

  nvtxRangePush("Vector Addition");
  double start = get_time_sec();
  int iterations = 0;
  while (get_time_sec() - start < 10) {
    sumVectors<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n,
                                                   valuesPerThread);
    checkCuda(cudaDeviceSynchronize());
    iterations++;
  }
  nvtxRangePop();
  double gpu_time = (get_time_sec() - start) / iterations;
  printf("%fs\n%.2f GB/s\n", gpu_time, 2 * n * sizeof(float) / gpu_time / 1e9);
  return 0;
}
