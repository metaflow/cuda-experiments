#include <cuda_runtime.h>
#include <stdio.h>

const int N = 1024;
const int M = 2048;

inline void checkCuda(cudaError_t res, const char *msg = "") {
  if (res != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(res));
    exit(1);
  }
}

void initMatrix(float *matrix, int N, int M) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      matrix[i * M + j] = rand() / (float)RAND_MAX;
    }
  }
}

void transposeMatrix(float *dst, float *src, int N, int M) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      dst[j * N + i] = src[i * M + j];
    }
  }
}

double get_time_sec() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

__global__ void transposePerThread(float *dst, float *src) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < N && j < M) {
    dst[j * N + i] = src[i * M + j];
  }
}

__global__ void transposeGridStrided(float *dst, float *src) {
  for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < N;
       x += gridDim.x * blockDim.x) {
    for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < M;
         y += blockDim.y * gridDim.y) {
      dst[y * N + x] = src[x * M + y];
    }
  }
}

#define INT_CEIL(x, y) ((x + y - 1) / y)

void verify(float *result, float *reference) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      if (result[i * N + j] != reference[i * N + j]) {
        printf("Mismatch at %d %d: %f <> %f\n", i, j, result[i * N + j],
               reference[i * N + j]);
        exit(1);
      }
    }
  }
  printf("Correctness: OK\n");
}

int main(int argc, char **argv) {
  int size = N * M * sizeof(float);
  float *h_a = (float *)malloc(size);
  float *h_b = (float *)malloc(size);
  float *h_b_ref = (float *)malloc(size);
  float *d_a;
  float *d_b;

  initMatrix(h_a, N, M);
  transposeMatrix(h_b_ref, h_a, N, M);

  checkCuda(cudaMalloc(&d_a, size));
  checkCuda(cudaMalloc(&d_b, size));

  checkCuda(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));

  int blockX = 32;
  if (argc > 1) {
    blockX = atoi(argv[1]);
  }
  int blockY = 32;
  if (argc > 2) {
    blockY = atoi(argv[2]);
  }

  dim3 blockSize(blockX, blockY, 1);
  dim3 gridSize(INT_CEIL(N, blockSize.x), INT_CEIL(M, blockSize.y), 1);
  transposePerThread<<<gridSize, blockSize>>>(d_b, d_a);
  checkCuda(cudaDeviceSynchronize());
  checkCuda(cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost));
  printf("Proper grid\n");
  verify(h_b, h_b_ref);

  double t1 = get_time_sec();
  int cnt = 0;
  while (get_time_sec() - t1 < 10.0) {
    transposePerThread<<<gridSize, blockSize>>>(d_b, d_a);
    cnt++;
  }
  double t2 = get_time_sec();
  printf("%d iterations, %.2f GB/s\n", cnt,
         ((double)cnt * N * M * sizeof(float) * 2) / (t2 - t1) / 1e9);

  transposeGridStrided<<<1024, blockSize>>>(d_b, d_a);
  checkCuda(cudaDeviceSynchronize());
  checkCuda(cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost));
  printf("\n");
  printf("Grid strided\n");
  verify(h_b, h_b_ref);

  t1 = get_time_sec();
  cnt = 0;
  while (get_time_sec() - t1 < 10.0) {
    transposeGridStrided<<<1024, blockSize>>>(d_b, d_a);
    cnt++;
  }
  t2 = get_time_sec();
  printf("%d iterations, %.2f GB/s\n", cnt,
         ((double)cnt * N * M * sizeof(float) * 2) / (t2 - t1) / 1e9);

  checkCuda(cudaFree(d_a));
  checkCuda(cudaFree(d_b));
  free(h_a);
  free(h_b);
  free(h_b_ref);

  return 0;
}
