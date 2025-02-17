#include "../include/utils.h"
#include <cuda_runtime.h>

#define NUM_RUNS 10

#define CUDA_CHECK(func)                                                     	   \
	do {                                                                           \
		cudaError_t status = (func);                                               \
		if (status != cudaSuccess) {                                               \
			printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__,   \
				cudaGetErrorString(status), status);                               \
			exit(EXIT_FAILURE);                                                    \
		}                                                                          \
	} while (0)

#define CHECK(name) \
	float *d_Aref_ ## name, *d_Bref_ ## name, *d_Cref_ ## name; \
	std::cerr << "checking " << #name << std::endl; \
	CUDA_CHECK(cudaMalloc(&d_Aref_ ## name, Ref::M * Ref::K * sizeof(float))); \
	CUDA_CHECK(cudaMalloc(&d_Bref_ ## name, Ref::K * Ref::N * sizeof(float))); \
	CUDA_CHECK(cudaMalloc(&d_Cref_ ## name, Ref::M * Ref::N * sizeof(float))); \
	CUDA_CHECK(cudaMemcpy(d_Aref_ ## name, ref.A, Ref::M * Ref::K * sizeof(float), cudaMemcpyHostToDevice)); \
	CUDA_CHECK(cudaMemcpy(d_Bref_ ## name, ref.B, Ref::K * Ref::N * sizeof(float), cudaMemcpyHostToDevice)); \
	float* d_Cref_INI_ ## name = new float[M * N](); \
	for (int i = 0; i < Ref::M; i++) { \
		for (int j = 0; j < Ref::N; j++) { \
			d_Cref_INI_ ## name[i * Ref::N + j] = 0; \
		} \
	} \
	CUDA_CHECK(cudaMemcpy(d_Cref_ ## name, d_Cref_INI_ ## name, Ref::M * Ref::N * sizeof(float), cudaMemcpyHostToDevice)); \
	name(d_Aref_ ## name, d_Bref_ ## name, d_Cref_ ## name, Ref::M, Ref::N, Ref::K); \
	cudaError_t err_c_ ## name = cudaGetLastError(); \
	if (err_c_ ## name != cudaSuccess) { \
		std::cerr << "CUDA Error: " << cudaGetErrorString(err_c_ ## name) << std::endl; \
	} \
	CUDA_CHECK(cudaMemcpy(refC, d_Cref_ ## name, Ref::M * Ref::N * sizeof(float), cudaMemcpyDeviceToHost)); \
	if (!ref.checkRef(refC)){ \
		std::cerr << "check ref failed!" << std::endl; \
	};

#define TIME(name) \
	float *d_A_ ## name, *d_B_ ## name, *d_C_ ## name; \
	CUDA_CHECK(cudaMalloc(&d_A_ ## name, M * K * sizeof(float))); \
	CUDA_CHECK(cudaMalloc(&d_B_ ## name, K * N * sizeof(float))); \
	CUDA_CHECK(cudaMalloc(&d_C_ ## name, M * N * sizeof(float))); \
	CUDA_CHECK(cudaMemcpy(d_A_ ## name, A, M * K * sizeof(float), cudaMemcpyHostToDevice)); \
	CUDA_CHECK(cudaMemcpy(d_B_ ## name, B, K * N * sizeof(float), cudaMemcpyHostToDevice)); \
	cudaEvent_t start_ ## name, end_ ## name; \
	cudaEventCreate(&start_ ## name); \
	cudaEventCreate(&end_ ## name); \
	float* d_C_INI_ ## name = new float[M * N](); \
	for (int i = 0; i < Ref::M; i++) { \
		for (int j = 0; j < Ref::N; j++) { \
			d_C_INI_ ## name[i * Ref::N + j] = 0; \
		} \
	} \
	for (int i = 0; i < 2; i++) \
	{ \
		CUDA_CHECK(cudaMemcpy(d_C_ ## name, d_C_INI_ ## name, Ref::M * Ref::N * sizeof(float), cudaMemcpyHostToDevice)); \
		name(d_A_ ## name, d_B_ ## name, d_C_ ## name, M, N, K); \
	} \
	cudaError_t err_t_ ## name = cudaGetLastError(); \
	if (err_t_ ## name != cudaSuccess) { \
		std::cerr << "CUDA Error: " << cudaGetErrorString(err_t_ ## name) << std::endl; \
	} \
	float milliseconds_ ## name = 0; \
	for (int i = 0; i < NUM_RUNS; i++) \
	{ \
		CUDA_CHECK(cudaMemcpy(d_C_ ## name, d_C_INI_ ## name, Ref::M * Ref::N * sizeof(float), cudaMemcpyHostToDevice)); \
		cudaDeviceSynchronize(); \
		cudaEventRecord(start_ ## name); \
		name(d_A_ ## name, d_B_ ## name, d_C_ ## name, M, N, K); \
		cudaEventRecord(end_ ## name); \
		cudaEventSynchronize(end_ ## name); \
		float milliseconds_ ## i = 0; \
		cudaEventElapsedTime(&milliseconds_ ## i, start_ ## name, end_ ## name); \
		milliseconds_ ## name += milliseconds_ ## i; \
	} \
	cudaMemcpy(C, d_C_ ## name, M * N * sizeof(float), cudaMemcpyDeviceToHost); \
	std::cout << "Time taken for GEMM (GPU, " << #name <<"): " << milliseconds_ ## name / (float)NUM_RUNS << "ms" << std::endl; \
	cudaFree(d_A_ ## name); \
	cudaFree(d_B_ ## name); \
	cudaFree(d_C_ ## name);

__global__ void gemm_gpu_o0_kernel(float* A, float* B, float *C, int M, int N, int K) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				for (int k = 0; k < K; k++) {
					C[i * N + j]  += A[i * K + k]  * B[k * N + j];
				}
			}
		}
    }
}

void gemm_gpu_o0(float* A, float* B, float* C, int M, int N, int K)
{
	// Init block and grid size
	dim3 blockSize(1);
	dim3 gridSize(1);
	gemm_gpu_o0_kernel<<<gridSize, blockSize>>>(A, B, C, M, N, K);
}

// =======================================================
// o1：简单并行化，每个线程负责计算 C 中的一个元素
// =======================================================
__global__ void gemm_gpu_o1_kernel(float* A, float* B, float* C, int M, int N, int K) {
    // 计算当前线程负责的输出矩阵 C 中的行和列
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        // 累加计算 C[row, col] 的值
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void gemm_gpu_o1(float* A, float* B, float* C, int M, int N, int K) {
    // 设置线程块尺寸为 16x16，保证足够的并行度
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (M + blockSize.y - 1) / blockSize.y);
    gemm_gpu_o1_kernel<<<gridSize, blockSize>>>(A, B, C, M, N, K);
}

// =======================================================
// o2：基于 tiling 的优化，利用共享内存提升数据复用
// =======================================================
__global__ void gemm_gpu_o2_kernel(float* A, float* B, float* C, int M, int N, int K) {
    // 定义 tile 尺寸，可根据硬件特性调优
    const int TILE_SIZE = 16;
    // 分配共享内存
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    // 计算全局索引
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;

    // 遍历所有 tile
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 将 A 的 tile 加载到共享内存（边界检查）
        if (row < M && (t * TILE_SIZE + threadIdx.x) < K)
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        
        // 将 B 的 tile 加载到共享内存（边界检查）
        if (col < N && (t * TILE_SIZE + threadIdx.y) < K)
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        // 对当前 tile 内进行乘累加
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // 将计算结果写回全局内存
    if (row < M && col < N)
        C[row * N + col] = sum;
}

void gemm_gpu_o2(float* A, float* B, float* C, int M, int N, int K) {
    const int TILE_SIZE = 16;
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE,
                  (M + TILE_SIZE - 1) / TILE_SIZE);
    gemm_gpu_o2_kernel<<<gridSize, blockSize>>>(A, B, C, M, N, K);
}

// =======================================================
// o3：进一步调优版本，在 tiling 基础上尝试较大 tile 尺寸并利用 unroll 提升性能
// =======================================================
__global__ void gemm_gpu_o3_kernel(float* A, float* B, float* C, int M, int N, int K) {
    // 这里选用更大的 tile 尺寸，视 GPU 资源而定
    const int TILE_SIZE = 32;
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 加载 A tile
        if (row < M && (t * TILE_SIZE + threadIdx.x) < K)
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        
        // 加载 B tile
        if (col < N && (t * TILE_SIZE + threadIdx.y) < K)
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        // 使用 unroll 优化内层循环
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N)
        C[row * N + col] = sum;
}

void gemm_gpu_o3(float* A, float* B, float* C, int M, int N, int K) {
    const int TILE_SIZE = 32;
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE,
                  (M + TILE_SIZE - 1) / TILE_SIZE);
    gemm_gpu_o3_kernel<<<gridSize, blockSize>>>(A, B, C, M, N, K);
}

int main(int argc, char* argv[]) {
	if (argc < 3) {
		std::cout << "Usage: mp1 <M> <N> <K>" << std::endl;
		return 1;
	}

	int M = atoi(argv[1]);
	int N = atoi(argv[2]);
	int K = atoi(argv[3]);

	// int runs = atoi(argv[3]);
	float* A = new float[M * K]();
	float* B = new float[K * N]();
	float* C = new float[M * N]();

	fillRandom(A, M * K);
	fillRandom(B, K * N);

	/// GPU Implementation
        // Check if implementation is correct
	auto ref = Ref();
	float* refC = new float[Ref::M * Ref::N]();
 	CHECK(gemm_gpu_o0)
	CHECK(gemm_gpu_o1)
	CHECK(gemm_gpu_o2)
	CHECK(gemm_gpu_o3)

	// Actual run
 	TIME(gemm_gpu_o0)
	TIME(gemm_gpu_o1)
	TIME(gemm_gpu_o2)
	TIME(gemm_gpu_o3)

	cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFreeHost(C);

	delete[] A;
	delete[] B;
	delete[] C;

	return 0;
}
