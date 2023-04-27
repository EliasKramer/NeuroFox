#include "gpu_matrix.hpp"

float* copy_to_gpu(const matrix& m)
{
	float* gpu_matrix = nullptr;
	cudaError_t cudaStatus = cudaMalloc((void**)&gpu_matrix, m.flat_readonly().size() * sizeof(float));

	if (cudaStatus != cudaSuccess)
	{
		throw std::runtime_error("copying values to gpu failed. cudaMalloc failed");
		return nullptr;
	}

	cudaStatus = cudaMemcpy(
		gpu_matrix,
		m.flat_readonly().data(),
		m.flat_readonly().size() * sizeof(float),
		cudaMemcpyHostToDevice);

	if (cudaStatus != cudaSuccess)
	{
		throw std::runtime_error("copying values to gpu failed. cudaMemcpy failed");
		return nullptr;
	}

	return gpu_matrix;
}

__global__ void gpu_add_matrices_kernel(const float* a, const float* b, float* result, unsigned int size)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size)
	{
		result[index] = a[index] + b[index];
	}
}

cudaError_t gpu_add_matrices(
	const float* gpu_matrix_a,
	const float* gpu_matrix_b,
	float* gpu_matrix_result,
	unsigned int size)
{
	if (size == 0)
	{
		throw std::invalid_argument("gpu_add_matrices failed. size must be greater than 0");
	}

	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		return cudaStatus;
	}

	//every block has 1024 threads
	unsigned int threads_per_block = 1024;
	//if we have 1024 elements, we need 1 block
	//if we have 1025 elements, we need 2 blocks
	//if we have 2048 elements, we need 2 blocks
	//and as long as it is under 1024 - 1 thread will still work
	unsigned int blocks = ((size - 1) / threads_per_block) + 1;

	gpu_add_matrices_kernel << <blocks, threads_per_block >> > (
		gpu_matrix_a,
		gpu_matrix_b,
		gpu_matrix_result,
		size);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		return cudaStatus;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		return cudaStatus;
	}

	return cudaStatus;
}
