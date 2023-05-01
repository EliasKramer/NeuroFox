#include "gpu_math.cuh"

#define THREADS_PER_BLOCK 1024

static unsigned int get_block_count(unsigned int size)
{
	//if we have 1024 elements, we need 1 block
	//if we have 1025 elements, we need 2 blocks
	//if we have 2048 elements, we need 2 blocks
	//and as long as it is under 1024 - 1 thread will still work
	return ((size - 1) / THREADS_PER_BLOCK) + 1;
}

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
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size)
	{
		result[index] = a[index] + b[index];
	}
}
cudaError_t gpu_add(
	const gpu_memory<float>& gpu_memory_a, 
	const gpu_memory<float>& gpu_memory_b, 
	gpu_memory<float>& gpu_memory_result)
{
	if (gpu_memory_a.count() == 0 ||
		gpu_memory_a.count() != gpu_memory_b.count() ||
		gpu_memory_a.count() != gpu_memory_result.count())
	{
		throw std::invalid_argument("gpu_add_matrices failed. size must be greater than 0");
	}

	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		return cudaStatus;
	}

	unsigned int size = gpu_memory_a.count();
	
	gpu_add_matrices_kernel <<< get_block_count(size), THREADS_PER_BLOCK >> > (
		gpu_memory_a.gpu_data_ptr(),
		gpu_memory_b.gpu_data_ptr(),
		gpu_memory_result.gpu_data_ptr(),
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

cudaError_t apply_activation_function(
	const gpu_memory<float>& gpu_memory,
	e_activation_t activation_function)
{
	//TODO
}