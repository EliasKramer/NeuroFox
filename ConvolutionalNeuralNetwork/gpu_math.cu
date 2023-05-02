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

__device__ int get_idx(int x, int y, int z, int height, int width)
{
	return x + y * width + z * width * height;
}

__global__ void gpu_dot_product_kernel(
	const float* weights,
	const float* input,
	const int input_size,
	float* activations,
	const int activations_size)
{
	unsigned int activation_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (activation_idx < activations_size)
	{
		float sum = 0;
		for (int i = 0; i < input_size; i++)
		{
			int weight_idx = get_idx(i, activation_idx, 0, 
									activations_size, input_size);
			sum += weights[weight_idx] * input[i];
		}
		activations[activation_idx] = sum;
	}
}

void gpu_dot_product(
	const gpu_memory<float>& gpu_weights,
	const gpu_memory<float>& gpu_input,
	gpu_memory<float>& gpu_activations)
{
	if (gpu_weights.item_count() == 0 ||
		gpu_input.item_count() == 0 ||
		gpu_activations.item_count() == 0)
	{
		throw std::invalid_argument("gpu_dot_product failed. size must be greater than 0");
	}
	if (gpu_activations.item_count() * gpu_input.item_count() != gpu_weights.item_count())
	{
		throw std::invalid_argument("gpu_dot_product failed. false format");
	}

	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		throw std::runtime_error("gpu_dot_product failed. cudaSetDevice failed");
	}

	unsigned int size = gpu_activations.item_count();
	unsigned int block_count = get_block_count(size);
	gpu_dot_product_kernel <<< block_count, THREADS_PER_BLOCK >>> (
		gpu_weights.gpu_data_ptr(),
		gpu_input.gpu_data_ptr(),
		gpu_input.item_count(),
		gpu_activations.gpu_data_ptr(),
		gpu_activations.item_count());

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		throw std::runtime_error("gpu_dot_product failed. gpu_dot_product_kernel failed");
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		throw std::runtime_error("gpu_dot_product failed. cudaDeviceSynchronize failed");
	}
}

__global__ void gpu_add_matrices_kernel(const float* a, const float* b, float* result, unsigned int size)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size)
	{
		result[index] = a[index] + b[index];
	}
}
void gpu_add(
	const gpu_memory<float>& gpu_memory_a, 
	const gpu_memory<float>& gpu_memory_b, 
	gpu_memory<float>& gpu_memory_result)
{
	if (gpu_memory_a.item_count() == 0 ||
		gpu_memory_a.item_count() != gpu_memory_b.item_count() ||
		gpu_memory_a.item_count() != gpu_memory_result.item_count())
	{
		throw std::invalid_argument("gpu_add_matrices failed. size must be greater than 0");
	}

	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		throw std::runtime_error("gpu_add_matrices failed. cudaSetDevice failed");
	}

	unsigned int size = gpu_memory_a.item_count();
	
	gpu_add_matrices_kernel <<< get_block_count(size), THREADS_PER_BLOCK >>> (
		gpu_memory_a.gpu_data_ptr(),
		gpu_memory_b.gpu_data_ptr(),
		gpu_memory_result.gpu_data_ptr(),
		size);
		
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		throw std::runtime_error("gpu_add_matrices failed. kernel launch failed");
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		throw std::runtime_error("gpu_add_matrices failed. cudaDeviceSynchronize failed");
	}
}

void gpu_apply_activation_function(
	const gpu_memory<float>& gpu_memory,
	e_activation_t activation_function)
{
	//TODO
}