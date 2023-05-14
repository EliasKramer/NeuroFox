#include "gpu_matrix.cuh"

void gpu_matrix::check_for_valid_args() const
{
	if (width == 0 || height == 0 || depth == 0)
	{
		throw std::invalid_argument("could not create gpu_matrix");
	}
}

void gpu_matrix::check_for_last_cuda_error() const
{
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		throw std::runtime_error(cudaGetErrorString(err));
	}
}

void gpu_matrix::free_owned_gpu_mem()
{
	if (owns_gpu_mem_ptr)
	{
		cudaFree(gpu_ptr);
		check_for_last_cuda_error();
	}
	owns_gpu_mem_ptr = false;
}

int gpu_matrix::get_idx(int x, int y, int z) const
{
	return x + y * width + z * width * height;
}

gpu_matrix::gpu_matrix(
	size_t width,
	size_t height,
	size_t depth
) :
	width(width),
	height(height),
	depth(depth),
	owns_gpu_mem_ptr(true)
{
	check_for_valid_args();
	cudaMalloc(&gpu_ptr, item_count() * sizeof(float));
	check_for_last_cuda_error();
}

gpu_matrix::gpu_matrix(
	float* given_gpu_ptr,
	size_t width,
	size_t height,
	size_t depth
) :
	gpu_ptr(given_gpu_ptr),
	width(width),
	height(height),
	depth(depth),
	owns_gpu_mem_ptr(false)
{
	check_for_valid_args();
	//TODO if there is a way to check how much is allocated on the given ptr, 
	//then check if that matches the given height, width and depth
}

gpu_matrix::gpu_matrix(const matrix& m, bool copy_values)
	:gpu_matrix(m.get_width(), m.get_height(), m.get_depth())
{
	if (copy_values)
	{
		set_values_cpu(m);
	}
}

gpu_matrix::~gpu_matrix()
{
	free_owned_gpu_mem();
}

const float* gpu_matrix::get_gpu_memory_readonly() const
{
	return gpu_ptr;
}

float* gpu_matrix::get_gpu_memory()
{
	return gpu_ptr;
}

size_t gpu_matrix::get_width() const
{
	return width;
}

size_t gpu_matrix::get_height() const
{
	return height;
}

size_t gpu_matrix::get_depth() const
{
	return depth;
}

void gpu_matrix::set_values_gpu(const gpu_matrix& m)
{
	if (!same_format(*this, m))
	{
		throw std::runtime_error("gpu_memory size mismatch");
	}

	cudaMemcpy(
		gpu_ptr,
		m.get_gpu_memory_readonly(),
		item_count() * sizeof(float),
		cudaMemcpyDeviceToDevice);

	check_for_last_cuda_error();
}

void gpu_matrix::set_values_cpu(const matrix& m)
{
	if (m.item_count() != item_count())
	{
		throw std::runtime_error("gpu_memory size mismatch");
	}

	cudaMemcpy(
		gpu_ptr,
		m.get_data_readonly(),
		item_count() * sizeof(float),
		cudaMemcpyHostToDevice);

	check_for_last_cuda_error();
}

std::unique_ptr<gpu_matrix> gpu_matrix::clone() const
{
	//this allocates new memory
	std::unique_ptr<gpu_matrix> clone = 
		std::make_unique<gpu_matrix>(width, height, depth);
	
	//this sets the values
	clone.get()->set_values_gpu(*this);

	return std::move(clone);
}

bool gpu_matrix::same_format(const gpu_matrix& m1, const gpu_matrix& m2)
{
	return m1.get_width() == m2.get_width() &&
		m1.get_height() == m2.get_height() &&
		m1.get_depth() == m2.get_depth();
}

void gpu_matrix::set_all(float value)
{
	std::vector<float> values(item_count(), value);
	cudaMemcpy(
		gpu_ptr,
		values.data(),
		item_count() * sizeof(float),
		cudaMemcpyHostToDevice);
	check_for_last_cuda_error();
}

void gpu_matrix::set_at(size_t width, size_t height, size_t depth, float value)
{
	if (width >= this->width || height >= this->height || depth >= this->depth)
	{
		throw std::invalid_argument("index out of bounds");
	}

	cudaMemcpy(
		gpu_ptr + get_idx(width, height, depth),
		&value,
		sizeof(float),
		cudaMemcpyHostToDevice);
	check_for_last_cuda_error();
}


std::unique_ptr<matrix> gpu_matrix::to_cpu() const
{
	std::unique_ptr<matrix> result = std::make_unique<matrix>(width, height, depth);
	cudaMemcpy(
		result->get_data(),
		gpu_ptr,
		item_count() * sizeof(float),
		cudaMemcpyDeviceToHost);
	check_for_last_cuda_error();
	return result;
}

float* gpu_matrix::get_gpu_ptr_layer(size_t depth_idx)
{
	return sub_ptr<float>(gpu_ptr, width * height, depth_idx);
}

float* gpu_matrix::get_gpu_ptr_row(size_t height_idx, size_t depth_idx)
{
	return get_gpu_ptr_layer(depth_idx) + height_idx * width;
}

float* gpu_matrix::get_gpu_ptr_item(size_t width_idx, size_t height_idx, size_t depth_idx)
{
	return get_gpu_ptr_row(height_idx, depth_idx) + width_idx;
}

size_t gpu_matrix::item_count() const
{
	return width * height * depth;
}