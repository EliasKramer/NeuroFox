#include "gpu_matrix.cuh"

void gpu_matrix::check_for_valid_args()
{
	if (gpu_mem_ptr == nullptr ||
		width == 0 || height == 0 || depth == 0 ||
		gpu_mem_ptr->item_count() != width * height * depth)
	{
		throw std::invalid_argument("could not create gpu_matrix");
	}
}

gpu_matrix::gpu_matrix(
	size_t width,
	size_t height,
	size_t depth
	):
	width(width),
	height(height),
	depth(depth),
	owns_gpu_mem_ptr(true)
{
	gpu_mem_ptr = new gpu_memory<float>(width * height * depth);
	check_for_valid_args();
}

void gpu_matrix::free_owned_gpu_mem()
{
	if (owns_gpu_mem_ptr)
	{
		delete gpu_mem_ptr;
	}
	owns_gpu_mem_ptr = false;
}

gpu_matrix::gpu_matrix(
	gpu_memory<float>* given_gpu_ptr,
	size_t width,
	size_t height,
	size_t depth
	):
	gpu_mem_ptr(given_gpu_ptr),
	width(width),
	height(height),
	depth(depth),
	owns_gpu_mem_ptr(false)
{
	check_for_valid_args();
}

gpu_matrix::gpu_matrix(const matrix& m, bool copy_values)
	:gpu_matrix(m.get_width(), m.get_height(), m.get_depth())
{
	if (copy_values)
	{
		gpu_mem_ptr->set_values_from_matrix(m);
	}
}

gpu_matrix::~gpu_matrix()
{
	free_owned_gpu_mem();
}

const gpu_memory<float>& gpu_matrix::get_gpu_memory_readonly() const
{
	return *gpu_mem_ptr;
}

gpu_memory<float>& gpu_matrix::get_gpu_memory()
{
	return *gpu_mem_ptr;
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

size_t gpu_matrix::item_count() const
{
	return width * height * depth;
}
