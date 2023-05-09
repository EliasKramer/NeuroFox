#pragma once

#include "gpu_memory.cuh"
#include "matrix.hpp"

class gpu_matrix 
{
private:
	gpu_memory<float>* gpu_mem_ptr = nullptr;
	size_t width = 0;
	size_t height = 0;
	size_t depth = 0;
	
	bool owns_gpu_mem_ptr;

	void check_for_valid_args();
	void free_owned_gpu_mem();
public:
	gpu_matrix(
		size_t width, 
		size_t height, 
		size_t depth);

	gpu_matrix(const matrix& m, bool copy_values);

	gpu_matrix(
		gpu_memory<float>* gpu_memory, 
		size_t width, 
		size_t height, 
		size_t depth);

	~gpu_matrix();

	const gpu_memory<float>& get_gpu_memory_readonly() const;
	gpu_memory<float>& get_gpu_memory();

	size_t get_width() const;
	size_t get_height() const;
	size_t get_depth() const;

	size_t item_count() const;
};