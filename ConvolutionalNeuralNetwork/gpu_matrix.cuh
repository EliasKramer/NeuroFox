#pragma once
#include "matrix.hpp"

class gpu_matrix
{
private:
	float* gpu_ptr = nullptr;
	size_t width = 0;
	size_t height = 0;
	size_t depth = 0;

	bool owns_gpu_mem_ptr;

	void if_not_initialized_throw() const;

	void check_for_valid_format() const;
	void check_for_last_cuda_error() const;
	void free_if_owned();

	int get_idx(int x, int y, int z) const;
public:
	gpu_matrix();
	gpu_matrix(
		size_t width,
		size_t height,
		size_t depth);

	gpu_matrix(const matrix& m, bool copy_values);

	gpu_matrix(
		float* gpu_memory,
		size_t width,
		size_t height,
		size_t depth);

	~gpu_matrix();
	
	gpu_matrix& operator=(const gpu_matrix& other);
	void set_gpu_ptr_as_source(float* new_ptr);

	static bool same_format(const gpu_matrix& m1, const gpu_matrix& m2);

	std::unique_ptr<gpu_matrix> clone() const;
	
	void set_values_gpu(const gpu_matrix& m);
	void set_values_from_cpu(const matrix& m);
	void set_all(float value);
	void set_at(size_t width, size_t height, size_t depth, float value);

	const float* get_gpu_memory_readonly() const;
	float* get_gpu_memory();

	float* get_gpu_ptr_layer(size_t depth_idx);
	float* get_gpu_ptr_row(size_t height_idx, size_t depth_idx);
	float* get_gpu_ptr_item(size_t width_idx, size_t height_idx, size_t depth_idx);

	std::unique_ptr<matrix> to_cpu() const;

	size_t get_width() const;
	size_t get_height() const;
	size_t get_depth() const;

	size_t item_count() const;
};