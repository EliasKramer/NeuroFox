#pragma once
#include <vector>
#include <iostream>
#include <string>
#include "util.hpp"
#include "math_functions.hpp"
#include "vector3.hpp"
//include cuda
#include <cuda_runtime.h>

class matrix {
private:
	vector3 format;
	
	bool owning_data;
	
	float* host_data;
	float* device_data;

	bool is_initialized() const;
	void if_not_initialized_throw() const;
	void if_not_owning_throw() const;

	bool is_device_mem_allocated() const;
	void if_gpu_not_allocated_throw() const;
	void allocate_device_mem();
	void copy_host_to_device();
	void copy_device_to_host();

	void if_cuda_error_throw() const;

	void check_for_valid_format() const;
	void allocate_host_mem();
	void set_own_host_data_from(const std::vector<float> src);
	void set_own_host_data_from(const matrix& src);
	void delete_data_if_owning();

	//given_ptr must be either host or device pointer
	float* get_ptr_layer(float* given_ptr, size_t depth_idx);
	float* get_ptr_row(float* given_ptr, size_t height_idx, size_t depth_idx);
	float* get_ptr_item(float* given_ptr, size_t width_idx, size_t height_idx, size_t depth_idx);
public:
	matrix();
	matrix(vector3 given_format);
	matrix(
		vector3 given_format,
		const std::vector<float>& given_vector);
	matrix(const matrix& source);

	matrix& operator=(const matrix& other);
	
	~matrix();	

	void enable_gpu();
	/*
	//deletes the old data and allocates new memory
	void initialize_format(vector3 new_format);
	//deletes the old data and allocates new memory
	//does not copy values
	void initialize_format(const matrix& source);
	*/

	//void set_ptr_as_source(float* given_ptr);

	void set_all(float value);
	void apply_noise(float range);
	void mutate(float range);

	vector3 get_format() const;
	size_t get_width() const;
	size_t get_height() const;
	size_t get_depth() const;
	size_t item_count() const;

	float get_at_flat(size_t idx) const;
	void set_at_flat(size_t idx, float value);
	void add_at_flat(size_t idx, float value);

	/*
	float* get_data();
	const float* get_data_readonly() const;
	*/
	float* get_device_ptr();
	const float* get_device_ptr_readonly() const;
	float* get_device_ptr_layer(size_t depth_idx);
	
	//the current matrix gets the data from a different matrix row
	//the current matrix must have the same amount of elements as this row
	//the current matrix will not own the data of the other matrix
	void observe_row(matrix& m, size_t row_idx);
	//the current matrix gets the data from a different matrix row at a certain element in this row
	//the current matrix must have the same amount of elements as this row from the given item index on
	//the current matrix will not own the data of the other matrix
	void observe_row(matrix& m, size_t row_idx, size_t item_idx);

	void set_row_from_matrix(const matrix& m, size_t row_idx);
	void set_row_from_matrix(const matrix& m, size_t row_idx, size_t item_idx);

	//setter
	void set_at(vector3 position, float value);
	void add_at(vector3 position, float value);

	//getter
	float get_at(vector3 pos) const;
	
	//TODO - implement for gpu
	static void dot_product(const matrix& a, const matrix& b, matrix& result);
	static void dot_product_flat(const matrix& a, const matrix& flat, matrix& result_flat);

	static void add(const matrix& a, const matrix& b, matrix& result);
	static void add_flat(const matrix& a, const matrix& b, matrix& result);

	static void subtract(const matrix& a, const matrix& b, matrix& result);

	static bool are_equal(const matrix& a, const matrix& b);
	static bool are_equal(const matrix& a, const matrix& b, float tolerance);
	static bool equal_format(const matrix& a, const matrix& b);

	static void valid_cross_correlation(
		const matrix& input,
		const std::vector<matrix>& kernels, 
		matrix& output, 
		int stride);
	//static void valid_convolution(const matrix& input, const matrix& kernel, matrix& output);
	//static void full_cross_correlation(const matrix& input, const matrix& kernel, matrix& output, int stride);

	void scalar_multiplication(float a);
	void apply_activation_function(e_activation_t activation_fn);

	std::string get_string() const;
};