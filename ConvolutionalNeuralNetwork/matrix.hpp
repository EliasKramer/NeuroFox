#pragma once
#include <vector>
#include <iostream>
#include <string>
#include "util.hpp"
#include "math_functions.hpp"

class matrix {
private:
	size_t width;
	size_t height;
	size_t depth;
	
	bool owning_data;
	float* data;

	size_t get_idx(size_t x, size_t y, size_t z) const;
	
	void if_not_initialized_throw() const;

	void check_for_valid_format() const;
	void allocate_mem();
	void set_own_data_from(const float* src);
	void set_own_data_from(const matrix& src);
	void delete_data_if_owning();
public:
	matrix();
	matrix(
		size_t width, 
		size_t height, 
		size_t depth);
	matrix(
		size_t width, 
		size_t height, 
		size_t depth,
		float* given_ptr, 
		bool copy);
	matrix(
		size_t width, 
		size_t height, 
		size_t depth,
		const std::vector<float>& given_vector);
	matrix(const matrix& source);

	matrix& operator=(const matrix& other);
	
	~matrix();	

	//deletes the old data and allocates new memory
	void initialize_format(size_t width, size_t height, size_t depth);
	//deletes the old data and allocates new memory
	//does not copy values
	void initialize_format(const matrix& source);

	void set_ptr_as_source(float* given_ptr);

	void set_all(float value);
	void apply_noise(float range);
	void mutate(float range);

	size_t get_width() const;
	size_t get_height() const;
	size_t get_depth() const;
	size_t item_count() const;

	float get_at_flat(size_t idx) const;
	void set_at_flat(size_t idx, float value);
	void add_at_flat(size_t idx, float value);

	float* get_data();
	const float* get_data_readonly() const;

	float* get_ptr_layer(size_t depth_idx);
	float* get_ptr_row(size_t height_idx, size_t depth_idx);
	float* get_ptr_item(size_t width_idx, size_t height_idx, size_t depth_idx);
	
	//setter
	void set_at(size_t x, size_t y, size_t z, float value);
	void add_at(size_t x, size_t y, size_t z, float value);
	//setting value where z = 0
	void set_at(size_t x, size_t y, float value);
	void add_at(size_t x, size_t y, float value);

	//getter
	float get_at(size_t x, size_t y, int z) const;
	//getting value where z = 0
	float get_at(size_t x, size_t y) const;

	//const matrix& rotate180copy() const;

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