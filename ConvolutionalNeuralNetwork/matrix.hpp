#pragma once
#include <vector>
#include <iostream>
#include <string>
#include "util.hpp"
#include "math_functions.hpp"

class matrix {
private:
	int width;
	int height;
	int depth;
	std::vector<float> data;

	int get_idx(int x, int y, int z) const;

public:
	matrix();
	matrix(int width, int height, int depth);
	size_t get_hash() const;

	void resize(int width, int height, int depth);
	void resize(const matrix& source);

	void set_all(float value);
	void apply_noise(float range);
	void mutate(float range);

	int get_width() const;
	int get_height() const;
	int get_depth() const;

	std::vector<float>& flat();
	const std::vector<float>& flat_readonly() const;

	//setter
	void set_at(int x, int y, int z, float value);
	void add_at(int x, int y, int z, float value);
	//setting value where z = 0
	void set_at(int x, int y, float value);
	void add_at(int x, int y, float value);

	//getter
	float get_at(int x, int y, int z) const;
	//getting value where z = 0
	float get_at(int x, int y) const;

	const matrix& rotate180copy() const;

	static void dot_product(const matrix& a, const matrix& b, matrix& result);
	static void dot_product_flat(const matrix& a, const matrix& flat, matrix& result_flat);

	static void add(const matrix& a, const matrix& b, matrix& result);
	static void add_flat(const matrix& a, const matrix& b, matrix& result);

	static void subtract(const matrix& a, const matrix& b, matrix& result);

	static bool are_equal(const matrix& a, const matrix& b);
	static bool equal_format(const matrix& a, const matrix& b);

	static void valid_cross_correlation(const matrix& input, const matrix& kernel, matrix& output);
	static void valid_convolution(const matrix& input, const matrix& kernel, matrix& output);
	static void full_cross_correlation(const matrix& input, const matrix& kernel, matrix& output);

	void scalar_multiplication(float a);
	void apply_activation_function(e_activation_t activation_fn);

	std::string get_string() const;
};