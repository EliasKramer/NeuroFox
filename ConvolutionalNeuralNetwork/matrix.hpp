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
	//matrix(int width, int height);
	//matrix(int height);
	size_t matrix_hash() const;

	void resize_matrix(int width, int height, int depth);
	void resize_matrix(const matrix& source);

	void set_all(float value);
	void matrix_apply_noise(float range);
	void matrix_mutate(float range);

	int get_width() const;
	int get_height() const;
	int get_depth() const;

	std::vector<float>& matrix_flat();
	const std::vector<float>& matrix_flat_readonly() const;

	//setter
	void set_at(int x, int y, int z, float value);
	//setting value where z = 0
	void set_at(int x, int y, float value);

	//getter
	float matrix_get_at(int x, int y, int z) const;
	//getting value where z = 0
	float matrix_get_at(int x, int y) const;

	static void matrix_dot(const matrix& a, const matrix& b, matrix& result);
	static void matrix_dot_flat(const matrix& a, const matrix& flat, matrix& result_flat);

	static void matrix_add(const matrix& a, const matrix& b, matrix& result);
	static void matrix_add_flat(const matrix& a, const matrix& b, matrix& result);

	static void matrix_subtract(const matrix& a, const matrix& b, matrix& result);

	static bool are_equal(const matrix& a, const matrix& b);
	static bool matrix_equal_format(const matrix& a, const matrix& b);
	
	void matrix_multiply(float a);
	void matrix_apply_activation(e_activation_t activation_fn);

	std::string get_matrix_string() const;
};