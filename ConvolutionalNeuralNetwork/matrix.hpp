#pragma once
#include <vector>
#include <iostream>
#include <string>
#include "util.hpp"
#include "math_functions.hpp"

struct _matrix {
	int width;
	int height;
	int depth;
	std::vector<float> data;
} typedef matrix;

matrix* create_matrix(int width, int height, int depth);
matrix get_matrix(int width, int height, int depth);

size_t matrix_hash(const matrix& m);

void resize_matrix(matrix& m, int width, int height, int depth);
void resize_matrix(matrix& resizing_matrix, const matrix& source);

bool matrix_equal_format(const matrix& a, const matrix& b);

void set_all(matrix& m, float value);
void matrix_apply_noise(matrix& m, float range);
void matrix_mutate(matrix& m, float range);

std::vector<float>& matrix_flat(matrix& m);
const std::vector<float>& matrix_flat_readonly(const matrix& m);

//setter
void set_at(matrix& m, int x, int y, int z, float value);
//setting value where z = 0
void set_at(matrix& m, int x, int y, int value);

//getter
float matrix_get_at(const matrix& m, int x, int y, int z);
//getting value where z = 0
float matrix_get_at(const matrix& m, int x, int y);

void matrix_dot(const matrix& a, const matrix& b, matrix& result);
void matrix_dot_flat(const matrix& a, const matrix& flat, matrix& result_flat);

void matrix_add(const matrix& a, const matrix& b, matrix& result);
void matrix_add_flat(const matrix& a, const matrix& b, matrix& result);

void matrix_subtract(const matrix& a, const matrix& b, matrix& result);
void matrix_multiply(matrix& a, float b);

void matrix_apply_activation(matrix& m, e_activation_t activation_fn);

bool are_equal(const matrix& a, const matrix& b);

std::string get_matrix_string(const matrix& m);