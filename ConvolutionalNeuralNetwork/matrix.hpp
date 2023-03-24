#pragma once
#include <vector>
#include <iostream>
#include <string>

struct _matrix {
	int width;
	int height;
	int depth;
	std::vector<float> data;
} typedef matrix;

matrix* create_matrix(int width, int height, int depth);

void set_all(matrix& m, float value);

std::string get_matrix_string(const matrix& m);

//setter
void set_at(matrix& m, int x, int y, int z, float value);
//setting value where z = 0
void set_at(matrix& m, int x, int y, int value);

//getter
float get_at(const matrix& m, int x, int y, int z);
//getting value where z = 0
float get_at(const matrix& m, int x, int z);

void dot(const matrix& a, const matrix& b, matrix& result);

bool are_equal(const matrix& a, const matrix& b);