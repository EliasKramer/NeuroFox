#pragma once
#include <vector>
#include <iostream>

struct _matrix {
	int width;
	int height;
	int depth;
	std::vector<float> data;
} typedef matrix;

matrix* create_matrix(int width, int height, int depth);

void print_matrix(matrix& m);

void dot(matrix& a, matrix& b, matrix& result);
