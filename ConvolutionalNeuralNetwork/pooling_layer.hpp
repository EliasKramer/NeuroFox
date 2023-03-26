#pragma once
#include "matrix.hpp"

enum _pooling_type
{
	max,
	min,
	average
} typedef pooling_type;

struct _pooling_layer {
	matrix* input;
	int kernel_width;
	int kernel_height;
	int stride;
	pooling_type type;
	matrix output;
} typedef pooling_layer;

pooling_layer* create_pooling_layer(int size, int stride, pooling_type type);
