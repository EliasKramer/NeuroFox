#pragma once
#include "matrix.hpp"

typedef enum _layer_type {
	convolution,
	pooling,
	fully_connected
} layer_type;

typedef struct _layer {
	layer_type type;
	matrix* input;
	matrix output;
	matrix* error_right;
	matrix error;
} layerd;