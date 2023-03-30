#pragma once
#include "matrix.hpp"

typedef enum _layer_type {
	convolution,
	pooling,
	fully_connected
} layer_type;

class layer {

protected:
	matrix* input;
	matrix output;
	matrix* error_right;
	matrix error;

public:
	layer(matrix* input);

	layer_type layer_type;

	virtual void set_input(matrix* input) = 0;
	virtual void set_error_right(matrix* error_right) = 0;

	virtual void forward_propagation() = 0;
	virtual void back_propagation() = 0;
};