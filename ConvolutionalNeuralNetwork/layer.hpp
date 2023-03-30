#pragma once
#include "matrix.hpp"

typedef enum _layer_type {
	convolution,
	pooling,
	fully_connected,
	NO_TYPE
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

	void set_input(matrix* input);
	void set_error_right(matrix* error_right);

	virtual void forward_propagation() = 0;
	virtual void back_propagation() = 0;
};