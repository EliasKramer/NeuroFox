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
	layer_type layer_type_;

	matrix* input;
	matrix activations;
	matrix* error_right;
	matrix error;

public:
	layer(matrix* input);

	const layer_type get_layer_type() const;
	const matrix* get_input_p() const;

	void set_input(matrix* input);
	void set_error_right(matrix* error_right);

	const matrix& get_activations() const;

	virtual void forward_propagation() = 0;
	virtual void back_propagation() = 0;
};