#pragma once
#include "matrix.hpp"

typedef enum _layer_type {
	convolution,
	pooling,
	fully_connected,
	NO_TYPE
} e_layer_type_t;

class layer {

protected:
	e_layer_type_t type;

	matrix* input;
	matrix activations;
	matrix* error_right;
	matrix error;

public:
	layer(matrix* input, e_layer_type_t given_layer_type);

	const e_layer_type_t get_layer_type() const;
	const matrix* get_input_p() const;

	void set_input(matrix* input);
	void set_error_right(matrix* error_right);

	const matrix& get_activations() const;
	matrix* get_activations_p();

	//set all weights and biases to that value
	virtual void set_all_parameter(float value) = 0;
	//a random value to the current weights and biases between -value and value
	virtual void apply_noise(float range) = 0;
	//add a random value between range and -range to one weight or bias 
	virtual void mutate(float range) = 0;

	virtual void forward_propagation() = 0;
	virtual void back_propagation() = 0;
};