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
	//the current matrix of neurons
	matrix activations;

	//the format, that the previous layer has
	matrix input_format;
	//the pointer to the neurons of the previous layer
	const matrix* input = nullptr;

	//the error has the same format as our neurons
	matrix error;
	//the passing error has the same format 
	//as the neurons of the previous layer
	matrix* passing_error = nullptr;

public:
	layer(e_layer_type_t given_layer_type);

	const e_layer_type_t get_layer_type() const;
	const matrix* get_input_p() const;

	void set_input(const matrix* input);
	virtual void set_input_format(const matrix& input_format);
	
	void set_previous_layer(layer& previous_layer);

	const matrix& get_activations() const;
	matrix* get_activations_p();

	void set_error_for_last_layer(const matrix& expected);

	//set all weights and biases to that value
	virtual void set_all_parameter(float value) = 0;
	//a random value to the current weights and biases between -value and value
	virtual void apply_noise(float range) = 0;
	//add a random value between range and -range to one weight or bias 
	virtual void mutate(float range) = 0;

	virtual void forward_propagation() = 0;
	virtual void back_propagation() = 0;

	//the deltas got calculated in the backprop function
	//all the deltas got summed up. now we need to apply the
	//average. this is done by dividing the deltas by the number of inputs
	virtual void apply_deltas(int number_of_inputs) = 0;
};