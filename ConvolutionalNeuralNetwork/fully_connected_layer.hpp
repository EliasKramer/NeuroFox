#pragma once
#include "matrix.hpp"
#include "math_functions.hpp"
#include "layer.hpp"

class fully_connected_layer : public layer {
private:
	matrix weights;
	matrix biases;
	
	matrix weight_deltas;
	matrix bias_deltas;

	activation activation_fn;
public:
	//constructor
	fully_connected_layer(
		int number_of_neurons,
		matrix* given_input,
		activation activation_function
	);

	void forward_propagation() override;
	void back_propagation() override;
};