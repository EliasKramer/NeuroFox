#include "fully_connected_layer.hpp"

fully_connected_layer::fully_connected_layer(
	int number_of_neurons,
	matrix* given_input,
	activation activation_function
)
	:layer(given_input, layer_type::fully_connected)
{
	if (given_input->width != 1 || given_input->depth != 1)
	{
		throw "Input matrix must be a vector (width and depth must be 1)";
	}

	//make an output matrix (1 Dimensional)
	resize_matrix(activations, 1, number_of_neurons, 1);

	//the error is needed for every output (we need it for learning)
	resize_matrix(error, 1, number_of_neurons, 1);
	//the right error, are the error values on the right side of the layer
	error_right = nullptr;

	//weights and biases are the parameter values
	resize_matrix(biases, 1, number_of_neurons, 1);
	resize_matrix(weights, input->height, number_of_neurons, 1);

	//when training the network, we need to know how much the weights and biases need to change
	resize_matrix(bias_deltas, 1, number_of_neurons, 1);
	resize_matrix(weight_deltas, input->height, number_of_neurons, 1);

	//the activation function for making the output not linear
	activation_fn = activation_function;
}

const matrix& fully_connected_layer::get_weights() const
{
	return weights;
}

const matrix& fully_connected_layer::get_biases() const
{
	return biases;
}

matrix& fully_connected_layer::get_weights_ref()
{
	return weights;
}

matrix& fully_connected_layer::get_biases_ref()
{
	return biases;
}

void fully_connected_layer::forward_propagation()
{
	//TODO - straighten out the input matrix
	matrix_dot(weights, *input, activations);
	matrix_add(activations, biases, activations);
	matrix_apply_activation(activations, activation_fn);
}

void fully_connected_layer::back_propagation()
{
	//TODO
}