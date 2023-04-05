#include "fully_connected_layer.hpp"

fully_connected_layer::fully_connected_layer(
	matrix* given_input,
	const matrix& input_format,
	int number_of_neurons,
	e_activation_t activation_function
)
	:fully_connected_layer(
		given_input, 
		input_format,
		get_matrix(1, number_of_neurons, 1), 
		activation_function)
{}

fully_connected_layer::fully_connected_layer(
	matrix* given_input,
	const matrix& input_format,
	const matrix& activation_format,
	e_activation_t activation_function
)
	:layer(given_input, e_layer_type_t::fully_connected)
{
	//make an output matrix (1 Dimensional)
	resize_matrix(activations, activation_format);

	//the error is needed for every output (we need it for learning)
	resize_matrix(error, activation_format);
	//the right error, are the error values on the right side of the layer
	error_right = nullptr;

	//weights and biases are the parameter values
	resize_matrix(biases, activation_format);
	resize_matrix(weights, input_format.data.size(), activations.data.size(), 1);

	//when training the network, we need to know how much the weights and biases need to change
	resize_matrix(bias_deltas, activation_format);
	resize_matrix(weight_deltas, input_format.data.size(), activations.data.size(), 1);

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
	matrix_dot_flat(weights, *input, activations);
	matrix_add_flat(activations, biases, activations);
	matrix_apply_activation(activations, activation_fn);
}

void fully_connected_layer::back_propagation()
{
	//TODO
}