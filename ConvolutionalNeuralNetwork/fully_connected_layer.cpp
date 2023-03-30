#include "fully_connected_layer.hpp"

fully_connected_layer create_fully_connected_layer(
	int number_of_neurons,
	matrix* input,
	activation activation
)
{
	if (input->width != 1 || input->depth != 1)
	{
		throw "Input matrix must be a vector (width and depth must be 1)";
	}

	fully_connected_layer layer;

	//make an output matrix (1 Dimensional)
	resize_matrix(layer.output, 1, number_of_neurons, 1);
	//set the input pointer to the given input
	layer.input = input;

	//the error is needed for every output (we need it for learning)
	resize_matrix(layer.error,	1, number_of_neurons, 1);
	//the right error, are the error values on the right side of the layer
	layer.error_right = nullptr;

	//weights and biases are the parameter values
	resize_matrix(layer.biases,			1, number_of_neurons, 1);
	resize_matrix(layer.weights,		layer.input->height, number_of_neurons, 1);
	
	//when training the network, we need to know how much the weights and biases need to change
	resize_matrix(layer.bias_deltas,	1, number_of_neurons, 1);
	resize_matrix(layer.weight_deltas,	layer.input->height, number_of_neurons, 1);

	//the activation function for making the output not linear
	layer.activation_fn = activation;

	return layer;
}

void feed_forward(fully_connected_layer& layer)
{
	matrix* input = layer.input;
	matrix* output = &layer.output;
	matrix* weights = &layer.weights;
	matrix* biases = &layer.biases;

	//TODO - straighten out the input matrix
	matrix_dot(*weights, *input, *output);
	matrix_add(*output, *biases, *output);
	matrix_apply_activation(*output, layer.activation_fn);
}

void learn(fully_connected_layer& layer)
{

}