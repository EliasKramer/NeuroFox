#include "fully_connected_layer.hpp"

fully_connected_layer* create_fully_connected_layer(
	int number_of_neurons,
	matrix* input,
	activation activation
)
{
	if (input->width != 1 || input->depth != 1)
	{
		throw "Input matrix must be a vector (width and depth must be 1)";
		return nullptr;
	}

	fully_connected_layer* layer = new fully_connected_layer;

	layer->input = input;

	resize_matrix(layer->output, 1, number_of_neurons, 1);

	resize_matrix(layer->biases, 1, number_of_neurons, 1);
	resize_matrix(layer->weights, layer->input->height, number_of_neurons, 1);

	layer->activation_fn = activation;

	return layer;
}

void feed_forward(fully_connected_layer& layer)
{
	matrix* input = layer.input;
	matrix* output = &layer.output;
	matrix* weights = &layer.weights;
	matrix* biases = &layer.biases;

	matrix_dot(*weights, *input, *output);
	matrix_add(*output, *biases, *output);
	matrix_apply_activation(*output, layer.activation_fn);
}
