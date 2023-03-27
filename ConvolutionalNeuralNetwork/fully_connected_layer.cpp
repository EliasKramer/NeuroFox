#include "fully_connected_layer.hpp"

fully_connected_layer* create_fully_connected_layer(
	int number_of_neurons, 
	activation activation,
	matrix* input
)
{
	fully_connected_layer* layer = new fully_connected_layer;

	layer->input = input;
	resize_matrix(layer->output, 1, number_of_neurons, 1);

	resize_matrix(layer->biases, 1, number_of_neurons, 1);
	resize_matrix(layer->weights, input->width, number_of_neurons, 1);

	layer->activation_fn = activation;

	return layer;
}

void forward_propagate_fully_connected_layer(fully_connected_layer& layer)
{
	matrix* input = layer.input;
	matrix* output = &layer.output;
	matrix* weights = &layer.weights;
	matrix* biases = &layer.biases;

	matrix_dot(*input, *weights, *output);
	matrix_add(*output, *biases, *output);
	matrix_apply_activation(*output, layer.activation_fn);

}
