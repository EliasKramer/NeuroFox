#include "convolutional_layer.hpp"

convolutional_layer* create_convolutional_layer(
	matrix* input,
    int kernel_size, 
    int number_of_kernels, 
    int stride, 
    activation activation_fn)
{
    convolutional_layer* layer = new convolutional_layer;

	layer->input = input;
	const int input_depth = input->depth;
	layer->stride = stride;
	layer->activation_fn = activation_fn;

	const int output_width = (input->width - kernel_size) / stride + 1;
	const int output_height = (input->height - kernel_size) / stride + 1;

	for (int i = 0; i < number_of_kernels; i++)
	{
		neural_kernel kernel;
		resize_matrix(kernel.weights, kernel_size, kernel_size, input_depth);
		layer->kernels.push_back(kernel);
	}

	resize_matrix(layer->output, output_width, output_height, number_of_kernels);

	return layer;
}