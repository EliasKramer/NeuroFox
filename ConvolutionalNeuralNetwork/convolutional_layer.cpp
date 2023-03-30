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

void feed_forward(convolutional_layer& layer)
{
	const int kernel_size = layer.kernels[0].weights.width;
	const int stride = layer.stride;
	const int number_of_kernels = layer.kernels.size();
	const int output_width = layer.output.width;
	const int output_height = layer.output.height;
	const int input_depth = layer.input->depth;

	for (int depth = 0; depth < number_of_kernels; depth++)
	{
		neural_kernel& kernel = layer.kernels[depth];
		for (int y = 0; y < output_height; y++)
		{
			for (int x = 0; x < output_width; x++)
			{
				float value = lay_kernel_over_matrix(
					*layer.input,
					kernel,
					x * stride,
					y * stride,
					kernel_size);
				set_at(layer.output, x, y, depth, value);
			}
		}
	}
	matrix_apply_activation(layer.output, layer.activation_fn);
}

convolutional_layer::convolutional_layer(matrix* input, int kernel_size, int number_of_kernels, int stride, activation activation_fn)
{
}

void convolutional_layer::set_input(matrix* input)
{
}

void convolutional_layer::set_error_right(matrix* output)
{
}

void convolutional_layer::forward_propagation()
{
}

void convolutional_layer::back_propagation()
{
}
