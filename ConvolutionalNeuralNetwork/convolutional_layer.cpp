#include "convolutional_layer.hpp"

convolutional_layer::convolutional_layer(
	matrix* input, 
	int kernel_size, 
	int number_of_kernels, 
	int stride, 
	activation activation_function
)
	:layer(input),
	stride(stride),
	kernels(),
	kernel_deltas(),
	activation_fn(activation_function)
{
	const int input_depth = input->depth;
	const int output_width = (input->width - kernel_size) / stride + 1;
	const int output_height = (input->height - kernel_size) / stride + 1;

	for (int i = 0; i < number_of_kernels; i++)
	{
		neural_kernel kernel;
		resize_matrix(kernel.weights, kernel_size, kernel_size, input_depth);
		kernels.push_back(kernel);
	}

	resize_matrix(output, output_width, output_height, number_of_kernels);
}

void convolutional_layer::forward_propagation()
{
	const int kernel_size = kernels[0].weights.width;
	const int number_of_kernels = kernels.size();
	const int output_width = output.width;
	const int output_height = output.height;
	const int input_depth = input->depth;

	for (int depth = 0; depth < number_of_kernels; depth++)
	{
		neural_kernel& kernel = kernels[depth];
		for (int y = 0; y < output_height; y++)
		{
			for (int x = 0; x < output_width; x++)
			{
				float value = lay_kernel_over_matrix(
					*input,
					kernel,
					x * stride,
					y * stride,
					kernel_size);
				set_at(output, x, y, depth, value);
			}
		}
	}
	matrix_apply_activation(output, activation_fn);
}

void convolutional_layer::back_propagation()
{
	//TODO
}