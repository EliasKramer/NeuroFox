#include "convolutional_layer.hpp"

convolutional_layer::convolutional_layer(
	matrix* input, 
	const matrix& input_format,
	int kernel_size, 
	int number_of_kernels, 
	int stride, 
	e_activation_t activation_function
)
	:layer(e_layer_type_t::convolution),
	stride(stride),
	kernels(),
	kernel_deltas(),
	activation_fn(activation_function)
{
	const int input_depth = input_format.depth;
	const int output_width = (input_format.width - kernel_size) / stride + 1;
	const int output_height = (input_format.height - kernel_size) / stride + 1;

	for (int i = 0; i < number_of_kernels; i++)
	{
		neural_kernel_t kernel;
		resize_matrix(kernel.weights, kernel_size, kernel_size, input_depth);
		kernels.push_back(kernel);
	}

	resize_matrix(activations, output_width, output_height, number_of_kernels);
}

void convolutional_layer::set_input_format(const matrix& input_format)
{
	layer::set_input_format(input_format);
	//TODO
}

void convolutional_layer::set_all_parameter(float value)
{
	for (neural_kernel_t& kernel : kernels)
	{
		set_all(kernel.weights, value);
		kernel.bias = value;
	}
}

void convolutional_layer::apply_noise(float range)
{
	for (neural_kernel_t& kernel : kernels)
	{
		matrix_apply_noise(kernel.weights, range);
		kernel.bias += random_float_incl(-range, range);
	}
}

void convolutional_layer::mutate(float range)
{
	//choose a random kernel
	int random_kernel_idx = random_idx(kernels.size());
	//choose if a weight or a bias is mutated
	if (biased_coin_toss(kernels[0].weights.data.size(), 1))
	{
		//choose a random weight to mutate
		int random_weight_idx = random_idx(kernels[0].weights.data.size());
		//mutate the weight
		kernels[random_kernel_idx].weights.data[random_weight_idx] += 
			random_float_incl(-range, range);
	}
	else
	{
		//mutate the bias
		kernels[random_kernel_idx].bias += random_float_incl(-range, range);
	}
}

void convolutional_layer::forward_propagation()
{
	const int kernel_size = kernels[0].weights.width;
	const size_t number_of_kernels = kernels.size();
	const int output_width = activations.width;
	const int output_height = activations.height;
	const int input_depth = input->depth;

	for (int depth = 0; depth < number_of_kernels; depth++)
	{
		neural_kernel_t& kernel = kernels[depth];
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
				set_at(activations, x, y, depth, value);
			}
		}
	}
	matrix_apply_activation(activations, activation_fn);
}

void convolutional_layer::back_propagation()
{
	//TODO
}

void convolutional_layer::apply_deltas(int number_of_inputs)
{
	//TODO
}
