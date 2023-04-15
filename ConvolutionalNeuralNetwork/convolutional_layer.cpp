#include "convolutional_layer.hpp"

convolutional_layer::convolutional_layer(
	int kernel_size,
	int number_of_kernels,
	int stride,
	e_activation_t activation_function
)
	:layer(e_layer_type_t::convolution),
	stride(stride),
	kernel_size(kernel_size),
	activation_fn(activation_function)
{
	for (int i = 0; i < number_of_kernels; i++)
	{
		//TODO delta
		//kernels.push_back(conv_kernel(kernel_size));
	}

	activations = matrix(0, 0, 0);
}

void convolutional_layer::set_input_format(const matrix& input_format)
{
	layer::set_input_format(input_format);

	const int input_depth = input_format.get_depth();
	const int output_width = (input_format.get_width() - kernel_size) / stride + 1;
	const int output_height = (input_format.get_height() - kernel_size) / stride + 1;

	/*
	for (int i = 0; i < kernels.size(); i++)
	{
		kernels[i].set_kernel_depth(input_depth);
		//resize delta
	}
	*/

	//resize_matrix(activations, output_width, output_height, (int)kernels.size());
}

void convolutional_layer::set_all_parameter(float value)
{
	/*
	for (conv_kernel& kernel : kernels)
	{
		set_all(kernel.get_weights(), value);
	}
	*/
}

void convolutional_layer::apply_noise(float range)
{
	/*
	for (conv_kernel& kernel : kernels)
	{
		matrix_apply_noise(kernel.get_weights(), range);

		kernel.set_bias(
			kernel.get_bias() +
			random_float_incl(-range, range));
	}
	*/
}

void convolutional_layer::mutate(float range)
{
	/*
	//choose a random kernel
	int random_kernel_idx = random_idx((int)kernels.size());
	//choose if a weight or a bias is mutated
	if (biased_coin_toss((float)kernels[0].get_weights_readonly().data.size(), 1))
	{
		//choose a random weight to mutate
		int random_weight_idx = random_idx((int)kernels[0].get_weights_readonly().data.size());
		//mutate the weight

		kernels[random_kernel_idx].get_weights().data[random_weight_idx] +=
			random_float_incl(-range, range);
	}
	else
	{
		//mutate the bias

		kernels[random_kernel_idx].set_bias(
			kernels[random_kernel_idx].get_bias() +
			random_float_incl(-range, range));
	}
	*/
}

void convolutional_layer::forward_propagation()
{
	/*
	const size_t kernel_size = kernels[0].get_kernel_size();
	const size_t number_of_kernels = kernels.size();
	const int output_width = activations.width;
	const int output_height = activations.height;
	const int input_depth = input->depth;

	for (int depth = 0; depth < number_of_kernels; depth++)
	{
		conv_kernel& kernel = kernels[depth];
		for (int y = 0; y < output_height; y++)
		{
			for (int x = 0; x < output_width; x++)
			{
				float value = kernel.lay_kernel_over_matrix(
					*input,
					x * stride,
					y * stride,
					(int)kernel_size);
				set_at(activations, x, y, depth, value);
			}
		}
	}
	matrix_apply_activation(activations, activation_fn);
	*/
}

void convolutional_layer::back_propagation()
{
	//TODO
}

void convolutional_layer::apply_deltas(int number_of_inputs)
{
	//TODO
}
