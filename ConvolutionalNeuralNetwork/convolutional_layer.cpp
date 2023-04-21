#include "convolutional_layer.hpp"

convolutional_layer::convolutional_layer(
	int number_of_kernels,
	int kernel_size,
	int stride,
	e_activation_t activation_function
)
	:layer(e_layer_type_t::convolution),
	stride(stride),
	kernel_size(kernel_size),
	activation_fn(activation_function)
{
	if (number_of_kernels <= 0)
		throw std::invalid_argument("number_of_kernels must be greater than 0");
	if (kernel_size <= 0)
		throw std::invalid_argument("kernel_size must be greater than 0");
	if (stride <= 0)
		throw std::invalid_argument("stride must be greater than 0");

	if (stride > kernel_size)
		throw std::invalid_argument("stride must be smaller or equal than the kernel_size");

	if (stride != 1)
		throw std::invalid_argument("stride must be 1 in convolutional layer - this version can only handle a stride of 1");

	for (int i = 0; i < number_of_kernels; i++)
	{
		kernels.push_back(conv_kernel(kernel_size));
		kernel_deltas.push_back(conv_kernel(kernel_size));
	}

	activations = matrix(0, 0, 0);
}

const std::vector<conv_kernel>& convolutional_layer::get_kernels_readonly() const
{
	return kernels;
}

std::vector<conv_kernel>& convolutional_layer::get_kernels()
{
	return kernels;
}

int convolutional_layer::get_kernel_size() const
{
	return kernel_size;
}

int convolutional_layer::get_stride() const
{
	return stride;
}

void convolutional_layer::set_input_format(const matrix& input_format)
{
	layer::set_input_format(input_format);

	const int input_depth = input_format.get_depth();

	const float output_width =
		(input_format.get_width() - kernel_size) / (float)stride + 1;

	const float output_height =
		(input_format.get_height() - kernel_size) / (float)stride + 1;

	if (!is_whole_number(output_width) ||
		!is_whole_number(output_height))
		throw std::invalid_argument("input format is not compatible with the kernel size and stride");

	activations.resize((int)output_width, (int)output_height, (int)kernels.size());
	
	for (int i = 0; i < kernels.size(); i++)
	{
		kernels[i].set_kernel_depth(input_depth);
		kernel_deltas[i].set_kernel_depth(input_depth);
		//BE WARY. THIS ONLY WORKS FOR A STRIDE OF 1
		kernels[i].set_bias_format(output_width);
		kernel_deltas[i].set_bias_format(output_width);
	}
}

void convolutional_layer::set_all_parameter(float value)
{
	for (conv_kernel& kernel : kernels)
	{
		kernel.get_weights().set_all(value);
	}
}

void convolutional_layer::apply_noise(float range)
{
	for (conv_kernel& kernel : kernels)
	{
		kernel.get_weights().apply_noise(range);
		kernel.get_bias().apply_noise(range);
	}
}

void convolutional_layer::mutate(float range)
{
	//choose a random kernel
	int random_kernel_idx = random_idx((int)kernels.size());
	//choose if a weight or a bias is mutated
	if (biased_coin_toss(
		(float)kernels[0].get_weights_readonly().flat_readonly().size(),
		(float)kernels[0].get_bias_readonly().flat_readonly().size()))
	{
		//choose a random weight to mutate
		int random_weight_idx =
			random_idx((int)kernels[0]
				.get_weights_readonly()
				.flat_readonly()
				.size());

		//mutate the weight
		kernels[random_kernel_idx]
			.get_weights()
			.flat()[random_weight_idx] +=
			random_float_incl(-range, range);
	}
	else
	{
		//choose a random bias to mutate
		int random_bias_idx =
			random_idx((int)kernels[0]
				.get_bias_readonly()
				.flat_readonly()
				.size());

		//mutate the weight
		kernels[random_kernel_idx]
			.get_bias()
			.flat()[random_bias_idx] +=
			random_float_incl(-range, range);
	}
}

void convolutional_layer::forward_propagation()
{
	const size_t kernel_size = kernels[0].get_kernel_size();
	const size_t number_of_kernels = kernels.size();
	const int output_width = activations.get_width();
	const int output_height = activations.get_height();
	const int input_depth = input->get_depth();

	if (activations.get_depth() != number_of_kernels)
		throw std::invalid_argument("activations depth must be equal to the number of kernels");

	activations.set_all(0);
	//iterate over all kernels
	for (int output_depth = 0; output_depth < number_of_kernels; output_depth++)
	{
		const conv_kernel& curr_kernel = kernels[output_depth];
		const matrix& kernel_bias = kernels[output_depth].get_bias();

		activations.set_all(0);
		matrix::valid_cross_correlation(
			*input, curr_kernel.get_weights_readonly(), activations);
		matrix::add(activations, kernel_bias, activations);
	}

	activations.apply_activation_function(activation_fn);
}

void convolutional_layer::back_propagation()
{
	//TODO
}

void convolutional_layer::apply_deltas(int number_of_inputs)
{
	//TODO
}
