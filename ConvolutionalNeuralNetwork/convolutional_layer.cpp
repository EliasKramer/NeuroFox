#include "convolutional_layer.hpp"

void convolutional_layer::forward_propagation_cpu()
{
	const int output_width = activations.get_width();
	const int output_height = activations.get_height();
	const int input_depth = input->get_depth();

	if (activations.get_depth() != kernel_count)
		throw std::invalid_argument("activations depth must be equal to the number of kernels");

	activations.set_all(0);

	matrix::valid_cross_correlation(
		*input, kernel_weights, activations, stride);

	matrix::add(activations, kernel_biases, activations);

	activations.apply_activation_function(activation_fn);
}

void convolutional_layer::back_propagation_cpu()
{
}

void convolutional_layer::forward_propagation_gpu()
{
	gpu_valid_cross_correlation(
		gpu_input,
		gpu_kernel_weights,
		*gpu_activations.get(),
		input->get_width(),
		input->get_depth(),
		kernel_size,
		kernel_weights.size(),
		stride,
		activations.get_width());

	gpu_add(
		*gpu_activations.get(),
		*gpu_kernel_biases.get(),
		*gpu_activations.get()
	);

	GPU_ACTIVATION[activation_fn](*gpu_activations.get());
}

void convolutional_layer::back_propagation_gpu()
{
}

convolutional_layer::convolutional_layer(
	int number_of_kernels,
	int kernel_size,
	int stride,
	e_activation_t activation_function
)
	:layer(e_layer_type_t::convolution),
	stride(stride),
	kernel_size(kernel_size),
	activation_fn(activation_function),
	kernel_count(number_of_kernels)
{
	if (number_of_kernels <= 0)
		throw std::invalid_argument("number_of_kernels must be greater than 0");
	if (kernel_size <= 0)
		throw std::invalid_argument("kernel_size must be greater than 0");
	if (stride <= 0)
		throw std::invalid_argument("stride must be greater than 0");

	if (stride > kernel_size)
		throw std::invalid_argument("stride must be smaller or equal than the kernel_size");

	for (int i = 0; i < number_of_kernels; i++)
	{
		kernel_weights.push_back(matrix(kernel_size, kernel_size, 1));
		kernel_weights_deltas.push_back(matrix(kernel_size, kernel_size, 1));
	}

	activations = matrix(0, 0, 0);
}

int convolutional_layer::get_kernel_size() const
{
	return kernel_size;
}

int convolutional_layer::get_stride() const
{
	return stride;
}

int convolutional_layer::get_kernel_count() const
{
	return kernel_count;
}

std::vector<matrix>& convolutional_layer::get_kernel_weights()
{
	return kernel_weights;
}

const std::vector<matrix>& convolutional_layer::get_kernel_weights_readonly() const
{
	return kernel_weights;
}

matrix& convolutional_layer::get_kernel_biases()
{
	return kernel_biases;
}

const matrix& convolutional_layer::get_kernel_biases_readonly() const
{
	return kernel_biases;
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

	activations.resize((int)output_width, (int)output_height, kernel_count);

	for (int i = 0; i < kernel_count; i++)
	{
		kernel_weights[i].resize(kernel_size, kernel_size, input_depth);
		kernel_weights_deltas[i].resize(kernel_size, kernel_size, input_depth);

	}
	kernel_biases.resize((int)output_width, (int)output_height, kernel_count);
	kernel_bias_deltas.resize((int)output_width, (int)output_height, kernel_count);
}

void convolutional_layer::set_all_parameter(float value)
{
	for (matrix& weights : kernel_weights)
	{
		weights.set_all(value);
	}
	kernel_biases.set_all(value);
}

void convolutional_layer::apply_noise(float range)
{
	for (matrix& weights : kernel_weights)
	{
		weights.apply_noise(range);
	}
	kernel_biases.apply_noise(range);
}

void convolutional_layer::mutate(float range)
{
	//choose if a weight or a bias is mutated
	if (biased_coin_toss(
		(float)kernel_weights[0].flat_readonly().size() * kernel_weights.size(),
		(float)kernel_biases.flat_readonly().size()))
	{
		//choose a random weight to mutate
		int random_weight_idx =
			random_idx((int)kernel_weights[0]
				.flat_readonly()
				.size());

		//mutate the weight
		kernel_weights[random_idx(kernel_weights.size())]
			.flat()[random_weight_idx] +=
			random_float_incl(-range, range);
	}
	else
	{
		kernel_biases.mutate(range);
	}
}

void convolutional_layer::apply_deltas(int number_of_inputs)
{
	//TODO
}

void convolutional_layer::enable_gpu()
{
	for (const matrix& curr : kernel_weights)
	{
		gpu_kernel_weights.emplace_back(std::make_unique<gpu_memory<float>>(curr));
	}
	gpu_kernel_biases = std::make_unique<gpu_memory<float>>(kernel_biases);
}

void convolutional_layer::disable_gpu()
{
	gpu_kernel_weights.clear();
	gpu_kernel_biases = nullptr;
}