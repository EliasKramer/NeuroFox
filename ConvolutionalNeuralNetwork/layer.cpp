#include "layer.hpp"

void layer::valid_input_check_cpu(const matrix* input) const
{
	if (input == nullptr)
	{
		throw std::runtime_error("input is nullptr");
	}
	if (input_format.item_count() == 0 || // the input format is not set
		matrix::equal_format(input_format, *input) == false)
	{
		throw std::runtime_error("input format is not set or does not match the input format");
	}
}

void layer::valid_passing_error_check_cpu(const matrix* passing_error) const
{
	if (passing_error == nullptr)
	{
		throw std::runtime_error("passing_error is nullptr");
	}
	//TODO check if the passing error format is correct
}

void layer::valid_input_check_gpu(const gpu_matrix* input) const
{
	if (input == nullptr)
	{
		throw std::runtime_error("input is nullptr");
	}
	//TODO check if the input format is correct
}

void layer::valid_passing_error_check_gpu(const gpu_matrix* passing_error) const
{
	if (passing_error == nullptr)
	{
		throw std::runtime_error("passing_error is nullptr");
	}
	//TODO check if the passing error format is correct
}

layer::layer(e_layer_type_t given_layer_type)
	:type(given_layer_type)
{}

const e_layer_type_t layer::get_layer_type() const
{
	return type;
}

void layer::set_input_format(const matrix& given_input_format)
{
	this->input_format.resize(given_input_format);
}

const matrix& layer::get_activations() const
{
	return activations;
}

matrix* layer::get_activations_p()
{
	return &activations;
}

void layer::set_error_for_last_layer_cpu(const matrix& expected)
{
	if (!matrix::equal_format(activations, expected))
	{
		throw std::runtime_error("setting error for the last layer could not be done. wrong expected matrix format");
	}
	//this calculates the const derivative
	matrix::subtract(activations, expected, error);
	error.scalar_multiplication(2);
}

void layer::enable_gpu()
{
	gpu_activations = std::make_unique<gpu_matrix>(activations, true);
}

void layer::disable_gpu()
{
	gpu_activations = nullptr;
	gpu_error = nullptr;
}

void layer::forward_propagation_cpu(const matrix* input)
{
	valid_input_check_cpu(input);
}

void layer::back_propagation_cpu(const matrix* input, const matrix* passing_error)
{
	valid_input_check_cpu(input);
	valid_passing_error_check_cpu(passing_error);
}

void layer::forward_propagation_gpu(const gpu_matrix* input)
{
	valid_input_check_gpu(input);
}

void layer::back_propagation_gpu(const gpu_matrix* input, const gpu_matrix* passing_error)
{
	valid_input_check_gpu(input);
	valid_passing_error_check_gpu(passing_error);
}