#include "layer.hpp"

void layer::valid_input_check_cpu(const matrix& input) const
{
	if (input_format.item_count() == 0 || // the input format is not set
		matrix::equal_format(input_format, input) == false)
	{
		throw std::runtime_error("input format is not set or does not match the input format");
	}
}

void layer::valid_passing_error_check_cpu(const matrix* passing_error) const
{
	if (passing_error == nullptr)
	{
		//a passing error is null when the layer is the very first in the network
		return;
	}
	//the passing error has the same format as the input format
	valid_input_check_cpu(*passing_error);
}

layer::layer(e_layer_type_t given_layer_type)
: type(given_layer_type)
{}

layer::layer(
	vector3 activation_format,
	e_layer_type_t given_layer_type
) :
	type(given_layer_type),
	activations(activation_format),
	error(activation_format)
{}

const e_layer_type_t layer::get_layer_type() const
{
	return type;
}

void layer::set_input_format(vector3 given_input_format)
{
	this->input_format = given_input_format;
}

const matrix& layer::get_activations() const
{
	return activations;
}

matrix* layer::get_activations_p()
{
	return &activations;
}

const matrix& layer::get_error() const
{
	return error;
}

matrix* layer::get_error_p()
{
	return &error;
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
	activations.enable_gpu();
	error.enable_gpu();
	
	//input_format.enable_gpu();
	//gpu_activations = std::make_unique<gpu_matrix>(activations, true);
}

void layer::disable_gpu()
{
	//gpu_activations = nullptr;
	//gpu_error = nullptr;
}

void layer::forward_propagation(const matrix& input)
{
	valid_input_check_cpu(input);
}

void layer::back_propagation(const matrix& input, matrix* passing_error)
{
	valid_input_check_cpu(input);
	valid_passing_error_check_cpu(passing_error);
}
/*
void layer::forward_propagation_gpu(const gpu_matrix& input)
{
	valid_input_check_gpu(input);
}

void layer::back_propagation_gpu(const gpu_matrix& input, gpu_matrix* passing_error)
{
	valid_input_check_gpu(input);
	valid_passing_error_check_gpu(passing_error);
}
*/