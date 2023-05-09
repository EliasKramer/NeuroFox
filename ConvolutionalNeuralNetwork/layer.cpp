#include "layer.hpp"

bool layer::should_use_gpu()
{
	return 
		gpu_activations != nullptr &&
		gpu_error != nullptr &&
		gpu_passing_error != nullptr;
}

layer::layer(e_layer_type_t given_layer_type)
	:type(given_layer_type)
{}

const e_layer_type_t layer::get_layer_type() const
{
	return type;
}

const matrix* layer::get_input_p() const
{
	return input;
}

void layer::set_input(const matrix* input)
{
	this->input = input;
}

void layer::set_input_format(const matrix& given_input_format)
{
	this->input_format.resize(given_input_format);
}

void layer::set_previous_layer(layer& previous_layer)
{
	input = &previous_layer.activations;
	//we are using a function here, because
	//some implementations might want to know if the input format has changed
	set_input_format(previous_layer.activations);
	passing_error = &previous_layer.error;
}

const matrix& layer::get_activations() const
{
	return activations;
}

matrix* layer::get_activations_p()
{
	return &activations;
}

void layer::set_error_for_last_layer(const matrix& expected)
{
	if (!matrix::equal_format(activations, expected))
	{
		throw std::runtime_error("setting error for the last layer could not be done. wrong expected matrix format");
	}
	//this calculates the const derivative
	matrix::subtract(activations, expected, error);
	error.scalar_multiplication(2);
}

void layer::forward_propagation()
{
	should_use_gpu() ?
		forward_propagation_gpu() :
		forward_propagation_cpu();
}

void layer::back_propagation()
{
	should_use_gpu() ?
		back_propagation_gpu() :
		back_propagation_cpu();
}
void layer::enable_gpu()
{
	gpu_activations = std::make_unique<gpu_matrix>(activations, true);
}

void layer::disable_gpu()
{
	gpu_activations = nullptr;
	gpu_error = nullptr;
	gpu_passing_error = nullptr;
}