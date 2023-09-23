#include "softmax_layer.hpp"

softmax_layer::softmax_layer(vector3 activation_format)
	: layer(activation_format, e_layer_type_t::softmax)
{
}

softmax_layer::softmax_layer(std::ifstream& file)
	: layer(file, e_layer_type_t::softmax)
{
}

softmax_layer::softmax_layer(const softmax_layer& other)
	: layer(other)
{
}

std::unique_ptr<layer> softmax_layer::clone() const
{
	return std::unique_ptr<layer>();
}

size_t softmax_layer::get_parameter_count() const
{
	return size_t();
}

void softmax_layer::set_input_format(vector3 input_format)
{
}

void softmax_layer::set_all_parameters(float value)
{
}

void softmax_layer::apply_noise(float range)
{
}

void softmax_layer::mutate(float range)
{
}

std::string softmax_layer::parameter_analysis() const
{
	return std::string();
}

void softmax_layer::forward_propagation(const matrix& input)
{
}

void softmax_layer::back_propagation(const matrix& input, matrix* passing_error)
{
}

void softmax_layer::apply_deltas(size_t training_data_count, float learning_rate)
{
}

void softmax_layer::enable_gpu_mode()
{
}

void softmax_layer::disable_gpu()
{
}

bool softmax_layer::equal_format(const layer& other)
{
	return false;
}

bool softmax_layer::equal_parameter(const layer& other)
{
	return false;
}

void softmax_layer::set_parameters(const layer& other)
{
}

void softmax_layer::write_to_ofstream(std::ofstream& file) const
{
}
