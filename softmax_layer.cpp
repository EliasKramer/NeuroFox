#include "softmax_layer.hpp"

softmax_layer::softmax_layer()
	: layer(e_layer_type_t::softmax)
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
	return std::make_unique<softmax_layer>(*this);
}

bool softmax_layer::is_parameter_layer() const
{
	return false;
}

size_t softmax_layer::get_parameter_count() const
{
	return size_t();
}

void softmax_layer::set_input_format(vector3 input_format)
{
	layer::set_input_format(input_format);

	activations = matrix(input_format);
	error = matrix(input_format);
}

void softmax_layer::set_error_for_last_layer(const matrix& expected)
{
	matrix::cross_entropy(activations, expected, error);
}

void softmax_layer::set_all_parameters(float value)
{
	throw std::logic_error("softmax has no parameter");
}

void softmax_layer::apply_noise(float range)
{
	throw std::logic_error("softmax has no parameter");
}

void softmax_layer::mutate(float range)
{
	throw std::logic_error("softmax has no parameter");
}

std::string softmax_layer::parameter_analysis() const
{
	return "softmax layer\n";
}

void softmax_layer::forward_propagation(const matrix& input)
{
	matrix::softmax(input, activations);
}

void softmax_layer::partial_forward_prop(const matrix& input, const matrix& prev_input, const vector3& change_idx)
{
	throw std::logic_error("softmax has no parameter");
}

void softmax_layer::back_propagation(const matrix& input, matrix* passing_error)
{
	if (passing_error == nullptr)
	{
		throw std::invalid_argument("softmax should not be the first layer");
	}

	for (int i = 0; i < error.item_count(); i++)
	{
		passing_error->set_at_flat_host(i, error.get_at_flat_host(i));
	}
}

void softmax_layer::apply_deltas(size_t training_data_count, float learning_rate)
{
	throw std::logic_error("softmax has no parameter");
}

void softmax_layer::enable_gpu_mode()
{
	throw std::runtime_error("not implemented");
}

void softmax_layer::disable_gpu()
{
	throw std::runtime_error("not implemented");
}

bool softmax_layer::equal_format(const layer& other)
{
	return layer::equal_format(other);
}

bool softmax_layer::equal_parameter(const layer& other)
{
	return layer::equal_format(other);
}

void softmax_layer::set_parameters(const layer& other)
{
	throw std::logic_error("softmax has no parameter");
}

void softmax_layer::write_to_ofstream(std::ofstream& file) const
{
	layer::write_to_ofstream(file);
}
