#include "neural_network.hpp"

neural_network::neural_network()
{
}

void neural_network::set_input_format(const matrix& given_input_format)
{
	if (input_format_set == false)
		input_format_set = true;
	else
		throw std::runtime_error("Cannot set input format twice.");

	resize_matrix(this->input_format, given_input_format);
}

void neural_network::set_output_format(const matrix& given_output_format)
{
	if (output_format_set == false)
		output_format_set = true;
	else
		throw std::runtime_error("Cannot set output format twice.");

	resize_matrix(this->output_format, given_output_format);
}

void neural_network::set_input(matrix* input)
{
	if (input == nullptr)
	{
		throw "Input is nullptr.";
	}

	if (input_format_set == false ||
		matrix_equal_format(input_format, *input) == false)
	{
		throw "Could not set Input. Input format is not set or does not match the input format.";
	}

	this->input_p = input;
}

const matrix& neural_network::get_output() const
{
	return *output_p;
}

static bool layer_can_be_added(
	const layer& given_layer,
	const layer* last_layer)
{
	//TODO

	return true;
}

void neural_network::add_layer(std::unique_ptr<layer>&& given_layer)
{
	const layer* last_layer = 
		layers.empty() ? nullptr : layers.back().get();

	if (layer_can_be_added(*given_layer.get(), last_layer))
	{
		layers.push_back(std::move(given_layer));
	}
	else
	{
		throw std::runtime_error("A layer could not be added");
	}
}

void neural_network::forward_propagation()
{
	//std::vector<std::unique_ptr<layer>>::iterator::value_type
	for (auto& layer : layers)
	{
		layer->forward_propagation();
	}
}

void neural_network::back_propagation(const matrix& expected_output)
{
	//TODO
}