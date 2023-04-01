#include "neural_network.hpp"

neural_network::neural_network()
{
}

void neural_network::set_input_format(const matrix& given_input_format)
{
	if (input_format_set == false)
		input_format_set = true;
	else
		throw "Cannot set input format twice.";

	resize_matrix(this->input_format, given_input_format);
}

void neural_network::set_output_format(const matrix& given_output_format)
{
	if (output_format_set == false)
		output_format_set = true;
	else
		throw "Cannot set output format twice.";

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
	// TODO: insert return statement here
}

void neural_network::add_layer(std::unique_ptr<layer> layer)
{
}

void neural_network::forward_propagation()
{
}

void neural_network::back_propagation(matrix* expected_output)
{
}