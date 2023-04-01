#include "layer.hpp"
#include "layer.hpp"

layer::layer(matrix* input)
	:input(input),
	error_right(nullptr),
	layer_type(_layer_type::NO_TYPE)
{}

void layer::set_input(matrix* input)
{
	if (!input)
	{
		throw "Input matrix cannot be null";
	}
	if (input->width != 1 || input->depth != 1)
	{
		throw "Input matrix must be a vector (width and depth must be 1)";
	}
	this->input = input;
}

void layer::set_error_right(matrix* error_right)
{
	if (!error_right)
	{
		throw "Error matrix cannot be null";
	}
	if (error_right->width != 1 || error_right->depth != 1)
	{
		throw "Error matrix must be a vector (width and depth must be 1)";
	}
	this->error_right = error_right;
}

const matrix& layer::get_output() const
{
	return output;
}
