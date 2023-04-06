#include "nn_data.hpp"

nn_data::nn_data()
{}

nn_data::nn_data(const matrix& data, const matrix& label)
	:data(data),
	label(label)
{}

const matrix& nn_data::get_data() const
{
	return data;
}

const matrix& nn_data::get_label() const
{
	return label;
}

matrix* nn_data::get_data_p()
{
	return &data;
}

matrix* nn_data::get_label_p()
{
	return &label;
}

void nn_data::set_data(const matrix& data)
{
	this->data = data;
}

void nn_data::set_label(const matrix& label)
{
	this->label = label;
}