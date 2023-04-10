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

const matrix* nn_data::get_data_p() const
{
	return &data;
}

const matrix* nn_data::get_label_p() const
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