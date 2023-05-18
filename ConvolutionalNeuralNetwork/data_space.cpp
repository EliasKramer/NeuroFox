#include "data_space.hpp"

data_space::data_space(
	const matrix& data_format,
	const matrix& label_format,
	const std::vector<matrix>& given_data,
	const std::vector<matrix>& given_label
) :
	data_iterator(data_format),
	label_iterator(label_format)
{
	if (given_data.size() == 0)
	{
		throw std::exception("no data given");
	}
	if (given_data.size() != given_label.size())
	{
		throw std::exception("data and label size mismatch");
	}
}

data_space::data_space(
	const matrix& data_format,
	const matrix& label_format,
	const std::vector<matrix>& given_data
) :
	data_iterator(data_format),
	label_iterator(label_format)
{
	if (given_data.size() == 0)
	{
		throw std::exception("no data given");
	}
}

void data_space::copy_to_gpu()
{
	if (copied_to_gpu)
		return;
	//gpu_data = gpu_matrix(data);
	//gpu_data_iterator = gpu_matrix(data_iterator);
	copied_to_gpu = true;
}

const matrix& data_space::get_next_data()
{
	// TODO: insert return statement here
	throw std::exception("not implemented");
}

const matrix& data_space::get_next_label()
{
	// TODO: insert return statement here
	throw std::exception("not implemented");
}

const matrix& data_space::get_next_gpu_data()
{
	copy_to_gpu();
	// TODO: insert return statement here
	throw std::exception("not implemented");

}

const matrix& data_space::get_next_gpu_label()
{
	copy_to_gpu();
	// TODO: insert return statement here
	throw std::exception("not implemented");
}
