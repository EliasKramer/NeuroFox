#include "data_space.hpp"

size_t data_space::label_item_count()
{
	return label_iterator.item_count();
}

size_t data_space::data_item_count()
{
	return data_iterator.item_count();
}

void data_space::set_data_in_table_at(const matrix& m, size_t idx)
{
	float* data_ptr = data_table.get_ptr_row(idx, 0);
	
	for (size_t i = 0; i < m.item_count(); i++)
	{
		data_ptr[i] = m.get_at_flat(i);
	}
}

void data_space::set_label_in_table_at(const matrix& m, size_t idx)
{
	float * label_ptr = data_table.get_ptr_item(data_item_count(), idx, 0);
	
	for (size_t i = 0; i < m.item_count(); i++)
	{
		label_ptr[i] = m.get_at_flat(i);
	}
}

data_space::data_space(
	const matrix& data_format,
	const matrix& label_format,
	const std::vector<matrix>& given_data,
	const std::vector<matrix>& given_label
) :
	data_iterator(data_format),
	label_iterator(label_format),
	item_count(given_data.size())
{
	if (given_data.size() == 0)
	{
		throw std::exception("no data given");
	}
	if (given_data.size() != given_label.size())
	{
		throw std::exception("data and label size mismatch");
	}

	data_table = matrix(
		data_item_count() + label_item_count(),
		given_data.size(),
		1);

	for (size_t i = 0; i < given_data.size(); i++)
	{
		set_data_in_table_at(given_data[i], i);
		set_label_in_table_at(given_label[i], i);
	}
}

data_space::data_space(
	const matrix& data_format,
	const std::vector<matrix>& given_data
) :
	data_iterator(data_format),
	item_count(given_data.size())
{
	if (given_data.size() == 0)
	{
		throw std::exception("no data given");
	}

	data_table = matrix(
		data_item_count(),
		given_data.size(),
		1);

	for (size_t i = 0; i < given_data.size(); i++)
	{
		set_data_in_table_at(given_data[i], i);
	}
}

size_t data_space::get_item_count() const
{
	return item_count;
}

void data_space::iterator_next()
{
	iterator_idx = (iterator_idx + 1) % item_count;
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
	float* data_ptr = data_table.get_ptr_row(iterator_idx, 0);
	data_iterator.set_ptr_as_source(data_ptr);

	return data_iterator;
}

const matrix& data_space::get_next_label()
{
	if (label_iterator.item_count() == 0)
	{
		throw std::exception("trying to access label, which is not set");
	}

	float* label_ptr = data_table.get_ptr_item(data_item_count(), iterator_idx, 0);
	label_iterator.set_ptr_as_source(label_ptr);

	return label_iterator;
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
