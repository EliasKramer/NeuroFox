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
	/*
	float* data_ptr = data_table.get_ptr_row(idx, 0);

	for (size_t i = 0; i < m.item_count(); i++)
	{
		data_ptr[i] = m.get_at_flat(i);
	}*/
	data_table.set_row_from_matrix(m, idx);
}

void data_space::set_label_in_table_at(const matrix& m, size_t idx)
{
	/*
	float* label_ptr = data_table.get_ptr_item(data_item_count(), idx, 0);

	for (size_t i = 0; i < m.item_count(); i++)
	{
		label_ptr[i] = m.get_at_flat(i);
	}
	*/
	data_table.set_row_from_matrix(m, data_item_count(), idx);
}

void data_space::if_not_initialized_throw() const
{
	if (data_table.item_count() == 0)
	{
		throw std::exception("data_space not initialized");
	}
}

data_space::data_space()
{}

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
		vector3(
			(size_t)(data_item_count() + label_item_count()),
			given_data.size(),
			(size_t)1));

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
		vector3(
			data_item_count(),
			given_data.size(),
			1));

	for (size_t i = 0; i < given_data.size(); i++)
	{
		set_data_in_table_at(given_data[i], i);
	}
}

data_space& data_space::operator=(const data_space& other)
{
	if (this != &other)
	{
		data_table = other.data_table;
		data_iterator = other.data_iterator;
		label_iterator = other.label_iterator;
		item_count = other.item_count;
		iterator_idx = other.iterator_idx;
		copied_to_gpu = other.copied_to_gpu;
		if (copied_to_gpu)
		{
			gpu_data_table = other.gpu_data_table;
			gpu_data_iterator = other.gpu_data_iterator;
			gpu_label_iterator = other.gpu_label_iterator;
		}
	}
	return *this;
}

size_t data_space::get_item_count() const
{
	return item_count;
}

void data_space::iterator_next()
{
	if_not_initialized_throw();
	iterator_idx = (iterator_idx + 1) % item_count;
}

void data_space::iterator_reset()
{
	if_not_initialized_throw();
	iterator_idx = 0;
}

bool data_space::iterator_has_next() const
{
	if_not_initialized_throw();

	return iterator_idx + 1 < item_count;
}

void data_space::copy_to_gpu()
{
	if_not_initialized_throw();

	if (copied_to_gpu)
		return;

	gpu_data_table = gpu_matrix(data_table, true);
	//no need to be copied. they get their values from gpu_data_table
	gpu_data_iterator = gpu_matrix(data_iterator, false);
	gpu_label_iterator = gpu_matrix(label_iterator, false);

	copied_to_gpu = true;
}

const matrix& data_space::get_next_data()
{
	if_not_initialized_throw();
	/*
	float* data_ptr = data_table.get_ptr_row(iterator_idx, 0);
	data_iterator.set_ptr_as_source(data_ptr);
	*/
	data_iterator.observe_row(data_table, iterator_idx);

	return data_iterator;
}

const matrix& data_space::get_next_label()
{
	if_not_initialized_throw();

	if (label_iterator.item_count() == 0)
	{
		throw std::exception("trying to access label, which is not set");
	}
	/*
	float* label_ptr = data_table.get_ptr_item(data_item_count(), iterator_idx, 0);
	label_iterator.set_ptr_as_source(label_ptr);
	*/
	label_iterator.observe_row(data_table, iterator_idx, data_item_count());

	return label_iterator;
}

//not tested
const gpu_matrix& data_space::get_next_gpu_data()
{
	if_not_initialized_throw();

	copy_to_gpu();

	float* data_ptr = gpu_data_table.get_gpu_ptr_row(iterator_idx, 0);
	gpu_data_iterator.set_gpu_ptr_as_source(data_ptr);

	return gpu_data_iterator;
}

//not tested
const gpu_matrix& data_space::get_next_gpu_label()
{
	if_not_initialized_throw();

	copy_to_gpu();

	if (gpu_label_iterator.item_count() == 0)
	{
		throw std::exception("trying to access label, which is not set");
	}

	float* label_ptr = gpu_data_table.get_gpu_ptr_item(data_item_count(), iterator_idx, 0);
	gpu_label_iterator.set_gpu_ptr_as_source(label_ptr);

	return gpu_label_iterator;
}