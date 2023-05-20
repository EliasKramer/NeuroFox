#pragma once
#include "gpu_matrix.cuh"
class data_space
{
private:
	size_t item_count = 0;
	bool copied_to_gpu = false;

	/*
		One piece of data is stored in the following format:
		+-------------+-----+
		| data        |label|
		+-------------+-----+
		data and label are stored in one dimension

		the data_iterator and label_iterator are used to
		point to current data and label
		these iterators also store the format of the data and label

		data_table looks like this:
		+-------------+-----+
		| data        |label|
		+-------------+-----+
		| data        |label|
		+-------------+-----+
		| data        |label|
		+-------------+-----+
	*/
	matrix data_table;
	//gpu_data_table is made the exact same way as data_table
	//just on the gpu
	gpu_matrix gpu_data_table;

	size_t iterator_idx = 0;
	
	matrix data_iterator;
	matrix label_iterator;

	gpu_matrix gpu_data_iterator;
	gpu_matrix gpu_label_iterator;

	size_t label_item_count();
	size_t data_item_count();

	void set_data_in_table_at(const matrix& m, size_t idx);
	void set_label_in_table_at(const matrix& m, size_t idx);

public:
	data_space(
		const matrix& data_format,
		const matrix& label_format,
		const std::vector<matrix>& given_data,
		const std::vector<matrix>& given_label);

	data_space(
		const matrix& data_format,
		const std::vector<matrix>& given_data);

	size_t get_item_count() const;

	void iterator_next();

	//load in file
	//save in file

	void copy_to_gpu();

	const matrix& get_next_data();
	const matrix& get_next_label();
	
	const matrix& get_next_gpu_data();
	const matrix& get_next_gpu_label();
};