#pragma once
#include "gpu_matrix.cuh"
class data_space
{
private:
	size_t item_count = 0;
	bool copied_to_gpu = false;

	matrix data;
	gpu_matrix gpu_data;
	
	matrix data_iterator;
	matrix label_iterator;

	gpu_matrix gpu_data_iterator;
	gpu_matrix gpu_label_iterator;

public:
	data_space(
		const matrix& data_format,
		const matrix& label_format,
		const std::vector<matrix>& given_data,
		const std::vector<matrix>& given_label);

	data_space(
		const matrix& data_format,
		const matrix& label_format,
		const std::vector<matrix>& given_data);

	//load in file
	//save in file

	void copy_to_gpu();

	const matrix& get_next_data();
	const matrix& get_next_label();
	
	const matrix& get_next_gpu_data();
	const matrix& get_next_gpu_label();
};