#pragma once
/*
#include "gpu_matrix.cuh"
#include "gpu_math.cuh"
#include "nn_data.hpp"

//this is just a bunch of memory on the gpu
//it looks like that:
//+---------+-----+
//|	 data	|label|
//+---------+-----+
//|	 data	|label|
//+---------+-----+
//|	 data	|label|
//+---------+-----+
class gpu_nn_data_block {
private:
	std::unique_ptr<gpu_matrix> data;

	size_t labels_per_block;
	size_t data_per_block;

	size_t elements_in_block() const;
public:
	gpu_nn_data_block(
		size_t num_of_blocks,
		size_t data_per_block,
		size_t labels_per_block);

	size_t blocks() const;
	size_t label_itmes_per_block() const;
	size_t data_items_per_block() const;
	size_t block_count() const;

	float* get_gpu_data_ptr(int idx);
	float* get_gpu_label_ptr(int idx);

	void set_data(int idx, const std::vector<float>& data);
	void set_label_data(int idx, const std::vector<float>& data);

	//set data with std::vector<nn_data> const iterators
	void set_data(
		std::vector<nn_data>::const_iterator begin,
		std::vector<nn_data>::const_iterator end);
};
*/