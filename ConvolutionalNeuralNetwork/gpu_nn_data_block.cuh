#pragma once

#include "gpu_memory.cuh"
#include "gpu_math.cuh"
#include "nn_data.hpp"

//this is just a bunch of memory on the gpu
//it looks like that:
//+---------+-----+---------+-----+---------+-----+---------+-----+
//|	 data	|label|	 data	|label|	 data	|label|	 data	|label|
//+---------+-----+---------+-----+---------+-----+---------+-----+
class gpu_nn_data_block {
private:
	std::unique_ptr<gpu_memory<float>> data;
	size_t num_of_blocks;

	size_t num_of_label_data;
	size_t num_of_data;

	size_t elements_in_block() const;

public:
	gpu_nn_data_block(
		size_t num_of_blocks, 
		size_t num_of_data,
		size_t num_of_label_data);

	size_t get_num_of_blocks() const;
	size_t get_num_of_label_data() const;
	size_t get_num_of_data() const;

	float* get_gpu_data_ptr(int idx);
	float* get_gpu_label_ptr(int idx);

	void set_data(int idx, const std::vector<float>& data);
	void set_label_data(int idx, const std::vector<float>& data);

	//set data with std::vector<nn_data> const iterators
	void set_data(
		std::vector<nn_data>::const_iterator begin,
		std::vector<nn_data>::const_iterator end);
};