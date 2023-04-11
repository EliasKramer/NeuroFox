#pragma once
#include "nn_data.hpp"
#include <vector>
class batch_handler
{
private:
	const std::vector<std::unique_ptr<nn_data>>& input_vector;
	int batch_size = 0;
	int batch_start_idx = 0;
	int batch_end_idx;
public:
	batch_handler(
		const std::vector<std::unique_ptr<nn_data>>& input, 
		int batch_size);

	const void calculate_new_batch();
	const std::vector<std::unique_ptr<nn_data>>::const_iterator get_batch_start() const;
	const std::vector<std::unique_ptr<nn_data>>::const_iterator get_batch_end() const;
};