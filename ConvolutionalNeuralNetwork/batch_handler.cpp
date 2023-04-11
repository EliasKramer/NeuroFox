#include "batch_handler.hpp"

batch_handler::batch_handler(
	const std::vector<std::unique_ptr<nn_data>>& input, 
	int batch_size
)
	: input_vector(input),
	batch_size(batch_size),
	batch_end_idx(batch_size)
{
	if(batch_size <= 0)
		throw std::invalid_argument("batch_size must be greater than 0");
	if(batch_size > input_vector.size())
		throw std::invalid_argument("batch_size must be less than or equal to the size of the input vector");
	if(batch_start_idx > batch_end_idx)
		throw std::invalid_argument("batch_start_idx must be less than or equal to batch_end_idx");
}

const void batch_handler::calculate_new_batch()
{
	if (batch_end_idx + batch_size >= input_vector.size())
	{
		batch_start_idx = 0;
		batch_end_idx = batch_size;
		return;
	}

	batch_start_idx = batch_end_idx;
	batch_end_idx += batch_size;
}

const std::vector<std::unique_ptr<nn_data>>::const_iterator batch_handler::get_batch_start() const
{
	return input_vector.cbegin() + batch_start_idx;
}

const std::vector<std::unique_ptr<nn_data>>::const_iterator batch_handler::get_batch_end() const
{
	return input_vector.cbegin() + batch_end_idx;
}