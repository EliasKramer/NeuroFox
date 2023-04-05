#include "pooling_layer.hpp"

pooling_layer::pooling_layer(
	matrix* input, 
	int filter_size, 
	int stride, 
	e_pooling_type_t pooling_fn
)
	:layer(input, e_layer_type_t::pooling),
	filter_size(filter_size),
	stride(stride),
	pooling_fn(pooling_fn)
{
	if (!input)
		throw std::invalid_argument("input cannot be null");
	if (filter_size <= 0)
		throw std::invalid_argument("filter size must be greater than 0");
	if (stride <= 0)
		throw std::invalid_argument("stride must be greater than 0");

	int output_width = (input->width - filter_size) / stride + 1;
	int output_height = (input->height - filter_size) / stride + 1;
	int output_depth = input->depth;

	resize_matrix(
		activations,
		output_width,
		output_height,
		output_depth);
}

int pooling_layer::get_filter_size() const
{
	return filter_size;
}

int pooling_layer::get_stride() const
{
	return stride;
}

e_pooling_type_t pooling_layer::get_pooling_fn() const
{
	return pooling_fn;
}

void pooling_layer::forward_propagation()
{

	//iterate over each depth
	for (int d = 0; d < activations.depth; d++)
	{
		//iterate over each row of the output
		for (int y = 0; y < activations.height; y++)
		{
			//calculate the start and end index of the filter on the y axis
			const int start_idx_y = y * stride;
			const int end_idx_y = start_idx_y + filter_size;

			for (int x = 0; x < activations.width; x++)
			{
				//calculate the start and end index of the filter on the x axis
				const int start_idx_x = x * stride;
				const int end_idx_x = start_idx_x + filter_size;

				//calculating the max, min, and average values
				//this could be improved by only calculating one of these values
				float max = FLT_MIN;
				float min = FLT_MAX;
				float sum = 0;

				//iterate over the filter
				for (int i = start_idx_y; i <= end_idx_y; i++)
				{
					if (i >= input->height)
						break;

					for (int j = start_idx_x; j <= end_idx_x; j++)
					{
						if (j >= input->width)
							break;

						//get the value of the input at the current index
						const float curr_val = matrix_get_at(*input, j, i, d);

						//if the current value is greater than the max value
						//set the max value to the current value
						if (curr_val > max)
						{
							max = curr_val;
						}
						if (curr_val < min)
						{
							min = curr_val;
						}
						sum += curr_val;
					}
				}

				switch (pooling_fn)
				{
				case max_pooling:
					set_at(activations, x, y, d, max);
					break;
				case min_pooling:
					set_at(activations, x, y, d, min);
					break;
				case average_pooling:
					set_at(activations, x, y, d, sum / (filter_size * filter_size));
					break;
				default:
					throw std::runtime_error("Invalid pooling type");
					break;
				}
			}
		}
	}
}

void pooling_layer::back_propagation()
{
	//TODO
}
