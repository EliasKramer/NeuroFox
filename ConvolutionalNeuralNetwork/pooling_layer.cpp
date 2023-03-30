#include "pooling_layer.hpp"

pooling_layer* create_pooling_layer(
	matrix* input,
	int filter_size,
	int stride,
	pooling_type pooling_fn)
{
	if(input == nullptr)
		throw "input cannot be null";
	if(filter_size <= 0)
		throw "filter size must be greater than 0";
	if(stride <= 0)
		throw "stride must be greater than 0";

	pooling_layer* layer = new pooling_layer;

	layer->input = input;
	layer->filter_size = filter_size;
	layer->stride = stride;
	layer->pooling_fn = pooling_fn;

	int output_width = (input->width - filter_size) / stride + 1;
	int output_height = (input->height - filter_size) / stride + 1;
	int output_depth = input->depth;

	resize_matrix(
		layer->output,
		output_width,
		output_height,
		output_depth);

	return layer;
}

void feed_forward(pooling_layer& layer)
{
	const matrix* input = layer.input;
	matrix* output = &layer.output;
	const int filter_size = layer.filter_size;
	const int stride = layer.stride;
	const pooling_type type = layer.pooling_fn;

	const int output_width = output->width;
	const int output_height = output->height;
	const int output_depth = output->depth;

	//iterate over each depth
	for (int d = 0; d < output_depth; d++)
	{
		//iterate over each row of the output
		for (int y = 0; y < output_height; y++)
		{
			//calculate the start and end index of the filter on the y axis
			const int start_idx_y = y * stride;
			const int end_idx_y = start_idx_y + filter_size;

			for (int x = 0; x < output_width; x++)
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
					if(i >= input->height)
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

				switch (type)
				{
				case max_pooling:
					set_at(*output, x, y, d, max);
					break;
				case min_pooling:
					set_at(*output, x, y, d, min);
					break;
				case average_pooling:
					set_at(*output, x, y, d, sum / (filter_size * filter_size));
					break;
				default:
					throw "Invalid pooling type";
					break;
				}
			}
		}
	}
}

pooling_layer::pooling_layer(matrix* input, int filter_size, int stride, pooling_type pooling_fn)
{
}

void pooling_layer::set_input(matrix* input)
{
}

void pooling_layer::set_error_right(matrix* output)
{
}

void pooling_layer::forward_propagation()
{
}

void pooling_layer::back_propagation()
{
}
