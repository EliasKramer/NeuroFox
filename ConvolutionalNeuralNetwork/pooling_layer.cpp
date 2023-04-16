#include "pooling_layer.hpp"

pooling_layer::pooling_layer(
	int filter_size,
	int stride,
	e_pooling_type_t pooling_fn
)
	:layer(e_layer_type_t::pooling),
	filter_size(filter_size),
	stride(stride),
	pooling_fn(pooling_fn)
{
	if (filter_size <= 0)
		throw std::invalid_argument("filter size must be greater than 0");
	if (stride <= 0)
		throw std::invalid_argument("stride must be greater than 0");
}

void pooling_layer::set_input_format(const matrix& input_format)
{
	layer::set_input_format(input_format);
	
	int output_width = (input_format.get_width() - filter_size) / stride + 1;
	int output_height = (input_format.get_height() - filter_size) / stride + 1;
	int output_depth = input_format.get_depth();

	activations.resize_matrix(
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

void pooling_layer::set_all_parameter(float value)
{
	throw std::invalid_argument("pooling layer does not have any parameters");
}

void pooling_layer::apply_noise(float range)
{
	throw std::invalid_argument("pooling layer does not have any parameters");
}

void pooling_layer::mutate(float range)
{
	throw std::invalid_argument("pooling layer does not have any parameters");
}

void pooling_layer::forward_propagation()
{

	//iterate over each depth
	for (int d = 0; d < activations.get_depth(); d++)
	{
		//iterate over each row of the output
		for (int y = 0; y < activations.get_height(); y++)
		{
			//calculate the start and end index of the filter on the y axis
			const int start_idx_y = y * stride;
			const int end_idx_y = start_idx_y + filter_size;

			for (int x = 0; x < activations.get_width(); x++)
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
					if (i >= input->get_height())
						break;

					for (int j = start_idx_x; j <= end_idx_x; j++)
					{
						if (j >= input->get_width())
							break;

						//get the value of the input at the current index
						const float curr_val = input->matrix_get_at(j, i, d);

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
					activations.set_at(x, y, d, max);
					break;
				case min_pooling:
					activations.set_at(x, y, d, min);
					break;
				case average_pooling:
					activations.set_at(x, y, d, sum / (filter_size * filter_size));
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

void pooling_layer::apply_deltas(int number_of_inputs)
{
	throw std::invalid_argument("pooling layer does not have any parameters");
}
