#include "pooling_layer.hpp"

pooling_layer::pooling_layer(
	size_t filter_size,
	size_t stride,
	e_pooling_type_t pooling_fn
)
	:layer(e_layer_type_t::pooling),
	filter_size(filter_size),
	stride(stride),
	pooling_fn(pooling_fn)
{
	if (filter_size == 0)
		throw std::invalid_argument("filter size must be greater than 0");
	if (stride == 0)
		throw std::invalid_argument("stride must be greater than 0");
}

std::unique_ptr<layer> pooling_layer::clone() const
{
	return std::make_unique<pooling_layer>(*this);
}

size_t pooling_layer::get_parameter_count() const
{
	throw std::invalid_argument("pooling layer does not have any parameters");
}

pooling_layer::pooling_layer(const pooling_layer& other)
	:layer(other),
	filter_size(other.filter_size),
	stride(other.stride),
	pooling_fn(other.pooling_fn)
{}

void pooling_layer::set_input_format(vector3 input_format)
{
	layer::set_input_format(input_format);

	int output_width = (input_format.x - filter_size) / stride + 1;
	int output_height = (input_format.y - filter_size) / stride + 1;
	int output_depth = input_format.z;

	activations = matrix(
		vector3(
			output_width,
			output_height,
			output_depth));
}

size_t pooling_layer::get_filter_size() const
{
	return filter_size;
}

size_t pooling_layer::get_stride() const
{
	return stride;
}

e_pooling_type_t pooling_layer::get_pooling_fn() const
{
	return pooling_fn;
}

void pooling_layer::set_all_parameters(float value)
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

void pooling_layer::sync_device_and_host()
{
	layer::sync_device_and_host();
}

void pooling_layer::forward_propagation(const matrix& input)
{
	layer::forward_propagation(input);

	//iterate over each depth
	for (size_t d = 0; d < activations.get_depth(); d++)
	{
		//iterate over each row of the output
		for (size_t y = 0; y < activations.get_height(); y++)
		{
			//calculate the start and end index of the filter on the y axis
			const size_t start_idx_y = y * stride;
			const size_t end_idx_y = start_idx_y + filter_size;

			for (size_t x = 0; x < activations.get_width(); x++)
			{
				//calculate the start and end index of the filter on the x axis
				const size_t start_idx_x = x * stride;
				const size_t end_idx_x = start_idx_x + filter_size;

				//calculating the max, min, and average values
				//this could be improved by only calculating one of these values
				float max = FLT_MIN;
				float min = FLT_MAX;
				float sum = 0;

				//iterate over the filter
				for (size_t i = start_idx_y; i <= end_idx_y; i++)
				{
					if (i >= input.get_height())
						break;

					for (size_t j = start_idx_x; j <= end_idx_x; j++)
					{
						if (j >= input.get_width())
							break;

						//get the value of the input at the current index
						const float curr_val = input.get_at_host(vector3(j, i, d));

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
					activations.set_at(vector3(x, y, d), max);
					break;
				case min_pooling:
					activations.set_at(vector3(x, y, d), min);
					break;
				case average_pooling:
					activations.set_at(vector3(x, y, d), sum / (filter_size * filter_size));
					break;
				default:
					throw std::runtime_error("Invalid pooling type");
					break;
				}
			}
		}
	}
}

void pooling_layer::back_propagation(const matrix& input, matrix* passing_error)
{
	layer::back_propagation(input, passing_error);
	throw std::exception("not implemented");
}
/*
void pooling_layer::forward_propagation_gpu(const gpu_matrix& input)
{
	layer::forward_propagation_gpu(input);
	throw std::exception("not implemented");
}

void pooling_layer::back_propagation_gpu(const gpu_matrix& input, gpu_matrix* passing_error)
{
	layer::back_propagation_gpu(input, passing_error);
	throw std::exception("not implemented");
}
*/

void pooling_layer::apply_deltas(size_t training_data_count, float learning_rate)
{
	throw std::invalid_argument("pooling layer does not have any parameters");
}

void pooling_layer::enable_gpu_mode()
{
	gpu_enabled = true;
	throw std::runtime_error("pooling layer has no implementation of enable gpu");
}

void pooling_layer::disable_gpu()
{
	gpu_enabled = false;
	throw std::runtime_error("pooling layer has no implementation of disable gpu");
}

bool pooling_layer::equal_format(const layer& other)
{
	if (layer::equal_format(other))
	{
		const pooling_layer& other_cast = dynamic_cast<const pooling_layer&>(other);

		return filter_size == other_cast.filter_size &&
			stride == other_cast.stride &&
			pooling_fn == other_cast.pooling_fn;
	}

	return false;
}

bool pooling_layer::equal_parameter(const layer& other)
{
	return equal_format(other);
}
