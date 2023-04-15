#include "conv_kernel.hpp"

conv_kernel::conv_kernel(int kernel_size)
	:weights(matrix(kernel_size, kernel_size, 0))
{}

void conv_kernel::set_kernel_depth(int depth)
{
	weights.resize_matrix(get_kernel_size(), get_kernel_size(), depth);
}

matrix& conv_kernel::get_weights()
{
	return weights;
}

const matrix& conv_kernel::get_weights_readonly() const
{
	return weights;
}

float conv_kernel::get_bias()
{
	return bias;
}

void conv_kernel::set_bias(float given_bias)
{
	bias = given_bias;
}

size_t conv_kernel::get_kernel_size() const
{
	return weights.get_width();
}

float conv_kernel::lay_kernel_over_matrix(const matrix& input_matrix, int start_x, int start_y, int kernel_size)
{
	//could be done with a matrix dot product,
	//but copying the input data into a matrix is too slow
	float sum = 0;
	for (int z = 0; z < input_matrix.get_depth(); z++)
	{
		//we add all the values at each depth
		for (int x = 0; x < kernel_size; x++)
		{
			for (int y = 0; y < kernel_size; y++)
			{
				sum +=
					input_matrix.matrix_get_at(start_x + x, start_y + y, z) *
					weights.matrix_get_at(x, y, z);
			}
		}
	}
	return sum + bias;
}
