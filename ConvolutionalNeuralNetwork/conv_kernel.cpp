#include "conv_kernel.hpp"

float lay_kernel_over_matrix(
	const matrix& input_matrix,
	const neural_kernel& kernel,
	int start_x,
	int start_y,
	int kernel_size)
{
	//could be done with a matrix dot product,
	//but copying the input data into a matrix is too slow
	float sum = 0;
	for (int z = 0; z < input_matrix.depth; z++)
	{
		//we add all the values at each depth
		for (int x = 0; x < kernel_size; x++)
		{
			for (int y = 0; y < kernel_size; y++)
			{
				sum +=
					matrix_get_at(input_matrix, start_x + x, start_y + y, z) *
					matrix_get_at(kernel.weights, x, y, z);
			}
		}
	}
	return sum + kernel.bias;
}