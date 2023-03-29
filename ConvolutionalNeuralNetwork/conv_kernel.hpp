#pragma once
#include "matrix.hpp"

struct _neural_kernel {
	matrix weights;
	float bias;
	matrix output;
} typedef neural_kernel;

/// <summary>
/// sums up all the values in the input matrix multiplied by the kernel
/// </summary>
/// <param name="start_x/y">the top left position in the input matrix, 
///		where the kernel gets laid over</param>
/// <param name="kernel_size">the size of the kernel, 
///		that will be laid over the matrix</param>
/// <returns></returns>

float lay_kernel_over_matrix(
	const matrix& input_matrix,
	const neural_kernel& kernel,
	int start_x,
	int start_y,
	int kernel_size);