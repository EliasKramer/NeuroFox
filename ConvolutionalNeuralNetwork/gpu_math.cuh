#pragma once
#include <stdexcept>
#include "matrix.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/// <summary>
/// performs a dot product on the gpu
/// the input is the vector A
/// the weights are the matrix B
/// the activations are the vector C
/// B has to be have the width of A
/// B has to have the height of C
/// 
/// tho it is only checked if B's size is A's size * C's size
/// since the gpu_memory class stores the data in one dimension
/// </summary>
void gpu_dot_product(
	const matrix& gpu_weights,
	const matrix& gpu_input,
	matrix& gpu_activations);

/// <summary>
/// adds the values of two gpu memory objects
/// these will be stored in the result object
/// //all have to be the same size
/// </summary>
void gpu_add(
	const matrix& gpu_memory_a,
	const matrix& gpu_memory_b,
	matrix& gpu_memory_result);

/// <summary>
/// performs a valid cross correlation
/// this is done by laying the kernels over the input one by one
/// and multiply overlaying values and summing them up
/// then the kernel will be moved by the stride and the process will be repeated
/// 
/// the kernels have to have the same depth as the input
/// the output will have the depth of the amount of kernels that exist
/// </summary>
void gpu_valid_cross_correlation(
	const matrix& gpu_input,
	const std::vector<matrix>& gpu_kernel_weights,
	matrix& gpu_activations,
	size_t input_width,
	size_t input_depth,
	size_t kernel_width,
	size_t kernel_count,
	size_t stride,
	size_t output_width);

/*
	activation functions
	performs a function that has one input and one output
	for example relu where x = max(0, x)
*/
using gpu_activation_fn = void(*)(matrix&);
void gpu_sigmoid(matrix& gpu_memory);
void gpu_relu(matrix& gpu_memory);

//this has to have the same indexing as the ACTIVATION function pointer array
const gpu_activation_fn GPU_ACTIVATION[] = {
	gpu_sigmoid,
	gpu_relu
};