#pragma once
#include "matrix.hpp"
#include "math_functions.hpp"
#include "conv_kernel.hpp"

struct _convolutional_layer {
	matrix* input;
	std::vector<neural_kernel> kernels;
	int stride;
	activation activation;
	matrix output;
} typedef convolutional_layer;

convolutional_layer* create_convolutional_layer(
	int kernel_size, 
	int number_of_kernels, 
	int stride, 
	activation activation);
