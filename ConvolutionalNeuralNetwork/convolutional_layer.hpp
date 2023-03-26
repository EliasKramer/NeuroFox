#pragma once
#include "matrix.hpp"
#include "layer.hpp"

struct _convolutional_layer {
	matrix* input;
	std::vector<neural_kernel> kernels;
	int stride;
	activation_function activation;
	matrix output;
} typedef convolutional_layer;

convolutional_layer* create_convolutional_layer(int kernel_size, int number_of_kernels, int stride, activation_function activation);
