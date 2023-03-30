#pragma once
#include "matrix.hpp"
#include "math_functions.hpp"
#include "conv_kernel.hpp"
#include "layer.hpp"

struct _convolutional_layer : layer {

	std::vector<neural_kernel> kernels;
	std::vector<neural_kernel> kernel_deltas;

	int stride;
	activation activation_fn;
} typedef convolutional_layer;

convolutional_layer* create_convolutional_layer(
	matrix* input,
	int kernel_size,
	int number_of_kernels,
	int stride,
	activation activation_fn);

void feed_forward(convolutional_layer& layer);