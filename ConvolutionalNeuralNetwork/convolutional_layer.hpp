#pragma once
#include "matrix.hpp"
#include "math_functions.hpp"
#include "conv_kernel.hpp"
#include "layer.hpp"

class convolutional_layer : public layer {
private:
	std::vector<neural_kernel> kernels;
	std::vector<neural_kernel> kernel_deltas;

	int stride;
	activation activation_fn;
public:
	//constructor
	convolutional_layer(
		matrix* input,
		int kernel_size,
		int number_of_kernels,
		int stride,
		activation activation_function
	);
	void forward_propagation() override;
	void back_propagation() override;	
};