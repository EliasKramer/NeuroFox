#pragma once
#include "matrix.hpp"
#include "math_functions.hpp"
#include "conv_kernel.hpp"
#include "layer.hpp"

class convolutional_layer : public layer {
private:
	std::vector<neural_kernel_t> kernels;
	std::vector<neural_kernel_t> kernel_deltas;

	int stride;
	e_activation_t activation_fn;
public:
	//constructor
	convolutional_layer(
		matrix* input,
		const matrix& input_format,
		int kernel_size,
		int number_of_kernels,
		int stride,
		e_activation_t activation_function
	);

	//set all weights and biases to that value
	void set_all_parameter(float value) override;
	//a random value to the current weights and biases between -value and value
	void apply_noise(float range) override;
	//add a random value between range and -range to one weight or bias 
	void mutate(float range) override;

	void forward_propagation() override;
	void back_propagation() override;	
};