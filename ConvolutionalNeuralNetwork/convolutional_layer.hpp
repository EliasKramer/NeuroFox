#pragma once
#include "matrix.hpp"
#include "math_functions.hpp"
#include "conv_kernel.hpp"
#include "layer.hpp"
#include <memory>
#include <vector>

class convolutional_layer : public layer {
private:
	std::vector<neural_kernel_t> kernels;

	int kernel_size;
	int stride;

	//if i uncomment this it will crash when deleting
	//std::vector<neural_kernel_t> kernel_delta;

	e_activation_t activation_fn;
public:
	//constructor
	convolutional_layer(
		int kernel_size,
		int number_of_kernels,
		int stride,
		e_activation_t activation_function
	);

	void set_input_format(const matrix& input_format) override;

	//set all weights and biases to that value
	void set_all_parameter(float value) override;
	//a random value to the current weights and biases between -value and value
	void apply_noise(float range) override;
	//add a random value between range and -range to one weight or bias 
	void mutate(float range) override;

	void forward_propagation() override;
	void back_propagation() override;	

	void apply_deltas(int number_of_inputs) override;
};