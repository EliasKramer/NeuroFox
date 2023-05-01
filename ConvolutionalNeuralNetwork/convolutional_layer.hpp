#pragma once
#include "matrix.hpp"
#include "math_functions.hpp"
#include "conv_kernel.hpp"
#include "layer.hpp"
#include <memory>
#include <vector>

class convolutional_layer : public layer {

private:
	std::vector<conv_kernel> kernels;
	std::vector<conv_kernel> kernel_deltas;

	int kernel_size;
	int stride;

	e_activation_t activation_fn;
public:
	//constructor
	convolutional_layer(
		int number_of_kernels,
		int kernel_size,
		int stride,
		e_activation_t activation_function
	);

	//getters
	const std::vector<conv_kernel>& get_kernels_readonly() const;
	std::vector<conv_kernel>& get_kernels();
	int get_kernel_size() const;
	int get_stride() const;

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

	void enable_gpu() override;
	void disable_gpu() override;
};