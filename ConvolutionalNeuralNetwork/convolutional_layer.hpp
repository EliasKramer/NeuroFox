#pragma once
#include "matrix.hpp"
#include "math_functions.hpp"
#include "layer.hpp"
#include <memory>
#include <vector>

class convolutional_layer : public layer {
private:
	std::vector<matrix> kernel_weights;
	matrix kernel_biases;
	std::vector<matrix> kernel_weights_deltas;
	matrix kernel_bias_deltas;

	int kernel_size;
	int stride;
	int kernel_count;

	e_activation_t activation_fn;
public:
	//constructor
	convolutional_layer(
		int number_of_kernels,
		int kernel_size,
		int stride,
		e_activation_t activation_function
	);

	convolutional_layer(const convolutional_layer& other);

	//getters
	int get_kernel_size() const;
	int get_stride() const;
	int get_kernel_count() const;

	std::vector<matrix>& get_kernel_weights();
	const std::vector<matrix>& get_kernel_weights_readonly() const;
	matrix& get_kernel_biases();
	const matrix& get_kernel_biases_readonly() const;

	void set_input_format(vector3 input_format) override;

	//set all weights and biases to that value
	void set_all_parameter(float value) override;
	//a random value to the current weights and biases between -value and value
	void apply_noise(float range) override;
	//add a random value between range and -range to one weight or bias 
	void mutate(float range) override;

	void sync_device_and_host() override;

	void forward_propagation(const matrix& input) override;
	void back_propagation(const matrix& input, matrix* passing_error) override;

	void apply_deltas(size_t training_data_count, float learning_rate) override;

	void enable_gpu_mode() override;
	void disable_gpu() override;
};