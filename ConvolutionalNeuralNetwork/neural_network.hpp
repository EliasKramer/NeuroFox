#pragma once
#include <chrono>
#include "matrix.hpp"
#include "math_functions.hpp"
#include "pooling_layer.hpp"
#include "fully_connected_layer.hpp"
#include "convolutional_layer.hpp"
#include "nn_data.hpp"
#include "batch_handler.hpp"
#include "test_result.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class neural_network {
private:
	matrix input_format;

	std::vector<std::unique_ptr<layer>> layers;
	//saves the indices of all layers tha have parameter
	//convolutional and fully connected 
	std::vector<int> parameter_layer_indices;

	layer* get_last_layer();

	void add_layer(std::unique_ptr<layer>&& given_layer);

	//we need the training_data_count for 
	//calculating the average of the deltas
	void apply_deltas(int training_data_count);

	float calculate_cost(const matrix& expected_output);

	bool gpu_enabled = false;
public:

	neural_network();

	//sets the input matrix to a certain format
	void set_input_format(const matrix& given_input_format);

	const matrix& get_output() const;

	void add_fully_connected_layer(int num_neurons, e_activation_t activation_fn);
	void add_fully_connected_layer(const matrix& neuron_format, e_activation_t activation_fn);
	
	void add_convolutional_layer(
		int number_of_kernels, 
		int kernel_size, 
		int stride, 
		e_activation_t activation_fn);
	void add_pooling_layer(int kernel_size, int stride, e_pooling_type_t pooling_type);

	//set all weights and biases to that value
	void set_all_parameter(float value);
	//a random value to the current weights and biases between -value and value
	void apply_noise(float range);
	//add a random value between range and -range to one weight or bias 
	void mutate(float range);

	//test_result test(const std::vector<std::unique_ptr<nn_data>>& training_data);
	void forward_propagation_cpu(const matrix& input);
	void forward_propagation_gpu(const gpu_matrix& input);

	void learn(const std::vector<std::unique_ptr<nn_data>>& training_data, int batch_size, int epochs);
	void learn_once(const std::unique_ptr<nn_data>& expected_output, bool apply_changes = true);

	void enable_gpu();
};