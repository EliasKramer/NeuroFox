#pragma once
#include "matrix.hpp"
#include "math_functions.hpp"
#include "layer.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class fully_connected_layer : public layer {
private:
	matrix weights;
	matrix biases;

	matrix weight_deltas;
	matrix bias_deltas;

	matrix weight_momentum;
	matrix bias_momentum;

	matrix weight_momentum_squared;
	matrix bias_momentum_squared;

	int time_step = 1;

	e_activation_t activation_fn;

public:
	fully_connected_layer(
		size_t number_of_neurons,
		e_activation_t activation_function);

	fully_connected_layer(
		vector3 neuron_format,
		e_activation_t activation_function);

	fully_connected_layer(
		std::ifstream& file);

	fully_connected_layer(const fully_connected_layer& other);

	std::unique_ptr<layer> clone() const override;

	bool is_parameter_layer() const override;
	size_t get_parameter_count() const override;

	void set_input_format(vector3 input_format) override;

	const matrix& get_weights() const;
	const matrix& get_biases() const;
	matrix& get_weights_ref();
	matrix& get_biases_ref();

	void set_error_for_last_layer(const matrix& expected) override;

	//set all weights and biases to that value
	void set_all_parameters(float value) override;
	//a random value to the current weights and biases between -value and value
	void apply_noise(float range) override;
	//add a random value between range and -range to one weight or bias 
	void mutate(float range) override;

	std::string parameter_analysis() const override;

	void sync_device_and_host() override;

	void forward_propagation(const matrix& input) override;
	void partial_forward_prop(const matrix& input, const matrix& prev_input, const vector3& change_idx) override;
	void partial_forward_prop(const matrix& input, float new_value, const vector3& change_idx) override;
	void back_propagation(const matrix& input, matrix* passing_error) override;

	void apply_deltas(size_t training_data_count, float learning_rate) override;

	void enable_gpu_mode() override;
	void disable_gpu() override;

	bool equal_format(const layer& other) override;
	bool equal_parameter(const layer& other) override;
	void set_parameters(const layer& other) override;

	void write_to_ofstream(std::ofstream& file) const override;
};